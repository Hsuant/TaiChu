"""Atomic document-aware pretraining dataset preparation."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from tokenizers import Tokenizer
from tqdm import tqdm

from ...storage.fs import atomic_replace_directory
from ..adapters import TextTokenizer
from ..cleaning import TextCleaner
from ..readers import RecordSourceReader
from .config import PretrainingConfig, PretrainingSource
from .deduplication import DocumentDeduplicator
from .fingerprinting import (
    FINGERPRINT_VERSION,
    DocumentFingerprint,
    compute_fingerprint,
    compute_fingerprints,
)
from .hashing import cluster_id, split_for_cluster
from .output import verify_split_isolation, write_splits
from .reporting import build_manifest
from .store import get_state, open_store, record_error, set_state
from .tokenization import initialize_tokenizer_worker, tokenize_batch


PROCESSING_VERSION = "document-aware"


def _process_batch(
    raw_texts: list[str],
    texts: list[str],
    *,
    ngram_size: int,
    num_hashes: int,
    seed: int,
    tokenizer: TextTokenizer | None = None,
) -> tuple[list[DocumentFingerprint], list[int]]:
    fingerprints = compute_fingerprints(
        texts,
        ngram_size=ngram_size,
        num_hashes=num_hashes,
        seed=seed,
    )
    raw_tokens = [
        len(token_ids)
        for token_ids in tokenize_batch(raw_texts, tokenizer=tokenizer)
    ]
    return fingerprints, raw_tokens


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parent_document_id(
    source: PretrainingSource,
    path_key: str,
    ordinal: int,
    record: Mapping[str, Any],
) -> str:
    record_id = record.get("document_id", record.get("id", ordinal))
    payload = (
        f"{source.name}\0{path_key}\0{ordinal}\0{record_id}"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _tokenizer_version(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()[:24]


class PretrainingDataPreparation:
    """Prepare raw documents into immutable pretraining binary datasets."""

    def __init__(self, config: PretrainingConfig) -> None:
        if not config.tokenizer_path.is_file():
            raise FileNotFoundError(f"tokenizer does not exist: {config.tokenizer_path}")
        self.config = config
        self.tokenizer = TextTokenizer(
            Tokenizer.from_file(str(config.tokenizer_path)),
            pad_token=config.pad_token,
            bos_token=config.bos_token,
            eos_token=config.eos_token,
        )
        if self.tokenizer.bos_token_id is None or self.tokenizer.eos_token_id is None:
            raise ValueError("pretraining preparation requires BOS and EOS")
        self.cleaner = TextCleaner(config.cleaning)
        self.reader = RecordSourceReader()
        self.tokenizer_version = _tokenizer_version(config.tokenizer_path)

    def run(self, *, restart: bool = False) -> dict[str, Any]:
        output = self.config.output_dir
        staging = output.with_name(f".{output.name}.staging")
        if output.exists():
            raise FileExistsError(f"pretraining output already exists: {output}")
        if restart and staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True, exist_ok=True)

        print(f"opening staging database: {staging}", flush=True)
        database = open_store(staging)
        print("staging database ready", flush=True)
        started_at = _now()
        try:
            deduplicator = self._scan(database)
            self._finalize_clusters(database, deduplicator, staging)
            split_manifests = write_splits(
                config=self.config,
                tokenizer=self.tokenizer,
                tokenizer_version=self.tokenizer_version,
                database=database,
                document_count=deduplicator.document_count,
                staging=staging,
                processing_version=PROCESSING_VERSION,
            )
            leakage = verify_split_isolation(staging, split_manifests)
            manifest = build_manifest(
                config=self.config,
                tokenizer=self.tokenizer,
                tokenizer_version=self.tokenizer_version,
                database=database,
                document_count=deduplicator.document_count,
                split_manifests=split_manifests,
                leakage=leakage,
                started_at=started_at,
                staging=staging,
                processing_version=PROCESSING_VERSION,
            )
            (staging / "dataset_manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            database.close()
            database = None
            (staging / "documents.sqlite3").unlink(missing_ok=True)
            atomic_replace_directory(staging, output)
            return manifest
        finally:
            if database is not None:
                database.close()

    def _scan(self, database) -> DocumentDeduplicator:
        scan_complete = get_state(database, "scan_complete") == "1"
        existing_documents = int(
            database.execute(
                "SELECT COALESCE(MAX(id), -1) + 1 FROM documents"
            ).fetchone()[0]
        )
        with tqdm(
            total=existing_documents,
            desc="restore dedup index",
            unit="doc",
            dynamic_ncols=True,
        ) as progress:
            deduplicator = DocumentDeduplicator.restore(
                database,
                self.config.dedup,
                seed=self.config.splits.seed,
                progress=progress.update,
                load_exact=not scan_complete,
            )
        stored_fingerprint_version = get_state(database, "fingerprint_version")
        if deduplicator.document_count and (
            stored_fingerprint_version != FINGERPRINT_VERSION
        ):
            raise RuntimeError(
                "staging directory uses an older fingerprint algorithm; "
                "rerun with --restart"
            )
        if stored_fingerprint_version is None:
            set_state(database, "fingerprint_version", FINGERPRINT_VERSION)
        if scan_complete:
            return deduplicator

        workers = min(self.config.processing_workers, os.cpu_count() or 1)
        executor = (
            ProcessPoolExecutor(
                max_workers=workers,
                initializer=initialize_tokenizer_worker,
                initargs=(
                    str(self.config.tokenizer_path),
                    self.config.pad_token,
                    self.config.bos_token,
                    self.config.eos_token,
                ),
            )
            if workers > 1
            else None
        )
        pending: deque[tuple[Future, list[dict[str, Any]]]] = deque()

        def submit_batch(batch: list[dict[str, Any]]) -> None:
            raw_texts = [item["raw_text"] for item in batch]
            texts = [item["text"] for item in batch]
            if executor is None:
                result = _process_batch(
                    raw_texts,
                    texts,
                    ngram_size=self.config.dedup.ngram_size,
                    num_hashes=self.config.dedup.num_hashes,
                    seed=self.config.splits.seed,
                    tokenizer=self.tokenizer,
                )
                future = Future()
                future.set_result(result)
                pending.append((future, batch))
                return
            future = executor.submit(
                _process_batch,
                raw_texts,
                texts,
                ngram_size=self.config.dedup.ngram_size,
                num_hashes=self.config.dedup.num_hashes,
                seed=self.config.splits.seed,
            )
            pending.append((future, batch))

        def consume_batch(progress=None) -> None:
            future, batch = pending.popleft()
            try:
                fingerprints, raw_token_counts = future.result()
            except Exception:
                fingerprints = [
                    compute_fingerprint(
                        item["text"],
                        ngram_size=self.config.dedup.ngram_size,
                        num_hashes=self.config.dedup.num_hashes,
                        seed=self.config.splits.seed,
                    )
                    for item in batch
                ]
                raw_token_counts = [
                    len(token_ids)
                    for token_ids in self.tokenizer.encode_batch(
                        [item["raw_text"] for item in batch]
                    )
                ]
            for item, fingerprint, raw_tokens in zip(
                batch,
                fingerprints,
                raw_token_counts,
                strict=True,
            ):
                deduplicator.add_document(
                    parent_document_id=item["parent_id"],
                    source=item["source"],
                    source_priority=item["priority"],
                    source_path=item["path"],
                    source_index=item["ordinal"],
                    language=item["source"].language,
                    text=item["text"],
                    raw_characters=item["raw_characters"],
                    raw_tokens=raw_tokens,
                    fingerprint=fingerprint,
                )
                if deduplicator.document_count % 1000 == 0:
                    database.commit()
            if progress is not None:
                progress.set_postfix(
                    documents=deduplicator.document_count,
                    pending_batches=len(pending),
                    candidates=deduplicator.candidate_documents,
                    jaccard=deduplicator.jaccard_checks,
                    max_candidates=deduplicator.max_candidates,
                )

        def drain_pending(progress=None) -> None:
            while pending:
                consume_batch(progress)

        try:
            for priority, source in enumerate(self.config.sources):
                for path in self.reader.resolve(source.paths):
                    batch: list[dict[str, Any]] = []
                    path_key = str(path.resolve())
                    total_bytes = path.stat().st_size
                    with tqdm(
                        total=total_bytes,
                        desc=f"scan {source.name}/{path.name}",
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1000,
                        dynamic_ncols=True,
                    ) as progress:
                        for ordinal, record in enumerate(
                            self.reader.read(
                                path,
                                progress_callback=progress.update,
                            )
                        ):
                            parent_id = _parent_document_id(
                                source,
                                path_key,
                                ordinal,
                                record,
                            )
                            if database.execute(
                                "SELECT 1 FROM documents WHERE parent_document_id = ?",
                                (parent_id,),
                            ).fetchone():
                                continue
                            try:
                                raw = record[self.config.text_column]
                                if not isinstance(raw, str):
                                    raise TypeError("text is not a string")
                                cleaned, reason = self.cleaner.clean(raw)
                                if cleaned is None:
                                    raise ValueError(reason)
                            except Exception as exc:
                                record_error(
                                    database,
                                    source=source,
                                    path=path,
                                    record_index=ordinal,
                                    stage="validate",
                                    reason=f"{type(exc).__name__}: {exc}",
                                )
                                if self.config.invalid_policy == "error":
                                    raise
                                continue
                            batch.append(
                                {
                                    "parent_id": parent_id,
                                    "source": source,
                                    "priority": priority,
                                    "path": str(path),
                                    "ordinal": ordinal,
                                    "text": cleaned,
                                    "raw_text": raw,
                                    "raw_characters": len(raw),
                                }
                            )
                            if len(batch) >= self.config.fingerprint_batch_size:
                                submit_batch(batch)
                                batch = []
                            while len(pending) >= workers * 2:
                                consume_batch()
                            progress.set_postfix(
                                documents=deduplicator.document_count,
                                pending_batches=len(pending),
                                candidates=deduplicator.candidate_documents,
                                jaccard=deduplicator.jaccard_checks,
                                max_candidates=deduplicator.max_candidates,
                            )
                    if batch:
                        submit_batch(batch)
                    drain_pending(progress)
                    database.commit()
            set_state(database, "scan_complete", "1")
            return deduplicator
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)

    def _finalize_clusters(
        self,
        database,
        deduplicator: DocumentDeduplicator,
        staging: Path,
    ) -> None:
        if get_state(database, "clusters_complete") == "1":
            return
        cluster_artifact = staging / "duplicate_clusters.jsonl"
        if (
            get_state(database, "clusters_assigned") != "1"
            and cluster_artifact.exists()
        ):
            # Older pipelines created this artifact only after cluster assignment.
            set_state(database, "clusters_assigned", "1")
        if get_state(database, "clusters_assigned") != "1":
            print("resolving canonical documents", flush=True)
            with tqdm(
                total=deduplicator.document_count,
                desc="resolve clusters",
                unit="doc",
                dynamic_ncols=True,
            ) as progress:
                canonical = deduplicator.canonical_document_ids(
                    progress=progress.update,
                )
            updates: list[tuple[str, str, int, int]] = []

            def flush_updates() -> None:
                if not updates:
                    return
                database.executemany(
                    """
                    UPDATE documents
                    SET cluster_id = ?, split = ?, canonical_id = ?
                    WHERE id = ?
                    """,
                    updates,
                )
                updates.clear()

            with tqdm(
                total=deduplicator.document_count,
                desc="assign clusters",
                unit="doc",
                dynamic_ncols=True,
            ) as progress:
                for document_id in range(deduplicator.document_count):
                    canonical_id = canonical[document_id]
                    cluster = cluster_id(deduplicator.parents[canonical_id])
                    split = split_for_cluster(
                        cluster,
                        validation_ratio=self.config.splits.validation_ratio,
                        diagnostic_ratio=self.config.splits.diagnostic_ratio,
                        seed=self.config.splits.seed,
                    )
                    updates.append((cluster, split, canonical_id, document_id))
                    progress.update()
                    if len(updates) >= 10_000:
                        flush_updates()
                    if document_id and document_id % 50_000 == 0:
                        database.commit()
            flush_updates()
            database.commit()
            set_state(database, "clusters_assigned", "1")
        print(
            "writing duplicate clusters: reading and sorting duplicate documents",
            flush=True,
        )
        self._write_clusters(database, cluster_artifact)
        set_state(database, "clusters_complete", "1")

    @staticmethod
    def _write_clusters(database, path: Path) -> None:
        current = None
        rows = []
        temporary = path.with_name(f".{path.name}.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            result = database.execute(
                """
                WITH clustered_document_ids(id) AS (
                    SELECT duplicate_id FROM duplicate_edges
                    UNION
                    SELECT canonical_id FROM duplicate_edges
                )
                SELECT d.id, d.parent_document_id, d.source_name, d.language,
                       d.cluster_id, d.split, d.canonical_id,
                       e.reason, e.similarity
                FROM clustered_document_ids c
                JOIN documents d ON d.id = c.id
                LEFT JOIN duplicate_edges e ON e.duplicate_id = d.id
                ORDER BY d.cluster_id, d.source_name, d.id
                """
            )
            with tqdm(
                desc="write duplicate clusters",
                unit="doc",
                dynamic_ncols=True,
            ) as progress:
                processed = 0
                for index, row in enumerate(result, start=1):
                    if current is not None and row["cluster_id"] != current:
                        PretrainingDataPreparation._emit_cluster(stream, rows)
                        rows = []
                    current = row["cluster_id"]
                    rows.append(row)
                    processed = index
                    if index % 1000 == 0:
                        progress.update(1000)
                if processed % 1000:
                    progress.update(processed % 1000)
            if current is not None:
                PretrainingDataPreparation._emit_cluster(stream, rows)
        temporary.replace(path)

    @staticmethod
    def _emit_cluster(stream, rows: list) -> None:
        if len(rows) <= 1:
            return
        canonical = next(
            row for row in rows if int(row["id"]) == int(row["canonical_id"])
        )
        payload = {
            "cluster_id": canonical["cluster_id"],
            "canonical_document_id": canonical["parent_document_id"],
            "split": canonical["split"],
            "documents": [
                {
                    "parent_document_id": row["parent_document_id"],
                    "source_id": row["source_name"],
                    "language": row["language"],
                    "canonical": int(row["id"]) == int(row["canonical_id"]),
                    "reason": row["reason"],
                    "similarity": row["similarity"],
                }
                for row in rows
            ],
        }
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")


__all__ = ["PROCESSING_VERSION", "PretrainingDataPreparation"]
