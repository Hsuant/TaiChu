"""Whole-document exact and MinHash-LSH near-duplicate clustering."""
from __future__ import annotations

import sqlite3
from collections.abc import Callable

from .config import DedupConfig, PretrainingSource
from .fingerprinting import DocumentFingerprint
from .hashing import (
    band_keys,
    digest_text,
    jaccard,
    shingle_hashes,
    signature_bytes,
)


class DocumentDeduplicator:
    def __init__(
        self,
        database: sqlite3.Connection,
        config: DedupConfig,
        *,
        seed: int,
    ) -> None:
        self.database = database
        self.config = config
        self.seed = seed
        self.parents: list[str] = []
        self.priorities: list[int] = []
        self._union_parent: list[int] = []
        self._union_rank: list[int] = []
        self._exact: dict[bytes, int] = {}
        self._candidate_shingles: dict[int, bytes] = {}
        self.candidate_documents = 0
        self.jaccard_checks = 0
        self.max_candidates = 0

    @classmethod
    def restore(
        cls,
        database: sqlite3.Connection,
        config: DedupConfig,
        *,
        seed: int,
        progress: Callable[[int], None] | None = None,
        load_exact: bool = True,
    ) -> "DocumentDeduplicator":
        result = cls(database, config, seed=seed)
        size = int(
            database.execute(
                "SELECT COALESCE(MAX(id), -1) + 1 FROM documents"
            ).fetchone()[0]
        )
        result.parents = [""] * size
        result.priorities = [0] * size
        result._union_parent = list(range(size))
        result._union_rank = [0] * size
        columns = "id, parent_document_id, source_priority"
        if load_exact:
            columns += ", text_sha"
        for row in database.execute(
            f"SELECT {columns} FROM documents ORDER BY id"
        ):
            document_id = int(row["id"])
            result.parents[document_id] = str(row["parent_document_id"])
            result.priorities[document_id] = int(row["source_priority"])
            if load_exact:
                result._exact.setdefault(bytes(row["text_sha"]), document_id)
            if progress is not None and (document_id + 1) % 1000 == 0:
                progress(1000)
        if progress is not None and size % 1000:
            progress(size % 1000)
        for row in database.execute(
            """
            SELECT canonical_id, duplicate_id
            FROM duplicate_edges ORDER BY duplicate_id
            """
        ):
            result._union(int(row["duplicate_id"]), int(row["canonical_id"]))
        return result

    @property
    def document_count(self) -> int:
        return len(self.parents)

    def _find(self, document_id: int) -> int:
        root = document_id
        while self._union_parent[root] != root:
            root = self._union_parent[root]
        while self._union_parent[document_id] != document_id:
            parent = self._union_parent[document_id]
            self._union_parent[document_id] = root
            document_id = parent
        return root

    def _union(self, left: int, right: int) -> None:
        left_root = self._find(left)
        right_root = self._find(right)
        if left_root == right_root:
            return
        if self._union_rank[left_root] < self._union_rank[right_root]:
            left_root, right_root = right_root, left_root
        self._union_parent[right_root] = left_root
        if self._union_rank[left_root] == self._union_rank[right_root]:
            self._union_rank[left_root] += 1

    def _record_edge(
        self,
        canonical_id: int,
        duplicate_id: int,
        reason: str,
        similarity: float,
    ) -> None:
        self.database.execute(
            """
            INSERT OR REPLACE INTO duplicate_edges(
                canonical_id, duplicate_id, reason, similarity
            ) VALUES (?, ?, ?, ?)
            """,
            (canonical_id, duplicate_id, reason, float(similarity)),
        )

    def _near_candidates(
        self,
        buckets: tuple[tuple[int, bytes], ...],
        document_id: int,
    ) -> set[int]:
        candidates: set[int] = set()
        if buckets:
            predicate = " OR ".join(
                "(band = ? AND bucket = ?)" for _ in buckets
            )
            parameters = tuple(
                value
                for band, bucket in buckets
                for value in (band, bucket)
            )
            for row in self.database.execute(
                f"SELECT doc_id FROM lsh_buckets WHERE {predicate}",
                parameters,
            ):
                candidates.add(int(row["doc_id"]))
        candidates.discard(document_id)
        self.candidate_documents += len(candidates)
        self.max_candidates = max(self.max_candidates, len(candidates))
        return candidates

    def add_document(
        self,
        *,
        parent_document_id: str,
        source: PretrainingSource,
        source_priority: int,
        source_path: str,
        source_index: int,
        language: str,
        text: str,
        raw_characters: int,
        raw_tokens: int,
        fingerprint: DocumentFingerprint | None = None,
    ) -> int:
        document_id = len(self.parents)
        fingerprint = fingerprint or DocumentFingerprint(
            digest=digest_text(text),
            shingles=shingle_hashes(text, self.config.ngram_size),
            signature=b"",
        )
        if not fingerprint.signature:
            fingerprint = DocumentFingerprint(
                digest=fingerprint.digest,
                shingles=fingerprint.shingles,
                signature=signature_bytes(
                    fingerprint.shingles,
                    num_hashes=self.config.num_hashes,
                    seed=self.seed,
                ),
            )
        digest = fingerprint.digest
        shingles = fingerprint.shingles
        signature = fingerprint.signature
        buckets = tuple(
            band_keys(
                signature,
                bands=self.config.bands,
                rows=self.config.rows,
            )
        )
        self.parents.append(parent_document_id)
        self.priorities.append(source_priority)
        self._union_parent.append(document_id)
        self._union_rank.append(0)

        exact_candidate = self._exact.get(digest)
        candidates = (
            self._near_candidates(buckets, document_id)
            if self.config.enabled and exact_candidate is None
            else set()
        )
        self.database.execute(
            """
            INSERT INTO documents(
                id, parent_document_id, source_name, source_priority,
                source_path, source_index, language, text, text_sha,
                signature, shingles, raw_characters, raw_tokens
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                document_id,
                parent_document_id,
                source.name,
                source_priority,
                source_path,
                source_index,
                language,
                text,
                digest,
                b"",
                b"",
                raw_characters,
                raw_tokens,
            ),
        )

        if exact_candidate is not None:
            self._union(document_id, exact_candidate)
            self._record_edge(exact_candidate, document_id, "exact", 1.0)
        else:
            candidate_shingles: dict[int, bytes] = {}
            ordered_candidates = sorted(candidates)
            self.jaccard_checks += len(ordered_candidates)
            for start in range(0, len(ordered_candidates), 500):
                chunk = ordered_candidates[start : start + 500]
                placeholders = ",".join("?" for _ in chunk)
                for row in self.database.execute(
                    f"""
                    SELECT id, text, shingles FROM documents
                    WHERE id IN ({placeholders})
                    """,
                    chunk,
                ):
                    candidate_id = int(row["id"])
                    payload = self._candidate_shingles.get(candidate_id)
                    if payload is None:
                        payload = bytes(row["shingles"])
                        if not payload:
                            payload = shingle_hashes(
                                str(row["text"]),
                                self.config.ngram_size,
                            )
                        if len(self._candidate_shingles) >= 4096:
                            self._candidate_shingles.clear()
                        self._candidate_shingles[candidate_id] = payload
                    candidate_shingles[candidate_id] = payload
            for candidate_id in ordered_candidates:
                candidate_shingles_bytes = candidate_shingles.get(candidate_id)
                if candidate_shingles_bytes is None:
                    continue
                similarity = jaccard(
                    shingles,
                    candidate_shingles_bytes,
                )
                if similarity >= self.config.similarity_threshold:
                    self._union(document_id, candidate_id)
                    self._record_edge(
                        candidate_id,
                        document_id,
                        "near",
                        similarity,
                    )
        self._exact.setdefault(digest, document_id)
        if self.config.enabled:
            self.database.executemany(
                """
                INSERT OR IGNORE INTO lsh_buckets(band, bucket, doc_id)
                VALUES (?, ?, ?)
                """,
                [
                    (band, bucket, document_id)
                    for band, bucket in buckets
                ],
            )
        return document_id

    def canonical_document_ids(
        self,
        progress: Callable[[int], None] | None = None,
    ) -> dict[int, int]:
        best: dict[int, int] = {}
        for document_id in range(len(self.parents)):
            root = self._find(document_id)
            current = best.get(root)
            candidate = (self.priorities[document_id], document_id)
            if current is None or candidate < (
                self.priorities[current],
                current,
            ):
                best[root] = document_id
            if progress is not None and (document_id + 1) % 1000 == 0:
                progress(1000)
        if progress is not None and len(self.parents) % 1000:
            progress(len(self.parents) % 1000)
        return {
            document_id: best[self._find(document_id)]
            for document_id in range(len(self.parents))
        }


__all__ = ["DocumentDeduplicator"]
