"""Whole-document exact and MinHash-LSH near-duplicate clustering."""
from __future__ import annotations

import sqlite3
from array import array
from collections.abc import Callable, Iterable, Mapping

import numpy as np

from .config import DedupConfig, PretrainingSource
from .fingerprinting import DocumentFingerprint
from .hashing import (
    band_keys,
    digest_text,
    jaccard,
    shingle_hashes,
    signature_bytes,
)


_BLOOM_BITS_PER_SLICE = 1 << 29
_BLOOM_SLICES = 4
_BLOOM_DOCUMENT_THRESHOLD = 50_000


class _BucketBloom:
    """Vectorized Bloom filter for LSH bucket keys."""

    def __init__(self) -> None:
        self._bits = [
            np.zeros(_BLOOM_BITS_PER_SLICE, dtype=np.bool_)
            for _ in range(_BLOOM_SLICES)
        ]
        self._mask = _BLOOM_BITS_PER_SLICE - 1
        self._salt = np.array(
            (0x9E3779B1, 0x85EBCA6B, 0xC2B2AE35, 0x27D4EB2F),
            dtype=np.uint32,
        )

    def _positions(
        self,
        bands: np.ndarray,
        bucket_payload: bytes,
    ) -> np.ndarray:
        positions = np.frombuffer(
            bucket_payload,
            dtype="<u4",
        ).reshape(-1, _BLOOM_SLICES).astype(np.uint32, copy=True)
        positions ^= bands.astype(np.uint32, copy=False)[:, None] * self._salt
        positions &= self._mask
        return positions

    def add_many(
        self,
        bands: np.ndarray,
        bucket_payload: bytes,
    ) -> None:
        positions = self._positions(bands, bucket_payload)
        for index, bits in enumerate(self._bits):
            bits[positions[:, index]] = True

    def add(self, band: int, bucket: bytes) -> None:
        self.add_many(
            np.array([band], dtype=np.uint32),
            bucket,
        )

    def might_contain(self, band: int, bucket: bytes) -> bool:
        positions = self._positions(
            np.array([band], dtype=np.uint32),
            bucket,
        )[0]
        return all(
            bool(bits[positions[index]])
            for index, bits in enumerate(self._bits)
        )

    def might_contain_any(self, buckets: tuple[tuple[int, bytes], ...]) -> bool:
        if not buckets:
            return False
        return bool(self.filter_pairs(list(buckets)))

    def filter_pairs(
        self,
        pairs: list[tuple[int, bytes]],
    ) -> list[tuple[int, bytes]]:
        if not pairs:
            return []
        bands = np.fromiter(
            (band for band, _ in pairs),
            dtype=np.uint32,
            count=len(pairs),
        )
        payload = b"".join(bucket for _, bucket in pairs)
        positions = self._positions(bands, payload)
        present = np.ones(len(pairs), dtype=np.bool_)
        for index, bits in enumerate(self._bits):
            present &= bits[positions[:, index]]
        return [
            pair
            for pair, candidate in zip(pairs, present, strict=True)
            if candidate
        ]

    def build(self, database: sqlite3.Connection) -> None:
        cursor = database.execute("SELECT band, bucket FROM lsh_buckets")
        while True:
            rows = cursor.fetchmany(100_000)
            if not rows:
                break
            bands = np.fromiter(
                (int(row[0]) for row in rows),
                dtype=np.uint32,
                count=len(rows),
            )
            payload = b"".join(bytes(row[1]) for row in rows)
            self.add_many(bands, payload)


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
        self.priorities = array("I")
        self._union_parent = array("I")
        self._union_rank = bytearray()
        self._exact: dict[bytes, int] = {}
        self._candidate_shingles: dict[int, bytes] = {}
        self._bucket_bloom: _BucketBloom | None = None
        self._pending_bucket_rows: list[tuple[int, bytes, int]] = []
        self._pending_bucket_docs: dict[tuple[int, bytes], list[int]] = {}
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
        load_bloom: bool = True,
    ) -> "DocumentDeduplicator":
        result = cls(database, config, seed=seed)
        size = int(
            database.execute(
                "SELECT COALESCE(MAX(id), -1) + 1 FROM documents"
            ).fetchone()[0]
        )
        result.parents = [""] * size
        result.priorities = array("I", [0]) * size
        result._union_parent = array("I", range(size))
        result._union_rank = bytearray(size)
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
        if (
            load_bloom
            and result.config.enabled
            and size >= _BLOOM_DOCUMENT_THRESHOLD
        ):
            result._ensure_bucket_bloom()
        return result

    def _ensure_bucket_bloom(self) -> None:
        if not self.config.enabled or self._bucket_bloom is not None:
            return
        print("building in-memory LSH bucket filter", flush=True)
        bloom = _BucketBloom()
        bloom.build(self.database)
        self._bucket_bloom = bloom

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
        self.add_documents(
            [
                {
                    "parent_id": parent_document_id,
                    "source": source,
                    "priority": source_priority,
                    "path": source_path,
                    "ordinal": source_index,
                    "language": language,
                    "text": text,
                    "raw_characters": raw_characters,
                    "raw_tokens": raw_tokens,
                    "fingerprint": fingerprint,
                }
            ]
        )
        return document_id

    def add_documents(
        self,
        documents: Iterable[Mapping[str, object]],
    ) -> None:
        items = list(documents)
        if not items:
            return
        batch_start = len(self.parents)
        if (
            self.config.enabled
            and self._bucket_bloom is None
            and batch_start >= _BLOOM_DOCUMENT_THRESHOLD
        ):
            self._ensure_bucket_bloom()
        prepared: list[dict[str, object]] = []
        for item in items:
            text = str(item["text"])
            raw_fingerprint = item.get("fingerprint")
            fingerprint = (
                raw_fingerprint
                if isinstance(raw_fingerprint, DocumentFingerprint)
                else None
            )
            fingerprint = fingerprint or DocumentFingerprint(
                digest=digest_text(text),
                shingles=b"",
                signature=b"",
            )
            if not fingerprint.signature:
                signature_shingles = fingerprint.shingles or shingle_hashes(
                    text,
                    self.config.ngram_size,
                )
                fingerprint = DocumentFingerprint(
                    digest=fingerprint.digest,
                    shingles=fingerprint.shingles,
                    signature=signature_bytes(
                        signature_shingles,
                        num_hashes=self.config.num_hashes,
                        seed=self.seed,
                    ),
                )
            exact_candidate = self._exact.get(fingerprint.digest)
            buckets = (
                tuple(
                    band_keys(
                        fingerprint.signature,
                        bands=self.config.bands,
                        rows=self.config.rows,
                    )
                )
                if self.config.enabled and exact_candidate is None
                else ()
            )
            prepared.append(
                {
                    **item,
                    "text": text,
                    "fingerprint": fingerprint,
                    "buckets": buckets,
                    "exact_candidate": exact_candidate,
                }
            )

        external_candidates: list[set[int]] = [
            set() for _ in prepared
        ]
        if self.config.enabled:
            pair_documents: dict[tuple[int, bytes], list[int]] = {}
            for offset, item in enumerate(prepared):
                for pair in item["buckets"]:
                    pair_documents.setdefault(pair, []).append(
                        batch_start + offset
                    )
            if self._pending_bucket_docs:
                for pair, document_ids in pair_documents.items():
                    pending_candidates = self._pending_bucket_docs.get(pair)
                    if not pending_candidates:
                        continue
                    for document_id in document_ids:
                        external_candidates[document_id - batch_start].update(
                            pending_candidates
                        )
            pairs = sorted(pair_documents)
            if self._bucket_bloom is not None:
                pairs = self._bucket_bloom.filter_pairs(pairs)
            for start in range(0, len(pairs), 350):
                chunk = pairs[start : start + 350]
                predicate = " OR ".join(
                    "(band = ? AND bucket = ?)" for _ in chunk
                )
                parameters = tuple(
                    value
                    for band, bucket in chunk
                    for value in (band, bucket)
                )
                for row in self.database.execute(
                    f"""
                    SELECT band, bucket, doc_id
                    FROM lsh_buckets
                    WHERE {predicate}
                    """,
                    parameters,
                ):
                    pair = (int(row[0]), bytes(row[1]))
                    for document_id in pair_documents.get(pair, ()):
                        external_candidates[document_id - batch_start].add(
                            int(row[2])
                        )

        external_payloads: dict[int, bytes] = {}
        external_ids = sorted(
            {
                document_id
                for candidates in external_candidates
                for document_id in candidates
            }
        )
        for start in range(0, len(external_ids), 500):
            chunk = external_ids[start : start + 500]
            placeholders = ",".join("?" for _ in chunk)
            for row in self.database.execute(
                f"""
                SELECT id, text, shingles
                FROM documents
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
                external_payloads[candidate_id] = payload

        local_pairs: dict[tuple[int, bytes], list[int]] = {}
        local_text: dict[int, str] = {}
        local_shingles: dict[int, bytes] = {}
        document_rows: list[tuple] = []
        edge_rows: list[tuple] = []
        bucket_rows: list[tuple[int, bytes, int]] = []

        for offset, item in enumerate(prepared):
            document_id = batch_start + offset
            text = str(item["text"])
            source = item["source"]
            if not isinstance(source, PretrainingSource):
                raise TypeError("document source must be a PretrainingSource")
            fingerprint = item["fingerprint"]
            if not isinstance(fingerprint, DocumentFingerprint):
                raise TypeError("document fingerprint is invalid")
            buckets = item["buckets"]
            if not isinstance(buckets, tuple):
                raise TypeError("document buckets are invalid")
            exact_candidate = item["exact_candidate"]
            if exact_candidate is not None and not isinstance(
                exact_candidate,
                int,
            ):
                raise TypeError("exact candidate must be an integer")

            candidates = set(external_candidates[offset])
            if self.config.enabled and exact_candidate is None:
                for pair in buckets:
                    candidates.update(local_pairs.get(pair, ()))
            candidates.discard(document_id)
            self.candidate_documents += len(candidates)
            self.max_candidates = max(self.max_candidates, len(candidates))

            shingles = fingerprint.shingles
            if candidates and not shingles:
                shingles = shingle_hashes(text, self.config.ngram_size)

            self.parents.append(str(item["parent_id"]))
            self.priorities.append(int(item["priority"]))
            self._union_parent.append(document_id)
            self._union_rank.append(0)

            if exact_candidate is not None:
                self._union(document_id, exact_candidate)
                edge_rows.append(
                    (exact_candidate, document_id, "exact", 1.0)
                )
            else:
                ordered_candidates = sorted(candidates)
                self.jaccard_checks += len(ordered_candidates)
                for candidate_id in ordered_candidates:
                    candidate_shingles = external_payloads.get(candidate_id)
                    if candidate_shingles is None:
                        candidate_shingles = local_shingles.get(candidate_id)
                        if not candidate_shingles:
                            candidate_text = local_text.get(candidate_id)
                            if candidate_text is None:
                                continue
                            candidate_shingles = shingle_hashes(
                                candidate_text,
                                self.config.ngram_size,
                            )
                    similarity = jaccard(
                        shingles,
                        candidate_shingles,
                    )
                    if similarity >= self.config.similarity_threshold:
                        self._union(document_id, candidate_id)
                        edge_rows.append(
                            (
                                candidate_id,
                                document_id,
                                "near",
                                similarity,
                            )
                        )

            self._exact.setdefault(fingerprint.digest, document_id)
            local_text[document_id] = text
            local_shingles[document_id] = shingles
            for pair in buckets:
                local_pairs.setdefault(pair, []).append(document_id)
                bucket_rows.append((pair[0], pair[1], document_id))
            document_rows.append(
                (
                    document_id,
                    str(item["parent_id"]),
                    source.name,
                    int(item["priority"]),
                    str(item["path"]),
                    int(item["ordinal"]),
                    str(item["language"]),
                    text,
                    fingerprint.digest,
                    b"",
                    b"",
                    int(item["raw_characters"]),
                    int(item["raw_tokens"]),
                )
            )

        self.database.executemany(
            """
            INSERT INTO documents(
                id, parent_document_id, source_name, source_priority,
                source_path, source_index, language, text, text_sha,
                signature, shingles, raw_characters, raw_tokens
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            document_rows,
        )
        if edge_rows:
            self.database.executemany(
                """
                INSERT OR REPLACE INTO duplicate_edges(
                    canonical_id, duplicate_id, reason, similarity
                ) VALUES (?, ?, ?, ?)
                """,
                edge_rows,
            )
        if bucket_rows:
            bucket_rows.sort(key=lambda row: (row[0], row[1], row[2]))
            if self._bucket_bloom is not None:
                self._bucket_bloom.add_many(
                    np.fromiter(
                        (row[0] for row in bucket_rows),
                        dtype=np.uint32,
                        count=len(bucket_rows),
                    ),
                    b"".join(row[1] for row in bucket_rows),
                )
            self._pending_bucket_rows.extend(bucket_rows)
            for band, bucket, document_id in bucket_rows:
                self._pending_bucket_docs.setdefault(
                    (band, bucket),
                    [],
                ).append(document_id)

    def flush_pending_buckets(self) -> None:
        if not self._pending_bucket_rows:
            return
        self._pending_bucket_rows.sort(
            key=lambda row: (row[0], row[1], row[2])
        )
        self.database.executemany(
            """
            INSERT OR IGNORE INTO lsh_buckets(band, bucket, doc_id)
            VALUES (?, ?, ?)
            """,
            self._pending_bucket_rows,
        )
        self._pending_bucket_rows.clear()
        self._pending_bucket_docs.clear()

    def canonical_document_ids(
        self,
        progress: Callable[[int], None] | None = None,
    ) -> array:
        size = len(self.parents)
        unset = 2**32 - 1
        best = array("I", [unset]) * size
        for document_id in range(size):
            root = self._find(document_id)
            current = best[root]
            candidate = (self.priorities[document_id], document_id)
            if current == unset or candidate < (
                self.priorities[current],
                current,
            ):
                best[root] = document_id
            if progress is not None and (document_id + 1) % 1000 == 0:
                progress(1000)
        if progress is not None and size % 1000:
            progress(size % 1000)
        canonical = array("I", [0]) * size
        for document_id in range(size):
            canonical[document_id] = best[self._find(document_id)]
        return canonical


__all__ = ["DocumentDeduplicator"]
