"""Deterministic fingerprints and MinHash primitives."""
from __future__ import annotations

import hashlib
from array import array
from functools import lru_cache
from typing import Iterator

import numpy as np


FINGERPRINT_VERSION = "rolling-u64-v1"
_ROLLING_BASE = np.uint64(1_099_511_628_211)


def digest_text(text: str) -> bytes:
    return hashlib.sha256(text.encode("utf-8")).digest()


def _stable_hash(value: bytes, salt: int = 0) -> int:
    payload = salt.to_bytes(8, "little") + value
    return int.from_bytes(
        hashlib.blake2b(payload, digest_size=8).digest(),
        "little",
    )


def shingle_hashes(text: str, ngram_size: int) -> bytes:
    normalized = " ".join(text.split()).casefold()
    if not normalized:
        values = np.array([0], dtype=np.uint64)
    elif len(normalized) <= ngram_size:
        values = np.array(
            [_stable_hash(normalized.encode("utf-8"))],
            dtype=np.uint64,
        )
    else:
        codepoints = np.frombuffer(
            normalized.encode("utf-32-le"),
            dtype="<u4",
        ).astype(np.uint64)
        width = codepoints.size - ngram_size + 1
        hashes = np.zeros(width, dtype=np.uint64)
        with np.errstate(over="ignore"):
            for offset in range(ngram_size):
                hashes = (
                    hashes * _ROLLING_BASE
                    + codepoints[offset : offset + width]
                )
        values = np.unique(hashes)
    return values.astype("<u8", copy=False).tobytes()


def decode_shingles(payload: bytes) -> array:
    result = array("Q")
    result.frombytes(payload)
    _to_little_endian(result)
    return result


def _to_little_endian(values: array) -> None:
    if int.from_bytes(array("H", [1]).tobytes(), "little") != 1:
        values.byteswap()


def jaccard(left: bytes | array, right: bytes | array) -> float:
    left_values = np.frombuffer(decode_shingles(left), dtype=np.uint64)
    right_values = np.frombuffer(decode_shingles(right), dtype=np.uint64)
    union = left_values.size + right_values.size
    if union == 0:
        return 1.0
    intersection = np.intersect1d(
        left_values,
        right_values,
        assume_unique=True,
    ).size
    return intersection / (union - intersection)


@lru_cache(maxsize=16)
def _permutation_params(
    num_hashes: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    a = np.empty(num_hashes, dtype=np.uint64)
    b = np.empty(num_hashes, dtype=np.uint64)
    for index in range(num_hashes):
        digest = hashlib.sha256(
            f"minhash:{seed}:{index}".encode("ascii")
        ).digest()
        a[index] = int.from_bytes(digest[:8], "little") | 1
        b[index] = int.from_bytes(digest[8:16], "little")
    return a, b


def signature_bytes(
    shingles: bytes,
    *,
    num_hashes: int,
    seed: int,
) -> bytes:
    values = np.frombuffer(decode_shingles(shingles), dtype=np.uint64)
    if values.size == 0:
        values = np.array([0], dtype=np.uint64)
    a, b = _permutation_params(num_hashes, seed)
    signature = np.empty(num_hashes, dtype=np.uint64)
    for start in range(0, num_hashes, 8):
        stop = min(start + 8, num_hashes)
        candidates = values[None, :] * a[start:stop, None] + b[start:stop, None]
        signature[start:stop] = candidates.min(axis=1)
    return signature.astype("<u8", copy=False).tobytes()


def band_keys(
    signature: bytes,
    *,
    bands: int,
    rows: int,
) -> Iterator[tuple[int, bytes]]:
    values = array("Q")
    values.frombytes(signature)
    _to_little_endian(values)
    for band in range(bands):
        payload = band.to_bytes(4, "little") + values[
            band * rows : (band + 1) * rows
        ].tobytes()
        yield band, hashlib.blake2b(payload, digest_size=16).digest()


def cluster_id(parent_document_id: str) -> str:
    return "cluster-" + hashlib.sha256(
        parent_document_id.encode("ascii")
    ).hexdigest()[:24]


def split_for_cluster(
    value: str,
    *,
    validation_ratio: float,
    diagnostic_ratio: float,
    seed: int,
) -> str:
    digest = hashlib.blake2b(
        f"{seed}:{value}".encode("utf-8"),
        digest_size=8,
    ).digest()
    fraction = int.from_bytes(digest, "big") / 2**64
    if fraction < validation_ratio:
        return "validation"
    if fraction < validation_ratio + diagnostic_ratio:
        return "diagnostic"
    return "train"


__all__ = [
    "FINGERPRINT_VERSION",
    "band_keys",
    "cluster_id",
    "decode_shingles",
    "digest_text",
    "jaccard",
    "shingle_hashes",
    "signature_bytes",
    "split_for_cluster",
]
