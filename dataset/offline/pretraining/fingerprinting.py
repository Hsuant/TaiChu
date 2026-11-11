"""Parallelizable document fingerprint computation."""
from __future__ import annotations

from dataclasses import dataclass

from .hashing import (
    FINGERPRINT_VERSION,
    digest_text,
    shingle_hashes,
    signature_bytes,
)


@dataclass(frozen=True, slots=True)
class DocumentFingerprint:
    digest: bytes
    shingles: bytes
    signature: bytes


def compute_fingerprint(
    text: str,
    *,
    ngram_size: int,
    num_hashes: int,
    seed: int,
) -> DocumentFingerprint:
    shingles = shingle_hashes(text, ngram_size)
    return DocumentFingerprint(
        digest=digest_text(text),
        shingles=shingles,
        signature=signature_bytes(
            shingles,
            num_hashes=num_hashes,
            seed=seed,
        ),
    )


def compute_fingerprints(
    texts: list[str],
    *,
    ngram_size: int,
    num_hashes: int,
    seed: int,
) -> list[DocumentFingerprint]:
    return [
        compute_fingerprint(
            text,
            ngram_size=ngram_size,
            num_hashes=num_hashes,
            seed=seed,
        )
        for text in texts
    ]


__all__ = [
    "DocumentFingerprint",
    "FINGERPRINT_VERSION",
    "compute_fingerprint",
    "compute_fingerprints",
]
