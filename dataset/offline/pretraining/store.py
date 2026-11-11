"""SQLite-backed document staging and pipeline state."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from .config import PretrainingSource


_SCHEMA = """
CREATE TABLE IF NOT EXISTS state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    parent_document_id TEXT UNIQUE NOT NULL,
    source_name TEXT NOT NULL,
    source_priority INTEGER NOT NULL,
    source_path TEXT NOT NULL,
    source_index INTEGER NOT NULL,
    language TEXT NOT NULL,
    text TEXT NOT NULL,
    text_sha BLOB NOT NULL,
    signature BLOB NOT NULL,
    shingles BLOB NOT NULL,
    raw_characters INTEGER NOT NULL,
    raw_tokens INTEGER NOT NULL,
    cleaned_tokens INTEGER NOT NULL DEFAULT 0,
    cluster_id TEXT NOT NULL DEFAULT '',
    split TEXT NOT NULL DEFAULT '',
    canonical_id INTEGER
);
CREATE TABLE IF NOT EXISTS lsh_buckets (
    band INTEGER NOT NULL,
    bucket BLOB NOT NULL,
    doc_id INTEGER NOT NULL,
    PRIMARY KEY (band, bucket, doc_id)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS duplicate_edges (
    canonical_id INTEGER NOT NULL,
    duplicate_id INTEGER NOT NULL,
    reason TEXT NOT NULL,
    similarity REAL NOT NULL,
    PRIMARY KEY (canonical_id, duplicate_id)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS errors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_name TEXT NOT NULL,
    source_path TEXT NOT NULL,
    record_index INTEGER NOT NULL,
    stage TEXT NOT NULL,
    reason TEXT NOT NULL
);
"""


def open_store(staging: Path) -> sqlite3.Connection:
    database_path = staging / "documents.sqlite3"
    database = sqlite3.connect(database_path)
    database.row_factory = sqlite3.Row
    database.execute("PRAGMA page_size=32768")
    database.execute("PRAGMA journal_mode=OFF")
    database.execute("PRAGMA synchronous=OFF")
    database.execute("PRAGMA locking_mode=EXCLUSIVE")
    database.execute("PRAGMA temp_store=MEMORY")
    database.execute("PRAGMA cache_size=-1048576")
    database.execute("PRAGMA mmap_size=8589934592")
    database.execute("PRAGMA cache_spill=OFF")
    database.executescript(_SCHEMA)
    database.execute("DROP INDEX IF EXISTS documents_text_sha")
    database.commit()
    return database


def get_state(database: sqlite3.Connection, key: str) -> str | None:
    row = database.execute(
        "SELECT value FROM state WHERE key = ?",
        (key,),
    ).fetchone()
    return None if row is None else str(row[0])


def set_state(database: sqlite3.Connection, key: str, value: str) -> None:
    database.execute(
        "INSERT OR REPLACE INTO state(key, value) VALUES (?, ?)",
        (key, value),
    )
    database.commit()


def record_error(
    database: sqlite3.Connection,
    *,
    source: PretrainingSource,
    path: Path,
    record_index: int,
    stage: str,
    reason: str,
) -> None:
    database.execute(
        """
        INSERT INTO errors(
            source_name, source_path, record_index, stage, reason
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (source.name, str(path), record_index, stage, reason),
    )


__all__ = ["get_state", "open_store", "record_error", "set_state"]
