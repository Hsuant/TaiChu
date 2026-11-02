"""Reusable text cleaning and exact deduplication."""
from __future__ import annotations

import hashlib
import re
import sqlite3
import unicodedata
from pathlib import Path
from typing import Any

from .config import CleaningConfig


class TextCleaner:
    def __init__(self, config: CleaningConfig) -> None:
        self.config = config

    def clean(self, value: Any) -> tuple[str | None, str]:
        if not isinstance(value, str):
            return None, "empty"
        text = unicodedata.normalize(self.config.unicode_form, value)
        text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
        if self.config.remove_control_chars:
            text = "".join(
                character
                for character in text
                if character in "\n\t"
                or not unicodedata.category(character).startswith("C")
            )
        if self.config.collapse_whitespace:
            text = re.sub(r"[^\S\n]+", " ", text)
            text = re.sub(r" *\n *", "\n", text)
            text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()
        if not text:
            return None, "empty"
        if len(text) > self.config.max_chars:
            if self.config.oversize_policy == "skip":
                return None, "length"
            text = text[: self.config.max_chars].rstrip()
        if len(text) < self.config.min_chars:
            return None, "length"
        return text, "accepted"


class ExactDeduplicator:
    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._database: sqlite3.Connection | None = None

    def __enter__(self) -> "ExactDeduplicator":
        if self.path is None:
            return self
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._database = sqlite3.connect(self.path)
        self._database.execute("PRAGMA journal_mode=OFF")
        self._database.execute("PRAGMA synchronous=OFF")
        self._database.execute(
            "CREATE TABLE fingerprints (digest BLOB PRIMARY KEY) WITHOUT ROWID"
        )
        return self

    def accept(self, text: str) -> bool:
        if self._database is None:
            return True
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()
        cursor = self._database.execute(
            "INSERT OR IGNORE INTO fingerprints(digest) VALUES (?)",
            (digest,),
        )
        return cursor.rowcount == 1

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._database is not None:
            self._database.commit()
            self._database.close()
            self._database = None
        if self.path is not None and self.path.exists():
            self.path.unlink()


__all__ = ["ExactDeduplicator", "TextCleaner"]
