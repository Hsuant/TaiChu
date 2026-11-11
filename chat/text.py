"""Incremental detokenization for streaming byte-level BPE output."""

from __future__ import annotations

from typing import Callable, Sequence


class IncrementalTextDecoder:
    """Emit stable text deltas while holding incomplete UTF-8 sequences."""

    def __init__(
        self,
        decode: Callable[[Sequence[int]], str],
        *,
        strip_leading: bool = True,
    ) -> None:
        self._decode = decode
        self._token_ids: list[int] = []
        self._emitted = ""
        self._started = False
        self._strip_leading = strip_leading

    def push(self, token_id: int) -> str:
        self._token_ids.append(token_id)
        decoded = self._decode(self._token_ids)
        stable = decoded.rstrip("\ufffd")
        if stable.startswith(self._emitted):
            delta = stable[len(self._emitted):]
        else:
            common = 0
            limit = min(len(stable), len(self._emitted))
            while common < limit and stable[common] == self._emitted[common]:
                common += 1
            delta = stable[common:]
        if not self._started and self._strip_leading:
            delta = delta.lstrip()
            if delta:
                self._started = True
        elif delta:
            self._started = True
        self._emitted = stable
        return delta

    def finish(self) -> str:
        decoded = self._decode(self._token_ids)
        if decoded.startswith(self._emitted):
            delta = decoded[len(self._emitted):]
        else:
            delta = decoded
        self._emitted = decoded
        if not self._started and self._strip_leading:
            delta = delta.lstrip()
            if delta:
                self._started = True
        elif delta:
            self._started = True
        return delta

    @property
    def text(self) -> str:
        return self._emitted


__all__ = ["IncrementalTextDecoder"]
