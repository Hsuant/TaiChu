"""Process-tree RSS and stage timing monitor."""

from __future__ import annotations

import json
import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional

import psutil

logger = logging.getLogger(__name__)


@dataclass
class StageMetrics:
    name: str
    elapsed_seconds: float
    rss_before_mb: float
    rss_after_mb: float
    rss_delta_mb: float
    records: int = 0
    chars: int = 0
    throughput_chars_per_second: float = 0.0


class ResourceMonitor:
    """Sample current process and descendants without retaining time-series data."""

    def __init__(self, interval_seconds: float = 0.1):
        self.interval_seconds = interval_seconds
        self.process = psutil.Process()
        self.peak_tree_rss = self.current_rss()[0]
        self.peak_children = 0
        self.stages: Dict[str, StageMetrics] = {}
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stage_started_at: Dict[str, float] = {}
        self._stage_rss_before: Dict[str, int] = {}

    def current_rss(self) -> tuple[int, int]:
        total = self.process.memory_info().rss
        children = 0
        for child in self.process.children(recursive=True):
            try:
                total += child.memory_info().rss
                children += 1
            except psutil.Error:
                continue
        return total, children

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def sample(self) -> tuple[int, int]:
        rss, children = self.current_rss()
        with self._lock:
            self.peak_tree_rss = max(self.peak_tree_rss, rss)
            self.peak_children = max(self.peak_children, children)
        return rss, children

    @contextmanager
    def stage(self, name: str) -> Iterator[StageMetrics]:
        rss_before, _ = self.sample()
        started = time.perf_counter()
        metrics = StageMetrics(
            name=name,
            elapsed_seconds=0.0,
            rss_before_mb=_mb(rss_before),
            rss_after_mb=_mb(rss_before),
            rss_delta_mb=0.0,
        )
        try:
            yield metrics
        finally:
            rss_after, _ = self.sample()
            metrics.elapsed_seconds = time.perf_counter() - started
            metrics.rss_after_mb = _mb(rss_after)
            metrics.rss_delta_mb = _mb(rss_after - rss_before)
            if metrics.elapsed_seconds > 0:
                metrics.throughput_chars_per_second = (
                    metrics.chars / metrics.elapsed_seconds
                )
            with self._lock:
                self.stages[name] = metrics
            logger.info(
                "stage=%s elapsed=%.3fs rss_before=%.1fMB rss_after=%.1fMB "
                "delta=%.1fMB records=%d chars=%d",
                name,
                metrics.elapsed_seconds,
                metrics.rss_before_mb,
                metrics.rss_after_mb,
                metrics.rss_delta_mb,
                metrics.records,
                metrics.chars,
            )

    def report(self) -> Dict[str, object]:
        rss, children = self.sample()
        return {
            "current_tree_rss_mb": _mb(rss),
            "peak_tree_rss_mb": _mb(self.peak_tree_rss),
            "peak_children": self.peak_children,
            "current_children": children,
            "stages": {
                name: asdict(metrics)
                for name, metrics in self.stages.items()
            },
        }

    def write_report(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.report(), handle, ensure_ascii=False, indent=2)

    def _sample_loop(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self.sample()


def _mb(value: int) -> float:
    return round(value / 1048576, 2)
