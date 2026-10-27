"""Search instrumentation and detailed per-instance metrics tracking."""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
import time
from contextlib import contextmanager


@dataclass
class SearchStats:
    """Comprehensive performance metrics collected during CSP search."""

    nodes: int = 0
    backtracks: int = 0
    propagation_calls: int = 0
    wipeouts: int = 0
    wall_time_sec: float = 0.0
    inference_time_sec: float = 0.0
    max_depth: int = 0
    status: str = "PENDING"  # "SOLVED", "TIMEOUT", "INFEASIBLE", "ERROR"
    seed: int = 42
    time_limit_sec: float = 10.0
    instance_name: str = ""

    @property
    def solved(self) -> bool:
        return self.status == "SOLVED"

    @property
    def timed_out(self) -> bool:
        return self.status == "TIMEOUT"

    @property
    def search_time_sec(self) -> float:
        """Wall time excluding model inference overhead."""
        return max(0.0, self.wall_time_sec - self.inference_time_sec)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["solved"] = self.solved
        d["timed_out"] = self.timed_out
        d["search_time_sec"] = self.search_time_sec
        return d


class InstrumentationTimer:
    """High-resolution wall and inference timer."""

    def __init__(self) -> None:
        self.wall_start: float = 0.0
        self.total_wall_time: float = 0.0
        self.total_inference_time: float = 0.0

    def start_wall(self) -> None:
        self.wall_start = time.perf_counter()

    def stop_wall(self) -> float:
        elapsed = time.perf_counter() - self.wall_start
        self.total_wall_time = elapsed
        return elapsed

    @contextmanager
    def measure_inference(self):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.total_inference_time += (time.perf_counter() - start)
