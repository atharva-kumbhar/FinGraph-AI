"""Metric helpers used by all three inference pipelines."""

from __future__ import annotations

import re
from dataclasses import dataclass
from time import perf_counter
from typing import Callable, TypeVar


T = TypeVar("T")
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def estimate_tokens(text: str) -> int:
    """Fast token estimate that tracks close enough for dashboard comparison."""
    if not text:
        return 0
    rough = len(TOKEN_RE.findall(text))
    return max(1, int(rough * 1.12))


def estimate_cost(tokens: int, price_per_1k_tokens_usd: float) -> float:
    return round((tokens / 1000.0) * price_per_1k_tokens_usd, 6)


def percent_reduction(baseline: int | float, candidate: int | float) -> float:
    if baseline <= 0:
        return 0.0
    return round(((baseline - candidate) / baseline) * 100.0, 1)


@dataclass(frozen=True)
class TimedResult:
    value: T
    latency_ms: int


def timed(fn: Callable[[], T]) -> TimedResult:
    start = perf_counter()
    value = fn()
    elapsed = int((perf_counter() - start) * 1000)
    return TimedResult(value=value, latency_ms=max(elapsed, 1))
