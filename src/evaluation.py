"""Evaluation helpers for semantic search behavior and latency."""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from .models import SearchResult
from .search_service import build_metadata_filter


class Searcher(Protocol):
    """Protocol for objects that can execute project searches."""

    def search(self, query: str, top_k: int, metadata_filter: dict[str, Any] | None = None) -> list[SearchResult]:
        """Run one search query."""


@dataclass(frozen=True)
class EvaluationQuery:
    """One configured evaluation query."""

    query: str
    expected_categories: list[str]
    expected_concepts: list[str]
    relevant_ids: list[str]
    filter_category: str | None = None
    min_year: int | None = None
    max_year: int | None = None


def load_evaluation_queries(path: Path) -> list[EvaluationQuery]:
    """Load evaluation queries from JSON."""
    raw_queries = json.loads(path.read_text(encoding="utf-8"))
    queries: list[EvaluationQuery] = []
    for item in raw_queries:
        queries.append(
            EvaluationQuery(
                query=item["query"],
                expected_categories=list(item.get("expected_categories", [])),
                expected_concepts=list(item.get("expected_concepts", [])),
                relevant_ids=list(item.get("relevant_ids", [])),
                filter_category=item.get("filter_category"),
                min_year=item.get("min_year"),
                max_year=item.get("max_year"),
            )
        )
    return queries


def evaluate_search(searcher: Searcher, queries: list[EvaluationQuery], top_k: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Execute evaluation queries and compute proxy metrics."""
    rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    for query in queries:
        metadata_filter = build_metadata_filter(query.filter_category, query.min_year, query.max_year)
        start = time.perf_counter()
        results = searcher.search(query.query, top_k=top_k, metadata_filter=metadata_filter)
        latency_ms = (time.perf_counter() - start) * 1000
        latencies.append(latency_ms)
        rows.append(_query_row(query, results, top_k, latency_ms))
    frame = pd.DataFrame(rows)
    summary = summarize_metrics(frame, latencies)
    return frame, summary


def category_hit_rate(results: list[SearchResult], expected_categories: list[str]) -> float:
    """Return 1.0 if any result is in an expected category, otherwise 0.0."""
    if not expected_categories:
        return float("nan")
    expected = set(expected_categories)
    return 1.0 if any(result.primary_category in expected for result in results) else 0.0


def precision_at_k(results: list[SearchResult], relevant_ids: list[str], k: int) -> float | None:
    """Compute precision@k when human labels are available."""
    if not relevant_ids:
        return None
    relevant = set(relevant_ids)
    retrieved = results[:k]
    if not retrieved:
        return 0.0
    return sum(1 for result in retrieved if result.record_id in relevant) / min(k, len(retrieved))


def filters_respected(results: list[SearchResult], query: EvaluationQuery) -> bool:
    """Check whether returned results satisfy the configured metadata filters."""
    for result in results:
        if query.filter_category and result.primary_category != query.filter_category:
            return False
        if query.min_year is not None and result.published_year < query.min_year:
            return False
        if query.max_year is not None and result.published_year > query.max_year:
            return False
    return True


def summarize_metrics(frame: pd.DataFrame, latencies_ms: list[float]) -> dict[str, float]:
    """Summarize latency and retrieval proxy metrics."""
    if not latencies_ms:
        return {}
    p95 = statistics.quantiles(latencies_ms, n=20)[18] if len(latencies_ms) >= 2 else latencies_ms[0]
    summary = {
        "mean_latency_ms": statistics.mean(latencies_ms),
        "median_latency_ms": statistics.median(latencies_ms),
        "p95_latency_ms": p95,
    }
    if "category_hit" in frame:
        summary["category_hit_rate_at_k"] = float(frame["category_hit"].dropna().mean())
    if "precision_at_k" in frame and frame["precision_at_k"].notna().any():
        summary["precision_at_k"] = float(frame["precision_at_k"].dropna().mean())
    if "filters_respected" in frame:
        summary["filter_respect_rate"] = float(frame["filters_respected"].mean())
    return summary


def write_markdown_summary(summary: dict[str, float], frame: pd.DataFrame, path: Path) -> None:
    """Write a readable Markdown summary for evaluation output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Evaluation Summary",
        "",
        "These metrics combine latency measurements with automatic proxy checks. Precision@k is only reported when manually labeled relevant IDs are present.",
        "",
        "## Aggregate Metrics",
        "",
    ]
    for key, value in summary.items():
        lines.append(f"- **{key}**: {value:.3f}")
    lines.extend(["", "## Per-query Results", "", frame.to_markdown(index=False)])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _query_row(query: EvaluationQuery, results: list[SearchResult], top_k: int, latency_ms: float) -> dict[str, Any]:
    precision = precision_at_k(results, query.relevant_ids, top_k)
    return {
        "query": query.query,
        "latency_ms": latency_ms,
        "result_count": len(results),
        "top_result_id": results[0].record_id if results else "",
        "top_result_title": results[0].title if results else "",
        "category_hit": category_hit_rate(results, query.expected_categories),
        "precision_at_k": precision,
        "filters_respected": filters_respected(results, query),
        "expected_categories": ", ".join(query.expected_categories),
    }

