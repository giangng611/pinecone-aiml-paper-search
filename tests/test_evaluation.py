from src.evaluation import EvaluationQuery, category_hit_rate, filters_respected, precision_at_k, summarize_metrics
from src.models import SearchResult


def test_category_hit_rate_detects_expected_category() -> None:
    assert category_hit_rate([_result("arxiv:1", "cs.CL", 2024)], ["cs.CL"]) == 1.0
    assert category_hit_rate([_result("arxiv:1", "cs.CV", 2024)], ["cs.CL"]) == 0.0


def test_precision_at_k_requires_labels() -> None:
    results = [_result("arxiv:1", "cs.CL", 2024), _result("arxiv:2", "cs.CL", 2024)]
    assert precision_at_k(results, [], 2) is None
    assert precision_at_k(results, ["arxiv:1"], 2) == 0.5


def test_filters_respected_checks_category_and_year() -> None:
    query = EvaluationQuery(
        query="recent NLP",
        expected_categories=["cs.CL"],
        expected_concepts=[],
        relevant_ids=[],
        filter_category="cs.CL",
        min_year=2020,
        max_year=2024,
    )
    assert filters_respected([_result("arxiv:1", "cs.CL", 2022)], query)
    assert not filters_respected([_result("arxiv:1", "cs.CV", 2022)], query)
    assert not filters_respected([_result("arxiv:1", "cs.CL", 2019)], query)


def test_summarize_metrics_handles_single_latency() -> None:
    import pandas as pd

    frame = pd.DataFrame([{"category_hit": 1.0, "filters_respected": True}])
    summary = summarize_metrics(frame, [12.5])
    assert summary["mean_latency_ms"] == 12.5
    assert summary["category_hit_rate_at_k"] == 1.0
    assert summary["filter_respect_rate"] == 1.0


def _result(record_id: str, category: str, year: int) -> SearchResult:
    return SearchResult(
        record_id=record_id,
        score=0.9,
        title="Title",
        abstract="Abstract",
        authors=["Author"],
        published_year=year,
        primary_category=category,
        categories=[category],
        paper_url="https://example.com",
    )

