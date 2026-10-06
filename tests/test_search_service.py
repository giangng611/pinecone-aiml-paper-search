import pytest

from src.models import SearchResult
from src.search_service import build_metadata_filter, format_authors, result_to_display_dict, validate_year_range


def test_build_metadata_filter_combines_conditions() -> None:
    assert build_metadata_filter("cs.CL", 2020, 2024) == {
        "$and": [
            {"primary_category": {"$eq": "cs.CL"}},
            {"published_year": {"$gte": 2020}},
            {"published_year": {"$lte": 2024}},
        ]
    }


def test_build_metadata_filter_returns_none_without_filters() -> None:
    assert build_metadata_filter() is None


def test_validate_year_range_rejects_inverted_range() -> None:
    with pytest.raises(ValueError):
        validate_year_range(2025, 2020)


def test_result_to_display_dict_rounds_score_and_authors() -> None:
    result = SearchResult(
        record_id="arxiv:1",
        score=0.987654,
        title="Title",
        abstract="Abstract",
        authors=["A", "B", "C", "D", "E"],
        published_year=2024,
        primary_category="cs.LG",
        categories=["cs.LG"],
        paper_url="https://example.com",
    )
    display = result_to_display_dict(result)
    assert display["score"] == 0.9877
    assert display["authors"] == "A, B, C, D, et al."


def test_format_authors_short_list() -> None:
    assert format_authors(["A", "B"]) == "A, B"

