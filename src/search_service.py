"""Search input validation, metadata filtering, and result formatting."""

from __future__ import annotations

from typing import Any

from .models import SearchResult


def validate_year_range(min_year: int | None, max_year: int | None) -> None:
    """Validate an optional inclusive publication-year range."""
    if min_year is not None and min_year < 1900:
        raise ValueError("Minimum year is unrealistically early.")
    if max_year is not None and max_year < 1900:
        raise ValueError("Maximum year is unrealistically early.")
    if min_year is not None and max_year is not None and min_year > max_year:
        raise ValueError("Minimum year cannot be greater than maximum year.")


def build_metadata_filter(
    primary_category: str | None = None,
    min_year: int | None = None,
    max_year: int | None = None,
) -> dict[str, Any] | None:
    """Build a Pinecone metadata filter expression from UI options."""
    validate_year_range(min_year, max_year)
    conditions: list[dict[str, Any]] = []
    if primary_category:
        conditions.append({"primary_category": {"$eq": primary_category}})
    if min_year is not None:
        conditions.append({"published_year": {"$gte": min_year}})
    if max_year is not None:
        conditions.append({"published_year": {"$lte": max_year}})
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def format_authors(authors: list[str], max_authors: int = 4) -> str:
    """Format author names compactly for result cards."""
    if len(authors) <= max_authors:
        return ", ".join(authors)
    visible = ", ".join(authors[:max_authors])
    return f"{visible}, et al."


def result_to_display_dict(result: SearchResult) -> dict[str, str | int | float]:
    """Convert a search result into display-friendly scalar values."""
    return {
        "record_id": result.record_id,
        "title": result.title,
        "authors": format_authors(result.authors),
        "year": result.published_year,
        "primary_category": result.primary_category,
        "score": round(result.score, 4),
        "paper_url": result.paper_url,
    }

