"""Domain models for arXiv papers and search results."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any


@dataclass(frozen=True)
class Paper:
    """Normalized metadata for one arXiv research paper."""

    arxiv_id: str
    title: str
    abstract: str
    authors: list[str]
    published_date: date
    published_year: int
    primary_category: str
    categories: list[str]
    paper_url: str
    source: str = "arxiv"

    @property
    def record_id(self) -> str:
        """Stable Pinecone record ID derived from the arXiv identifier."""
        return f"arxiv:{self.arxiv_id}"

    @property
    def embedding_text(self) -> str:
        """Text field embedded by Pinecone integrated inference."""
        return f"Title: {self.title}\n\nAbstract: {self.abstract}"

    def to_json(self) -> dict[str, Any]:
        """Serialize the paper to JSON-compatible values."""
        return {
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "abstract": self.abstract,
            "authors": self.authors,
            "published_date": self.published_date.isoformat(),
            "published_year": self.published_year,
            "primary_category": self.primary_category,
            "categories": self.categories,
            "paper_url": self.paper_url,
            "source": self.source,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Paper":
        """Deserialize a paper from JSON-compatible values."""
        return cls(
            arxiv_id=str(data["arxiv_id"]),
            title=str(data["title"]),
            abstract=str(data["abstract"]),
            authors=list(data["authors"]),
            published_date=date.fromisoformat(str(data["published_date"])),
            published_year=int(data["published_year"]),
            primary_category=str(data["primary_category"]),
            categories=list(data["categories"]),
            paper_url=str(data["paper_url"]),
            source=str(data.get("source", "arxiv")),
        )


@dataclass(frozen=True)
class SearchResult:
    """Application-level representation of a Pinecone hit."""

    record_id: str
    score: float
    title: str
    abstract: str
    authors: list[str]
    published_year: int
    primary_category: str
    categories: list[str]
    paper_url: str


def paper_from_pinecone_fields(record_id: str, score: float, fields: dict[str, Any]) -> SearchResult:
    """Convert Pinecone hit fields into a display-ready result."""
    authors = fields.get("authors", [])
    categories = fields.get("categories", [])
    if isinstance(authors, str):
        authors = [authors]
    if isinstance(categories, str):
        categories = [categories]
    return SearchResult(
        record_id=record_id,
        score=float(score),
        title=str(fields.get("title", "")),
        abstract=str(fields.get("abstract", "")),
        authors=[str(author) for author in authors],
        published_year=int(fields.get("published_year", 0) or 0),
        primary_category=str(fields.get("primary_category", "")),
        categories=[str(category) for category in categories],
        paper_url=str(fields.get("paper_url", "")),
    )

