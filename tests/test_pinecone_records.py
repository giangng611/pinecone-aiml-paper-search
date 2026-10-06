from datetime import date

from src.models import SearchResult, paper_from_pinecone_fields
from src.pinecone_store import build_pinecone_record


def test_build_pinecone_record_contains_text_and_metadata() -> None:
    paper = _paper()
    record = build_pinecone_record(paper, text_field="paper_text")
    assert record["_id"] == "arxiv:1706.03762"
    assert "Title: Attention Is All You Need" in record["paper_text"]
    assert record["published_year"] == 2017
    assert record["categories"] == ["cs.CL", "cs.LG"]


def test_paper_from_pinecone_fields_formats_lists() -> None:
    result = paper_from_pinecone_fields(
        "arxiv:1",
        0.9,
        {
            "title": "Title",
            "abstract": "Abstract",
            "authors": ["A", "B"],
            "published_year": 2024,
            "primary_category": "cs.AI",
            "categories": ["cs.AI"],
            "paper_url": "https://example.com",
        },
    )
    assert isinstance(result, SearchResult)
    assert result.authors == ["A", "B"]
    assert result.score == 0.9


def _paper():
    from src.models import Paper

    return Paper(
        arxiv_id="1706.03762",
        title="Attention Is All You Need",
        abstract="Transformer paper.",
        authors=["Ashish Vaswani"],
        published_date=date(2017, 6, 12),
        published_year=2017,
        primary_category="cs.CL",
        categories=["cs.CL", "cs.LG"],
        paper_url="https://arxiv.org/abs/1706.03762",
    )

