from datetime import date

from src.arxiv_client import deduplicate_papers, normalize_arxiv_id, normalize_whitespace, parse_arxiv_feed
from src.models import Paper


def test_normalize_whitespace_collapses_spaces() -> None:
    assert normalize_whitespace("A\n  title\twith   spaces") == "A title with spaces"


def test_normalize_arxiv_id_from_url() -> None:
    assert normalize_arxiv_id("http://arxiv.org/abs/1706.03762v7") == "1706.03762v7"


def test_parse_arxiv_feed_normalizes_entry() -> None:
    xml = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
      <entry>
        <id>http://arxiv.org/abs/1234.56789v1</id>
        <title> A   Paper\n Title </title>
        <summary> An abstract\n with spacing. </summary>
        <published>2024-01-02T00:00:00Z</published>
        <author><name>Ada Lovelace</name></author>
        <arxiv:primary_category term="cs.LG" />
        <category term="cs.LG" />
        <link href="https://arxiv.org/abs/1234.56789" rel="alternate" type="text/html" />
      </entry>
    </feed>
    """
    papers = parse_arxiv_feed(xml)
    assert len(papers) == 1
    assert papers[0].title == "A Paper Title"
    assert papers[0].published_year == 2024
    assert papers[0].primary_category == "cs.LG"


def test_deduplicate_papers_preserves_first() -> None:
    first = _paper("1", "First")
    duplicate = _paper("1", "Duplicate")
    second = _paper("2", "Second")
    assert deduplicate_papers([first, duplicate, second]) == [first, second]


def _paper(arxiv_id: str, title: str) -> Paper:
    return Paper(
        arxiv_id=arxiv_id,
        title=title,
        abstract="Abstract",
        authors=["Author"],
        published_date=date(2024, 1, 1),
        published_year=2024,
        primary_category="cs.LG",
        categories=["cs.LG"],
        paper_url=f"https://arxiv.org/abs/{arxiv_id}",
    )

