"""arXiv API data collection and paper normalization."""

from __future__ import annotations

import json
import logging
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Iterable

from .models import Paper

LOGGER = logging.getLogger(__name__)

ARXIV_API_URL = "https://export.arxiv.org/api/query"
DEFAULT_CATEGORIES = ("cs.AI", "cs.LG", "cs.CL", "cs.CV")
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


class ArxivClientError(RuntimeError):
    """Raised when the arXiv API cannot be used successfully."""


def normalize_whitespace(value: str) -> str:
    """Collapse repeated whitespace in arXiv text fields."""
    return " ".join(value.split())


def normalize_arxiv_id(raw_id: str) -> str:
    """Extract the stable arXiv ID from an API entry URL."""
    arxiv_id = raw_id.rstrip("/").split("/")[-1]
    return arxiv_id.removeprefix("abs/")


def parse_arxiv_feed(xml_text: str) -> list[Paper]:
    """Parse an arXiv Atom feed into normalized papers."""
    root = ET.fromstring(xml_text)
    papers: list[Paper] = []
    for entry in root.findall("atom:entry", ATOM_NS):
        paper = parse_arxiv_entry(entry)
        if paper is not None:
            papers.append(paper)
    return papers


def parse_arxiv_entry(entry: ET.Element) -> Paper | None:
    """Parse one arXiv Atom entry, returning None for API error entries."""
    title = _entry_text(entry, "atom:title")
    if title == "Error":
        LOGGER.warning("Skipping arXiv API error entry: %s", _entry_text(entry, "atom:summary"))
        return None

    raw_id = _entry_text(entry, "atom:id")
    abstract = _entry_text(entry, "atom:summary")
    published_raw = _entry_text(entry, "atom:published")
    primary = entry.find("arxiv:primary_category", ATOM_NS)
    categories = [category.attrib.get("term", "") for category in entry.findall("atom:category", ATOM_NS)]
    authors = [
        normalize_whitespace(author.findtext("atom:name", default="", namespaces=ATOM_NS))
        for author in entry.findall("atom:author", ATOM_NS)
    ]
    paper_url = _alternate_link(entry) or f"https://arxiv.org/abs/{normalize_arxiv_id(raw_id)}"

    if not (raw_id and title and abstract and published_raw and authors and primary is not None):
        LOGGER.warning("Skipping incomplete arXiv record: %s", raw_id or title)
        return None

    published_date = datetime.fromisoformat(published_raw.replace("Z", "+00:00")).date()
    return Paper(
        arxiv_id=normalize_arxiv_id(raw_id),
        title=normalize_whitespace(title),
        abstract=normalize_whitespace(abstract),
        authors=[author for author in authors if author],
        published_date=published_date,
        published_year=published_date.year,
        primary_category=primary.attrib["term"],
        categories=[category for category in categories if category],
        paper_url=paper_url,
    )


def fetch_papers(
    limit: int,
    categories: Iterable[str] = DEFAULT_CATEGORIES,
    batch_size: int = 100,
    delay_seconds: float = 3.0,
) -> list[Paper]:
    """Download and deduplicate recent AI/ML papers from the official arXiv API."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    seen: dict[str, Paper] = {}
    query = " OR ".join(f"cat:{category}" for category in categories)
    start = 0
    while len(seen) < limit:
        page_size = min(batch_size, limit - len(seen))
        params = {
            "search_query": query,
            "start": str(start),
            "max_results": str(page_size),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        xml_text = _fetch_api(params)
        papers = parse_arxiv_feed(xml_text)
        if not papers:
            break
        for paper in papers:
            seen.setdefault(paper.arxiv_id, paper)
        start += page_size
        if len(seen) < limit:
            time.sleep(delay_seconds)
    return list(seen.values())[:limit]


def deduplicate_papers(papers: Iterable[Paper]) -> list[Paper]:
    """Deduplicate papers by arXiv ID while preserving first-seen order."""
    seen: dict[str, Paper] = {}
    for paper in papers:
        seen.setdefault(paper.arxiv_id, paper)
    return list(seen.values())


def load_papers_jsonl(path: Path) -> list[Paper]:
    """Load cached papers from a JSONL file."""
    papers: list[Paper] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                papers.append(Paper.from_json(json.loads(line)))
    return papers


def save_papers_jsonl(papers: Iterable[Paper], path: Path) -> None:
    """Write papers to a JSONL cache file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for paper in papers:
            handle.write(json.dumps(paper.to_json(), ensure_ascii=False) + "\n")


def fetch_or_load_cached(limit: int, cache_path: Path) -> list[Paper]:
    """Fetch papers, falling back to an existing cache if the API is unavailable."""
    try:
        papers = fetch_papers(limit=limit)
        save_papers_jsonl(papers, cache_path)
        return papers
    except (ArxivClientError, urllib.error.URLError, ET.ParseError) as exc:
        if cache_path.exists():
            LOGGER.warning("arXiv API unavailable; using cached data at %s: %s", cache_path, exc)
            return load_papers_jsonl(cache_path)
        raise ArxivClientError(f"arXiv API unavailable and no cache exists at {cache_path}") from exc


def _fetch_api(params: dict[str, str]) -> str:
    url = f"{ARXIV_API_URL}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": "uga-csci4370-pinecone-paper-search/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=30, context=_ssl_context()) as response:
            return response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise ArxivClientError(f"arXiv API returned HTTP {exc.code}") from exc


def _ssl_context() -> ssl.SSLContext:
    """Return an SSL context that works reliably in local virtual environments."""
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _entry_text(entry: ET.Element, path: str) -> str:
    return normalize_whitespace(entry.findtext(path, default="", namespaces=ATOM_NS))


def _alternate_link(entry: ET.Element) -> str | None:
    for link in entry.findall("atom:link", ATOM_NS):
        if link.attrib.get("rel") == "alternate" and link.attrib.get("href"):
            return link.attrib["href"]
    return None
