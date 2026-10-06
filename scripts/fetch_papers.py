"""Download AI/ML paper metadata from arXiv and cache it as JSONL."""

from __future__ import annotations

import argparse
import logging

from pathlib import Path

from src.arxiv_client import fetch_or_load_cached, save_papers_jsonl
from src.config import load_settings


def main() -> None:
    """Run the arXiv collection script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Number of papers to collect.")
    parser.add_argument("--output", type=str, default=None, help="JSONL cache path.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = load_settings()
    limit = args.limit or settings.dataset_size
    output = settings.cache_path if args.output is None else Path(args.output)

    papers = fetch_or_load_cached(limit=limit, cache_path=output)
    save_papers_jsonl(papers, output)
    logging.info("Saved %s papers to %s", len(papers), output)


if __name__ == "__main__":
    main()
