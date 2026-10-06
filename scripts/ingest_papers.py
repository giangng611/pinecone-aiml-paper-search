"""Ingest cached arXiv papers into Pinecone."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.arxiv_client import load_papers_jsonl
from src.config import ConfigError, load_settings
from src.pinecone_store import PineconePaperStore


def main() -> None:
    """Run Pinecone ingestion."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None, help="JSONL paper cache path.")
    parser.add_argument("--batch-size", type=int, default=96, help="Pinecone upsert batch size.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = load_settings()
    input_path = args.input or settings.cache_path

    try:
        settings.validate(require_key=True)
    except ConfigError as exc:
        raise SystemExit(str(exc)) from exc

    if not input_path.exists():
        raise SystemExit(f"Paper cache not found: {input_path}. Run scripts/fetch_papers.py first.")

    papers = load_papers_jsonl(input_path)
    store = PineconePaperStore(settings)
    store.create_or_connect_index()
    count = store.upsert_papers(papers, batch_size=args.batch_size)
    logging.info("Submitted %s records to namespace %s", count, settings.namespace)


if __name__ == "__main__":
    main()
