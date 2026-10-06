"""Run configured search evaluation queries against Pinecone."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ConfigError, load_settings
from src.evaluation import evaluate_search, load_evaluation_queries, write_markdown_summary
from src.pinecone_store import PineconePaperStore


def main() -> None:
    """Run semantic-search evaluation and write CSV plus Markdown outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=Path("evaluation/queries.json"))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-csv", type=Path, default=Path("evaluation/results.csv"))
    parser.add_argument("--output-md", type=Path, default=Path("evaluation/summary.md"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = load_settings()
    try:
        settings.validate(require_key=True)
    except ConfigError as exc:
        raise SystemExit(str(exc)) from exc

    store = PineconePaperStore(settings)
    store.create_or_connect_index()
    queries = load_evaluation_queries(args.queries)
    frame, summary = evaluate_search(store, queries, top_k=args.top_k)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_csv, index=False)
    write_markdown_summary(summary, frame, args.output_md)
    logging.info("Wrote %s and %s", args.output_csv, args.output_md)


if __name__ == "__main__":
    main()
