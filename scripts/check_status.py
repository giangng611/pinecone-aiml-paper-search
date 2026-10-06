"""Check local dataset and Pinecone namespace status."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.arxiv_client import load_papers_jsonl
from src.config import ConfigError, load_settings
from src.pinecone_store import PineconePaperStore


def main() -> None:
    """Print a concise project status report without exposing secrets."""
    settings = load_settings()
    print(f"Index: {settings.index_name}")
    print(f"Namespace: {settings.namespace}")
    print(f"Embedding model: {settings.embedding_model}")
    print(f"Local cache: {settings.cache_path}")
    if settings.cache_path.exists():
        papers = load_papers_jsonl(settings.cache_path)
        print(f"Local cached papers: {len(papers)}")
        if papers:
            print(f"First cached paper: {papers[0].title} ({papers[0].arxiv_id})")
    else:
        print("Local cached papers: 0")

    try:
        settings.validate(require_key=True)
    except ConfigError as exc:
        print(f"Pinecone: not checked ({exc})")
        return

    store = PineconePaperStore(settings)
    store.create_or_connect_index()
    stats = store.describe_stats()
    namespaces = stats.get("namespaces", {})
    namespace_stats = namespaces.get(settings.namespace, {})
    if isinstance(namespace_stats, dict):
        count = namespace_stats.get("vector_count", 0)
    else:
        count = getattr(namespace_stats, "vector_count", "unknown")
    print(f"Pinecone namespace records: {count}")


if __name__ == "__main__":
    main()

