import os
import time
from datetime import date

import pytest

from src.config import Settings
from src.models import Paper
from src.pinecone_store import PineconePaperStore


@pytest.mark.skipif(
    not os.getenv("PINECONE_API_KEY") or os.getenv("RUN_PINECONE_INTEGRATION") != "1",
    reason="Set PINECONE_API_KEY and RUN_PINECONE_INTEGRATION=1 to run Pinecone smoke test.",
)
def test_pinecone_integration_smoke() -> None:
    namespace = f"integration-test-{int(time.time())}"
    settings = Settings(
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        index_name=os.getenv("PINECONE_INDEX_NAME", "aiml-paper-search"),
        namespace=namespace,
        embedding_model=os.getenv("PINECONE_EMBEDDING_MODEL", "llama-text-embed-v2"),
        cloud=os.getenv("PINECONE_CLOUD", "aws"),
        region=os.getenv("PINECONE_REGION", "us-east-1"),
    )
    store = PineconePaperStore(settings)
    store.create_or_connect_index()
    count = store.upsert_papers([_paper()])
    assert count == 1
    time.sleep(10)
    results = store.search("attention models for translation", top_k=1)
    assert results
    store.delete_paper(_paper().record_id)


def _paper() -> Paper:
    return Paper(
        arxiv_id="integration.0001",
        title="Attention Models for Translation",
        abstract="A tiny integration-test paper about attention and translation.",
        authors=["Test Author"],
        published_date=date(2024, 1, 1),
        published_year=2024,
        primary_category="cs.CL",
        categories=["cs.CL"],
        paper_url="https://example.com/integration",
        source="integration-test",
    )

