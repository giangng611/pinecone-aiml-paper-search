"""Pinecone vector database operations for integrated text records."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable
from typing import Any, TypeVar

from .config import Settings
from .models import Paper, SearchResult, paper_from_pinecone_fields

LOGGER = logging.getLogger(__name__)
T = TypeVar("T")


class PineconeStoreError(RuntimeError):
    """Raised when Pinecone operations fail in an application-facing way."""


class PineconePaperStore:
    """Wrapper around Pinecone integrated embedding record APIs."""

    def __init__(self, settings: Settings) -> None:
        settings.validate(require_key=True)
        self.settings = settings
        from pinecone import EmbedConfig, Pinecone

        self._embed_config_cls = EmbedConfig
        self.pc = Pinecone(api_key=settings.require_pinecone_key())
        self.index = None

    def create_or_connect_index(self) -> Any:
        """Create the Pinecone index if needed, then return an index handle."""
        if not self.pc.indexes.exists(self.settings.index_name):
            LOGGER.info("Creating Pinecone index %s", self.settings.index_name)
            self.pc.indexes.create_for_model(
                name=self.settings.index_name,
                cloud=self.settings.cloud,
                region=self.settings.region,
                embed=self._embed_config_cls(
                    model=self.settings.embedding_model,
                    field_map={"text": self.settings.text_field},
                    write_parameters={"input_type": "passage"},
                    read_parameters={"input_type": "query"},
                ),
            )
        self.wait_until_ready()
        self.index = self.pc.index(self.settings.index_name)
        return self.index

    def wait_until_ready(self, timeout_seconds: int = 180) -> None:
        """Wait until the configured Pinecone index is ready."""
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            description = self.pc.indexes.describe(self.settings.index_name)
            ready = bool(getattr(getattr(description, "status", None), "ready", False))
            if ready:
                return
            time.sleep(3)
        raise PineconeStoreError(f"Timed out waiting for index {self.settings.index_name} to become ready.")

    def upsert_papers(self, papers: Iterable[Paper], batch_size: int = 96) -> int:
        """Batch-upsert paper records into the configured namespace."""
        index = self._index()
        total = 0
        batch: list[dict[str, Any]] = []
        for paper in papers:
            batch.append(build_pinecone_record(paper, self.settings.text_field))
            if len(batch) >= batch_size:
                total += self._upsert_batch(index, batch)
                batch = []
        if batch:
            total += self._upsert_batch(index, batch)
        return total

    def search(self, query: str, top_k: int, metadata_filter: dict[str, Any] | None = None) -> list[SearchResult]:
        """Search papers by natural-language text."""
        response = with_retries(
            lambda: self._index().search(
                namespace=self.settings.namespace,
                top_k=top_k,
                inputs={"text": query},
                filter=metadata_filter,
            )
        )
        return _hits_to_results(response)

    def similar_to_record(
        self,
        record_id: str,
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Find records semantically similar to an existing record ID."""
        response = with_retries(
            lambda: self._index().search(
                namespace=self.settings.namespace,
                top_k=top_k + 1,
                id=record_id,
                filter=metadata_filter,
            )
        )
        return [result for result in _hits_to_results(response) if result.record_id != record_id][:top_k]

    def fetch_paper(self, record_id: str) -> dict[str, Any] | None:
        """Fetch one Pinecone record by ID."""
        response = with_retries(lambda: self._index().fetch(ids=[record_id], namespace=self.settings.namespace))
        vectors = getattr(response, "vectors", None) or {}
        record = vectors.get(record_id)
        if record is None and isinstance(response, dict):
            record = response.get("vectors", {}).get(record_id)
        return record

    def delete_paper(self, record_id: str) -> None:
        """Delete one Pinecone record by ID."""
        with_retries(lambda: self._index().delete(ids=[record_id], namespace=self.settings.namespace))

    def describe_stats(self) -> dict[str, Any]:
        """Return index and namespace statistics."""
        stats = with_retries(lambda: self._index().describe_index_stats())
        if hasattr(stats, "to_dict"):
            return stats.to_dict()
        if isinstance(stats, dict):
            return stats
        return {"raw": stats}

    def describe_index(self) -> Any:
        """Return Pinecone index description metadata."""
        return self.pc.indexes.describe(self.settings.index_name)

    def _upsert_batch(self, index: Any, records: list[dict[str, Any]]) -> int:
        response = with_retries(
            lambda: index.upsert_records(namespace=self.settings.namespace, records=records)
        )
        return int(getattr(response, "record_count", len(records)))

    def _index(self) -> Any:
        if self.index is None:
            return self.create_or_connect_index()
        return self.index


def build_pinecone_record(paper: Paper, text_field: str = "paper_text") -> dict[str, Any]:
    """Build the text-plus-metadata record consumed by Pinecone integrated embedding."""
    return {
        "_id": paper.record_id,
        text_field: paper.embedding_text,
        "arxiv_id": paper.arxiv_id,
        "title": paper.title,
        "abstract": paper.abstract,
        "authors": paper.authors,
        "published_date": paper.published_date.isoformat(),
        "published_year": paper.published_year,
        "primary_category": paper.primary_category,
        "categories": paper.categories,
        "paper_url": paper.paper_url,
        "source": paper.source,
    }


def with_retries(operation: Callable[[], T], attempts: int = 4, base_delay: float = 0.8) -> T:
    """Run a transient service operation with bounded exponential backoff."""
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return operation()
        except Exception as exc:  # Pinecone SDK exception hierarchy varies by version.
            last_error = exc
            if attempt == attempts - 1:
                break
            time.sleep(min(base_delay * (2**attempt), 8.0))
    raise PineconeStoreError(f"Pinecone operation failed after {attempts} attempts: {last_error}") from last_error


def _hits_to_results(response: Any) -> list[SearchResult]:
    hits = getattr(getattr(response, "result", None), "hits", [])
    results: list[SearchResult] = []
    for hit in hits:
        record_id = str(getattr(hit, "id", ""))
        score = float(getattr(hit, "score", 0.0) or 0.0)
        fields = getattr(hit, "fields", {}) or {}
        results.append(paper_from_pinecone_fields(record_id, score, fields))
    return results
