"""Configuration loading and validation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_INDEX_NAME = "aiml-paper-search"
DEFAULT_NAMESPACE = "arxiv-papers"
DEFAULT_EMBEDDING_MODEL = "llama-text-embed-v2"
DEFAULT_CLOUD = "aws"
DEFAULT_REGION = "us-east-1"
DEFAULT_DATASET_SIZE = 1000
DEFAULT_TEXT_FIELD = "paper_text"


@dataclass(frozen=True)
class Settings:
    """Runtime settings sourced from environment variables."""

    pinecone_api_key: str | None
    index_name: str = DEFAULT_INDEX_NAME
    namespace: str = DEFAULT_NAMESPACE
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    cloud: str = DEFAULT_CLOUD
    region: str = DEFAULT_REGION
    dataset_size: int = DEFAULT_DATASET_SIZE
    text_field: str = DEFAULT_TEXT_FIELD
    data_dir: Path = Path("data")

    @property
    def cache_path(self) -> Path:
        """Default JSONL cache path for downloaded papers."""
        return self.data_dir / "arxiv_papers.jsonl"

    def require_pinecone_key(self) -> str:
        """Return the API key or raise a user-facing configuration error."""
        if not self.pinecone_api_key:
            raise ConfigError(
                "PINECONE_API_KEY is not set. Add it to your shell environment before using Pinecone."
            )
        return self.pinecone_api_key

    def validate(self, require_key: bool = False) -> None:
        """Validate settings that can be checked without contacting services."""
        if require_key:
            self.require_pinecone_key()
        if not self.index_name:
            raise ConfigError("PINECONE_INDEX_NAME cannot be empty.")
        if not self.namespace:
            raise ConfigError("PINECONE_NAMESPACE cannot be empty.")
        if self.dataset_size <= 0:
            raise ConfigError("ARXIV_DATASET_SIZE must be a positive integer.")
        if not self.embedding_model:
            raise ConfigError("PINECONE_EMBEDDING_MODEL cannot be empty.")


class ConfigError(ValueError):
    """Raised when project configuration is invalid."""


def load_settings() -> Settings:
    """Load application settings from environment variables."""
    return Settings(
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        index_name=os.getenv("PINECONE_INDEX_NAME", DEFAULT_INDEX_NAME),
        namespace=os.getenv("PINECONE_NAMESPACE", DEFAULT_NAMESPACE),
        embedding_model=os.getenv("PINECONE_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        cloud=os.getenv("PINECONE_CLOUD", DEFAULT_CLOUD),
        region=os.getenv("PINECONE_REGION", DEFAULT_REGION),
        dataset_size=_get_int_env("ARXIV_DATASET_SIZE", DEFAULT_DATASET_SIZE),
        text_field=os.getenv("PINECONE_TEXT_FIELD", DEFAULT_TEXT_FIELD),
        data_dir=Path(os.getenv("DATA_DIR", "data")),
    )


def _get_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer.") from exc

