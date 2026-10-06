import pytest

from src.config import ConfigError, Settings, load_settings


def test_settings_validate_requires_positive_dataset_size() -> None:
    settings = Settings(pinecone_api_key=None, dataset_size=0)
    with pytest.raises(ConfigError):
        settings.validate()


def test_settings_require_key_message() -> None:
    settings = Settings(pinecone_api_key=None)
    with pytest.raises(ConfigError, match="PINECONE_API_KEY"):
        settings.require_pinecone_key()


def test_load_settings_reads_env(monkeypatch) -> None:
    monkeypatch.setenv("PINECONE_INDEX_NAME", "test-index")
    monkeypatch.setenv("ARXIV_DATASET_SIZE", "12")
    settings = load_settings()
    assert settings.index_name == "test-index"
    assert settings.dataset_size == 12

