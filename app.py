"""Streamlit interface for AI/ML research paper semantic search."""

from __future__ import annotations

import logging
from typing import Any

import streamlit as st

from src.config import ConfigError, load_settings
from src.pinecone_store import PineconePaperStore, PineconeStoreError
from src.search_service import build_metadata_filter, result_to_display_dict

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

CATEGORIES = ["", "cs.AI", "cs.LG", "cs.CL", "cs.CV"]


@st.cache_resource(show_spinner=False)
def get_store() -> PineconePaperStore:
    """Create and cache the Pinecone store for the Streamlit session."""
    settings = load_settings()
    settings.validate(require_key=True)
    store = PineconePaperStore(settings)
    store.create_or_connect_index()
    return store


def main() -> None:
    """Render the Streamlit app."""
    st.set_page_config(page_title="AI/ML Paper Semantic Search", layout="wide")
    st.title("AI/ML Research Paper Semantic Search Using Pinecone")
    st.caption(
        "Search arXiv AI/ML papers by semantic meaning, filter by metadata, and find papers similar to a selected result."
    )

    try:
        settings = load_settings()
        store = get_store()
    except (ConfigError, PineconeStoreError, Exception) as exc:
        st.error(_safe_error(exc))
        st.info("Set PINECONE_API_KEY, ingest the dataset, then restart Streamlit.")
        return

    with st.sidebar:
        st.header("Database status")
        stats = _safe_stats(store)
        st.write("Index:", settings.index_name)
        st.write("Namespace:", settings.namespace)
        st.write("Embedding model:", settings.embedding_model)
        st.write("Ready:", _index_ready(store))
        st.write("Approx. records:", _namespace_count(stats, settings.namespace))

        st.header("Filters")
        category = st.selectbox("Primary category", CATEGORIES, format_func=lambda value: value or "Any")
        min_year = st.number_input("Minimum year", min_value=1900, max_value=2100, value=2018)
        max_year = st.number_input("Maximum year", min_value=1900, max_value=2100, value=2100)
        top_k = st.slider("Number of results", min_value=5, max_value=20, value=10)

    query = st.text_input(
        "Research question",
        placeholder="Efficient techniques for running large language models on devices with limited memory",
    )
    search_clicked = st.button("Search papers", type="primary")

    if search_clicked:
        if not query.strip():
            st.warning("Enter a research question to search.")
            return
        try:
            metadata_filter = build_metadata_filter(category or None, int(min_year), int(max_year))
        except ValueError as exc:
            st.error(str(exc))
            return
        with st.spinner("Searching Pinecone..."):
            results = store.search(query.strip(), top_k=top_k, metadata_filter=metadata_filter)
        st.session_state["last_results"] = results

    results = st.session_state.get("last_results", [])
    if results:
        st.subheader("Search results")
        for result in results:
            display = result_to_display_dict(result)
            with st.container(border=True):
                st.markdown(f"### [{result.title}]({result.paper_url})")
                st.write(
                    f"{display['authors']} · {result.published_year} · {result.primary_category} · Score {display['score']}"
                )
                with st.expander("Abstract"):
                    st.write(result.abstract)
                if st.button("Find similar papers", key=f"similar-{result.record_id}"):
                    with st.spinner("Finding similar papers..."):
                        similar = store.similar_to_record(result.record_id, top_k=min(top_k, 10))
                    st.session_state["similar_results"] = (result.title, similar)
    elif search_clicked:
        st.info("No matching papers found. Try a broader query or loosen the filters.")

    similar_state = st.session_state.get("similar_results")
    if similar_state:
        title, similar_results = similar_state
        st.subheader(f"Similar papers to: {title}")
        if not similar_results:
            st.info("No similar papers found.")
        for result in similar_results:
            with st.container(border=True):
                st.markdown(f"### [{result.title}]({result.paper_url})")
                st.write(
                    f"{', '.join(result.authors[:4])} · {result.published_year} · {result.primary_category} · Score {result.score:.4f}"
                )
                with st.expander("Abstract", expanded=False):
                    st.write(result.abstract)


def _safe_error(exc: Exception) -> str:
    text = str(exc)
    return text.replace(load_settings().pinecone_api_key or "", "[redacted]")


def _safe_stats(store: PineconePaperStore) -> dict[str, Any]:
    try:
        return store.describe_stats()
    except Exception:
        return {}


def _namespace_count(stats: dict[str, Any], namespace: str) -> int | str:
    namespaces = stats.get("namespaces", {})
    summary = namespaces.get(namespace, {})
    if isinstance(summary, dict):
        return int(summary.get("vector_count", 0))
    return getattr(summary, "vector_count", "unknown")


def _index_ready(store: PineconePaperStore) -> bool | str:
    try:
        status = getattr(store.describe_index(), "status", None)
        return bool(getattr(status, "ready", False))
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
