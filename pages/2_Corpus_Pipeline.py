from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cultural_mood_tracker.config import settings
from cultural_mood_tracker.dlt_pipeline import inspect_local_pipeline, query_local_pipeline

st.set_page_config(
    page_title="Cultural Mood Tracker Corpus Pipeline", page_icon="🧰", layout="wide"
)
st.title("Corpus pipeline")
st.caption(
    "A read-only view of the local DuckDB dataset, attached from persisted dlt state."
)

source = st.selectbox("Pipeline source", ("demo", "tmdb"))
config = settings()

try:
    details = inspect_local_pipeline(config, source)
    by_kind = query_local_pipeline(
        config,
        source,
        """
        SELECT media_kind, COUNT(*) AS documents
        FROM documents
        GROUP BY media_kind
        ORDER BY documents DESC
        """,
    )
    by_type = query_local_pipeline(
        config,
        source,
        """
        SELECT document_type, COUNT(*) AS documents
        FROM documents
        GROUP BY document_type
        ORDER BY documents DESC
        """,
    )
    by_year = query_local_pipeline(
        config,
        source,
        """
        SELECT year, COUNT(*) AS documents
        FROM documents
        WHERE year IS NOT NULL
        GROUP BY year
        ORDER BY year
        """,
    )
except Exception as exc:  # noqa: BLE001
    st.info(f"No persisted {source} pipeline is available yet: {exc}")
    st.stop()

metrics = st.columns(3)
metrics[0].metric("Documents", details["document_rows"])
metrics[1].metric("Dataset", details["dataset"])
metrics[2].metric("Tables", len(details["tables"]))

left, right = st.columns(2)
with left:
    st.subheader("Documents by medium")
    st.bar_chart(by_kind.set_index("media_kind"))
with right:
    st.subheader("Documents by content type")
    st.bar_chart(by_type.set_index("document_type"))

st.subheader("Documents by release year")
st.bar_chart(by_year.set_index("year"))

with st.expander("Pipeline metadata"):
    st.json(details)
