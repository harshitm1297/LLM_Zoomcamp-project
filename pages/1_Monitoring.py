from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moodlens.factory import telemetry

st.set_page_config(page_title="MoodLens Monitoring", page_icon="📈", layout="wide")
st.title("MoodLens monitoring")
summary = telemetry().summary()
rows = telemetry().rows()

metrics = st.columns(6)
metrics[0].metric("Requests", int(summary.get("requests") or 0))
metrics[1].metric("Errors", int(summary.get("errors") or 0))
metrics[2].metric("Mean latency", f"{float(summary.get('average_latency_ms') or 0):.0f} ms")
metrics[3].metric("Mean similarity", f"{float(summary.get('average_similarity') or 0):.3f}")
metrics[4].metric("Tracked tokens", int(summary.get("tokens") or 0))
metrics[5].metric("Positive feedback", f"{100 * float(summary.get('positive_rate') or 0):.0f}%")

if not rows:
    st.info("No interactions have been recorded yet.")
    st.stop()

frame = pd.DataFrame(rows)
frame["created_at"] = pd.to_datetime(frame["created_at"], utc=True)
frame["date"] = frame["created_at"].dt.date
frame["total_tokens"] = frame["prompt_tokens"].fillna(0) + frame["completion_tokens"].fillna(0)
frame["is_error"] = frame["error"].notna().astype(int)

left, right = st.columns(2)
with left:
    st.subheader("Requests by day")
    st.line_chart(frame.groupby("date").size().rename("requests"))
with right:
    st.subheader("Latency by day")
    st.line_chart(frame.groupby("date")["latency_ms"].mean())

left, right = st.columns(2)
with left:
    st.subheader("Retrieval similarity")
    similarity = frame.dropna(subset=["mean_similarity"])
    if similarity.empty:
        st.info("No similarity measurements yet.")
    else:
        st.line_chart(similarity.set_index("created_at")["mean_similarity"])
with right:
    st.subheader("User feedback")
    feedback = frame.dropna(subset=["feedback_score"])
    if feedback.empty:
        st.info("No user feedback yet.")
    else:
        st.bar_chart(feedback.groupby("feedback_score").size().rename("responses"))

left, right = st.columns(2)
with left:
    st.subheader("Token usage by day")
    st.line_chart(frame.groupby("date")["total_tokens"].sum())
with right:
    st.subheader("Errors by day")
    st.line_chart(frame.groupby("date")["is_error"].sum())

st.subheader("Recent interactions")
st.dataframe(
    frame[
        [
            "created_at",
            "question",
            "model",
            "latency_ms",
            "prompt_tokens",
            "completion_tokens",
            "mean_similarity",
            "feedback_score",
            "error",
        ]
    ],
    use_container_width=True,
    hide_index=True,
)
