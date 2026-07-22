from __future__ import annotations

import sys
import uuid
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from cultural_mood_tracker.config import settings
from cultural_mood_tracker.factory import assistant, database, telemetry

st.set_page_config(page_title="Cultural Mood Tracker", page_icon="🎞️", layout="centered")
st.markdown(
    """
    <style>
    :root { --ink:#172033; --muted:#5d6778; --accent:#5b4bdb; --line:#dce1ea; }
    .stApp { background:#f5f6fa; color:var(--ink); }
    .block-container { max-width:900px; padding-top:2rem; }
    .hero { background:linear-gradient(130deg,#302a73,#5b4bdb); border-radius:22px;
            padding:1.7rem 1.9rem; color:white; margin-bottom:1rem; }
    .hero h1 { color:white!important; margin:0; font-size:2.15rem; }
    .hero p { color:#eeecff!important; margin:.45rem 0 0; }
    [data-testid="stChatMessage"] { background:white; border:1px solid var(--line);
      border-radius:15px; box-shadow:0 3px 10px rgba(23,32,51,.04); }
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li { color:var(--ink)!important; }
    [data-testid="stChatMessage"][aria-label="Chat message from user"] { background:#ebe9ff; }
    .source { color:var(--muted); font-size:.86rem; border-left:3px solid #bcb6ff;
              padding-left:.7rem; margin:.55rem 0; }
    section[data-testid="stSidebar"] { background:white; }
    </style>
    <div class="hero">
      <h1>Cultural Mood Tracker</h1>
      <p>Explore plots, themes, emotions, and cultural tensions through grounded evidence.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def render_evidence(hits: list[dict]) -> None:
    with st.expander(f"Evidence ({len(hits)} passages)"):
        for number, hit in enumerate(hits, start=1):
            metadata = hit["metadata"]
            st.markdown(
                '<div class="source"><strong>'
                f'{number}. {metadata.get("title", "Untitled")}</strong>'
                f' · similarity {hit["score"]:.3f}<br>{hit["text"]}</div>',
                unsafe_allow_html=True,
            )


if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

status = database().corpus_status()
config = settings()
with st.sidebar:
    st.header("Knowledge base")
    if status:
        st.success(
            f"{status['document_count']} documents · {status['chunk_count']} passages\n\n"
            f"Source: {status['source']}"
        )
    else:
        st.warning("No corpus is indexed yet. Run the demo ingestion command.")
    st.caption(
        f"Every answer uses the evaluated {config.retrieval_strategy} RAG strategy. "
        "There is no SQL or agent route."
    )
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.caption("Try: Which story explores environmental inequality? · Find a title about grief "
           "and imagination.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("hits"):
            render_evidence(message["hits"])
        if interaction_id := message.get("interaction_id"):
            selection = st.feedback("thumbs", key=f"feedback-{interaction_id}")
            if selection is not None:
                telemetry().feedback(interaction_id, 1 if selection == 1 else -1)

question = st.chat_input("Ask about the indexed stories...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Finding evidence and composing an answer..."):
            try:
                result = assistant().answer(question)
                hit_rows = [
                    {
                        "chunk_id": hit.chunk.chunk_id,
                        "text": hit.chunk.text,
                        "metadata": hit.chunk.metadata,
                        "score": hit.score,
                    }
                    for hit in result.hits
                ]
                interaction_id = telemetry().record(
                    session_id=st.session_state.session_id,
                    question=question,
                    answer=result,
                )
                st.markdown(result.text)
                st.caption(
                    f"{len(result.hits)} passages · {result.latency_ms / 1000:.2f}s · "
                    f"{result.model}"
                )
                render_evidence(hit_rows)
                message = {
                    "role": "assistant",
                    "content": result.text,
                    "hits": hit_rows,
                    "interaction_id": interaction_id,
                }
            except Exception as exc:  # noqa: BLE001
                friendly = str(exc)
                telemetry().record(
                    session_id=st.session_state.session_id,
                    question=question,
                    answer=None,
                    error=friendly,
                )
                st.error(friendly)
                message = {"role": "assistant", "content": friendly}
    st.session_state.messages.append(message)
