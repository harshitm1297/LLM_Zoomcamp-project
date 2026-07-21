from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from .chunking import chunk_documents
from .config import Settings
from .database import Database
from .dlt_pipeline import LocalPipelineResult, load_demo_source, load_document_resource
from .embeddings import Embedder, make_embedder
from .models import Document
from .sample_data import demo_source_path
from .tmdb import download_documents


def _timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _run_id(created_at: str) -> str:
    return created_at.replace("-", "").replace(":", "").replace(".", "").replace("Z", "Z")


def _write_documents(path: Path, documents: Sequence[Document]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        for document in documents:
            handle.write(
                json.dumps(
                    {**document.metadata(), "text": document.text}, ensure_ascii=False
                )
                + "\n"
            )


def _index_pipeline_result(
    result: LocalPipelineResult,
    *,
    source: str,
    created_at: str,
    run_id: str,
    config: Settings,
    embedder: Embedder | None,
) -> dict[str, object]:
    documents = result.documents
    chunks = chunk_documents(
        documents, words=config.chunk_words, overlap=config.chunk_overlap
    )
    if not chunks:
        raise ValueError("The dlt dataset did not produce any searchable chunks")
    encoder = embedder or make_embedder(config.embedding_backend, config.embedding_model)
    matrix = encoder.documents([chunk.text for chunk in chunks])
    Database(config.database_path).replace_corpus(
        run_id=run_id,
        source=source,
        created_at=created_at,
        document_count=len(documents),
        chunks=chunks,
        embeddings=matrix,
        embedding_model=getattr(encoder, "model_name", config.embedding_model),
    )

    run_dir = config.artifacts_dir / "corpora" / run_id
    _write_documents(run_dir / "documents.jsonl", documents)
    with (run_dir / "chunks.jsonl").open("w", encoding="utf-8", newline="") as handle:
        for chunk in chunks:
            handle.write(
                json.dumps(
                    {
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "position": chunk.position,
                        "text": chunk.text,
                        "metadata": chunk.metadata,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    manifest: dict[str, object] = {
        "run_id": run_id,
        "created_at": created_at,
        "source": source,
        "documents": len(documents),
        "chunks": len(chunks),
        "embedding_model": getattr(encoder, "model_name", config.embedding_model),
        "embedding_dimensions": int(matrix.shape[1]),
        "database": str(config.database_path),
        "dlt": result.metadata,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def ingest_documents(
    documents: Sequence[Document],
    *,
    source: str,
    config: Settings,
    embedder: Embedder | None = None,
) -> dict[str, object]:
    created_at = _timestamp()
    run_id = _run_id(created_at)
    result = load_document_resource(
        documents, source=source, config=config, run_id=run_id
    )
    return _index_pipeline_result(
        result,
        source=source,
        created_at=created_at,
        run_id=run_id,
        config=config,
        embedder=embedder,
    )


def ingest_demo(config: Settings, *, embedder: Embedder | None = None) -> dict[str, object]:
    created_at = _timestamp()
    run_id = _run_id(created_at)
    result = load_demo_source(demo_source_path(), config=config, run_id=run_id)
    return _index_pipeline_result(
        result,
        source="demo",
        created_at=created_at,
        run_id=run_id,
        config=config,
        embedder=embedder,
    )


def ingest_tmdb(config: Settings, *, embedder: Embedder | None = None) -> dict[str, object]:
    documents = download_documents(
        api_key=config.tmdb_api_key,
        language=config.tmdb_language,
        start_date=config.tmdb_start_date,
        end_date=config.tmdb_end_date,
        movie_limit=config.tmdb_movie_limit,
        tv_limit=config.tmdb_tv_limit,
    )
    return ingest_documents(documents, source="tmdb", config=config, embedder=embedder)
