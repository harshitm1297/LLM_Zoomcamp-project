from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Settings
from .models import Document

PIPELINE_PREFIX = "cultural_mood_tracker"
TABLE_NAME = "documents"
DOCUMENT_COLUMNS: dict[str, dict[str, object]] = {
    "document_id": {"data_type": "text", "nullable": False},
    "title": {"data_type": "text", "nullable": False},
    "media_kind": {"data_type": "text", "nullable": False},
    "text": {"data_type": "text", "nullable": False},
    "source": {"data_type": "text", "nullable": False},
    "source_url": {"data_type": "text", "nullable": False},
    "year": {"data_type": "bigint", "nullable": True},
    "genres": {"data_type": "json", "nullable": False},
    "document_type": {"data_type": "text", "nullable": False},
    "ingestion_run_id": {"data_type": "text", "nullable": False},
}


@dataclass(frozen=True)
class LocalPipelineResult:
    documents: tuple[Document, ...]
    metadata: dict[str, object]


def _dlt() -> Any:
    # dlt initializes its runtime on import, so configure telemetry first.
    os.environ.setdefault("RUNTIME__DLTHUB_TELEMETRY", "false")
    import dlt

    return dlt


def _record(document: Document, *, ingestion_run_id: str) -> dict[str, object]:
    return {
        **document.metadata(),
        "text": document.text,
        "ingestion_run_id": ingestion_run_id,
    }


def _pipeline(config: Settings, source: str) -> Any:
    dlt = _dlt()
    config.dlt_database_path.parent.mkdir(parents=True, exist_ok=True)
    pipelines_dir = config.artifacts_dir / "dlt_state"
    pipelines_dir.mkdir(parents=True, exist_ok=True)
    return dlt.pipeline(
        pipeline_name=f"{PIPELINE_PREFIX}_{source}_ingestion",
        destination=dlt.destinations.duckdb(str(config.dlt_database_path.resolve())),
        dataset_name=f"{PIPELINE_PREFIX}_{source}",
        pipelines_dir=str(pipelines_dir.resolve()),
        dev_mode=False,
    )


def _apply_document_hints(resource: Any, *, write_disposition: str) -> Any:
    resource.apply_hints(
        primary_key="document_id",
        write_disposition=write_disposition,
        columns=DOCUMENT_COLUMNS,
    )
    return resource


def _read_documents(
    pipeline: Any, *, ingestion_run_id: str | None = None
) -> tuple[Document, ...]:
    frame = pipeline.dataset().table(TABLE_NAME).df()
    if ingestion_run_id is not None:
        frame = frame[frame["ingestion_run_id"] == ingestion_run_id]
    rows = frame.to_dict(orient="records")
    documents = tuple(Document.from_record(row) for row in rows)
    if not documents:
        raise ValueError("dlt loaded no documents")
    return tuple(sorted(documents, key=lambda document: document.document_id))


def _metadata(
    pipeline: Any,
    load_info: Any,
    documents: Sequence[Document],
    *,
    write_disposition: str,
    incremental: bool,
) -> dict[str, object]:
    tables = sorted(
        name for name in pipeline.default_schema.tables if not name.startswith("_dlt")
    )
    normalize_info = getattr(
        getattr(pipeline, "last_trace", None), "last_normalize_info", None
    )
    return {
        "pipeline": pipeline.pipeline_name,
        "dataset": pipeline.dataset_name,
        "destination": "duckdb",
        "database": str(pipeline.destination.config_params.get("credentials", "duckdb")),
        "load_ids": list(load_info.loads_ids),
        "write_disposition": write_disposition,
        "incremental": incremental,
        "tables": tables,
        "table_count": len(tables),
        "document_rows": len(documents),
        "load_info": str(load_info),
        "normalize_info": str(normalize_info) if normalize_info is not None else None,
    }


def load_demo_source(
    source_path: Path,
    *,
    config: Settings,
    run_id: str,
) -> LocalPipelineResult:
    """Load versioned JSONL through dlt's incremental filesystem source."""
    dlt = _dlt()
    from dlt.sources.filesystem import filesystem, read_jsonl

    files = filesystem(
        bucket_url=str(source_path.parent.resolve()),
        file_glob=source_path.name,
        incremental=dlt.sources.incremental("modification_date"),
    )
    reader = (files | read_jsonl()).with_name(TABLE_NAME)
    reader.add_map(lambda row: {**row, "ingestion_run_id": run_id})
    resource = _apply_document_hints(reader, write_disposition="merge")
    pipeline = _pipeline(config, "demo")
    load_info = pipeline.run(resource)
    documents = _read_documents(pipeline)
    return LocalPipelineResult(
        documents,
        _metadata(
            pipeline,
            load_info,
            documents,
            write_disposition="merge",
            incremental=True,
        ),
    )


def load_document_resource(
    documents: Sequence[Document],
    *,
    source: str,
    config: Settings,
    run_id: str,
) -> LocalPipelineResult:
    """Load API-derived documents as a keyed dlt resource with merge semantics."""
    if not documents:
        raise ValueError("Cannot load an empty dlt resource")
    dlt = _dlt()

    @dlt.resource(
        name=TABLE_NAME,
        primary_key="document_id",
        write_disposition="merge",
        columns=DOCUMENT_COLUMNS,
    )
    def document_resource() -> Any:
        for document in documents:
            yield _record(document, ingestion_run_id=run_id)

    pipeline = _pipeline(config, source)
    load_info = pipeline.run(document_resource())
    loaded = _read_documents(pipeline, ingestion_run_id=run_id)
    return LocalPipelineResult(
        loaded,
        _metadata(
            pipeline,
            load_info,
            loaded,
            write_disposition="merge",
            incremental=False,
        ),
    )


def load_tmdb_source(
    *,
    config: Settings,
    run_id: str,
    sample: bool = False,
) -> LocalPipelineResult:
    """Extract TMDB with dlt's declarative REST API source and load it locally."""
    from .tmdb_source import build_tmdb_source

    source = build_tmdb_source(
        api_key=config.tmdb_api_key,
        language=config.tmdb_language,
        start_date=config.tmdb_start_date,
        end_date=config.tmdb_end_date,
        movie_limit=config.tmdb_movie_limit,
        tv_limit=config.tmdb_tv_limit,
        run_id=run_id,
        columns=DOCUMENT_COLUMNS,
        sample=sample,
    )
    pipeline = _pipeline(config, "tmdb")
    load_info = pipeline.run(source)
    loaded = _read_documents(pipeline, ingestion_run_id=run_id)
    return LocalPipelineResult(
        loaded,
        _metadata(
            pipeline,
            load_info,
            loaded,
            write_disposition="merge",
            incremental=False,
        ),
    )


def _attach_pipeline(config: Settings, source: str) -> Any:
    dlt = _dlt()
    return dlt.attach(
        pipeline_name=f"{PIPELINE_PREFIX}_{source}_ingestion",
        pipelines_dir=str((config.artifacts_dir / "dlt_state").resolve()),
        destination=dlt.destinations.duckdb(str(config.dlt_database_path.resolve())),
        dataset_name=f"{PIPELINE_PREFIX}_{source}",
    )


def inspect_local_pipeline(config: Settings, source: str) -> dict[str, object]:
    """Inspect persisted dlt schema and rows without running extraction."""
    pipeline = _attach_pipeline(config, source)
    documents = _read_documents(pipeline)
    tables = sorted(
        name for name in pipeline.default_schema.tables if not name.startswith("_dlt")
    )
    return {
        "pipeline": pipeline.pipeline_name,
        "dataset": pipeline.dataset_name,
        "database": str(config.dlt_database_path),
        "tables": tables,
        "document_rows": len(documents),
        "sample_document_ids": [document.document_id for document in documents[:5]],
    }


def query_local_pipeline(config: Settings, source: str, query: str) -> Any:
    """Run a read-only analytical query against an attached local dlt dataset."""
    pipeline = _attach_pipeline(config, source)
    return pipeline.dataset()(query).df()
