from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

from .models import Document


def _record(document: Document, *, ingestion_run_id: str) -> dict[str, object]:
    """Convert a domain document into a flat, schema-friendly dlt record."""
    return {
        **document.metadata(),
        "text": document.text,
        "ingestion_run_id": ingestion_run_id,
    }


def load_documents_locally(
    documents: Sequence[Document],
    *,
    destination_dir: Path,
    pipelines_dir: Path,
    run_id: str,
) -> dict[str, object]:
    """Extract, normalize, and load source documents with dlt to local files."""
    destination_dir.mkdir(parents=True, exist_ok=True)
    pipelines_dir.mkdir(parents=True, exist_ok=True)

    # Configure telemetry before importing dlt, which initializes its runtime on import.
    os.environ.setdefault("RUNTIME__DLTHUB_TELEMETRY", "false")
    import dlt
    from dlt.destinations import filesystem

    pipeline = dlt.pipeline(
        pipeline_name="moodlens_ingestion",
        destination=filesystem(bucket_url=str(destination_dir.resolve())),
        dataset_name="corpus",
        pipelines_dir=str(pipelines_dir.resolve()),
    )
    load_info = pipeline.run(
        (_record(document, ingestion_run_id=run_id) for document in documents),
        table_name="documents",
        write_disposition="replace",
        loader_file_format="jsonl",
    )
    return {
        "pipeline": pipeline.pipeline_name,
        "dataset": pipeline.dataset_name,
        "load_ids": list(load_info.loads_ids),
        "destination": str(destination_dir.resolve()),
    }
