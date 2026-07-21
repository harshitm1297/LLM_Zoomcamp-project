from __future__ import annotations

import json
from pathlib import Path

from .models import Document


def demo_source_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "corpus" / "demo_documents.jsonl"


def demo_documents(path: Path | None = None) -> list[Document]:
    """Read the deterministic demo corpus from its versioned JSONL source."""
    source = path or demo_source_path()
    with source.open(encoding="utf-8") as handle:
        return [Document.from_record(json.loads(line)) for line in handle if line.strip()]
