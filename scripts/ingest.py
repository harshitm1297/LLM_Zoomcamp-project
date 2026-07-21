from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moodlens.config import settings
from moodlens.ingestion import ingest_demo, ingest_tmdb


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the MoodLens local search index.")
    parser.add_argument("--source", choices=("demo", "tmdb"), default="demo")
    args = parser.parse_args()
    config = settings()
    manifest = ingest_demo(config) if args.source == "demo" else ingest_tmdb(config)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

