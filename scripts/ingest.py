from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cultural_mood_tracker.config import settings
from cultural_mood_tracker.ingestion import ingest_demo, ingest_tmdb


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the Cultural Mood Tracker local search index."
    )
    parser.add_argument("--source", choices=("demo", "tmdb"), default="demo")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--sample",
        action="store_true",
        help="For TMDB, load at most three movies and three TV shows.",
    )
    mode.add_argument(
        "--full",
        action="store_true",
        help="Load the configured full source limits (the default).",
    )
    args = parser.parse_args()
    config = settings()
    manifest = (
        ingest_demo(config)
        if args.source == "demo"
        else ingest_tmdb(config, sample=args.sample)
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
