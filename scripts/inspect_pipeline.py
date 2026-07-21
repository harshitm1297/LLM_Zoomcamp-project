from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moodlens.config import settings
from moodlens.dlt_pipeline import inspect_local_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a persisted MoodLens dlt dataset.")
    parser.add_argument("--source", choices=("demo", "tmdb"), default="demo")
    args = parser.parse_args()
    print(json.dumps(inspect_local_pipeline(settings(), args.source), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
