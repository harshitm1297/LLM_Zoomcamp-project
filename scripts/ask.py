from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moodlens.factory import assistant


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask the MoodLens RAG assistant a question.")
    parser.add_argument("question")
    args = parser.parse_args()
    result = assistant().answer(args.question)
    print(
        json.dumps(
            {
                "answer": result.text,
                "model": result.model,
                "latency_ms": result.latency_ms,
                "evidence": [
                    {
                        "chunk_id": hit.chunk.chunk_id,
                        "title": hit.chunk.metadata.get("title"),
                        "similarity": hit.score,
                        "text": hit.chunk.text,
                    }
                    for hit in result.hits
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

