from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cultural_mood_tracker.config import settings
from cultural_mood_tracker.evaluation import evaluate_retrieval_strategies, load_retrieval_cases
from cultural_mood_tracker.factory import database, embedder
from cultural_mood_tracker.retrieval import Retriever


def main() -> int:
    config = settings()
    cases = load_retrieval_cases(ROOT / "data/evaluation/retrieval_golden.jsonl")
    strategies = ("bm25", "vector", "hybrid", "vector_reranked")
    report = evaluate_retrieval_strategies(
        Retriever(
            database(),
            embedder(),
            strategy=config.retrieval_strategy,
            enable_query_rewriting=config.enable_query_rewriting,
        ),
        cases,
        strategies=strategies,
        top_k=config.retrieval_top_k,
    )
    report["generated_at"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    report["embedding_backend"] = config.embedding_backend
    report["embedding_model"] = config.embedding_model
    output = ROOT / "data/evaluation/results/retrieval_evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = {
        strategy: values["metrics"] for strategy, values in report["strategies"].items()
    }
    print(json.dumps({"best_strategy": report["best_strategy"], "results": summary}, indent=2))
    print(f"Full report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
