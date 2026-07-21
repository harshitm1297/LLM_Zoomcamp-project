from __future__ import annotations

import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from groq import Groq

from cultural_mood_tracker.assistant import CulturalMoodTrackerAssistant
from cultural_mood_tracker.config import settings
from cultural_mood_tracker.evaluation import judge_answer
from cultural_mood_tracker.factory import database, embedder
from cultural_mood_tracker.generation import GroqGenerator
from cultural_mood_tracker.retrieval import Retriever

REFUSAL = "I don't have enough indexed evidence to answer that."
CITATION = re.compile(r"\[\d+\]")


def load_cases() -> list[dict[str, object]]:
    with (ROOT / "data/evaluation/answer_golden.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def evaluate_variant(
    variant: str,
    *,
    client: Groq,
    cases: list[dict[str, object]],
) -> dict[str, object]:
    config = settings()
    retriever = Retriever(
        database(),
        embedder(),
        strategy=config.retrieval_strategy,
        enable_query_rewriting=config.enable_query_rewriting,
    )
    application = CulturalMoodTrackerAssistant(
        retriever,
        GroqGenerator(
            config.groq_api_key,
            config.groq_model,
            prompt_variant=variant,
            temperature=0.0,
        ),
        top_k=config.retrieval_top_k,
    )
    rows: list[dict[str, object]] = []
    for case in cases:
        question = str(case["question"])
        result = application.answer(question)
        evidence = "\n".join(hit.chunk.text for hit in result.hits)
        judgement = judge_answer(
            client=client,
            model=config.groq_model,
            question=question,
            reference=str(case["reference"]),
            evidence=evidence,
            answer=result.text,
        )
        answerable = bool(case["answerable"])
        policy_compliant = (
            bool(CITATION.search(result.text))
            if answerable
            else result.text.strip() == REFUSAL
        )
        rows.append(
            {
                **case,
                "answer": result.text,
                "retrieved_chunk_ids": [hit.chunk.chunk_id for hit in result.hits],
                "judgement": judgement,
                "policy_compliant": policy_compliant,
            }
        )
    relevance = sum(int(row["judgement"]["relevance"]) for row in rows) / (2 * len(rows))
    groundedness = sum(int(row["judgement"]["groundedness"]) for row in rows) / (
        2 * len(rows)
    )
    policy_compliance = sum(bool(row["policy_compliant"]) for row in rows) / len(rows)
    return {
        "prompt_variant": variant,
        "metrics": {
            "relevance": relevance,
            "groundedness": groundedness,
            "policy_compliance": policy_compliance,
            "composite": (relevance + groundedness + policy_compliance) / 3,
        },
        "cases": rows,
    }


def main() -> int:
    config = settings()
    if not config.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is required for answer evaluation")
    client = Groq(api_key=config.groq_api_key)
    cases = load_cases()
    variants = ("baseline", "strict")
    results = {
        variant: evaluate_variant(variant, client=client, cases=cases)
        for variant in variants
    }
    winner = max(
        variants,
        key=lambda variant: (
            float(results[variant]["metrics"]["composite"]),
            -variants.index(variant),
        ),
    )
    report = {
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "model": config.groq_model,
        "retrieval_strategy": config.retrieval_strategy,
        "embedding_backend": config.embedding_backend,
        "case_count": len(cases),
        "selection_metric": "composite",
        "best_prompt_variant": winner,
        "best_score": results[winner]["metrics"]["composite"],
        "prompt_variants": results,
    }
    output = ROOT / "data/evaluation/results/llm_evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = {
        variant: values["metrics"] for variant, values in report["prompt_variants"].items()
    }
    print(json.dumps({"best_prompt_variant": winner, "results": summary}, indent=2))
    print(f"Full report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
