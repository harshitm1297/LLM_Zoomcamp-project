from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .retrieval import Retriever


@dataclass(frozen=True)
class RetrievalCase:
    case_id: str
    question: str
    relevant_chunk_ids: frozenset[str]


def load_retrieval_cases(path: Path) -> list[RetrievalCase]:
    output: list[RetrievalCase] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            relevant = frozenset(str(value) for value in row["relevant_chunk_ids"])
            if not relevant:
                raise ValueError(f"evaluation row {number} has no relevant chunks")
            output.append(
                RetrievalCase(
                    case_id=str(row["case_id"]),
                    question=str(row["question"]),
                    relevant_chunk_ids=relevant,
                )
            )
    if not output:
        raise ValueError("retrieval golden set is empty")
    return output


def reciprocal_rank(retrieved: list[str], relevant: frozenset[str]) -> float:
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def evaluate_retrieval(
    retriever: Retriever,
    cases: list[RetrievalCase],
    *,
    top_k: int = 5,
    strategy: str | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        started = time.perf_counter()
        if strategy is None:
            hits = retriever.search(case.question, top_k=top_k)
        else:
            hits = retriever.search(case.question, top_k=top_k, strategy=strategy)
        latency_ms = (time.perf_counter() - started) * 1000
        ids = [hit.chunk.chunk_id for hit in hits]
        found = set(ids) & case.relevant_chunk_ids
        rows.append(
            {
                "case_id": case.case_id,
                "question": case.question,
                "relevant_chunk_ids": sorted(case.relevant_chunk_ids),
                "retrieved_chunk_ids": ids,
                "hit_rate": float(bool(found)),
                "mrr": reciprocal_rank(ids, case.relevant_chunk_ids),
                "recall_at_k": len(found) / len(case.relevant_chunk_ids),
                "precision_at_k": len(found) / len(ids) if ids else 0.0,
                "latency_ms": latency_ms,
            }
        )

    def mean(field: str) -> float:
        return sum(float(row[field]) for row in rows) / len(rows)

    return {
        "strategy": strategy or getattr(retriever, "strategy", "default"),
        "top_k": top_k,
        "case_count": len(rows),
        "metrics": {
            "hit_rate": mean("hit_rate"),
            "mrr": mean("mrr"),
            "recall_at_k": mean("recall_at_k"),
            "precision_at_k": mean("precision_at_k"),
            "mean_latency_ms": mean("latency_ms"),
        },
        "cases": rows,
    }


def evaluate_retrieval_strategies(
    retriever: Retriever,
    cases: list[RetrievalCase],
    *,
    strategies: tuple[str, ...],
    top_k: int = 5,
) -> dict[str, Any]:
    reports = {
        strategy: evaluate_retrieval(
            retriever, cases, top_k=top_k, strategy=strategy
        )
        for strategy in strategies
    }
    winner = max(
        strategies,
        key=lambda strategy: (
            float(reports[strategy]["metrics"]["mrr"]),
            float(reports[strategy]["metrics"]["hit_rate"]),
            -strategies.index(strategy),
        ),
    )
    return {
        "selection_metric": "mrr",
        "best_strategy": winner,
        "best_score": reports[winner]["metrics"]["mrr"],
        "strategies": reports,
    }


JUDGE_PROMPT = """Evaluate a grounded assistant answer.
Return only JSON with fields relevance, groundedness, and explanation.
relevance and groundedness must each be one of: 0, 1, 2.
0 means poor, 1 means partial, 2 means strong.
If the reference says the requested fact is absent and the evidence does not provide it, a clear
refusal is the fully relevant and fully grounded response and should receive 2 for both scores.

Question: {question}
Reference facts: {reference}
Retrieved evidence: {evidence}
Assistant answer: {answer}
"""


def judge_answer(
    *,
    client: Any,
    model: str,
    question: str,
    reference: str,
    evidence: str,
    answer: str,
) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "user",
                "content": JUDGE_PROMPT.format(
                    question=question,
                    reference=reference,
                    evidence=evidence,
                    answer=answer,
                ),
            }
        ],
    )
    result = json.loads(response.choices[0].message.content)
    for field in ("relevance", "groundedness"):
        score = int(result[field])
        if score not in {0, 1, 2}:
            raise ValueError(f"judge returned invalid {field}: {score}")
        result[field] = score
    return result
