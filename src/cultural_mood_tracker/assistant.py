from __future__ import annotations

import time

from .generation import Generator
from .models import Answer
from .retrieval import Retriever, normalize_query


class CulturalMoodTrackerAssistant:
    def __init__(self, retriever: Retriever, generator: Generator, *, top_k: int = 5) -> None:
        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k

    def answer(self, question: str) -> Answer:
        normalized = normalize_query(question)
        started = time.perf_counter()
        hits = self.retriever.search(normalized, top_k=self.top_k)
        generation = self.generator.generate(normalized, hits)
        return Answer(
            text=generation.text,
            hits=tuple(hits),
            model=self.generator.model,
            latency_ms=(time.perf_counter() - started) * 1000,
            prompt_tokens=generation.prompt_tokens,
            completion_tokens=generation.completion_tokens,
        )

