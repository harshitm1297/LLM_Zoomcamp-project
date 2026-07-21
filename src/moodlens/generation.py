from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .models import SearchHit

BASELINE_SYSTEM_PROMPT = """You are MoodLens, an assistant that explores stories and cultural
themes. Answer the question using the supplied evidence. Keep the response concise."""

STRICT_SYSTEM_PROMPT = """You are MoodLens, an assistant that explores stories and cultural themes.
Answer using only the supplied evidence. Every factual claim must include the supporting evidence
number in brackets, for example: "The residents organize for environmental justice [1]."
Never provide a factual answer without at least one bracketed evidence citation.
Do not use outside knowledge to fill gaps. If the evidence does not support an answer, respond with
exactly: "I don't have enough indexed evidence to answer that."
Keep the answer clear and concise."""

SYSTEM_PROMPT = STRICT_SYSTEM_PROMPT
SYSTEM_PROMPTS = {"baseline": BASELINE_SYSTEM_PROMPT, "strict": STRICT_SYSTEM_PROMPT}


def build_user_prompt(question: str, hits: list[SearchHit], *, max_chars: int = 7000) -> str:
    blocks: list[str] = []
    used = 0
    for number, hit in enumerate(hits, start=1):
        metadata = hit.chunk.metadata
        header = (
            f"[{number}] {metadata.get('title', 'Untitled')} | "
            f"{metadata.get('media_kind', 'unknown')} | {metadata.get('document_type', 'text')}"
        )
        block = f"{header}\n{hit.chunk.text.strip()}"
        if blocks and used + len(block) > max_chars:
            break
        blocks.append(block[: max_chars - used])
        used += len(block)
    evidence = "\n\n".join(blocks) or "No evidence was retrieved."
    return f"Question:\n{question.strip()}\n\nEvidence:\n{evidence}"


@dataclass(frozen=True)
class GenerationResult:
    text: str
    prompt_tokens: int | None
    completion_tokens: int | None


class Generator(Protocol):
    model: str

    def generate(self, question: str, hits: list[SearchHit]) -> GenerationResult: ...


class GroqGenerator:
    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        prompt_variant: str = "strict",
        temperature: float = 0.0,
    ) -> None:
        if not api_key:
            raise ValueError("GROQ_API_KEY is required to generate answers")
        if prompt_variant not in SYSTEM_PROMPTS:
            raise ValueError(f"Unknown prompt variant: {prompt_variant}")
        from groq import Groq

        self.model = model
        self.prompt_variant = prompt_variant
        self.temperature = temperature
        self._client = Groq(api_key=api_key)

    def generate(self, question: str, hits: list[SearchHit]) -> GenerationResult:
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=500,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPTS[self.prompt_variant]},
                {"role": "user", "content": build_user_prompt(question, hits)},
            ],
        )
        usage = getattr(response, "usage", None)
        return GenerationResult(
            text=response.choices[0].message.content.strip(),
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
        )
