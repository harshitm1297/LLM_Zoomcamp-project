from __future__ import annotations

from functools import lru_cache

from .assistant import MoodLensAssistant
from .config import settings
from .database import Database
from .embeddings import Embedder, make_embedder
from .generation import GroqGenerator
from .retrieval import Retriever
from .telemetry import Telemetry


@lru_cache(maxsize=1)
def database() -> Database:
    return Database(settings().database_path)


@lru_cache(maxsize=1)
def embedder() -> Embedder:
    config = settings()
    return make_embedder(config.embedding_backend, config.embedding_model)


@lru_cache(maxsize=1)
def assistant() -> MoodLensAssistant:
    config = settings()
    return MoodLensAssistant(
        Retriever(
            database(),
            embedder(),
            strategy=config.retrieval_strategy,
            enable_query_rewriting=config.enable_query_rewriting,
        ),
        GroqGenerator(
            config.groq_api_key,
            config.groq_model,
            prompt_variant=config.prompt_variant,
        ),
        top_k=config.retrieval_top_k,
    )


@lru_cache(maxsize=1)
def telemetry() -> Telemetry:
    return Telemetry(database())
