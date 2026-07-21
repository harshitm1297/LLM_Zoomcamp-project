from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_dotenv(path: Path | None = None) -> None:
    dotenv = path or project_root() / ".env"
    if not dotenv.is_file():
        return
    for raw_line in dotenv.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def _integer(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _path(name: str, default: str) -> Path:
    value = Path(os.getenv(name, default))
    return value if value.is_absolute() else project_root() / value


@dataclass(frozen=True)
class Settings:
    groq_api_key: str
    tmdb_api_key: str
    tmdb_language: str
    tmdb_start_date: str
    tmdb_end_date: str
    tmdb_movie_limit: int
    tmdb_tv_limit: int
    embedding_model: str
    embedding_backend: str
    groq_model: str
    prompt_variant: str
    retrieval_top_k: int
    retrieval_strategy: str
    enable_query_rewriting: bool
    chunk_words: int
    chunk_overlap: int
    database_path: Path
    artifacts_dir: Path


def settings() -> Settings:
    load_dotenv()
    result = Settings(
        groq_api_key=os.getenv("GROQ_API_KEY", "").strip(),
        tmdb_api_key=os.getenv("TMDB_API_KEY", "").strip(),
        tmdb_language=os.getenv("TMDB_LANGUAGE", "en-US").strip() or "en-US",
        tmdb_start_date=os.getenv("TMDB_START_DATE", "2025-01-01").strip(),
        tmdb_end_date=os.getenv("TMDB_END_DATE", "2026-12-31").strip(),
        tmdb_movie_limit=_integer("TMDB_MOVIE_LIMIT", 100),
        tmdb_tv_limit=_integer("TMDB_TV_LIMIT", 100),
        embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5").strip(),
        embedding_backend=os.getenv("EMBEDDING_BACKEND", "sentence-transformers").strip().lower(),
        groq_model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant").strip(),
        prompt_variant=os.getenv("PROMPT_VARIANT", "strict").strip().lower(),
        retrieval_top_k=_integer("RETRIEVAL_TOP_K", 5),
        retrieval_strategy=os.getenv("RETRIEVAL_STRATEGY", "bm25").strip().lower(),
        enable_query_rewriting=os.getenv("ENABLE_QUERY_REWRITING", "true").strip().lower()
        in {"1", "true", "yes", "on"},
        chunk_words=_integer("CHUNK_WORDS", 180),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "30")),
        database_path=_path("MOODLENS_DB_PATH", "runtime/moodlens.sqlite3"),
        artifacts_dir=_path("MOODLENS_ARTIFACTS_DIR", "artifacts"),
    )
    if result.chunk_overlap < 0 or result.chunk_overlap >= result.chunk_words:
        raise ValueError("CHUNK_OVERLAP must be non-negative and smaller than CHUNK_WORDS")
    if result.embedding_backend not in {"sentence-transformers", "hashing"}:
        raise ValueError("EMBEDDING_BACKEND must be sentence-transformers or hashing")
    if result.prompt_variant not in {"baseline", "strict"}:
        raise ValueError("PROMPT_VARIANT must be baseline or strict")
    if result.retrieval_strategy not in {"bm25", "vector", "hybrid", "vector_reranked"}:
        raise ValueError("Unsupported RETRIEVAL_STRATEGY")
    return result
