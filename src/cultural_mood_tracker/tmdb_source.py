from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

BASE_URL = "https://api.themoviedb.org/3/"
USER_AGENT = "cultural-mood-tracker-learning-project/0.2"
TMDB_PAGE_SIZE = 20


def _year(value: Any) -> int | None:
    text = str(value or "")
    return int(text[:4]) if len(text) >= 4 and text[:4].isdigit() else None


def _first(limit: int) -> Callable[[dict[str, Any]], bool]:
    seen = 0

    def keep(_: dict[str, Any]) -> bool:
        nonlocal seen
        seen += 1
        return seen <= limit

    return keep


def _overview(kind: str, run_id: str) -> Callable[[dict[str, Any]], dict[str, object]]:
    def transform(record: dict[str, Any]) -> dict[str, object]:
        tmdb_id = int(record["id"])
        title = str(record.get("title") or record.get("name") or f"TMDB {tmdb_id}")
        return {
            "document_id": f"tmdb:{kind}:{tmdb_id}:overview",
            "title": title,
            "media_kind": kind,
            "text": str(record.get("overview") or "").strip(),
            "source": "tmdb",
            "source_url": f"https://www.themoviedb.org/{kind}/{tmdb_id}",
            "year": _year(record.get("release_date") or record.get("first_air_date")),
            "genres": [
                str(item["name"])
                for item in record.get("genres") or []
                if isinstance(item, dict) and item.get("name")
            ],
            "document_type": "overview",
            "ingestion_run_id": run_id,
        }

    return transform


def _review(
    kind: str, parent: str, run_id: str
) -> Callable[[dict[str, Any]], dict[str, object]]:
    prefix = f"_{parent}_"

    def transform(record: dict[str, Any]) -> dict[str, object]:
        tmdb_id = int(record[f"{prefix}id"])
        review_id = str(record["id"])
        title = str(
            record.get(f"{prefix}title")
            or record.get(f"{prefix}name")
            or f"TMDB {tmdb_id}"
        )
        return {
            "document_id": f"tmdb:{kind}:{tmdb_id}:review:{review_id}",
            "title": title,
            "media_kind": kind,
            "text": str(record.get("content") or "").strip(),
            "source": "tmdb",
            "source_url": f"https://www.themoviedb.org/{kind}/{tmdb_id}",
            "year": _year(
                record.get(f"{prefix}release_date")
                or record.get(f"{prefix}first_air_date")
            ),
            "genres": [],
            "document_type": "user_review",
            "ingestion_run_id": run_id,
        }

    return transform


def build_tmdb_source(
    *,
    api_key: str,
    language: str,
    start_date: str,
    end_date: str,
    movie_limit: int,
    tv_limit: int,
    run_id: str,
    columns: dict[str, dict[str, object]],
    sample: bool = False,
) -> Any:
    """Build the declarative TMDB source used by the local dlt pipeline."""
    if not api_key:
        raise ValueError("TMDB_API_KEY is required for a live refresh")
    import dlt
    from dlt.sources.rest_api import RESTAPIConfig, rest_api_resources

    movie_limit = min(movie_limit, 3) if sample else movie_limit
    tv_limit = min(tv_limit, 3) if sample else tv_limit
    common_params = {"api_key": api_key, "language": language}

    def details_resource(name: str, parent: str, kind: str) -> dict[str, Any]:
        return {
            "name": name,
            "table_name": "documents",
            "primary_key": "document_id",
            "write_disposition": "merge",
            "max_table_nesting": 0,
            "columns": columns,
            "endpoint": {
                "path": f"{kind}/{{resources.{parent}.id}}",
                "params": common_params,
                "paginator": {"type": "single_page"},
            },
            "processing_steps": [
                {"filter": lambda record: bool(str(record.get("overview") or "").strip())},
                {"map": _overview(kind, run_id)},
            ],
        }

    def reviews_resource(name: str, parent: str, kind: str) -> dict[str, Any]:
        parent_fields = (
            ["id", "title", "release_date"]
            if kind == "movie"
            else ["id", "name", "first_air_date"]
        )
        return {
            "name": name,
            "table_name": "documents",
            "primary_key": "document_id",
            "write_disposition": "merge",
            "max_table_nesting": 0,
            "columns": columns,
            "parallelized": True,
            "endpoint": {
                "path": f"{kind}/{{resources.{parent}.id}}/reviews",
                "params": {**common_params, "page": 1},
                "data_selector": "results",
                "paginator": {"type": "single_page"},
            },
            "include_from_parent": parent_fields,
            "processing_steps": [
                {"filter": lambda record: bool(str(record.get("content") or "").strip())},
                {"map": _review(kind, parent, run_id)},
            ],
        }

    config: RESTAPIConfig = {
        "client": {
            "base_url": BASE_URL,
            "headers": {"User-Agent": USER_AGENT},
        },
        "resources": [
            {
                "name": "movies",
                "selected": False,
                "primary_key": "id",
                "endpoint": {
                    "path": "discover/movie",
                    "params": {
                        **common_params,
                        "sort_by": "popularity.desc",
                        "vote_count.gte": 10,
                        "primary_release_date.gte": start_date,
                        "primary_release_date.lte": end_date,
                    },
                    "data_selector": "results",
                    "paginator": {
                        "type": "page_number",
                        "base_page": 1,
                        "page_param": "page",
                        "total_path": "total_pages",
                        "maximum_page": max(1, math.ceil(movie_limit / TMDB_PAGE_SIZE)),
                    },
                },
                "processing_steps": [{"filter": _first(movie_limit)}],
            },
            details_resource("movie_details", "movies", "movie"),
            reviews_resource("movie_reviews", "movies", "movie"),
            {
                "name": "tv_shows",
                "selected": False,
                "primary_key": "id",
                "endpoint": {
                    "path": "discover/tv",
                    "params": {
                        **common_params,
                        "sort_by": "popularity.desc",
                        "vote_count.gte": 10,
                        "first_air_date.gte": start_date,
                        "first_air_date.lte": end_date,
                    },
                    "data_selector": "results",
                    "paginator": {
                        "type": "page_number",
                        "base_page": 1,
                        "page_param": "page",
                        "total_path": "total_pages",
                        "maximum_page": max(1, math.ceil(tv_limit / TMDB_PAGE_SIZE)),
                    },
                },
                "processing_steps": [{"filter": _first(tv_limit)}],
            },
            details_resource("tv_details", "tv_shows", "tv"),
            reviews_resource("tv_reviews", "tv_shows", "tv"),
        ],
    }

    @dlt.source(name="cultural_mood_tracker_tmdb")
    def tmdb_source() -> Any:
        yield from rest_api_resources(config)

    return tmdb_source()
