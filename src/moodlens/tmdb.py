from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from .models import Document, MediaKind

BASE_URL = "https://api.themoviedb.org/3"
USER_AGENT = "moodlens-learning-project/0.1"


def _get(path: str, *, api_key: str, params: dict[str, Any]) -> dict[str, Any]:
    query = urllib.parse.urlencode({**params, "api_key": api_key})
    request = urllib.request.Request(
        f"{BASE_URL}/{path.lstrip('/')}?{query}", headers={"User-Agent": USER_AGENT}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"TMDB request failed with HTTP {exc.code} for {path}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"TMDB request failed for {path}: {exc.reason}") from exc


def _discover(
    media_kind: MediaKind,
    *,
    api_key: str,
    language: str,
    start_date: str,
    end_date: str,
    limit: int,
) -> list[dict[str, Any]]:
    date_field = "primary_release_date" if media_kind == "movie" else "first_air_date"
    output: list[dict[str, Any]] = []
    page = 1
    while len(output) < limit:
        payload = _get(
            f"discover/{media_kind}",
            api_key=api_key,
            params={
                "language": language,
                "sort_by": "popularity.desc",
                "vote_count.gte": 10,
                f"{date_field}.gte": start_date,
                f"{date_field}.lte": end_date,
                "page": page,
            },
        )
        results = payload.get("results") or []
        output.extend(item for item in results if isinstance(item, dict))
        if not results or page >= int(payload.get("total_pages") or page):
            break
        page += 1
    return output[:limit]


def _year(value: Any) -> int | None:
    text = str(value or "")
    return int(text[:4]) if len(text) >= 4 and text[:4].isdigit() else None


def download_documents(
    *,
    api_key: str,
    language: str,
    start_date: str,
    end_date: str,
    movie_limit: int,
    tv_limit: int,
) -> list[Document]:
    if not api_key:
        raise ValueError("TMDB_API_KEY is required for a live refresh")
    documents: list[Document] = []
    for media_kind, limit in (("movie", movie_limit), ("tv", tv_limit)):
        titles = _discover(
            media_kind,
            api_key=api_key,
            language=language,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
        )
        for title in titles:
            tmdb_id = int(title["id"])
            details = _get(
                f"{media_kind}/{tmdb_id}", api_key=api_key, params={"language": language}
            )
            name = str(details.get("title") or details.get("name") or f"TMDB {tmdb_id}")
            overview = str(details.get("overview") or "").strip()
            common = {
                "title": name,
                "media_kind": media_kind,
                "year": _year(details.get("release_date") or details.get("first_air_date")),
                "genres": tuple(
                    str(item["name"])
                    for item in details.get("genres") or []
                    if isinstance(item, dict) and item.get("name")
                ),
                "source": "tmdb",
                "source_url": f"https://www.themoviedb.org/{media_kind}/{tmdb_id}",
            }
            if overview:
                documents.append(
                    Document(
                        document_id=f"tmdb:{media_kind}:{tmdb_id}:overview",
                        text=overview,
                        document_type="overview",
                        **common,
                    )
                )
            reviews = _get(
                f"{media_kind}/{tmdb_id}/reviews",
                api_key=api_key,
                params={"language": language, "page": 1},
            )
            for review in reviews.get("results") or []:
                review_id = str(review.get("id") or "").strip()
                review_text = str(review.get("content") or "").strip()
                if review_id and review_text:
                    documents.append(
                        Document(
                            document_id=f"tmdb:{media_kind}:{tmdb_id}:review:{review_id}",
                            text=review_text,
                            document_type="user_review",
                            **common,
                        )
                    )
    return documents

