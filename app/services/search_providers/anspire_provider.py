from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

import requests

from app.services.search_providers.base import SearchHit, SearchProvider


class AnspireSearchProvider(SearchProvider):
    def __init__(self, *, api_key: str, base_url: str, timeout_seconds: int = 20) -> None:
        self.api_key = api_key.strip()
        self.base_url = base_url.strip()
        self.timeout_seconds = max(5, int(timeout_seconds))

    def is_available(self) -> bool:
        return bool(self.api_key and self.base_url)

    def search(self, query: str, *, limit: int = 5) -> list[SearchHit]:
        if not self.is_available() or not query.strip():
            return []
        payload = {
            "query": query.strip(),
            "q": query.strip(),
            "limit": int(limit),
            "num": int(limit),
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        response = requests.post(
            self.base_url,
            json=payload,
            headers=headers,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        return self._parse_hits(data, limit=limit)

    def _parse_hits(self, data: Any, *, limit: int) -> list[SearchHit]:
        candidates: list[dict[str, Any]] = []
        if isinstance(data, list):
            candidates = [item for item in data if isinstance(item, dict)]
        elif isinstance(data, dict):
            for key in ("results", "items", "data", "organic_results", "web", "hits"):
                value = data.get(key)
                if isinstance(value, list):
                    candidates = [item for item in value if isinstance(item, dict)]
                    if candidates:
                        break

        output: list[SearchHit] = []
        for item in candidates:
            url = str(
                item.get("url")
                or item.get("link")
                or item.get("href")
                or ""
            ).strip()
            if not url:
                continue
            title = str(item.get("title") or item.get("name") or item.get("headline") or url).strip()
            snippet = str(
                item.get("snippet")
                or item.get("description")
                or item.get("summary")
                or item.get("content")
                or ""
            ).strip()
            domain = urlparse(url).netloc.lower().split("@")[-1]
            if ":" in domain:
                domain = domain.split(":", 1)[0]
            output.append(
                SearchHit(
                    title=title[:300],
                    url=url,
                    snippet=snippet[:500],
                    domain=domain,
                )
            )
            if len(output) >= limit:
                break
        return output
