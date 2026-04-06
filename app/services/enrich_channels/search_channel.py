from __future__ import annotations

from typing import Any

from app.services.enrich_channels.base import EnrichChannel


class SearchChannel(EnrichChannel):
    name = "search"

    def can_handle(self, item: dict[str, Any]) -> bool:
        return bool(item.get("url"))

    def normalize(self, item: dict[str, Any]) -> dict[str, Any]:
        return {
            "title": str(item.get("title", "") or ""),
            "url": str(item.get("url", "") or ""),
            "snippet": str(item.get("snippet", "") or ""),
            "domain": str(item.get("domain", "") or ""),
            "source_type": str(item.get("source_type", "context") or "context"),
            "content_text": str(item.get("content_text", "") or ""),
        }
