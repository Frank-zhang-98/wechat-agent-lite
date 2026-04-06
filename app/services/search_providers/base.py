from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SearchHit:
    title: str
    url: str
    snippet: str = ""
    domain: str = ""
    source_type: str = "context"


class SearchProvider:
    def is_available(self) -> bool:
        raise NotImplementedError

    def search(self, query: str, *, limit: int = 5) -> list[SearchHit]:
        raise NotImplementedError
