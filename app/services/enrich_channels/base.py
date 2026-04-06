from __future__ import annotations

from typing import Any


class EnrichChannel:
    name = "base"

    def can_handle(self, item: dict[str, Any]) -> bool:
        raise NotImplementedError

    def normalize(self, item: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError
