from __future__ import annotations

from pathlib import Path
from typing import Any

from app.core.config import CONFIG
from app.services.programmatic_visual_service import ProgrammaticVisualService


class BodyIllustrationService:
    def __init__(self, visual_renderer: ProgrammaticVisualService) -> None:
        self.visual_renderer = visual_renderer

    def generate(
        self,
        *,
        run_id: str,
        article_title: str,
        visual_strategy: dict[str, Any],
        size: str,
    ) -> list[dict[str, Any]]:
        output_root = CONFIG.data_dir / "runs" / run_id / "illustrations"
        output_root.mkdir(parents=True, exist_ok=True)
        assets: list[dict[str, Any]] = []
        for idx, brief in enumerate(list(visual_strategy.get("body_illustrations") or []), start=1):
            output_path = output_root / f"illus-{idx}.png"
            asset = self.visual_renderer.render_body_illustration(
                article_title=article_title,
                brief=brief,
                output_path=output_path,
                size=size,
            )
            assets.append(
                {
                    "index": idx,
                    "section": str(brief.get("section", "") or ""),
                    "title": str(brief.get("title", "") or ""),
                    "caption": str(brief.get("caption", "") or ""),
                    "type": str(brief.get("type", "architecture_diagram") or "architecture_diagram"),
                    "requested_size": size,
                    **asset,
                }
            )
        return assets
