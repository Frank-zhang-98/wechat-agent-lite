from __future__ import annotations

import json
from typing import Any

from app.services.llm_gateway import LLMGateway
from app.services.localization_service import LocalizationService


class VisualStrategyService:
    def build_strategy(
        self,
        *,
        run_id: str,
        topic: dict[str, Any],
        fact_pack: dict[str, Any],
        fact_grounding: dict[str, Any],
        source_structure: dict[str, Any],
        llm: LLMGateway,
        max_body_illustrations: int = 2,
    ) -> dict[str, Any]:
        prompt = (
            "你是中文技术文章的视觉策略规划器。\n"
            "只返回严格 JSON，不要输出任何解释。\n"
            "请同时设计两种互补视觉模式：\n"
            "1. 封面海报模式：视觉冲击更强，但必须有内容锚点，不能是空洞 AI 图。\n"
            "2. 正文技术插图模式：强调结构、关系、流程和信息层级。\n"
            "除产品名、专有名词、命令行命令外，所有文本字段都必须使用中文。\n"
            "输出结构：\n"
            "{\n"
            '  "cover_family": "structure|comparison|thesis|command",\n'
            '  "cover_brief": {"main_claim": "...", "subject_hint": "...", "scene_hint": "...", "mood_hint": "...", "title_safe_zone": "left_top|left_center|left_bottom", "must_show": ["..."], "must_avoid": ["..."]},\n'
            '  "body_illustrations": [\n'
            '    {"type": "architecture_diagram|workflow_diagram|comparison_card", "section": "...", "title": "...", "caption": "...", "must_show": ["..."], "must_avoid": ["..."]}\n'
            "  ]\n"
            "}\n\n"
            f"Topic:\n{json.dumps(topic, ensure_ascii=False)}\n\n"
            f"Fact Pack:\n{json.dumps(fact_pack, ensure_ascii=False)[:5000]}\n\n"
            f"Fact Grounding:\n{json.dumps(fact_grounding, ensure_ascii=False)[:5000]}\n\n"
            f"Source Structure:\n{json.dumps(source_structure, ensure_ascii=False)[:5000]}\n\n"
            f"正文插图最多生成 {max_body_illustrations} 张。"
        )
        result = llm.call(run_id, "VISUAL_STRATEGY", "decision", prompt, temperature=0.2)
        parsed = self._parse_strategy(result.text, max_body_illustrations=max_body_illustrations)
        return parsed or self._fallback_strategy(
            topic=topic,
            fact_pack=fact_pack,
            fact_grounding=fact_grounding,
            source_structure=source_structure,
            max_body_illustrations=max_body_illustrations,
        )

    def build_cover_prompt_request(self, *, article_title: str, strategy: dict[str, Any], cover_5d: dict[str, Any]) -> str:
        brief = dict(strategy.get("cover_brief") or {})
        must_show = LocalizationService.localize_visual_items(brief.get("must_show") or [])
        must_avoid = LocalizationService.localize_visual_items(brief.get("must_avoid") or [])
        subject_hint = LocalizationService.localize_visual_text(str(brief.get("subject_hint", "") or "").strip())
        scene_hint = LocalizationService.localize_visual_text(str(brief.get("scene_hint", "") or "").strip())
        mood_hint = LocalizationService.localize_visual_text(str(brief.get("mood_hint", "") or "").strip())
        title_safe_zone = str(brief.get("title_safe_zone", "left_bottom") or "left_bottom").strip()
        return (
            "请为微信公众号技术文章生成一段适合文生图模型的中文封面提示词。\n"
            "这次只生成主题视觉图，不要在图片中直接写文章标题，标题会在后处理中叠加。\n"
            "要求：横版、科技感、适合公众号封面缩略图、主体明确、画面干净、有氛围感。\n"
            "必须预留适合叠加标题的干净区域，优先左侧或左下区域，避免该区域出现复杂主体、密集纹理或难辨认小字。\n"
            "禁止在图里出现任何可读文字、logo、水印、伪文字、UI 截图、信息图卡片。\n"
            "不要机器人脸、不要抽象芯片、不要蓝紫泛光模板感，避免廉价 AI 海报风。\n"
            f"封面类型：{strategy.get('cover_family', 'structure')}\n"
            f"文章标题：{LocalizationService.localize_visual_text(article_title)}\n"
            f"封面 5D：{json.dumps(cover_5d, ensure_ascii=False)}\n"
            f"核心表达：{LocalizationService.localize_visual_text(str(brief.get('main_claim', '') or ''))}\n"
            f"主体线索：{subject_hint or '围绕文章核心对象构图'}\n"
            f"场景线索：{scene_hint or '使用干净的科技主题场景'}\n"
            f"氛围线索：{mood_hint or '克制、专业、可信'}\n"
            f"标题留白区：{title_safe_zone}\n"
            f"必须展示：{json.dumps(must_show, ensure_ascii=False)}\n"
            f"必须避免：{json.dumps(must_avoid, ensure_ascii=False)}"
        )

    def build_body_prompt_request(self, *, article_title: str, item: dict[str, Any]) -> str:
        must_show = LocalizationService.localize_visual_items(item.get("must_show") or [])
        must_avoid = LocalizationService.localize_visual_items(item.get("must_avoid") or [])
        return (
            "请为微信公众号正文插图生成一段适合文生图模型的中文提示词。\n"
            "这是正文技术插图，不是封面，不是装饰图。画面要像高质量信息设计插图，强调结构、关系与内容。\n"
            "不要使用 draw.io 工具截图感，不要生成空洞抽象科技背景。\n"
            f"文章标题：{LocalizationService.localize_visual_text(article_title)}\n"
            f"插图类型：{item.get('type', 'architecture_diagram')}\n"
            f"对应章节：{LocalizationService.localize_visual_text(str(item.get('section', '') or ''))}\n"
            f"插图标题：{LocalizationService.localize_visual_text(str(item.get('title', '') or ''))}\n"
            f"插图说明：{LocalizationService.localize_visual_text(str(item.get('caption', '') or ''))}\n"
            f"必须展示：{json.dumps(must_show, ensure_ascii=False)}\n"
            f"必须避免：{json.dumps(must_avoid, ensure_ascii=False)}"
        )

    @staticmethod
    def _parse_strategy(text: str, *, max_body_illustrations: int) -> dict[str, Any] | None:
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start < 0 or end <= start:
                return None
            data = json.loads(text[start : end + 1])
        except Exception:
            return None
        if not isinstance(data, dict):
            return None
        cover_family = str(data.get("cover_family", "structure") or "structure").strip().lower()
        if cover_family not in {"structure", "comparison", "thesis", "command"}:
            cover_family = "structure"
        cover_brief = dict(data.get("cover_brief") or {})
        body_illustrations: list[dict[str, Any]] = []
        for item in list(data.get("body_illustrations") or [])[:max_body_illustrations]:
            if not isinstance(item, dict):
                continue
            body_type = str(item.get("type", "architecture_diagram") or "architecture_diagram").strip()
            body_illustrations.append(
                {
                    "type": body_type,
                    "section": LocalizationService.localize_visual_text(str(item.get("section", "") or "").strip()),
                    "title": LocalizationService.localize_visual_text(str(item.get("title", "") or "").strip()),
                    "caption": LocalizationService.localize_visual_text(str(item.get("caption", "") or "").strip()),
                    "must_show": LocalizationService.localize_visual_items(item.get("must_show") or []),
                    "must_avoid": LocalizationService.localize_visual_items(item.get("must_avoid") or []),
                }
            )
        return {
            "cover_family": cover_family,
            "cover_brief": {
                "main_claim": LocalizationService.localize_visual_text(str(cover_brief.get("main_claim", "") or "").strip()),
                "subject_hint": LocalizationService.localize_visual_text(str(cover_brief.get("subject_hint", "") or "").strip()),
                "scene_hint": LocalizationService.localize_visual_text(str(cover_brief.get("scene_hint", "") or "").strip()),
                "mood_hint": LocalizationService.localize_visual_text(str(cover_brief.get("mood_hint", "") or "").strip()),
                "title_safe_zone": VisualStrategyService._normalize_title_safe_zone(cover_brief.get("title_safe_zone", "left_bottom")),
                "must_show": LocalizationService.localize_visual_items(cover_brief.get("must_show") or []),
                "must_avoid": LocalizationService.localize_visual_items(cover_brief.get("must_avoid") or []),
            },
            "body_illustrations": body_illustrations,
        }

    @staticmethod
    def _fallback_strategy(
        *,
        topic: dict[str, Any],
        fact_pack: dict[str, Any],
        fact_grounding: dict[str, Any],
        source_structure: dict[str, Any],
        max_body_illustrations: int,
    ) -> dict[str, Any]:
        title = LocalizationService.localize_visual_text(str(topic.get("title", "") or "").strip())
        section_blueprint = [item for item in (fact_pack.get("section_blueprint") or []) if isinstance(item, dict)]
        content_type = str(fact_pack.get("content_type", "") or "")
        cover_family = "comparison" if content_type == "industry_analysis" else "command" if content_type == "tutorial" else "structure"
        cover_brief = {
            "main_claim": title,
            "subject_hint": title[:24] or "主题主体",
            "scene_hint": "简洁的科技主题场景，主体集中，背景留白充足",
            "mood_hint": "专业、克制、可信，有一定未来感",
            "title_safe_zone": "left_bottom",
            "must_show": [title] if title else [],
            "must_avoid": ["机器人脸", "廉价 AI 光效", "抽象芯片背景"],
        }
        body_illustrations: list[dict[str, Any]] = []
        if section_blueprint:
            first = section_blueprint[0]
            body_illustrations.append(
                {
                    "type": "workflow_diagram" if content_type == "tutorial" else "architecture_diagram",
                    "section": LocalizationService.localize_visual_text(str(first.get("heading", "") or "")),
                    "title": title[:36] or "技术总览",
                    "caption": LocalizationService.localize_visual_text(str(first.get("summary", "") or "")[:120]),
                    "must_show": LocalizationService.localize_visual_items([str(first.get("heading", "") or "").strip()]),
                    "must_avoid": ["机器人脸", "廉价 AI 光效"],
                }
            )
        if len(section_blueprint) > 1 and max_body_illustrations > 1:
            second = section_blueprint[1]
            body_illustrations.append(
                {
                    "type": "comparison_card" if content_type == "industry_analysis" else "workflow_diagram",
                    "section": LocalizationService.localize_visual_text(str(second.get("heading", "") or "")),
                    "title": LocalizationService.localize_visual_text(str(second.get("heading", "") or "")[:36]),
                    "caption": LocalizationService.localize_visual_text(str(second.get("summary", "") or "")[:120]),
                    "must_show": LocalizationService.localize_visual_items([str(second.get("heading", "") or "").strip()]),
                    "must_avoid": ["装饰性粒子", "空洞未来科技背景"],
                }
            )
        return {
            "cover_family": cover_family,
            "cover_brief": cover_brief,
            "body_illustrations": body_illustrations[:max_body_illustrations],
        }

    @staticmethod
    def _normalize_title_safe_zone(value: Any) -> str:
        raw = str(value or "").strip().lower().replace("-", "_")
        aliases = {
            "top_left": "left_top",
            "lefttop": "left_top",
            "left_middle": "left_center",
            "middle_left": "left_center",
            "center_left": "left_center",
            "leftcenter": "left_center",
            "bottom_left": "left_bottom",
            "leftbottom": "left_bottom",
        }
        normalized = aliases.get(raw, raw)
        if normalized not in {"left_top", "left_center", "left_bottom"}:
            return "left_bottom"
        return normalized
