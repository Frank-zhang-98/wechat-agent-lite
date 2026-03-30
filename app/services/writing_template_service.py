from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

from app.core.config import CONFIG


class WritingTemplateService:
    def __init__(self) -> None:
        self.templates = self._load_templates()

    def _load_templates(self) -> dict[str, Any]:
        path = Path(CONFIG.data_dir).parents[0] / "config" / "writing_templates.yaml"
        if not path.exists():
            path = Path(__file__).resolve().parents[2] / "config" / "writing_templates.yaml"
        with path.open("r", encoding="utf-8") as file:
            return yaml.safe_load(file) or {}

    def get_audience(self, audience_key: str) -> dict[str, Any]:
        audiences = self.templates.get("audiences", {})
        if audience_key in audiences:
            return {"key": audience_key, **dict(audiences[audience_key] or {})}
        if audiences:
            fallback_key = next(iter(audiences))
            return {"key": fallback_key, **dict(audiences[fallback_key] or {})}
        return {
            "key": audience_key or "default",
            "label": "通用读者",
            "description": "关注 AI 产品、工具和落地价值的通用读者。",
            "focus": ["产品价值", "落地场景", "使用边界"],
            "tone": "专业、清晰、克制",
        }

    def get_content_type(self, content_type: str) -> dict[str, Any]:
        content_types = self.templates.get("content_types", {})
        if content_type in content_types:
            return {"key": content_type, **dict(content_types[content_type] or {})}
        if content_types:
            fallback_key = next(iter(content_types))
            return {"key": fallback_key, **dict(content_types[fallback_key] or {})}
        return {
            "key": content_type or "tool_review",
            "label": "工具 / 产品解读",
            "objective": "讲清楚产品价值、机制和落地方式。",
            "sections": [],
            "emphasis": [],
        }

    def infer_content_type(self, topic: dict[str, Any]) -> str:
        text = " ".join(
            str(topic.get(key, "") or "")
            for key in ("title", "summary", "source", "url", "rerank_reason")
        ).lower()
        tutorial_keywords = [
            "教程", "指南", "实战", "工作流", "prompt", "提示词", "how to", "guide", "tutorial", "workflow",
        ]
        industry_keywords = [
            "融资", "收购", "趋势", "财报", "报告", "industry", "analysis", "发布会", "研究", "政策",
        ]
        if any(keyword in text for keyword in tutorial_keywords):
            return "tutorial"
        if any(keyword in text for keyword in industry_keywords):
            return "industry_analysis"
        return "tool_review"

    def build_fact_pack(self, ctx: dict[str, Any], audience_key: str) -> dict[str, Any]:
        topic = dict(ctx.get("selected_topic") or {})
        related_candidates = list(ctx.get("top_k") or [])
        source_pack = dict(ctx.get("source_pack") or {})
        primary_source = dict(source_pack.get("primary") or {})
        related_sources = [item for item in (source_pack.get("related") or []) if isinstance(item, dict)]
        audience = self.get_audience(audience_key)
        content_type = self.infer_content_type(topic)

        related_topics: list[dict[str, Any]] = []
        seen_urls = {str(topic.get("url", "") or "").strip()}
        for item in related_candidates:
            url = str(item.get("url", "") or "").strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            related_topics.append(
                {
                    "title": str(item.get("title", "") or ""),
                    "summary": str(item.get("summary", "") or ""),
                    "source": str(item.get("source", "") or ""),
                    "url": url,
                    "final_score": item.get("final_score"),
                }
            )
            if len(related_topics) >= 3:
                break

        primary_text = " ".join(
            str(topic.get(key, "") or "")
            for key in ("title", "summary", "rerank_reason")
        )
        primary_text = f"{primary_text} {str(primary_source.get('content_text', '') or '')}".strip()
        related_text = " ".join(
            " ".join(str(item.get(key, "") or "") for key in ("title", "summary"))
            for item in related_topics
        )
        related_text = (
            f"{related_text} "
            + " ".join(str(item.get("content_text", "") or "") for item in related_sources)
        ).strip()
        combined_text = f"{primary_text} {related_text}".strip()

        key_points = self._build_key_points(topic, related_topics, primary_source, related_sources)
        numbers = self._extract_numbers(combined_text)
        keywords = self._extract_keywords(combined_text)

        return {
            "topic_title": str(topic.get("title", "") or ""),
            "topic_summary": str(topic.get("summary", "") or ""),
            "topic_url": str(topic.get("url", "") or ""),
            "topic_source": str(topic.get("source", "") or ""),
            "published": str(topic.get("published", "") or ""),
            "primary_excerpt": str(primary_source.get("content_text", "") or "")[:2400],
            "source_status": str(primary_source.get("status", "") or ""),
            "audience_key": audience["key"],
            "audience_label": audience.get("label", audience["key"]),
            "content_type": content_type,
            "content_type_label": self.get_content_type(content_type).get("label", content_type),
            "key_points": key_points,
            "related_topics": related_topics,
            "related_excerpts": [
                {
                    "title": str(item.get("title", "") or ""),
                    "url": str(item.get("url", "") or ""),
                    "content_text": str(item.get("content_text", "") or "")[:1200],
                }
                for item in related_sources[:2]
            ],
            "numbers": numbers,
            "keywords": keywords,
        }

    def build_write_prompt(
        self,
        *,
        topic: dict[str, Any],
        fact_pack: dict[str, Any],
        audience_key: str,
        content_type: str,
    ) -> str:
        audience = self.get_audience(audience_key)
        type_cfg = self.get_content_type(content_type)
        quality = dict(self.templates.get("quality_requirements") or {})
        style = dict(self.templates.get("writing_style") or {})
        constraints = list(self.templates.get("global_constraints") or [])

        sections = []
        for section in type_cfg.get("sections", []) or []:
            heading = str(section.get("heading", "") or "").strip()
            purpose = str(section.get("purpose", "") or "").strip()
            if heading:
                sections.append(f"{heading}：{purpose}")

        focus = [str(item) for item in audience.get("focus", []) if str(item).strip()]
        key_points = [str(item) for item in fact_pack.get("key_points", []) if str(item).strip()]
        related_topics = [item for item in fact_pack.get("related_topics", []) if isinstance(item, dict)]
        numbers = [str(item) for item in fact_pack.get("numbers", []) if str(item).strip()]
        keywords = [str(item) for item in fact_pack.get("keywords", []) if str(item).strip()]

        related_text = "\n".join(
            f"- {item.get('title', '')} | 来源：{item.get('source', '-')}"
            + (f" | 摘要：{item.get('summary', '')}" if item.get("summary") else "")
            for item in related_topics
        ) or "- 暂无额外相关线索"

        prompt = f"""你是一名专业的中文科技作者，要为微信公众号写一篇高信息密度的技术解读文章。

【目标读者】
- 读者类型：{audience.get('label', audience_key)}
- 读者描述：{audience.get('description', '')}
- 读者最关心的点：
{self._bullet_block(focus)}

【文章任务】
- 文章类型：{type_cfg.get('label', content_type)}
- 写作目标：{type_cfg.get('objective', '')}
- 主题：{topic.get('title', '')}
- 原始摘要：{topic.get('summary', '')}
- 原始链接：{topic.get('url', '')}

【事实包】
- 主来源：{fact_pack.get('topic_source', '-') or '-'}
- 发布时间：{fact_pack.get('published', '-') or '-'}
- 已知关键点：
{self._bullet_block(key_points)}

- 相关线索：
{related_text}

- 可直接引用的数字 / 量化信息：
{self._bullet_block(numbers)}

- 关键词：
{self._bullet_block(keywords)}

【强制结构】
{self._bullet_block(sections)}

【该类型需要特别强调】
{self._bullet_block(type_cfg.get('emphasis', []) or [])}

【必须包含】
{self._bullet_block(quality.get('must_have', []) or [])}

【必须避免】
{self._bullet_block(quality.get('must_avoid', []) or [])}

【最佳实践】
{self._bullet_block(quality.get('best_practices', []) or [])}

【写作风格】
- 整体语气：{style.get('tone', '')}
- 优先写法：
{self._bullet_block(style.get('prefer', []) or [])}
- 段落结构：
{self._bullet_block(style.get('paragraph_structure', []) or [])}

【全局约束】
{self._bullet_block(constraints)}

【额外要求】
- 开头第一段必须直接解释“它是什么”和“为什么值得关注”，不要用“随着 AI 的发展”这类套话开场。
- 文章必须写出产品机制、工作流机制或判断依据，不能只写功能列表。
- 如果事实包里没有的数据或信息，请不要编造；可以明确写“现有公开信息尚不足以判断”。
- 至少给出 3 条真正有信息密度的干货结论。
- 结尾给出面向读者的可执行建议，而不是空泛总结。
- 全文输出为简体中文 Markdown，只输出正文，不要输出解释。
"""
        return prompt

    @staticmethod
    def _build_key_points(
        topic: dict[str, Any],
        related_topics: list[dict[str, Any]],
        primary_source: dict[str, Any],
        related_sources: list[dict[str, Any]],
    ) -> list[str]:
        points: list[str] = []
        title = str(topic.get("title", "") or "").strip()
        summary = str(topic.get("summary", "") or "").strip()
        if title:
            points.append(f"主题本身：{title}")
        if summary:
            points.append(f"主摘要：{summary}")
        if topic.get("rerank_reason"):
            points.append(f"入选原因：{topic.get('rerank_reason')}")
        if topic.get("final_score") is not None:
            points.append(f"综合评分：{topic.get('final_score')}")
        for paragraph in (primary_source.get("paragraphs") or [])[:3]:
            text = str(paragraph or "").strip()
            if text:
                points.append(f"正文线索：{text}")
        for item in related_topics[:2]:
            if item.get("summary"):
                points.append(f"相关线索：{item['summary']}")
        for item in related_sources[:1]:
            text = str(item.get("content_text", "") or "").strip()
            if text:
                points.append(f"相关正文：{text[:220]}")
        return points[:6]

    @staticmethod
    def _extract_numbers(text: str) -> list[str]:
        values = re.findall(r"\d+(?:\.\d+)?(?:%|倍|x|X|万|亿|k|K|m|M|分钟|小时|天|年|美元|元)?", text or "")
        output: list[str] = []
        for value in values:
            if value not in output:
                output.append(value)
            if len(output) >= 8:
                break
        return output

    @staticmethod
    def _extract_keywords(text: str) -> list[str]:
        candidates = re.findall(r"[A-Za-z][A-Za-z0-9._/-]{2,}|[\u4e00-\u9fff]{2,8}", text or "")
        blacklist = {
            "这是", "这个", "可以", "一个", "我们", "他们", "因为", "所以", "如果", "但是",
            "进行", "以及", "相关", "功能", "能力", "产品", "工具", "文章", "工作流",
        }
        output: list[str] = []
        for item in candidates:
            if item in blacklist:
                continue
            if item not in output:
                output.append(item)
            if len(output) >= 12:
                break
        return output

    @staticmethod
    def _bullet_block(items: list[Any]) -> str:
        cleaned = [str(item).strip() for item in items if str(item).strip()]
        if not cleaned:
            return "- 暂无"
        return "\n".join(f"- {item}" for item in cleaned)

    @staticmethod
    def preview_fact_pack(fact_pack: dict[str, Any], limit: int = 4000) -> str:
        text = json.dumps(fact_pack, ensure_ascii=False, indent=2)
        return text if len(text) <= limit else text[:limit] + f"\n... [truncated, total {len(text)} chars]"
