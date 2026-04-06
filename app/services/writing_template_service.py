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
            "objective": "把产品是什么、能做什么、为什么值得关注讲清楚。",
            "sections": [],
            "emphasis": [],
        }

    def infer_content_type(
        self,
        topic: dict[str, Any],
        *,
        source_structure: dict[str, Any] | None = None,
        primary_source: dict[str, Any] | None = None,
    ) -> str:
        source_structure = dict(source_structure or {})
        primary_source = dict(primary_source or {})
        section_text = " ".join(
            " ".join(
                [
                    str(section.get("heading", "") or ""),
                    str(section.get("summary", "") or ""),
                ]
            )
            for section in (source_structure.get("sections") or [])
            if isinstance(section, dict)
        )
        text = " ".join(
            [
                " ".join(
                    str(topic.get(key, "") or "")
                    for key in ("title", "summary", "source", "url", "rerank_reason")
                ),
                str(source_structure.get("lead", "") or ""),
                section_text,
                str(primary_source.get("content_text", "") or "")[:2400],
            ]
        ).lower()

        tutorial_keywords = [
            "教程",
            "指南",
            "实战",
            "工作流",
            "prompt",
            "提示词",
            "how to",
            "guide",
            "tutorial",
            "workflow",
        ]
        industry_keywords = [
            "融资",
            "收购",
            "趋势",
            "财报",
            "报告",
            "industry",
            "analysis",
            "发布会",
            "研究",
            "政策",
        ]
        technical_keywords = [
            "langgraph",
            "mcp",
            "rag",
            "agent",
            "graph",
            "pipeline",
            "workflow",
            "architecture",
            "orchestration",
            "implementation",
            "code",
            "sdk",
            "api",
            "state",
            "session",
            "ttl",
            "renewal",
            "lifecycle",
            "模块",
            "架构",
            "实现",
            "代码",
            "编排",
            "链路",
            "流程",
        ]

        sections = [item for item in (source_structure.get("sections") or []) if isinstance(item, dict)]
        section_count = len(sections)
        code_count = len(source_structure.get("code_blocks") or [])
        coverage_count = len(source_structure.get("coverage_checklist") or [])
        implementation_hits = 0
        architecture_hits = 0
        for section in sections:
            haystack = " ".join(
                [
                    str(section.get("heading", "") or ""),
                    str(section.get("summary", "") or ""),
                ]
            ).lower()
            if re.search(
                r"(step|步骤|阶段|流程|workflow|pipeline|graph|mcp|rag|agent|ttl|renewal|lifecycle)",
                haystack,
                flags=re.IGNORECASE,
            ):
                implementation_hits += 1
            if re.search(
                r"(architecture|架构|模块|组件|agent|mcp|rag|graph|workflow|pipeline|session)",
                haystack,
                flags=re.IGNORECASE,
            ):
                architecture_hits += 1

        technical_hits = sum(1 for keyword in technical_keywords if keyword in text)
        strong_technical_structure = (
            section_count >= 4
            and (
                implementation_hits >= 2
                or architecture_hits >= 2
                or code_count >= 1
                or coverage_count >= 4
            )
        ) or (
            section_count >= 2
            and code_count >= 1
            and (implementation_hits >= 1 or architecture_hits >= 1 or coverage_count >= 3)
        )
        if strong_technical_structure and (technical_hits >= 2 or code_count >= 1 or implementation_hits >= 2):
            return "technical_walkthrough"
        if any(keyword in text for keyword in tutorial_keywords):
            return "tutorial"
        if any(keyword in text for keyword in industry_keywords):
            return "industry_analysis"
        return "tool_review"

    def build_fact_pack(self, ctx: dict[str, Any], audience_key: str) -> dict[str, Any]:
        topic = dict(ctx.get("selected_topic") or {})
        related_candidates = list(ctx.get("top_k") or [])
        source_pack = dict(ctx.get("source_pack") or {})
        source_structure = dict(ctx.get("source_structure") or {})
        fact_grounding = dict(ctx.get("fact_grounding") or {})
        primary_source = dict(source_pack.get("primary") or {})
        related_sources = [item for item in (source_pack.get("related") or []) if isinstance(item, dict)]
        audience = self.get_audience(audience_key)

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

        primary_text = " ".join(str(topic.get(key, "") or "") for key in ("title", "summary", "rerank_reason"))
        primary_text = f"{primary_text} {str(primary_source.get('content_text', '') or '')}".strip()
        related_text = " ".join(
            " ".join(str(item.get(key, "") or "") for key in ("title", "summary")) for item in related_topics
        )
        related_text = (
            f"{related_text} " + " ".join(str(item.get("content_text", "") or "") for item in related_sources)
        ).strip()
        combined_text = f"{primary_text} {related_text}".strip()

        key_points = self._build_key_points(topic, related_topics, primary_source, related_sources)
        grounded_hard_facts = [str(item).strip() for item in (fact_grounding.get("hard_facts") or []) if str(item).strip()]
        grounded_official_facts = [str(item).strip() for item in (fact_grounding.get("official_facts") or []) if str(item).strip()]
        grounded_context_facts = [str(item).strip() for item in (fact_grounding.get("context_facts") or []) if str(item).strip()]
        soft_inferences = [str(item).strip() for item in (fact_grounding.get("soft_inferences") or []) if str(item).strip()]
        unknowns = [str(item).strip() for item in (fact_grounding.get("unknowns") or []) if str(item).strip()]
        forbidden_claims = [str(item).strip() for item in (fact_grounding.get("forbidden_claims") or []) if str(item).strip()]
        industry_context_points = grounded_context_facts[:6] or [
            str(item.get("summary", "") or "").strip()
            for item in related_topics[:3]
            if str(item.get("summary", "") or "").strip()
        ]
        if grounded_hard_facts or grounded_official_facts:
            key_points = (grounded_hard_facts[:4] + grounded_official_facts[:2])[:6]
        numbers = self._extract_numbers(combined_text)
        keywords = self._extract_keywords(combined_text)
        section_blueprint = self._build_section_blueprint(source_structure)
        implementation_steps = self._build_implementation_steps(source_structure)
        architecture_points = self._build_architecture_points(source_structure)
        code_artifacts = self._build_code_artifacts(source_structure)
        preserved_command_blocks = [item for item in code_artifacts if str(item.get("kind", "") or "") == "command"]
        preserved_code_blocks = [item for item in code_artifacts if str(item.get("kind", "") or "") != "command"]
        coverage_checklist = list(source_structure.get("coverage_checklist") or [])
        content_type = self.infer_content_type(
            topic,
            source_structure=source_structure,
            primary_source=primary_source,
        )

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
            "evidence_mode": str(fact_grounding.get("evidence_mode", "") or ""),
            "key_points": key_points,
            "grounded_hard_facts": grounded_hard_facts[:8],
            "grounded_official_facts": grounded_official_facts[:8],
            "grounded_context_facts": grounded_context_facts[:8],
            "industry_context_points": industry_context_points[:8],
            "soft_inferences": soft_inferences[:8],
            "unknowns": unknowns[:8],
            "forbidden_claims": forbidden_claims[:8],
            "source_lead": str(source_structure.get("lead", "") or "")[:1200],
            "section_blueprint": section_blueprint,
            "implementation_steps": implementation_steps,
            "architecture_points": architecture_points,
            "code_artifacts": code_artifacts,
            "preserved_command_blocks": preserved_command_blocks[:8],
            "preserved_code_blocks": preserved_code_blocks[:8],
            "coverage_checklist": coverage_checklist[:12],
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
        section_blueprint = [item for item in fact_pack.get("section_blueprint", []) if isinstance(item, dict)]
        implementation_steps = [item for item in fact_pack.get("implementation_steps", []) if isinstance(item, dict)]
        architecture_points = [item for item in fact_pack.get("architecture_points", []) if isinstance(item, dict)]
        code_artifacts = [item for item in fact_pack.get("code_artifacts", []) if isinstance(item, dict)]
        preserved_command_blocks = [item for item in fact_pack.get("preserved_command_blocks", []) if isinstance(item, dict)]
        preserved_code_blocks = [item for item in fact_pack.get("preserved_code_blocks", []) if isinstance(item, dict)]
        coverage_checklist = [str(item) for item in fact_pack.get("coverage_checklist", []) if str(item).strip()]
        source_lead = str(fact_pack.get("source_lead", "") or "").strip()
        evidence_mode = str(fact_pack.get("evidence_mode", "") or "").strip().lower() or "analysis"
        grounded_hard_facts = [str(item) for item in fact_pack.get("grounded_hard_facts", []) if str(item).strip()]
        grounded_official_facts = [str(item) for item in fact_pack.get("grounded_official_facts", []) if str(item).strip()]
        grounded_context_facts = [str(item) for item in fact_pack.get("grounded_context_facts", []) if str(item).strip()]
        industry_context_points = [str(item) for item in fact_pack.get("industry_context_points", []) if str(item).strip()]
        soft_inferences = [str(item) for item in fact_pack.get("soft_inferences", []) if str(item).strip()]
        unknowns = [str(item) for item in fact_pack.get("unknowns", []) if str(item).strip()]
        forbidden_claims = [str(item) for item in fact_pack.get("forbidden_claims", []) if str(item).strip()]

        related_text = "\n".join(
            f"- {item.get('title', '')} | 来源：{item.get('source', '-')}"
            + (f" | 摘要：{item.get('summary', '')}" if item.get("summary") else "")
            for item in related_topics
        ) or "- 暂无额外相关线索"
        section_text = "\n".join(
            f"- {item.get('heading', '')}：{item.get('summary', '')}" for item in section_blueprint
        ) or "- 暂无明确章节结构"
        implementation_steps_text = "\n".join(
            f"- {item.get('title', '')}：{item.get('summary', '')}"
            + (f" | 细节：{'；'.join(item.get('details', [])[:3])}" if item.get("details") else "")
            for item in implementation_steps
        ) or "- 暂无明确实现步骤"
        architecture_text = "\n".join(
            f"- {item.get('component', '')}：{item.get('responsibility', '')}" for item in architecture_points
        ) or "- 暂无明确架构拆解"
        code_text = "\n".join(
            f"- {item.get('section', '') or item.get('language', '代码片段')}：{item.get('summary', '')}"
            + (f" | 代码语言：{item.get('language', '')}" if item.get("language") else "")
            for item in code_artifacts
        ) or "- 暂无明显代码片段"

        has_strong_source_structure = len(section_blueprint) >= 4
        has_dense_implementation = (
            len(implementation_steps) >= 2
            or len(architecture_points) >= 2
            or len(code_artifacts) >= 1
            or len(coverage_checklist) >= 4
        )
        structure_strategy: list[str] = []
        if has_strong_source_structure:
            structure_strategy.append("优先沿用“原文章节结构”的顺序组织正文，只允许合并相邻小节，不要重排实现链路。")
        else:
            structure_strategy.append("如果原文结构不够完整，再参考文章类型建议自行组织结构。")
        if has_dense_implementation or content_type == "technical_walkthrough":
            structure_strategy.extend(
                [
                    "正文优先使用 `##` 和 `###` 标题展开技术链路，不要把全文写成多个反复从 1 开始的顶层编号列表。",
                    "不要把每个小节都套成同一种“机制 / 价值 / 场景 / 工作流”模板句式，要根据该节内容自然展开。",
                    "凡是原文提到的实现步骤、系统角色、代码片段，都要解释它在整条链路里的作用、依赖关系和工程取舍。",
                ]
            )
        else:
            structure_strategy.append("列表只用于并列信息，不要为了显得工整而把所有段落都改写成清单。")
        section_plan_label = "【结构建议】"
        if has_strong_source_structure:
            section_plan_label = "【兜底结构建议】"

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
- 原文导语：{source_lead or '-'}
- 已知关键点：
{self._bullet_block(key_points)}

- 相关线索：
{related_text}

- 原文章节结构：
{section_text}

- 原文实现步骤：
{implementation_steps_text}

- 原文架构拆解：
{architecture_text}

- 原文代码实现概括：
{code_text}

- 必须覆盖的实现清单：
{self._bullet_block(coverage_checklist)}

- 可直接引用的数字 / 量化信息：
{self._bullet_block(numbers)}

- 关键词：
{self._bullet_block(keywords)}

{section_plan_label}
{self._bullet_block(sections)}

【结构保留策略】
{self._bullet_block(structure_strategy)}

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
- 如果原文结构已经很强，优先按原文顺序用二级/三级标题组织，而不是另起一套整齐但失真的新提纲。
- 如果原文存在明确步骤、章节、架构层次或代码实现细节，你必须逐段概括出来，不能只写大概概述。
- 对代码块不要原样大段粘贴，但必须说明：代码实现了什么、它在整体链路中的作用、以及这部分实现的独到之处。
- “必须覆盖的实现清单”中的项目必须全部出现在正文里；可以合并表达，但不能遗漏。
- 除非原文本身就是操作清单，否则不要把全文写成多个 `1.` `2.` `3.` 的大编号块。
- 不要为了格式工整而在每一个小节重复“它是什么 / 为什么重要 / 应用场景”这种同构句式。
- 如果事实包里没有的数据或信息，请不要编造；可以明确写“现有公开信息尚不足以判断”。
- 至少给出 3 条真正有信息密度的干货结论。
- 结尾给出面向读者的可执行建议，而不是空泛总结。
- 全文输出为简体中文 Markdown，只输出正文，不要输出解释。"""
        prompt += (
            "\n\n【Fact Grounding】\n"
            f"- evidence_mode: {evidence_mode}\n"
            "- Hard facts:\n"
            f"{self._bullet_block(grounded_hard_facts)}\n"
            "- Official facts:\n"
            f"{self._bullet_block(grounded_official_facts)}\n"
            "- Context facts:\n"
            f"{self._bullet_block(grounded_context_facts)}\n"
            "- Soft inferences:\n"
            f"{self._bullet_block(soft_inferences)}\n"
            "- Unknowns:\n"
            f"{self._bullet_block(unknowns)}\n"
            "- Forbidden claims:\n"
            f"{self._bullet_block(forbidden_claims)}\n"
            "- Use hard_facts and official_facts as the only definite factual basis.\n"
            "- Context facts can only be used for background or comparison, not as direct product facts.\n"
            "- Soft inferences must be written as cautious judgments such as 可能 / 可推测 / 从公开信息看.\n"
            "- Unknowns must remain unknown; do not fill them in with invented system design.\n"
            "- Forbidden claims must never appear in the final article as facts.\n"
        )
        prompt += (
            "\n\n[Industry Context Integration]\n"
            "- Use context facts and industry context points only as inline analysis, comparison, or background.\n"
            "- Do not output a standalone section titled 相关阅读 / 延伸阅读 / 参考资料 / Related Reading / Further Reading.\n"
            "- When using external context, explicitly connect it to the current topic instead of listing links or titles.\n"
            "- If a context point is not useful for the current argument, omit it instead of appending it as a reading list.\n"
            "- Industry context points:\n"
            f"{self._bullet_block(industry_context_points)}\n"
        )
        prompt += (
            "\n\n[Code Preservation]\n"
            "- Treat original command/code blocks as article assets, not as summaries.\n"
            "- Keep command blocks and code blocks verbatim whenever possible.\n"
            "- Do not rewrite, simplify, translate, or convert code into pseudo-code.\n"
            "- Place each code block under the most relevant section heading, then explain it before or after the block.\n"
            "- Code blocks do not count toward prose density; keep enough explanation text around them.\n"
            "- Every fenced block must use standard Markdown: opening ```lang on its own line, code on following lines, closing ``` on its own line.\n"
            "- Never put prose, headings, or list items inside a code fence unless they are truly part of the original file content.\n"
        )
        if preserved_command_blocks or preserved_code_blocks:
            prompt += "\n\n[Preserved Blocks]\n"
            for idx, item in enumerate(preserved_command_blocks[:4], start=1):
                language = str(item.get("language", "") or "bash").strip() or "bash"
                section = str(item.get("section", "") or "未指定章节").strip()
                code_text = str(item.get("code_text", "") or "").strip()
                if not code_text:
                    continue
                prompt += f"\nCommand Block {idx} | Section: {section}\n```{language}\n{code_text}\n```\n"
            for idx, item in enumerate(preserved_code_blocks[:4], start=1):
                language = str(item.get("language", "") or "text").strip() or "text"
                section = str(item.get("section", "") or "未指定章节").strip()
                code_text = str(item.get("code_text", "") or "").strip()
                if not code_text:
                    continue
                prompt += f"\nCode Block {idx} | Section: {section}\n```{language}\n{code_text}\n```\n"
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
            points.append(f"主题摘要：{summary}")
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
    def _build_section_blueprint(source_structure: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {
                "heading": str(section.get("heading", "") or ""),
                "summary": str(section.get("summary", "") or ""),
            }
            for section in (source_structure.get("sections") or [])[:8]
            if str(section.get("heading", "") or "").strip()
        ]

    @staticmethod
    def _build_implementation_steps(source_structure: dict[str, Any]) -> list[dict[str, Any]]:
        steps: list[dict[str, Any]] = []
        for section in (source_structure.get("sections") or [])[:8]:
            heading = str(section.get("heading", "") or "").strip()
            if not heading:
                continue
            if not re.search(
                r"(step|步骤|阶段|流程|workflow|pipeline|graph|mcp|rag|agent|ttl|renewal|lifecycle)",
                heading,
                flags=re.IGNORECASE,
            ):
                continue
            paragraphs = [str(item).strip() for item in (section.get("paragraphs") or []) if str(item).strip()]
            steps.append(
                {
                    "title": heading,
                    "summary": str(section.get("summary", "") or ""),
                    "details": paragraphs[:3],
                }
            )
        return steps[:6]

    @staticmethod
    def _build_architecture_points(source_structure: dict[str, Any]) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for section in (source_structure.get("sections") or [])[:8]:
            heading = str(section.get("heading", "") or "").strip()
            summary = str(section.get("summary", "") or "").strip()
            haystack = f"{heading} {summary}"
            if not heading or not summary:
                continue
            if not re.search(
                r"(architecture|模块|组件|架构|agent|mcp|rag|graph|workflow|pipeline|session|编排)",
                haystack,
                flags=re.IGNORECASE,
            ):
                continue
            output.append({"component": heading, "responsibility": summary})
        return output[:6]

    @staticmethod
    def _build_code_artifacts(source_structure: dict[str, Any]) -> list[dict[str, Any]]:
        sections = list(source_structure.get("sections") or [])
        code_blocks = list(source_structure.get("code_blocks") or [])
        output: list[dict[str, Any]] = []
        for idx, code in enumerate(code_blocks[:6]):
            section_title = ""
            for section in sections:
                if idx in (section.get("code_refs") or []):
                    section_title = str(section.get("heading", "") or "")
                    break
            excerpt = str(code.get("code_excerpt", "") or "").strip()
            if not excerpt:
                continue
            output.append(
                {
                    "section": section_title,
                    "language": str(code.get("language", "") or ""),
                    "summary": excerpt.splitlines()[0][:160],
                    "code_text": str(code.get("code_text", "") or excerpt),
                    "kind": str(code.get("kind", "code") or "code"),
                    "line_count": int(code.get("line_count", 0) or 0),
                    "preserve_verbatim": True,
                }
            )
        return output

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
            "这是",
            "这个",
            "可以",
            "一个",
            "我们",
            "他们",
            "因为",
            "所以",
            "如果",
            "但是",
            "进行",
            "以及",
            "相关",
            "功能",
            "能力",
            "产品",
            "工具",
            "文章",
            "工作流",
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
