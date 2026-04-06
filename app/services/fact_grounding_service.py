from __future__ import annotations

import json
from typing import Any

from app.services.llm_gateway import LLMGateway


class FactGroundingService:
    def ground(
        self,
        *,
        run_id: str,
        topic: dict[str, Any],
        source_pack: dict[str, Any],
        source_structure: dict[str, Any],
        web_enrich: dict[str, Any],
        llm: LLMGateway,
    ) -> dict[str, Any]:
        prompt = (
            "You are a factual grounding analyst.\n"
            "Return strict JSON only.\n"
            "Separate facts into: hard_facts, official_facts, context_facts, soft_inferences, unknowns, forbidden_claims.\n"
            "Each fact item must be a short Chinese sentence.\n"
            "Only put directly supported facts into hard_facts or official_facts.\n"
            "Put any guess or likely interpretation into soft_inferences.\n"
            "Put all missing but important details into unknowns.\n"
            "Put risky unsupported implementation claims into forbidden_claims.\n"
            "Also return evidence_mode as one of: deep_dive, analysis, brief.\n\n"
            f"Topic:\n{json.dumps(topic, ensure_ascii=False)}\n\n"
            f"Source Pack:\n{json.dumps(source_pack, ensure_ascii=False)[:5000]}\n\n"
            f"Source Structure:\n{json.dumps(source_structure, ensure_ascii=False)[:5000]}\n\n"
            f"Web Enrich:\n{json.dumps(web_enrich, ensure_ascii=False)[:5000]}"
        )
        result = llm.call(run_id, "FACT_GROUNDING", "decision", prompt, temperature=0.1)
        parsed = self._parse_grounding(result.text)
        if parsed:
            return parsed
        return self._fallback_grounding(topic=topic, source_pack=source_pack, source_structure=source_structure, web_enrich=web_enrich)

    @staticmethod
    def _parse_grounding(text: str) -> dict[str, Any] | None:
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

        def _string_list(key: str) -> list[str]:
            return [str(item).strip() for item in (data.get(key) or []) if str(item).strip()]

        evidence_mode = str(data.get("evidence_mode", "") or "").strip().lower()
        if evidence_mode not in {"deep_dive", "analysis", "brief"}:
            evidence_mode = "analysis"
        return {
            "hard_facts": _string_list("hard_facts"),
            "official_facts": _string_list("official_facts"),
            "context_facts": _string_list("context_facts"),
            "soft_inferences": _string_list("soft_inferences"),
            "unknowns": _string_list("unknowns"),
            "forbidden_claims": _string_list("forbidden_claims"),
            "evidence_mode": evidence_mode,
        }

    @staticmethod
    def _fallback_grounding(
        *,
        topic: dict[str, Any],
        source_pack: dict[str, Any],
        source_structure: dict[str, Any],
        web_enrich: dict[str, Any],
    ) -> dict[str, Any]:
        hard_facts: list[str] = []
        title = str(topic.get("title", "") or "").strip()
        summary = str(topic.get("summary", "") or "").strip()
        if title:
            hard_facts.append(f"主题标题：{title}")
        if summary:
            hard_facts.append(f"原文摘要：{summary}")
        primary = dict(source_pack.get("primary") or {})
        for paragraph in (primary.get("paragraphs") or [])[:3]:
            text = str(paragraph or "").strip()
            if text:
                hard_facts.append(text)
        official_facts = [
            str(item.get("snippet", "") or "").strip()
            for item in (web_enrich.get("official_sources") or [])[:3]
            if str(item.get("snippet", "") or "").strip()
        ]
        context_facts = [
            str(item.get("snippet", "") or "").strip()
            for item in (web_enrich.get("context_sources") or [])[:2]
            if str(item.get("snippet", "") or "").strip()
        ]
        sections = list(source_structure.get("sections") or [])
        evidence_mode = "deep_dive" if len(sections) >= 5 else "analysis" if sections else "brief"
        return {
            "hard_facts": hard_facts[:6],
            "official_facts": official_facts[:4],
            "context_facts": context_facts[:3],
            "soft_inferences": [],
            "unknowns": ["公开材料未披露的实现细节不要写成事实。"],
            "forbidden_claims": [
                "不要把未公开的网关、调度器、计费链路写成确定事实。",
            ],
            "evidence_mode": evidence_mode,
        }
