from __future__ import annotations

import json
import re
from typing import Any

from app.services.llm_gateway import LLMGateway


class HallucinationCheckService:
    def check(
        self,
        *,
        run_id: str,
        article_markdown: str,
        fact_grounding: dict[str, Any],
        llm: LLMGateway,
    ) -> dict[str, Any]:
        prompt = (
            "You are a factual verifier for a Chinese tech article.\n"
            "Return strict JSON only.\n"
            "Find: unsupported_claims, inference_written_as_fact, forbidden_claim_violations.\n"
            "Return severity: low|medium|high and rewrite_required: true|false.\n"
            "Be conservative. Only flag issues that are not grounded in the provided facts.\n\n"
            f"Fact Grounding:\n{json.dumps(fact_grounding, ensure_ascii=False)[:5000]}\n\n"
            f"Article:\n{article_markdown[:6000]}"
        )
        result = llm.call(run_id, "HALLUCINATION_CHECK", "decision", prompt, temperature=0.1)
        parsed = self._parse_result(result.text)
        if parsed:
            result = parsed
        else:
            result = {
                "unsupported_claims": [],
                "inference_written_as_fact": [],
                "forbidden_claim_violations": [],
                "severity": "low",
                "rewrite_required": False,
            }

        appendix_violations = self._detect_appendix_sections(article_markdown)
        if appendix_violations:
            result["forbidden_claim_violations"] = list(result.get("forbidden_claim_violations") or []) + appendix_violations
            result["rewrite_required"] = True
            if result.get("severity") != "high":
                result["severity"] = "medium"
        return result

    @staticmethod
    def _detect_appendix_sections(article_markdown: str) -> list[str]:
        violations: list[str] = []
        for line in str(article_markdown or "").splitlines():
            stripped = line.strip()
            match = re.match(r"^#{1,6}\s+(.+?)\s*$", stripped)
            if not match:
                continue
            heading = match.group(1).strip().lower()
            normalized = re.sub(r"\s+", " ", heading)
            if normalized in {
                "相关阅读",
                "延伸阅读",
                "参考资料",
                "related reading",
                "further reading",
                "references",
            }:
                violations.append(f"Standalone appendix-style section detected: {match.group(1).strip()}")
        return violations

    @staticmethod
    def _empty_result() -> dict[str, Any]:
        return {
            "unsupported_claims": [],
            "inference_written_as_fact": [],
            "forbidden_claim_violations": [],
            "severity": "low",
            "rewrite_required": False,
        }

    @staticmethod
    def _parse_result(text: str) -> dict[str, Any] | None:
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
        severity = str(data.get("severity", "low") or "low").strip().lower()
        if severity not in {"low", "medium", "high"}:
            severity = "low"
        return {
            "unsupported_claims": [str(item).strip() for item in (data.get("unsupported_claims") or []) if str(item).strip()],
            "inference_written_as_fact": [str(item).strip() for item in (data.get("inference_written_as_fact") or []) if str(item).strip()],
            "forbidden_claim_violations": [str(item).strip() for item in (data.get("forbidden_claim_violations") or []) if str(item).strip()],
            "severity": severity,
            "rewrite_required": bool(data.get("rewrite_required", False)),
        }
