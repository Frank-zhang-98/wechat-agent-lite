import unittest
from types import SimpleNamespace

from app.services.fact_grounding_service import FactGroundingService
from app.services.hallucination_check_service import HallucinationCheckService


class GroundingAndHallucinationServiceTests(unittest.TestCase):
    def test_fact_grounding_parses_structured_json(self) -> None:
        llm = SimpleNamespace(
            call=lambda *args, **kwargs: SimpleNamespace(
                text="""{
                    "hard_facts": ["事实 A"],
                    "official_facts": ["官方事实 B"],
                    "context_facts": ["背景事实 C"],
                    "soft_inferences": ["推测 D"],
                    "unknowns": ["未知 E"],
                    "forbidden_claims": ["禁止 F"],
                    "evidence_mode": "analysis"
                }"""
            )
        )
        service = FactGroundingService()
        result = service.ground(
            run_id="run-1",
            topic={"title": "Topic"},
            source_pack={},
            source_structure={},
            web_enrich={},
            llm=llm,
        )

        self.assertEqual(result["hard_facts"], ["事实 A"])
        self.assertEqual(result["evidence_mode"], "analysis")

    def test_hallucination_check_parses_structured_json(self) -> None:
        llm = SimpleNamespace(
            call=lambda *args, **kwargs: SimpleNamespace(
                text="""{
                    "unsupported_claims": ["claim-1"],
                    "inference_written_as_fact": ["claim-2"],
                    "forbidden_claim_violations": [],
                    "severity": "high",
                    "rewrite_required": true
                }"""
            )
        )
        service = HallucinationCheckService()
        result = service.check(
            run_id="run-1",
            article_markdown="article",
            fact_grounding={"hard_facts": ["A"]},
            llm=llm,
        )

        self.assertEqual(result["severity"], "high")
        self.assertTrue(result["rewrite_required"])
        self.assertEqual(result["unsupported_claims"], ["claim-1"])

    def test_hallucination_check_flags_related_reading_section(self) -> None:
        llm = SimpleNamespace(
            call=lambda *args, **kwargs: SimpleNamespace(
                text="""{
                    "unsupported_claims": [],
                    "inference_written_as_fact": [],
                    "forbidden_claim_violations": [],
                    "severity": "low",
                    "rewrite_required": false
                }"""
            )
        )
        service = HallucinationCheckService()
        result = service.check(
            run_id="run-1",
            article_markdown="# 标题\n\n## 相关阅读\n\n- 条目 A",
            fact_grounding={"hard_facts": ["A"]},
            llm=llm,
        )

        self.assertTrue(result["rewrite_required"])
        self.assertEqual(result["severity"], "medium")
        self.assertTrue(result["forbidden_claim_violations"])


if __name__ == "__main__":
    unittest.main()
