import unittest
from types import SimpleNamespace

from app.services.visual_strategy_service import VisualStrategyService


class VisualStrategyServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = VisualStrategyService()

    def test_build_strategy_parses_json_response(self) -> None:
        llm = SimpleNamespace(
            call=lambda *args, **kwargs: SimpleNamespace(
                text="""{
                    "cover_family": "structure",
                    "cover_brief": {
                        "main_claim": "Agent Runtime is infrastructure, not a plugin.",
                        "subject_hint": "Agent Runtime 核心主体",
                        "scene_hint": "深色科技背景中的模块化主体构图",
                        "mood_hint": "克制、专业、可靠",
                        "title_safe_zone": "left_center",
                        "must_show": ["Agent", "Sandbox"],
                        "must_avoid": ["robot face"]
                    },
                    "body_illustrations": [
                        {
                            "type": "architecture_diagram",
                            "section": "系统结构",
                            "title": "Agent Runtime 总览",
                            "caption": "关键模块关系",
                            "must_show": ["Agent", "Gateway"],
                            "must_avoid": ["generic ai glow"]
                        }
                    ]
                }"""
            )
        )
        strategy = self.service.build_strategy(
            run_id="run-1",
            topic={"title": "Agent Runtime"},
            fact_pack={},
            fact_grounding={},
            source_structure={},
            llm=llm,
            max_body_illustrations=2,
        )

        self.assertEqual(strategy["cover_family"], "structure")
        self.assertEqual(len(strategy["body_illustrations"]), 1)
        self.assertEqual(strategy["body_illustrations"][0]["type"], "architecture_diagram")
        self.assertEqual(strategy["cover_brief"]["main_claim"], "智能体运行时不是插件，而是基础设施。")
        self.assertEqual(strategy["cover_brief"]["subject_hint"], "智能体运行时核心主体")
        self.assertEqual(strategy["cover_brief"]["scene_hint"], "深色科技背景中的模块化主体构图")
        self.assertEqual(strategy["cover_brief"]["mood_hint"], "克制、专业、可靠")
        self.assertEqual(strategy["cover_brief"]["title_safe_zone"], "left_center")
        self.assertEqual(strategy["cover_brief"]["must_show"], ["智能体", "沙箱"])
        self.assertEqual(strategy["body_illustrations"][0]["title"], "智能体运行时总览")
        self.assertEqual(strategy["body_illustrations"][0]["must_show"], ["智能体", "网关"])


if __name__ == "__main__":
    unittest.main()
