import unittest

from app.services.title_generation_service import TitleGenerationService


class FakeLLMResult:
    def __init__(self, text: str):
        self.text = text


class FakeLLM:
    def __init__(self, text: str):
        self.text = text

    def call(self, run_id: str, step_name: str, role: str, prompt: str, temperature: float = 0.35):
        return FakeLLMResult(self.text)


class TitleGenerationServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = TitleGenerationService()

    def test_fallback_generates_short_wechat_title_for_english_topic(self) -> None:
        plan = self.service.generate(
            run_id="run-1",
            topic={
                "title": "I Analyzed My Claude Code Session and Found 5 Repeated Mistakes",
                "summary": "A practical review of repeated issues and workflow lessons from real Claude Code usage.",
            },
            fact_pack={"key_points": ["Focused on workflow efficiency and repeated mistakes."]},
            fact_compress={"one_sentence_summary": "复盘 Claude Code 会话中的重复问题和工作流改法。", "numbers": ["5"]},
            content_type="tool_review",
            llm=FakeLLM("not-json"),
        )

        self.assertTrue(plan.article_title)
        self.assertTrue(plan.wechat_title)
        self.assertLessEqual(len(plan.wechat_title), 32)
        self.assertLessEqual(len(plan.wechat_title.encode("utf-8")), 96)
        self.assertNotIn("Sessio", plan.wechat_title)

    def test_llm_generated_titles_are_used_when_valid(self) -> None:
        plan = self.service.generate(
            run_id="run-2",
            topic={"title": "How Multi-Agent Systems Are Reshaping Software Development", "summary": "A practical analysis."},
            fact_pack={"key_points": ["Multi-agent systems reshape development workflows."]},
            fact_compress={"one_sentence_summary": "多代理系统正在重塑软件开发流程。"},
            content_type="tool_review",
            llm=FakeLLM(
                '{"article_title":"多代理系统正在重塑软件开发：实战解读","wechat_title":"多代理系统重塑开发","reason":"更适合公众号阅读"}'
            ),
        )

        self.assertEqual(plan.source, "llm")
        self.assertEqual(plan.article_title, "多代理系统正在重塑软件开发：实战解读")
        self.assertEqual(plan.wechat_title, "多代理系统重塑开发")

    def test_long_wechat_title_is_compacted_instead_of_hard_cut(self) -> None:
        plan = self.service.generate(
            run_id="run-3",
            topic={
                "title": "Proxy-Pointer RAG：不用向量库，如何实现98.7%准确率",
                "summary": "A hybrid retrieval design that combines structure-aware precision with vector efficiency.",
            },
            fact_pack={"key_points": ["The article compares PageIndex with Proxy-Pointer RAG."]},
            fact_compress={"one_sentence_summary": "Proxy-Pointer RAG 试图在准确率与规模成本之间取得平衡。", "numbers": ["98.7%"]},
            content_type="technical_walkthrough",
            llm=FakeLLM(
                '{"article_title":"Proxy-Pointer RAG：如何融合结构感知精度与向量检索效率？","wechat_title":"Proxy-Pointer RAG：不用向量库，如何实现98.7%准确率","reason":"保留核心数字信息"}'
            ),
        )

        self.assertEqual(plan.source, "llm")
        self.assertEqual(plan.wechat_title, "Proxy-Pointer RAG：如何实现98.7%准确率")
        self.assertLessEqual(len(plan.wechat_title), 32)
        self.assertNotEqual(plan.wechat_title, "Proxy-Pointer RAG：不用向量库，如何实现98.7")


if __name__ == "__main__":
    unittest.main()
