import json
import unittest
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.models import Run, RunStatus
from app.services.orchestrator import Orchestrator


class OrchestratorWechatRedraftTests(unittest.TestCase):
    def setUp(self) -> None:
        engine = create_engine("sqlite:///:memory:", future=True)
        Base.metadata.create_all(engine)
        SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
        self.session = SessionLocal()
        self.orch = Orchestrator(self.session)

    def tearDown(self) -> None:
        self.session.close()

    def _create_base_run(self, *, article_markdown: str) -> Run:
        run = Run(
            run_type="main",
            status=RunStatus.partial_success.value,
            trigger_source="manual-test",
            article_title="测试标题",
            article_markdown=article_markdown,
            draft_status="pending_manual",
            quality_score=88.5,
            quality_threshold=78.0,
            quality_attempts=2,
            quality_fallback_used=False,
            summary_json=json.dumps(
                {
                    "selected_topic": {
                        "title": "测试主题",
                        "url": "https://example.com/article",
                        "source": "example",
                    },
                    "cover_asset": {
                        "path": "F:/covers/test-cover.png",
                        "status": "ready",
                    },
                    "cover_5d": {
                        "总分": 92,
                    },
                },
                ensure_ascii=False,
            ),
        )
        self.session.add(run)
        self.session.flush()
        return run

    def test_retry_wechat_draft_reuses_existing_article_context(self) -> None:
        base = self._create_base_run(article_markdown="# 测试标题\n\n正文内容")

        def fake_execute_step(run, name, handler, ctx, policy):
            self.assertEqual(name, "WECHAT_DRAFT")
            self.assertEqual(ctx["article_title"], "测试标题")
            self.assertIn("正文内容", ctx["article_markdown"])
            self.assertEqual(ctx["selected_topic"]["url"], "https://example.com/article")
            self.assertEqual(ctx["cover_asset"]["path"], "F:/covers/test-cover.png")
            ctx["draft_status"] = "saved"
            ctx["wechat_result"] = {
                "success": True,
                "draft_id": "draft-123",
                "reason": "ok",
            }

        with patch.object(self.orch, "_resolve_cover_asset", return_value={"path": "F:/covers/test-cover.png"}):
            with patch.object(self.orch, "_execute_step", side_effect=fake_execute_step):
                new_run = self.orch.rerun_from_action(base.id, "retry_wechat_draft")

        self.assertEqual(new_run.run_type, "manual")
        self.assertEqual(new_run.status, RunStatus.success.value)
        self.assertEqual(new_run.article_title, "测试标题")
        self.assertEqual(new_run.article_markdown, "# 测试标题\n\n正文内容")
        self.assertEqual(new_run.draft_status, "saved")

        summary = json.loads(new_run.summary_json)
        self.assertEqual(summary["source_run_id"], base.id)
        self.assertEqual(summary["redraft_mode"], "wechat_draft_only")
        self.assertEqual(summary["wechat"]["draft_id"], "draft-123")

    def test_retry_wechat_draft_sends_mail_report(self) -> None:
        base = self._create_base_run(article_markdown="# 测试标题\n\n正文内容")

        def fake_execute_step(run, name, handler, ctx, policy):
            ctx["draft_status"] = "saved"
            ctx["wechat_result"] = {
                "success": True,
                "draft_id": "draft-123",
                "reason": "ok",
            }

        with patch.object(self.orch, "_resolve_cover_asset", return_value={"path": "F:/covers/test-cover.png"}):
            with patch.object(self.orch, "_execute_step", side_effect=fake_execute_step):
                with patch.object(self.orch, "_send_daily_report") as mail_mock:
                    self.orch.rerun_from_action(base.id, "retry_wechat_draft")

        mail_mock.assert_called_once()

    def test_retry_wechat_draft_requires_saved_article(self) -> None:
        base = self._create_base_run(article_markdown="")

        with self.assertRaisesRegex(ValueError, "article_markdown"):
            self.orch.rerun_from_action(base.id, "retry_wechat_draft")

    def test_prepare_article_markdown_splits_inline_fence_and_keeps_command_block(self) -> None:
        raw = (
            "## 实战示例\n\n"
            "随后执行：```text claude code --file project_brief.md -t \"Read brief\"\n\n"
            "Claude Code 会进入执行模式。"
        )

        normalized = self.orch._prepare_article_markdown(raw)

        self.assertIn("随后执行：\n```text\nclaude code --file project_brief.md -t \"Read brief\"\n```", normalized)
        self.assertIn("Claude Code 会进入执行模式。", normalized)
        self.assertNotIn("随后执行：```text", normalized)

    def test_prepare_article_markdown_unwraps_prose_swallowed_by_code_fence(self) -> None:
        raw = (
            "```text\n"
            "Claude Code 会进入代理模式，读取概要并拆分任务。\n"
            "## 限制、风险与适用边界\n"
            "当前实践存在明确的工程约束。\n"
            "1. 工具链兼容性有限。\n"
            "```"
        )

        normalized = self.orch._prepare_article_markdown(raw)

        self.assertNotIn("```text", normalized)
        self.assertIn("## 限制、风险与适用边界", normalized)
        self.assertIn("1. 工具链兼容性有限。", normalized)

    def test_prepare_article_markdown_preserves_real_markdown_file_example(self) -> None:
        raw = (
            "## 示例\n\n"
            "```markdown\n"
            "## Project: TaskMaster Mobile\n"
            "**Goal:** Build a cross-platform task manager.\n"
            "**Phase 1:** Set up auth screen.\n"
            "```"
        )

        normalized = self.orch._prepare_article_markdown(raw)

        self.assertIn("```markdown", normalized)
        self.assertIn("## Project: TaskMaster Mobile", normalized)
        self.assertIn("**Goal:** Build a cross-platform task manager.", normalized)

    def test_prepare_article_markdown_localizes_standalone_english_headings(self) -> None:
        raw = (
            "## 按链路拆解实现\n\n"
            "Use Case Setup\n"
            "系统验证使用了一份 131 页的世界银行报告。\n\n"
            "How does PageIndex work?\n"
            "PageIndex 采用了一种根本不同的文档检索方法。\n"
        )

        normalized = self.orch._prepare_article_markdown(raw)

        self.assertIn("### 用例设置", normalized)
        self.assertIn("### PageIndex 如何工作？", normalized)
        self.assertNotIn("Use Case Setup\n", normalized)
        self.assertNotIn("How does PageIndex work?", normalized)

    def test_prepare_article_markdown_preserves_structured_interaction_examples_as_code_block(self) -> None:
        raw = (
            "```text\n"
            "=== VECTORLESS RAG INTERACTION ===\n"
            "Question: what are the questions answered by chapter 2\n"
            "Nodes Retrieved: 0098\n"
            "Response:\n"
            "Based on the provided excerpts, Chapter 2 addresses the following questions:\n"
            "```"
        )

        normalized = self.orch._prepare_article_markdown(raw)

        self.assertIn("```text", normalized)
        self.assertIn("=== VECTORLESS RAG INTERACTION ===", normalized)
        self.assertIn("Question: what are the questions answered by chapter 2", normalized)
        self.assertIn("Response:", normalized)
        self.assertIn("```", normalized)

    def test_execute_existing_manual_url_run_uses_url_pipeline(self) -> None:
        run = Run(
            run_type="manual_url",
            status=RunStatus.pending.value,
            trigger_source="manual-url-ui",
            summary_json=json.dumps(
                {
                    "manual_input": {
                        "source_url": "https://example.com/manual-article"
                    }
                },
                ensure_ascii=False,
            ),
        )
        self.session.add(run)
        self.session.flush()

        executed_steps: list[str] = []

        def fake_execute_step(run_obj, name, handler, ctx, policy):
            executed_steps.append(name)
            if name == "SOURCE_ENRICH":
                self.assertEqual(ctx["selected_topic"]["url"], "https://example.com/manual-article")
                ctx["source_pack"] = {
                    "primary": {
                        "title": "补抓后的文章标题",
                        "url": "https://example.com/manual-article",
                        "summary": "补抓后的摘要",
                        "status": "ok",
                    },
                    "related": [],
                }
            elif name == "SOURCE_STRUCTURE":
                self.assertEqual(ctx["selected_topic"]["title"], "补抓后的文章标题")
                ctx["source_structure"] = {"sections": [], "coverage_checklist": []}
            elif name == "WEB_SEARCH_PLAN":
                ctx["web_search_plan"] = {"should_search": False}
            elif name == "WEB_SEARCH_FETCH":
                ctx["web_enrich"] = {"status": "skipped"}
            elif name == "FACT_GROUNDING":
                ctx["fact_grounding"] = {"evidence_mode": "analysis"}
            elif name == "FACT_PACK":
                ctx["fact_pack"] = {"content_type": "tool_review"}
                ctx["content_type"] = "tool_review"
                ctx["target_audience"] = "ai_product_manager"
            elif name == "FACT_COMPRESS":
                ctx["fact_compress"] = {"one_sentence_summary": "摘要"}
            elif name == "WRITE":
                ctx["article_title"] = "生成文章标题"
                ctx["wechat_title"] = "生成公众号标题"
                ctx["title_plan"] = {"article_title": "生成文章标题", "wechat_title": "生成公众号标题"}
                ctx["article_markdown"] = "# 生成文章标题\n\n正文内容"
            elif name == "HALLUCINATION_CHECK":
                ctx["hallucination_check"] = {"severity": "low"}
            elif name == "VISUAL_STRATEGY":
                ctx["visual_strategy"] = {}
            elif name == "BODY_ILLUSTRATION_GEN":
                ctx["body_illustrations"] = []
            elif name == "QUALITY_CHECK":
                ctx["quality_score"] = 88.0
                ctx["quality_attempts"] = 1
                ctx["quality_fallback_used"] = False
            elif name == "ARTICLE_RENDER":
                ctx["article_layout"] = {"name": "clean_reading", "label": "清晰阅读"}
                ctx["article_render"] = {"html_path": "F:/runs/article.html"}
                ctx["article_html"] = "<p>正文内容</p>"
            elif name == "COVER_5D":
                ctx["cover_5d"] = {"总分": 90}
            elif name == "COVER_GEN":
                ctx["cover_asset"] = {"path": "F:/covers/test-cover.png"}
            elif name == "WECHAT_DRAFT":
                ctx["draft_status"] = "saved"
                ctx["wechat_result"] = {"success": True, "draft_id": "draft-1"}

        with patch.object(self.orch.fetch, "extract_article_metadata", return_value={"title": "", "summary": "", "published": ""}):
            with patch.object(self.orch, "_execute_step", side_effect=fake_execute_step):
                finished = self.orch.execute_existing(run.id)

        self.assertEqual(finished.status, RunStatus.success.value)
        self.assertEqual(finished.article_title, "生成文章标题")
        self.assertEqual(json.loads(finished.summary_json)["selected_topic"]["title"], "补抓后的文章标题")
        self.assertEqual(
            executed_steps,
            [
                "SOURCE_ENRICH",
                "SOURCE_STRUCTURE",
                "WEB_SEARCH_PLAN",
                "WEB_SEARCH_FETCH",
                "FACT_GROUNDING",
                "FACT_PACK",
                "FACT_COMPRESS",
                "WRITE",
                "HALLUCINATION_CHECK",
                "VISUAL_STRATEGY",
                "BODY_ILLUSTRATION_GEN",
                "QUALITY_CHECK",
                "ARTICLE_RENDER",
                "COVER_5D",
                "COVER_GEN",
                "COVER_CHECK",
                "WECHAT_DRAFT",
            ],
        )


if __name__ == "__main__":
    unittest.main()
