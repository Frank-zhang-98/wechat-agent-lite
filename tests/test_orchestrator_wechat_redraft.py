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


if __name__ == "__main__":
    unittest.main()
