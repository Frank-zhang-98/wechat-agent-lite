import json
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.models import Run, RunStatus
from app.services.orchestrator import Orchestrator


class TopicSelectionFilterTests(unittest.TestCase):
    def setUp(self) -> None:
        engine = create_engine("sqlite:///:memory:", future=True)
        Base.metadata.create_all(engine)
        SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
        self.session = SessionLocal()
        self.orch = Orchestrator(self.session)

    def tearDown(self) -> None:
        self.session.close()

    def _add_historical_run(self, *, title: str, source: str, url: str, days_ago: float) -> None:
        started_at = datetime.now(timezone.utc) - timedelta(days=days_ago)
        run = Run(
            run_type="main",
            status=RunStatus.success.value,
            trigger_source="test",
            started_at=started_at,
            finished_at=started_at + timedelta(minutes=3),
            summary_json=json.dumps(
                {
                    "selected_topic": {
                        "title": title,
                        "source": source,
                        "url": url,
                    }
                },
                ensure_ascii=False,
            ),
        )
        self.session.add(run)
        self.session.flush()

    def test_should_reject_workshop_registration_topic(self) -> None:
        item = {
            "title": "April 8 - Getting Started with Computer Vision Workflows Workshop",
            "summary": "Join us on April 8 at 10 AM Pacific for a free, 60-minute, virtual hands-on workshop. Register for the Zoom.",
            "url": "https://dev.to/voxel51/april-8-getting-started-with-computer-vision-workflows-workshop-1mf8",
        }

        self.assertTrue(Orchestrator._should_reject_topic(item))
        self.assertGreaterEqual(Orchestrator._topic_editorial_penalty_score(item), 85.0)

    def test_should_reject_direct_sales_topic(self) -> None:
        item = {
            "title": "Cycle 248: Launching the P2P API Monetization Stack ($19) - Direct Honor System Sales",
            "summary": "PayPal and IBAN included for $19 direct honor system sales. A monetization stack for AI agents and solo developers.",
            "url": "https://dev.to/universe7creator/cycle-248-launching-the-p2p-api-monetization-stack-19-direct-honor-system-sales-4agh",
        }

        self.assertTrue(Orchestrator._should_reject_topic(item))
        self.assertGreaterEqual(Orchestrator._topic_editorial_penalty_score(item), 85.0)

    def test_rule_score_filters_workshop_and_sales_topics(self) -> None:
        now = datetime.now(timezone.utc).isoformat()
        ctx = {
            "deduped_items": [
                {
                    "title": "April 8 - Getting Started with Computer Vision Workflows Workshop",
                    "summary": "Join us for a free virtual workshop. Register for the Zoom.",
                    "url": "https://example.com/workshop",
                    "published": now,
                    "source": "Dev.to AI",
                    "source_weight": 0.8,
                },
                {
                    "title": "Cycle 248: Launching the P2P API Monetization Stack ($19) - Direct Honor System Sales",
                    "summary": "PayPal and IBAN included for $19 direct honor system sales.",
                    "url": "https://example.com/direct-sales",
                    "published": now,
                    "source": "Dev.to AI",
                    "source_weight": 0.8,
                },
                {
                    "title": "How we built a multi-agent evaluation pipeline",
                    "summary": "An engineering walkthrough covering architecture, retries, evaluation, and production guardrails.",
                    "url": "https://example.com/agent-pipeline",
                    "published": now,
                    "source": "OpenAI Blog",
                    "source_weight": 1.0,
                },
            ]
        }

        self.orch._step_rule_score(SimpleNamespace(id="run-1"), ctx)

        titles = [item["title"] for item in ctx["top_n"]]
        self.assertEqual(titles, ["How we built a multi-agent evaluation pipeline"])

    def test_rule_score_skips_url_like_titles_even_if_published_is_recent(self) -> None:
        now = datetime.now(timezone.utc).isoformat()
        ctx = {
            "deduped_items": [
                {
                    "title": "https://example.com/blog/2024-ai-first",
                    "summary": "",
                    "url": "https://example.com/blog/2024-ai-first",
                    "published": now,
                    "source": "HTML Source",
                    "source_weight": 0.9,
                },
                {
                    "title": "Real Engineering Deep Dive",
                    "summary": "A real architecture walkthrough.",
                    "url": "https://example.com/deep-dive",
                    "published": now,
                    "source": "OpenAI Blog",
                    "source_weight": 1.0,
                },
            ]
        }

        self.orch._step_rule_score(SimpleNamespace(id="run-url-like-title"), ctx)

        titles = [item["title"] for item in ctx["top_n"]]
        self.assertEqual(titles, ["Real Engineering Deep Dive"])

    def test_rule_score_uses_soft_topic_gate_warning_instead_of_failing(self) -> None:
        self.orch.settings.set("quality.min_topic_score", "90")
        self.session.flush()
        now = datetime.now(timezone.utc).isoformat()
        ctx = {
            "deduped_items": [
                {
                    "title": "Useful but ordinary post",
                    "summary": "A solid engineering post that should still continue to rerank.",
                    "url": "https://example.com/ordinary",
                    "published": now,
                    "source": "OpenAI Blog",
                    "source_weight": 1.0,
                }
            ]
        }

        self.orch._step_rule_score(SimpleNamespace(id="run-soft-gate"), ctx)

        self.assertEqual(len(ctx["top_n"]), 1)
        self.assertIn("No topic passed minimum topic score 90.0", ctx.get("topic_gate_warning", ""))

    def test_rule_score_applies_source_diversity_cap(self) -> None:
        self.orch.settings.set("selection.top_n_per_source_family", "1")
        self.session.flush()
        now = datetime.now(timezone.utc).isoformat()
        ctx = {
            "deduped_items": [
                {
                    "title": "Dev Post A",
                    "summary": "Architecture walkthrough for agents.",
                    "url": "https://dev.to/post-a",
                    "published": now,
                    "source": "Dev.to AI",
                    "source_weight": 0.8,
                },
                {
                    "title": "Dev Post B",
                    "summary": "Workflow and API design.",
                    "url": "https://dev.to/post-b",
                    "published": now,
                    "source": "Dev.to Machine Learning",
                    "source_weight": 0.8,
                },
                {
                    "title": "OpenAI Deep Dive",
                    "summary": "Model architecture and production details.",
                    "url": "https://openai.com/blog/deep-dive",
                    "published": now,
                    "source": "OpenAI Blog",
                    "source_weight": 1.0,
                },
            ]
        }

        self.orch._step_rule_score(SimpleNamespace(id="run-diversity"), ctx)

        families = [self.orch._topic_source_family(item) for item in ctx["top_n"]]
        self.assertEqual(len(families), len(set(families)))

    def test_rule_score_filters_stale_low_value_topic(self) -> None:
        now = datetime.now(timezone.utc)
        stale = (now - timedelta(days=18)).isoformat()
        fresh = now.isoformat()
        ctx = {
            "deduped_items": [
                {
                    "title": "AI Weekly News Roundup",
                    "summary": "This week in AI: product updates, launch news, and hot takes.",
                    "url": "https://example.com/weekly-roundup",
                    "published": stale,
                    "source": "AI Weekly",
                    "source_weight": 0.8,
                },
                {
                    "title": "How we built a multi-agent evaluation pipeline",
                    "summary": "An engineering walkthrough covering architecture, retries, evaluation, and production guardrails.",
                    "url": "https://example.com/agent-pipeline",
                    "published": fresh,
                    "source": "OpenAI Blog",
                    "source_weight": 1.0,
                },
            ]
        }

        self.orch._step_rule_score(SimpleNamespace(id="run-stale-filter"), ctx)

        titles = [item["title"] for item in ctx["top_n"]]
        self.assertIn("How we built a multi-agent evaluation pipeline", titles)
        self.assertNotIn("AI Weekly News Roundup", titles)

    def test_rule_score_keeps_old_evergreen_walkthrough(self) -> None:
        old = (datetime.now(timezone.utc) - timedelta(days=12)).isoformat()
        ctx = {
            "deduped_items": [
                {
                    "title": "LangGraph architecture deep dive",
                    "summary": "A technical walkthrough covering implementation details, workflow patterns, APIs, and benchmark tradeoffs.",
                    "url": "https://example.com/langgraph-deep-dive",
                    "published": old,
                    "source": "OpenAI Blog",
                    "source_weight": 1.0,
                }
            ]
        }

        self.orch._step_rule_score(SimpleNamespace(id="run-old-evergreen"), ctx)

        self.assertEqual(len(ctx["top_n"]), 1)
        self.assertEqual(ctx["top_n"][0]["title"], "LangGraph architecture deep dive")
        self.assertGreater(ctx["top_n"][0].get("evergreen_score", 0), 58.0)

    def test_rule_score_filters_week_old_low_value_roundup_under_stricter_stale_gate(self) -> None:
        stale = (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()
        ctx = {
            "deduped_items": [
                {
                    "title": "This Week in AI roundup",
                    "summary": "Daily and weekly launch announcements, breaking AI news, and hot takes.",
                    "url": "https://example.com/weekly-ai-roundup",
                    "published": stale,
                    "source": "AI Weekly",
                    "source_weight": 0.8,
                }
            ]
        }

        with self.assertRaisesRegex(RuntimeError, "No suitable items left after topic filtering"):
            self.orch._step_rule_score(SimpleNamespace(id="run-week-old-roundup"), ctx)

    def test_timeliness_profile_uses_news_product_and_technical_windows(self) -> None:
        self.assertEqual(
            self.orch._topic_timeliness_profile(
                {"title": "This Week in AI roundup", "summary": "Daily news brief", "url": "https://example.com/news"}
            ),
            "news",
        )
        self.assertEqual(
            self.orch._topic_timeliness_profile(
                {"title": "MiMo V2 release review", "summary": "Hands-on benchmark and product update", "url": "https://example.com/review"}
            ),
            "product",
        )
        self.assertEqual(
            self.orch._topic_timeliness_profile(
                {"title": "LangGraph architecture deep dive", "summary": "Implementation walkthrough and guide", "url": "https://example.com/guide"}
            ),
            "technical",
        )

        self.assertEqual(self.orch._timeliness_thresholds("news"), (72.0, 168.0))
        self.assertEqual(self.orch._timeliness_thresholds("product"), (168.0, 504.0))
        self.assertEqual(self.orch._timeliness_thresholds("technical"), (720.0, 1440.0))

    def test_rule_score_keeps_month_old_technical_tutorial(self) -> None:
        old = (datetime.now(timezone.utc) - timedelta(days=35)).isoformat()
        ctx = {
            "deduped_items": [
                {
                    "title": "Agent memory architecture deep dive",
                    "summary": "A technical walkthrough and implementation guide for long-term memory design.",
                    "url": "https://example.com/agent-memory-guide",
                    "published": old,
                    "source": "OpenAI Blog",
                    "source_weight": 1.0,
                }
            ]
        }

        self.orch._step_rule_score(SimpleNamespace(id="run-month-old-technical"), ctx)

        self.assertEqual(len(ctx["top_n"]), 1)
        self.assertEqual(ctx["top_n"][0]["timeliness_profile"], "technical")

    def test_fatigue_penalty_is_gentle_and_recovers_over_time(self) -> None:
        item = {
            "title": "Engineering Deep Dive",
            "summary": "Architecture walkthrough.",
            "url": "https://openai.com/blog/deep-dive",
            "source": "OpenAI Blog",
        }
        self._add_historical_run(
            title="Engineering Deep Dive",
            source="OpenAI Blog",
            url="https://openai.com/blog/deep-dive",
            days_ago=1.0,
        )
        recent_penalty = self.orch._topic_fatigue_penalty_score(item, current_run_id="run-fatigue")

        self.session.query(Run).delete()
        self.session.flush()
        self._add_historical_run(
            title="Engineering Deep Dive",
            source="OpenAI Blog",
            url="https://openai.com/blog/deep-dive",
            days_ago=7.0,
        )
        old_penalty = self.orch._topic_fatigue_penalty_score(item, current_run_id="run-fatigue")

        self.assertGreater(recent_penalty, old_penalty)
        self.assertGreater(recent_penalty, 0.0)
        self.assertLessEqual(recent_penalty, 12.0)
        self.assertLess(old_penalty, recent_penalty * 0.5)

    def test_step_select_prompt_includes_exclusion_rules(self) -> None:
        ctx = {
            "top_k": [
                {
                    "title": "How we built a multi-agent evaluation pipeline",
                    "summary": "An engineering walkthrough with real implementation details.",
                    "url": "",
                    "published": datetime.now(timezone.utc).isoformat(),
                    "source": "OpenAI Blog",
                    "rule_score": 82.0,
                    "freshness_score": 80.0,
                    "depth_score": 85.0,
                    "value_score": 78.0,
                    "novelty_score": 70.0,
                    "editorial_penalty_score": 0.0,
                    "fatigue_penalty_score": 0.0,
                    "rerank_excerpt": "",
                }
            ]
        }

        with patch.object(self.orch.llm, "call", return_value=SimpleNamespace(text='{"index": 0, "reason": "ok"}')) as call_mock:
            with patch.object(self.orch, "_probe_topic_evidence", return_value={"score": 72.0, "summary": "sections=6, code=2", "status": "ok"}):
                self.orch._step_select(SimpleNamespace(id="run-select"), ctx)

        prompt = call_mock.call_args.args[3]
        self.assertIn("workshop/webinar/conference", prompt)
        self.assertIn("卖代码/卖模板/引导付款", prompt)
        self.assertIn("原文证据分", prompt)

    def test_step_select_fallback_uses_evidence_score(self) -> None:
        ctx = {
            "top_k": [
                {
                    "title": "Shallow Post",
                    "summary": "Looks trendy but has little structure.",
                    "url": "https://example.com/shallow",
                    "published": datetime.now(timezone.utc).isoformat(),
                    "source": "Dev.to AI",
                    "rule_score": 80.0,
                    "freshness_score": 90.0,
                    "depth_score": 60.0,
                    "value_score": 60.0,
                    "novelty_score": 75.0,
                    "editorial_penalty_score": 0.0,
                    "fatigue_penalty_score": 0.0,
                    "rerank_excerpt": "",
                },
                {
                    "title": "Real Engineering Deep Dive",
                    "summary": "A real architecture walkthrough.",
                    "url": "https://example.com/deep",
                    "published": datetime.now(timezone.utc).isoformat(),
                    "source": "OpenAI Blog",
                    "rule_score": 76.0,
                    "freshness_score": 75.0,
                    "depth_score": 88.0,
                    "value_score": 80.0,
                    "novelty_score": 68.0,
                    "editorial_penalty_score": 0.0,
                    "fatigue_penalty_score": 0.0,
                    "rerank_excerpt": "",
                },
            ]
        }

        evidence_side_effect = [
            {"score": 15.0, "summary": "sections=1, code=0", "status": "ok"},
            {"score": 82.0, "summary": "sections=6, code=2", "status": "ok"},
        ]

        with patch.object(self.orch.llm, "call", return_value=SimpleNamespace(text="not-json")):
            with patch.object(self.orch, "_probe_topic_evidence", side_effect=evidence_side_effect):
                self.orch._step_select(SimpleNamespace(id="run-fallback"), ctx)

        self.assertEqual(ctx["selected_topic"]["title"], "Real Engineering Deep Dive")

    def test_probe_topic_evidence_penalizes_podcast_page_without_transcript(self) -> None:
        item = {
            "title": "Episode #289: Limitations in Human and Automated Code Review",
            "summary": "The Real Python Podcast episode page.",
            "url": "https://realpython.com/podcasts/rpp/289/#t=775",
        }

        with patch.object(
            self.orch.fetch,
            "extract_article_structure",
            return_value={
                "status": "ok",
                "title": "Episode #289: Limitations in Human and Automated Code Review – The Real Python Podcast",
                "lead": "",
                "sections": [
                    {"heading": "Episode 289", "summary": ""},
                    {"heading": "The Real Python Podcast", "summary": "RSS Apple Spotify Download MP3"},
                    {"heading": "Level Up Your Python Skills", "summary": "Course links"},
                ],
                "code_blocks": [],
                "lists": [],
                "tables": [],
                "coverage_checklist": ["Episode 289", "The Real Python Podcast", "Level Up Your Python Skills"],
            },
        ):
            evidence = self.orch._probe_topic_evidence(item)

        self.assertTrue(evidence["is_audio_page"])
        self.assertFalse(evidence["has_transcript_signal"])
        self.assertLess(evidence["score"], 40.0)

    def test_probe_topic_evidence_penalizes_data_service_page(self) -> None:
        item = {
            "title": "机器之心·数据服务",
            "summary": "data service landing page for paid/reference access.",
            "url": "https://pro.jiqizhixin.com/reference/e2d2143f-d160-4756-88b1-966801a41a4b",
        }

        with patch.object(
            self.orch.fetch,
            "extract_article_structure",
            return_value={
                "status": "ok",
                "title": "机器之心·数据服务",
                "lead": "",
                "sections": [
                    {"heading": "还在费劲爬数据？机器之心数据服务已上线 直接获取数据，高效又稳定！", "summary": "深入合作请联系：zhaoyunfeng@jiqizhixin.com"},
                ],
                "code_blocks": [],
                "lists": [],
                "tables": [],
                "coverage_checklist": ["还在费劲爬数据？机器之心数据服务已上线"],
            },
        ):
            evidence = self.orch._probe_topic_evidence(item)

        self.assertTrue(evidence["has_data_service_signal"])
        self.assertLess(evidence["score"], 20.0)

    def test_step_fact_pack_auto_switches_builder_audience_for_technical_topic(self) -> None:
        ctx = {
            "selected_topic": {
                "title": "Session Management for AI Agents",
                "summary": "TTL, renewals, absolute lifetime, and implementation walkthrough.",
                "url": "https://example.com/session",
                "source": "dev.to",
                "published": datetime.now(timezone.utc).isoformat(),
            },
            "source_pack": {
                "primary": {
                    "status": "ok",
                    "content_text": "lead text",
                    "paragraphs": ["para1", "para2", "para3"],
                },
                "related": [],
            },
            "top_k": [],
            "source_structure": {
                "lead": "This system explains TTL, renewal controls, and hard expiry.",
                "coverage_checklist": ["TTL timeout", "Renewal control", "Absolute lifetime"],
                "sections": [
                    {
                        "heading": "Step 1: TTL",
                        "summary": "Idle expiry logic.",
                        "paragraphs": ["Track last active time"],
                        "code_refs": [0],
                    },
                    {
                        "heading": "Session architecture",
                        "summary": "Gateway, policy engine, and wallet adapter.",
                        "paragraphs": ["Role split", "Guardrails"],
                        "code_refs": [],
                    },
                ],
                "code_blocks": [{"language": "ts", "code_excerpt": "const session = createSession({...})"}],
            },
        }

        self.orch._step_fact_pack(SimpleNamespace(id="run-fact-pack"), ctx)

        self.assertEqual(ctx["content_type"], "technical_walkthrough")
        self.assertEqual(ctx["target_audience"], "ai_builder")


if __name__ == "__main__":
    unittest.main()
