import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import yaml
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.models import RunStatus, SourceHealthState
from app.services.fetch_service import FetchService
from app.services.orchestrator import Orchestrator
from app.services.settings_service import SettingsService
from app.services.source_maintenance_service import SourceMaintenanceService


class FakeResponse:
    def __init__(self, *, status_code: int, text: str, content_type: str, url: str):
        self.status_code = status_code
        self.text = text
        self.content = text.encode("utf-8")
        self.headers = {"Content-Type": content_type}
        self.url = url


class FakeScrapling:
    def __init__(self, discoveries: dict[str, dict] | None = None):
        self.discoveries = discoveries or {}
        self.calls: list[dict[str, object]] = []

    def discover_page(
        self,
        url: str,
        max_articles: int = 12,
        timeout_seconds: float | None = None,
        proxy: str | None = None,
    ) -> dict:
        self.calls.append(
            {
                "url": url,
                "max_articles": max_articles,
                "timeout_seconds": timeout_seconds,
                "proxy": proxy,
            }
        )
        payload = dict(self.discoveries.get(url, {}))
        payload.setdefault("enabled", True)
        payload.setdefault("available", True)
        payload.setdefault("used", True)
        payload.setdefault("page_url", url)
        payload.setdefault("feed_links", [])
        payload.setdefault("articles", [])
        payload.setdefault("error", "")
        return payload

    def build_html_list_items(self, *, url: str, source_name: str, source_weight: float, max_items: int) -> list[dict]:
        payload = self.discover_page(url, max_articles=max_items)
        items = []
        for article in payload.get("articles", [])[:max_items]:
            items.append(
                {
                    "title": article.get("title", ""),
                    "url": article.get("url", ""),
                    "summary": article.get("summary", ""),
                    "published": article.get("published", "2026-03-29T00:00:00+00:00"),
                    "source": source_name,
                    "source_weight": source_weight,
                    "type": "html_list",
                }
            )
        return items


def build_feed(*titles: str) -> str:
    items = "".join(
        f"<item><title>{title}</title><link>https://example.com/{idx}</link>"
        f"<pubDate>Mon, 01 Jan 2024 00:0{idx}:00 GMT</pubDate></item>"
        for idx, title in enumerate(titles, start=1)
    )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Example Feed</title>
    <link>https://example.com/</link>
    <description>Example</description>
    {items}
  </channel>
</rss>
"""


class SourceMaintenanceServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        engine = create_engine("sqlite:///:memory:", future=True)
        Base.metadata.create_all(engine)
        SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
        self.session = SessionLocal()
        self.settings = SettingsService(self.session)
        self.settings.ensure_defaults()
        self.fetch = FetchService()

    def tearDown(self) -> None:
        self.session.close()

    def _write_sources(self, root: Path, payload: dict) -> Path:
        path = root / "sources.yaml"
        path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
        return path

    def test_updates_source_url_when_same_host_candidate_feed_is_found(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_sources(
                Path(tmpdir),
                {
                    "ai_companies": [
                        {
                            "name": "Example Feed",
                            "url": "https://example.com/blog/rss",
                            "enabled": True,
                            "weight": 1.0,
                        }
                    ],
                    "tech_media": [],
                    "tutorial_communities": [],
                },
            )

            def fake_request(url: str, timeout: int = 15):
                if url == "https://example.com/blog/rss":
                    return FakeResponse(status_code=404, text="not found", content_type="text/html", url=url)
                if url == "https://example.com/blog":
                    return FakeResponse(
                        status_code=200,
                        text='<html><head><link rel="alternate" type="application/rss+xml" href="/feed.xml" /></head></html>',
                        content_type="text/html",
                        url=url,
                    )
                if url == "https://example.com/":
                    return FakeResponse(status_code=200, text="<html></html>", content_type="text/html", url=url)
                if url == "https://example.com/feed.xml":
                    return FakeResponse(
                        status_code=200,
                        text=build_feed("alpha", "beta"),
                        content_type="application/rss+xml",
                        url=url,
                    )
                raise AssertionError(f"unexpected url: {url}")

            llm = Mock()
            llm.call.return_value = SimpleNamespace(
                text=(
                    '[{"source_key":"ai_companies:example-feed","action":"update_url",'
                    '"candidate_url":"https://example.com/feed.xml","reason":"same host feed","confidence":0.96}]'
                )
            )
            self.settings.set("source_maintenance.max_llm_cases", "1")
            self.settings.set("source_maintenance.llm_low_confidence_threshold", "0.9")
            self.session.flush()
            service = SourceMaintenanceService(self.session, self.settings, self.fetch, llm)

            with patch.object(FetchService, "sources_path", return_value=config_path):
                with patch.object(self.fetch, "_request", side_effect=fake_request):
                    report = service.run(run_id="run-1")

            self.session.flush()
            saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            source = saved["ai_companies"][0]
            state = self.session.get(SourceHealthState, report["actions"][0]["source_key"])

            self.assertEqual(report["changed_sources"], 1)
            self.assertTrue(report["llm_used"])
            self.assertEqual(source["url"], "https://example.com/feed.xml")
            self.assertEqual(state.current_url, "https://example.com/feed.xml")
            self.assertEqual(state.consecutive_failures, 0)
            self.assertEqual(state.last_status, "updated_url")

    def test_disables_source_after_repeated_404_without_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_sources(
                Path(tmpdir),
                {
                    "ai_companies": [],
                    "tech_media": [
                        {
                            "name": "Dead Feed",
                            "url": "https://dead.example.com/rss",
                            "enabled": True,
                            "weight": 0.9,
                        }
                    ],
                    "tutorial_communities": [],
                },
            )
            self.session.add(
                SourceHealthState(
                    source_key="tech_media:dead-feed",
                    source_name="Dead Feed",
                    category="tech_media",
                    current_url="https://dead.example.com/rss",
                    enabled=True,
                    weight=0.9,
                    consecutive_failures=2,
                )
            )
            self.session.flush()

            def fake_request(url: str, timeout: int = 15):
                return FakeResponse(status_code=404, text="not found", content_type="text/html", url=url)

            llm = Mock()
            llm.call.side_effect = RuntimeError("model unavailable")
            service = SourceMaintenanceService(self.session, self.settings, self.fetch, llm)

            with patch.object(FetchService, "sources_path", return_value=config_path):
                with patch.object(self.fetch, "_request", side_effect=fake_request):
                    report = service.run(run_id="run-2")

            self.session.flush()
            saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            source = saved["tech_media"][0]
            action = report["actions"][0]
            state = self.session.get(SourceHealthState, "tech_media:dead-feed")

            self.assertEqual(report["changed_sources"], 1)
            self.assertFalse(source["enabled"])
            self.assertEqual(action["final_action"], "disable")
            self.assertEqual(action["applied_action"], "disable")
            self.assertEqual(state.consecutive_failures, 3)
            self.assertEqual(state.last_status, "disable")

    def test_switches_to_html_list_when_scrapling_finds_articles(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_sources(
                Path(tmpdir),
                {
                    "ai_companies": [
                        {
                            "name": "HTML Fallback Source",
                            "url": "https://example.com/blog/rss",
                            "enabled": True,
                            "weight": 0.8,
                        }
                    ],
                    "tech_media": [],
                    "tutorial_communities": [],
                },
            )

            def fake_request(url: str, timeout: int = 15):
                if url == "https://example.com/blog/rss":
                    return FakeResponse(status_code=404, text="not found", content_type="text/html", url=url)
                if url == "https://example.com/blog":
                    return FakeResponse(status_code=200, text="<html><body>No feed here</body></html>", content_type="text/html", url=url)
                if url == "https://example.com/":
                    return FakeResponse(status_code=200, text="<html></html>", content_type="text/html", url=url)
                raise AssertionError(f"unexpected url: {url}")

            llm = Mock()
            llm.call.side_effect = RuntimeError("model unavailable")
            scrapling = FakeScrapling(
                {
                    "https://example.com/blog": {
                        "page_url": "https://example.com/blog",
                        "articles": [
                            {
                                "url": "https://example.com/blog/post-one",
                                "title": "Post One",
                                "summary": "Summary one",
                            },
                            {
                                "url": "https://example.com/blog/post-two",
                                "title": "Post Two",
                                "summary": "Summary two",
                            },
                        ],
                    }
                }
            )
            service = SourceMaintenanceService(self.session, self.settings, self.fetch, llm, scrapling=scrapling)

            with patch.object(FetchService, "sources_path", return_value=config_path):
                with patch.object(self.fetch, "_request", side_effect=fake_request):
                    report = service.run(run_id="run-3")

            self.session.flush()
            saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            source = saved["ai_companies"][0]
            action = report["actions"][0]

            self.assertEqual(report["changed_sources"], 1)
            self.assertEqual(action["final_action"], "switch_to_html_list")
            self.assertEqual(action["applied_action"], "switch_to_html_list")
            self.assertEqual(source["mode"], "html_list")
            self.assertEqual(source["url"], "https://example.com/blog")
            self.assertEqual(scrapling.calls[0]["proxy"], None)

    def test_skips_llm_decision_when_max_cases_is_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_sources(
                Path(tmpdir),
                {
                    "ai_companies": [
                        {
                            "name": "Dead Feed",
                            "url": "https://example.com/feed.xml",
                            "enabled": True,
                            "weight": 1.0,
                        }
                    ],
                    "tech_media": [],
                    "tutorial_communities": [],
                },
            )
            self.settings.set("source_maintenance.max_llm_cases", "0")
            self.session.flush()

            def fake_request(url: str, timeout: int = 15):
                return FakeResponse(status_code=404, text="not found", content_type="text/html", url=url)

            llm = Mock()
            service = SourceMaintenanceService(self.session, self.settings, self.fetch, llm)

            with patch.object(FetchService, "sources_path", return_value=config_path):
                with patch.object(self.fetch, "_request", side_effect=fake_request):
                    report = service.run(run_id="run-4")

            self.assertFalse(report["llm_used"])
            llm.call.assert_not_called()

    def test_select_llm_review_items_only_keeps_manual_review_and_low_confidence(self) -> None:
        service = SourceMaintenanceService(self.session, self.settings, self.fetch, Mock())
        failed_items = [
            {
                "source_key": "high",
                "state": SimpleNamespace(consecutive_failures=0),
                "probe": {"reason": "http_404"},
                "candidates": [{"url": "https://example.com/feed.xml"}],
                "html_fallback": {},
                "mode": "rss",
            },
            {
                "source_key": "low",
                "state": SimpleNamespace(consecutive_failures=2),
                "probe": {"reason": "timeout"},
                "candidates": [],
                "html_fallback": {},
                "mode": "rss",
            },
            {
                "source_key": "manual",
                "state": SimpleNamespace(consecutive_failures=0),
                "probe": {"reason": "request_error"},
                "candidates": [],
                "html_fallback": {},
                "mode": "rss",
            },
        ]

        selected = service._select_llm_review_items(failed_items)

        self.assertEqual([item["source_key"] for item in selected], ["low", "manual"])
        self.assertEqual(selected[0]["_llm_review_reason"], "low_confidence")
        self.assertEqual(selected[1]["_llm_review_reason"], "manual_review")

    def test_probe_feed_uses_remaining_budget_as_timeout_cap(self) -> None:
        seen: list[float] = []

        def fake_request(url: str, timeout: int = 15):
            seen.append(float(timeout))
            return FakeResponse(status_code=404, text="not found", content_type="text/html", url=url)

        service = SourceMaintenanceService(self.session, self.settings, self.fetch, Mock())
        deadline = time.monotonic() + 2.4
        with patch.object(self.fetch, "_request", side_effect=fake_request):
            service._probe_feed("https://example.com/feed.xml", deadline=deadline)

        self.assertEqual(len(seen), 1)
        self.assertLessEqual(seen[0], 2.5)

    def test_fetch_source_supports_html_list_mode(self) -> None:
        scrapling = FakeScrapling(
            {
                "https://example.com/blog": {
                    "articles": [
                        {
                            "url": "https://example.com/blog/post-one",
                            "title": "Post One",
                            "summary": "Summary one",
                            "published": "2026-03-29T00:00:00+00:00",
                        },
                        {
                            "url": "https://example.com/blog/post-two",
                            "title": "Post Two",
                            "summary": "Summary two",
                            "published": "2026-03-28T00:00:00+00:00",
                        },
                    ]
                }
            }
        )
        items = self.fetch.fetch_source(
            {
                "name": "HTML Source",
                "url": "https://example.com/blog",
                "mode": "html_list",
                "weight": 0.9,
            },
            max_age_hours=168,
            max_items=5,
            scrapling=scrapling,
        )

        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]["type"], "html_list")
        self.assertEqual(items[0]["source"], "HTML Source")
        self.assertEqual(items[0]["url"], "https://example.com/blog/post-one")

    def test_scrapling_fallback_forwards_proxy(self) -> None:
        from app.services.scrapling_fallback_service import ScraplingFallbackService

        class FakeFetcher:
            captured: dict[str, object] = {}

            @classmethod
            def get(cls, url: str, **kwargs):
                cls.captured = {"url": url, **kwargs}
                return SimpleNamespace(url=url, status=200, css=lambda selector: [])

        service = ScraplingFallbackService(enabled=True, proxy="socks5h://127.0.0.1:10808")
        service._fetcher_cls = FakeFetcher

        result = service.discover_page("https://example.com/blog", timeout_seconds=4)

        self.assertTrue(result["used"])
        self.assertEqual(FakeFetcher.captured["proxy"], "socks5h://127.0.0.1:10808")

    def test_source_key_is_unique_for_non_ascii_names(self) -> None:
        key_a = SourceMaintenanceService._source_key(category="ai_companies", name="百川智能")
        key_b = SourceMaintenanceService._source_key(category="ai_companies", name="阶跃星辰")

        self.assertNotEqual(key_a, key_b)
        self.assertTrue(key_a.startswith("ai_companies:source-"))
        self.assertTrue(key_b.startswith("ai_companies:source-"))

    def test_get_state_reuses_pending_state_with_same_source_key(self) -> None:
        service = SourceMaintenanceService(self.session, self.settings, self.fetch, Mock())
        source = {"name": "百川智能", "url": "https://example.com/a", "enabled": True, "weight": 0.9}

        first = service._get_state(source=source, category="ai_companies")
        second = service._get_state(source=source, category="ai_companies")

        self.assertIs(first, second)
        self.assertEqual(first.current_url, "https://example.com/a")


class OrchestratorSourceMaintenanceTests(unittest.TestCase):
    def setUp(self) -> None:
        engine = create_engine("sqlite:///:memory:", future=True)
        Base.metadata.create_all(engine)
        SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
        self.session = SessionLocal()
        self.orch = Orchestrator(self.session)

    def tearDown(self) -> None:
        self.session.close()

    def test_main_flow_runs_source_maintenance_by_default(self) -> None:
        run = self.orch.create_run(run_type="main", trigger_source="test", status=RunStatus.running.value)
        ctx = {"quality_scores": [], "failed_logs": []}
        executed_steps: list[str] = []

        def fake_execute_step(run_obj, name, handler, ctx_obj, policy):
            executed_steps.append(name)

        with patch.object(self.orch, "_execute_step", side_effect=fake_execute_step):
            self.orch._run_main(run, ctx)

        self.assertEqual(executed_steps[:3], ["HEALTH_CHECK", "SOURCE_MAINTENANCE", "FETCH"])

    def test_main_flow_skips_source_maintenance_when_disabled(self) -> None:
        run = self.orch.create_run(run_type="main", trigger_source="test", status=RunStatus.running.value)
        ctx = {"quality_scores": [], "failed_logs": []}
        executed_steps: list[str] = []
        self.orch.settings.set("source_maintenance.run_on_main", "false")
        self.session.flush()

        def fake_execute_step(run_obj, name, handler, ctx_obj, policy):
            executed_steps.append(name)

        with patch.object(self.orch, "_execute_step", side_effect=fake_execute_step):
            self.orch._run_main(run, ctx)

        self.assertEqual(executed_steps[:2], ["HEALTH_CHECK", "FETCH"])
        self.assertNotIn("SOURCE_MAINTENANCE", executed_steps)

    def test_step_fetch_keeps_source_order_with_parallel_jobs(self) -> None:
        run = self.orch.create_run(run_type="main", trigger_source="test", status=RunStatus.running.value)
        ctx = {"quality_scores": [], "failed_logs": []}
        cfg = {
            "ai_companies": [
                {"name": "Source A", "url": "https://a.example.com/feed.xml", "enabled": True},
                {"name": "Source B", "url": "https://b.example.com/feed.xml", "enabled": True},
            ],
            "tech_media": [],
            "tutorial_communities": [],
            "github": {"enabled": True},
            "max_age_hours": 168,
            "max_hotspots_per_source": 10,
        }

        def fake_fetch_source(source, **kwargs):
            return [
                {
                    "title": source["name"],
                    "url": f"https://example.com/{source['name'].lower().replace(' ', '-')}",
                    "summary": source["name"],
                    "published": "2026-03-29T00:00:00+00:00",
                    "source": source["name"],
                    "source_weight": 1.0,
                    "type": "rss",
                }
            ]

        def fake_fetch_github(*args, **kwargs):
            return [
                {
                    "title": "GitHub Repo",
                    "url": "https://github.com/example/repo",
                    "summary": "repo",
                    "published": "2026-03-29T00:00:00+00:00",
                    "source": "GitHub Trending",
                    "source_weight": 0.85,
                    "type": "github",
                }
            ]

        with patch.object(self.orch.fetch, "load_sources", return_value=cfg):
            with patch.object(self.orch.fetch, "fetch_source", side_effect=fake_fetch_source):
                with patch.object(self.orch.fetch, "fetch_github", side_effect=fake_fetch_github):
                    with patch.object(self.orch.fetch, "dump_debug"):
                        self.orch._step_fetch(run, ctx)

        self.assertEqual(
            [item["title"] for item in ctx["fetched_items"]],
            ["Source A", "Source B", "GitHub Repo"],
        )


if __name__ == "__main__":
    unittest.main()
