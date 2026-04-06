import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.services.fetch_service import FetchService
from app.services.search_providers.base import SearchHit
from app.services.settings_service import SettingsService
from app.services.web_enrich_service import WebEnrichService


class WebEnrichServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        engine = create_engine("sqlite:///:memory:", future=True)
        Base.metadata.create_all(engine)
        SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
        self.session = SessionLocal()
        self.settings = SettingsService(self.session)
        self.settings.ensure_defaults()
        self.fetch = FetchService()
        self.service = WebEnrichService(self.settings, self.fetch)

    def tearDown(self) -> None:
        self.session.close()

    def test_build_search_plan_skips_when_evidence_is_strong(self) -> None:
        plan = self.service.build_search_plan(
            run_id="run-1",
            topic={"title": "Strong article"},
            source_pack={},
            source_structure={},
            content_type="technical_walkthrough",
            evidence_score=80.0,
            llm=Mock(),
        )

        self.assertFalse(plan["should_search"])
        self.assertIn("above_threshold", plan["reason"])

    def test_fetch_search_results_filters_official_domains(self) -> None:
        fake_provider = Mock()
        fake_provider.is_available.return_value = True
        fake_provider.search.return_value = [
            SearchHit(title="Official Doc", url="https://docs.example.com/plan", snippet="official", domain="docs.example.com"),
            SearchHit(title="Random Blog", url="https://blog.other.com/post", snippet="context", domain="blog.other.com"),
        ]
        with patch.object(self.service, "_build_provider", return_value=fake_provider):
            with patch.object(
                self.fetch,
                "extract_article_content",
                side_effect=lambda url, max_chars=2500: {"status": "ok", "reason": "", "content_text": f"content for {url}"},
            ):
                result = self.service.fetch_search_results(
                    plan={
                        "should_search": True,
                        "official_domains": ["example.com"],
                        "queries": [
                            {"q": "official pricing", "source_type": "official"},
                            {"q": "context coverage", "source_type": "context"},
                        ],
                    }
                )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(result["official_sources"]), 1)
        self.assertEqual(result["official_sources"][0]["domain"], "docs.example.com")


if __name__ == "__main__":
    unittest.main()
