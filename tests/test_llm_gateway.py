import unittest
from unittest.mock import patch

import requests
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.services.llm_gateway import LLMGateway
from app.services.settings_service import SettingsService


class LLMGatewaySourceMaintenanceTests(unittest.TestCase):
    def setUp(self) -> None:
        engine = create_engine("sqlite:///:memory:", future=True)
        Base.metadata.create_all(engine)
        SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
        self.session = SessionLocal()
        self.settings = SettingsService(self.session)
        self.settings.ensure_defaults()
        self.session.flush()
        self.gateway = LLMGateway(self.session, self.settings)

    def tearDown(self) -> None:
        self.session.close()

    def test_source_maintenance_uses_dedicated_timeout(self) -> None:
        self.settings.set("source_maintenance.llm_timeout_seconds", "18")
        self.session.flush()

        timeout, tier = self.gateway._resolve_timeout_plan(
            role="decision",
            step_name="SOURCE_MAINTENANCE",
            prompt_text="x" * 9000,
        )

        self.assertEqual(timeout, 18)
        self.assertEqual(tier, "high")

    def test_source_maintenance_uses_dedicated_retry_policy(self) -> None:
        self.settings.set("source_maintenance.llm_retry_count", "0")
        self.settings.set("source_maintenance.llm_retry_backoff_seconds", "1")
        self.session.flush()

        with patch("app.services.llm_gateway.requests.request", side_effect=requests.exceptions.Timeout("boom")) as request_mock:
            with patch("app.services.llm_gateway.time.sleep") as sleep_mock:
                with self.assertRaises(RuntimeError):
                    self.gateway._request_with_timeout_retry(
                        "get",
                        "https://example.com",
                        timeout=12,
                        role="decision",
                        step_name="SOURCE_MAINTENANCE",
                    )

        self.assertEqual(request_mock.call_count, 1)
        sleep_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
