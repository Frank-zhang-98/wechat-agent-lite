import unittest

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.services.settings_service import SettingsService


class SettingsServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        engine = create_engine("sqlite:///:memory:", future=True)
        Base.metadata.create_all(engine)
        SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
        self.session = SessionLocal()
        self.settings = SettingsService(self.session)

    def tearDown(self) -> None:
        self.session.close()

    def test_get_reads_pending_defaults_before_flush(self) -> None:
        self.settings.ensure_defaults()

        self.assertEqual(self.settings.get("selection.stale_hard_hours_news"), "168")
        self.assertEqual(self.settings.get_float("selection.stale_hard_hours_news", 336.0), 168.0)
        self.assertEqual(self.settings.get("selection.stale_hard_hours_technical"), "1440")
        self.assertEqual(self.settings.get_float("selection.stale_hard_hours_technical", 336.0), 1440.0)


if __name__ == "__main__":
    unittest.main()
