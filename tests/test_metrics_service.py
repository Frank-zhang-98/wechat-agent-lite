import unittest
from datetime import datetime, timedelta, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.models import LLMCall, Run, RunStatus
from app.services.metrics_service import _token_bucket, get_token_overview


class MetricsServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        engine = create_engine("sqlite:///:memory:", future=True)
        Base.metadata.create_all(engine)
        SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
        self.session = SessionLocal()

    def tearDown(self) -> None:
        self.session.close()

    def test_token_bucket_excludes_mock_and_blank_model_records(self) -> None:
        run = Run(
            id="run-1",
            run_type="main",
            status=RunStatus.success.value,
            started_at=datetime.now(timezone.utc),
        )
        self.session.add(run)
        self.session.flush()
        self.session.add_all(
            [
                LLMCall(
                    run_id=run.id,
                    step_name="WRITE",
                    role="writer",
                    provider="alibaba-bailian",
                    model="qwen-plus",
                    prompt_tokens=120,
                    completion_tokens=30,
                    total_tokens=150,
                ),
                LLMCall(
                    run_id=run.id,
                    step_name="WRITE",
                    role="writer",
                    provider="alibaba-bailian",
                    model="mock-model",
                    prompt_tokens=80,
                    completion_tokens=20,
                    total_tokens=100,
                ),
                LLMCall(
                    run_id=run.id,
                    step_name="WRITE",
                    role="writer",
                    provider="alibaba-bailian",
                    model="",
                    prompt_tokens=50,
                    completion_tokens=10,
                    total_tokens=60,
                ),
            ]
        )
        self.session.commit()

        bucket = _token_bucket(self.session, label="test", run_id=run.id, run=run)

        self.assertEqual(bucket["calls_count"], 1)
        self.assertEqual(bucket["total_tokens"], 150)
        self.assertEqual(bucket["excluded_mock_calls_count"], 1)
        self.assertEqual(bucket["excluded_invalid_calls_count"], 1)
        self.assertEqual(bucket["excluded_non_real_calls_count"], 2)
        self.assertEqual(bucket["costs"]["supported_calls_count"], 1)
        self.assertEqual(bucket["costs"]["unsupported_calls_count"], 0)

    def test_current_run_ignores_newer_blank_model_history(self) -> None:
        real_run = Run(
            id="run-real",
            run_type="main",
            status=RunStatus.success.value,
            started_at=datetime.now(timezone.utc) - timedelta(hours=2),
        )
        invalid_run = Run(
            id="run-invalid",
            run_type="main",
            status=RunStatus.partial_success.value,
            started_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        self.session.add_all([real_run, invalid_run])
        self.session.flush()
        self.session.add_all(
            [
                LLMCall(
                    run_id=real_run.id,
                    step_name="WRITE",
                    role="writer",
                    provider="alibaba-bailian",
                    model="qwen-plus",
                    prompt_tokens=100,
                    completion_tokens=20,
                    total_tokens=120,
                ),
                LLMCall(
                    run_id=invalid_run.id,
                    step_name="WRITE",
                    role="writer",
                    provider="alibaba-bailian",
                    model="",
                    prompt_tokens=90,
                    completion_tokens=20,
                    total_tokens=110,
                ),
            ]
        )
        self.session.commit()

        overview = get_token_overview(self.session)

        self.assertEqual(overview["windows"]["current_run"]["run_id"], real_run.id)
        self.assertEqual(overview["windows"]["current_run"]["calls_count"], 1)


if __name__ == "__main__":
    unittest.main()
