from __future__ import annotations

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from app.core.config import CONFIG
from app.db import get_session
from app.services.model_pricing_service import sync_pricing_catalog
from app.services.orchestrator import Orchestrator
from app.services.settings_service import SettingsService


class SchedulerService:
    def __init__(self) -> None:
        self.scheduler = BackgroundScheduler(timezone=CONFIG.timezone)
        self.started = False

    def start(self) -> None:
        if self.started:
            return
        with get_session() as session:
            settings = SettingsService(session)
            settings.ensure_defaults()
            health_cron = settings.get("general.health_cron", "30 8 * * *")
            main_cron = settings.get("general.main_cron", "0 9 * * *")

        self.scheduler.add_job(
            self._run_health_job,
            trigger=CronTrigger.from_crontab(health_cron, timezone=CONFIG.timezone),
            id="health-check-job",
            replace_existing=True,
        )
        self.scheduler.add_job(
            self._run_main_job,
            trigger=CronTrigger.from_crontab(main_cron, timezone=CONFIG.timezone),
            id="main-run-job",
            replace_existing=True,
        )
        self.scheduler.add_job(
            self._sync_pricing_job,
            trigger=IntervalTrigger(hours=12, timezone=CONFIG.timezone),
            id="pricing-sync-job",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )
        self.scheduler.start()
        self.started = True

    def reload_jobs(self) -> None:
        if not self.started:
            return
        with get_session() as session:
            settings = SettingsService(session)
            health_cron = settings.get("general.health_cron", "30 8 * * *")
            main_cron = settings.get("general.main_cron", "0 9 * * *")
        self.scheduler.reschedule_job(
            "health-check-job", trigger=CronTrigger.from_crontab(health_cron, timezone=CONFIG.timezone)
        )
        self.scheduler.reschedule_job(
            "main-run-job", trigger=CronTrigger.from_crontab(main_cron, timezone=CONFIG.timezone)
        )

    def stop(self) -> None:
        if not self.started:
            return
        self.scheduler.shutdown(wait=False)
        self.started = False

    @staticmethod
    def _run_health_job() -> None:
        with get_session() as session:
            Orchestrator(session).trigger(run_type="health", trigger_source="scheduler")

    @staticmethod
    def _run_main_job() -> None:
        with get_session() as session:
            Orchestrator(session).trigger(run_type="main", trigger_source="scheduler")

    @staticmethod
    def _sync_pricing_job() -> None:
        sync_pricing_catalog()
