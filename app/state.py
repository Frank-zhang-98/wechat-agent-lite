from __future__ import annotations

from threading import Event, Lock
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.scheduler_service import SchedulerService


scheduler: SchedulerService | None = None
source_health_snapshot: dict = {}
run_cancel_events: dict[str, Event] = {}
run_cancel_lock = Lock()


def register_run_cancel(run_id: str) -> None:
    if not run_id:
        return
    with run_cancel_lock:
        run_cancel_events.setdefault(run_id, Event())


def request_run_cancel(run_id: str) -> bool:
    if not run_id:
        return False
    with run_cancel_lock:
        event = run_cancel_events.setdefault(run_id, Event())
        event.set()
        return True


def is_run_cancel_requested(run_id: str) -> bool:
    if not run_id:
        return False
    with run_cancel_lock:
        event = run_cancel_events.get(run_id)
        return bool(event and event.is_set())


def clear_run_cancel(run_id: str) -> None:
    if not run_id:
        return
    with run_cancel_lock:
        run_cancel_events.pop(run_id, None)
