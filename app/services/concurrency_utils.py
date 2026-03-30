from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import BoundedSemaphore
from urllib.parse import urlparse
from typing import Any, Callable, Iterable, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def normalized_host(url: str) -> str:
    host = urlparse(str(url or "")).netloc.lower().split("@")[-1]
    if ":" in host:
        host = host.split(":", 1)[0]
    return host or "__default__"


def iter_host_limited_results(
    tasks: Iterable[T],
    *,
    worker_fn: Callable[[T], R],
    host_getter: Callable[[T], str],
    max_workers: int,
    per_host_limit: int,
) -> Iterable[tuple[T, R | None, Exception | None]]:
    task_list = list(tasks)
    if not task_list:
        return []

    worker_count = max(1, int(max_workers))
    host_limit = max(1, int(per_host_limit))
    semaphores = {
        host: BoundedSemaphore(host_limit)
        for host in {host_getter(task) or "__default__" for task in task_list}
    }

    def guarded_worker(task: T) -> R:
        host = host_getter(task) or "__default__"
        semaphore = semaphores[host]
        with semaphore:
            return worker_fn(task)

    def generator() -> Iterable[tuple[T, R | None, Exception | None]]:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {executor.submit(guarded_worker, task): task for task in task_list}
            for future in as_completed(future_map):
                task = future_map[future]
                try:
                    yield task, future.result(), None
                except Exception as exc:  # pragma: no cover - exercised via callers
                    yield task, None, exc

    return generator()
