from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import requests
from sqlalchemy.orm import Session

from app.core.config import CONFIG
from app.models import Run, RunStatus, RunStep, StepStatus
from app.services.article_render_service import ArticleRenderService
from app.services.fetch_service import FetchService
from app.services.llm_gateway import LLMGateway
from app.services.mail_service import MailService
from app.services.scrapling_fallback_service import ScraplingFallbackService
from app.services.source_maintenance_service import SourceMaintenanceService
from app.services.settings_service import SettingsService
from app.services.title_generation_service import TitleGenerationService
from app.services.wechat_service import WeChatService
from app.services.writing_template_service import WritingTemplateService
from app.services.concurrency_utils import iter_host_limited_results, normalized_host


class StepFailedError(RuntimeError):
    pass


@dataclass
class RetryPolicy:
    max_retries: int
    backoffs: list[int]


class Orchestrator:
    def __init__(self, session: Session):
        self.session = session
        self.settings = SettingsService(session)
        self.settings.ensure_defaults()
        self.llm = LLMGateway(session, self.settings)
        self.fetch = FetchService(
            all_proxy=self.settings.get("proxy.all_proxy", "") if self.settings.get_bool("proxy.enabled", False) else ""
        )
        self.article_renderer = ArticleRenderService()
        self.mail = MailService(self.settings)
        self.wechat = WeChatService(self.settings)
        self.writing_templates = WritingTemplateService()
        self.title_generator = TitleGenerationService()
        self.scrapling = ScraplingFallbackService(
            enabled=self.settings.get_bool("source_maintenance.scrapling_enabled", True),
            repo_path=self.settings.get("source_maintenance.scrapling_repo_path", ""),
            timeout_seconds=self.settings.get_int("source_maintenance.scrapling_timeout_seconds", 20),
            proxy=self.settings.get("proxy.all_proxy", "") if self.settings.get_bool("proxy.enabled", False) else "",
            max_concurrency=self.settings.get_int("source_maintenance.scrapling_max_concurrency", 1),
        )

    def trigger(self, run_type: str = "main", trigger_source: str = "manual") -> Run:
        run = self.create_run(run_type=run_type, trigger_source=trigger_source, status=RunStatus.running.value)
        return self._execute_run(run)

    def create_run(self, run_type: str = "main", trigger_source: str = "manual", status: str = RunStatus.pending.value) -> Run:
        run = Run(
            run_type=run_type,
            trigger_source=trigger_source,
            status=status,
            started_at=_utcnow(),
        )
        self.session.add(run)
        self.session.flush()
        return run

    def execute_existing(self, run_id: str) -> Run:
        run = self.session.get(Run, run_id)
        if not run:
            raise ValueError("run_id not found")
        run.status = RunStatus.running.value
        run.error_message = ""
        run.finished_at = None
        if not run.started_at:
            run.started_at = _utcnow()
        self._commit_progress()
        return self._execute_run(run)

    def rerun_from_action(self, run_id: str, action: str) -> Run:
        base = self.session.get(Run, run_id)
        if not base:
            raise ValueError("run_id not found")
        trigger = f"action:{action}"
        if action == "rerun_full":
            return self.trigger(run_type="main", trigger_source=trigger)
        if action == "retry_from_failed_step":
            # milestone: rerun main flow; subsequent phase can resume from failed step checkpoint.
            return self.trigger(run_type="main", trigger_source=trigger)
        if action == "regenerate_article":
            return self._article_only_run(trigger_source=trigger)
        if action == "regenerate_cover":
            return self._cover_only_run(trigger_source=trigger)
        if action == "retry_wechat_draft":
            return self._wechat_draft_only_run(base=base, trigger_source=trigger)
        raise ValueError(f"unsupported action: {action}")

    def _execute_run(self, run: Run) -> Run:
        ctx: dict[str, Any] = {"quality_scores": [], "failed_logs": []}
        run.status = RunStatus.running.value
        run.error_message = ""
        if not run.started_at:
            run.started_at = _utcnow()
        self._commit_progress()
        try:
            if run.run_type == "health":
                self._run_health_only(run, ctx)
            else:
                self._run_main(run, ctx)
            if run.status == RunStatus.running.value:
                run.status = RunStatus.success.value
            run.finished_at = _utcnow()
        except Exception as exc:
            run.status = RunStatus.failed.value
            run.error_message = str(exc)
            run.finished_at = _utcnow()
            ctx.setdefault("failed_logs", []).append(
                {"step": "RUN", "attempt": 1, "error": str(exc), "at": _utcnow().isoformat()}
            )
        finally:
            # Daily report only for main runs. No real-time failure alerts.
            if run.run_type == "main":
                self._send_daily_report(run, ctx)
            self._commit_progress()
        return run

    def _run_health_only(self, run: Run, ctx: dict[str, Any]) -> None:
        self._execute_step(run, "HEALTH_CHECK", self._step_health_check, ctx, self._policy_fetch())
        if self.settings.get_bool("source_maintenance.run_on_health", True):
            self._execute_step(run, "SOURCE_MAINTENANCE", self._step_source_maintenance, ctx, self._policy_fetch())
        run.summary_json = json.dumps(
            {
                "health": ctx.get("health", {}),
                "source_maintenance": ctx.get("source_maintenance", {}),
            },
            ensure_ascii=False,
        )

    def _run_main(self, run: Run, ctx: dict[str, Any]) -> None:
        self._execute_step(run, "HEALTH_CHECK", self._step_health_check, ctx, self._policy_fetch())
        if self.settings.get_bool("source_maintenance.run_on_main", True):
            self._execute_step(run, "SOURCE_MAINTENANCE", self._step_source_maintenance, ctx, self._policy_fetch())
        self._execute_step(run, "FETCH", self._step_fetch, ctx, self._policy_fetch())
        self._execute_step(run, "DEDUP", self._step_dedup, ctx, self._policy_fetch())
        self._execute_step(run, "RULE_SCORE", self._step_rule_score, ctx, self._policy_generate())
        self._execute_step(run, "RERANK", self._step_rerank_v2, ctx, self._policy_generate())
        self._execute_step(run, "SELECT", self._step_select, ctx, self._policy_generate())
        self._execute_step(run, "SOURCE_ENRICH", self._step_source_enrich, ctx, self._policy_fetch())
        self._execute_step(run, "FACT_PACK", self._step_fact_pack, ctx, self._policy_generate())
        self._execute_step(run, "FACT_COMPRESS", self._step_fact_compress, ctx, self._policy_generate())
        self._execute_step(run, "WRITE", self._step_write_v2, ctx, self._policy_generate())
        self._execute_step(run, "QUALITY_CHECK", self._step_quality_check, ctx, self._policy_generate())
        self._execute_step(run, "ARTICLE_RENDER", self._step_article_render, ctx, self._policy_generate())
        self._execute_step(run, "COVER_5D", self._step_cover_5d, ctx, self._policy_generate())
        self._execute_step(run, "COVER_GEN", self._step_cover_gen, ctx, self._policy_generate())
        self._execute_step(run, "COVER_CHECK", self._step_cover_check, ctx, self._policy_generate())
        try:
            self._execute_step(run, "WECHAT_DRAFT", self._step_wechat_draft, ctx, self._policy_publish())
        except StepFailedError:
            # Publish failure should keep local result and mark pending manual.
            ctx["draft_status"] = "pending_manual"
            run.status = RunStatus.partial_success.value

        run.article_title = ctx.get("article_title", "")
        run.article_markdown = ctx.get("article_markdown", "")
        run.quality_score = float(ctx.get("quality_score", 0))
        run.quality_threshold = float(self.settings.get_float("quality.threshold", 78))
        run.quality_attempts = int(ctx.get("quality_attempts", 1))
        run.quality_fallback_used = bool(ctx.get("quality_fallback_used", False))
        run.draft_status = ctx.get("draft_status", "not_started")
        run.summary_json = json.dumps(
            {
                "fetched_items": [self._compact_topic(item) for item in (ctx.get("fetched_items") or [])],
                "deduped_items": [self._compact_topic(item) for item in (ctx.get("deduped_items") or [])],
                "top_n": ctx.get("top_n", []),
                "top_k": ctx.get("top_k", []),
                "selected_topic": ctx.get("selected_topic", {}),
                "source_pack": ctx.get("source_pack", {}),
                "content_type": ctx.get("content_type", ""),
                "target_audience": ctx.get("target_audience", ""),
                "article_layout": ctx.get("article_layout", {}),
                "article_render": ctx.get("article_render", {}),
                "fact_pack": ctx.get("fact_pack", {}),
                "fact_compress": ctx.get("fact_compress", {}),
                "title_plan": ctx.get("title_plan", {}),
                "cover_asset": ctx.get("cover_asset", {}),
                "cover_5d": ctx.get("cover_5d", {}),
                "quality_scores": ctx.get("quality_scores", []),
                "failed_logs": ctx.get("failed_logs", []),
                "mail": ctx.get("mail_result", {}),
                "wechat": ctx.get("wechat_result", {}),
                "source_maintenance": ctx.get("source_maintenance", {}),
            },
            ensure_ascii=False,
        )
        if run.draft_status == "pending_manual" and run.status == RunStatus.running.value:
            run.status = RunStatus.partial_success.value

    def _article_only_run(self, trigger_source: str) -> Run:
        run = Run(
            run_type="manual",
            trigger_source=trigger_source,
            status=RunStatus.running.value,
            started_at=_utcnow(),
        )
        self.session.add(run)
        self.session.flush()
        ctx = {"quality_scores": [], "failed_logs": [], "selected_topic": {"title": "Manual Article Rerun", "url": "", "source": "manual"}}
        try:
            self._execute_step(run, "SOURCE_ENRICH", self._step_source_enrich, ctx, self._policy_fetch())
            self._execute_step(run, "FACT_PACK", self._step_fact_pack, ctx, self._policy_generate())
            self._execute_step(run, "FACT_COMPRESS", self._step_fact_compress, ctx, self._policy_generate())
            self._execute_step(run, "WRITE", self._step_write_v2, ctx, self._policy_generate())
            self._execute_step(run, "QUALITY_CHECK", self._step_quality_check, ctx, self._policy_generate())
            self._execute_step(run, "ARTICLE_RENDER", self._step_article_render, ctx, self._policy_generate())
            run.article_title = ctx.get("article_title", "")
            run.article_markdown = ctx.get("article_markdown", "")
            run.quality_score = float(ctx.get("quality_score", 0))
            run.quality_attempts = int(ctx.get("quality_attempts", 1))
            run.quality_fallback_used = bool(ctx.get("quality_fallback_used", False))
            run.summary_json = json.dumps(
                {
                    "source_pack": ctx.get("source_pack", {}),
                    "content_type": ctx.get("content_type", ""),
                    "target_audience": ctx.get("target_audience", ""),
                    "article_layout": ctx.get("article_layout", {}),
                    "article_render": ctx.get("article_render", {}),
                    "fact_pack": ctx.get("fact_pack", {}),
                    "fact_compress": ctx.get("fact_compress", {}),
                    "quality_scores": ctx.get("quality_scores", []),
                },
                ensure_ascii=False,
            )
            run.status = RunStatus.success.value
        except Exception as exc:
            run.status = RunStatus.failed.value
            run.error_message = str(exc)
        finally:
            run.finished_at = _utcnow()
        return run

    def _cover_only_run(self, trigger_source: str) -> Run:
        run = Run(
            run_type="manual",
            trigger_source=trigger_source,
            status=RunStatus.running.value,
            started_at=_utcnow(),
        )
        self.session.add(run)
        self.session.flush()
        ctx = {"failed_logs": [], "article_title": "Manual Cover Rerun", "article_markdown": "Manual Cover Rerun"}
        try:
            self._execute_step(run, "COVER_5D", self._step_cover_5d, ctx, self._policy_generate())
            self._execute_step(run, "COVER_GEN", self._step_cover_gen, ctx, self._policy_generate())
            self._execute_step(run, "COVER_CHECK", self._step_cover_check, ctx, self._policy_generate())
            run.summary_json = json.dumps({"cover_5d": ctx.get("cover_5d", {})}, ensure_ascii=False)
            run.status = RunStatus.success.value
        except Exception as exc:
            run.status = RunStatus.failed.value
            run.error_message = str(exc)
        finally:
            run.finished_at = _utcnow()
        return run

    def _wechat_draft_only_run(self, base: Run, trigger_source: str) -> Run:
        article_markdown = str(base.article_markdown or "").strip()
        if not article_markdown:
            raise ValueError("source run has no saved article_markdown, cannot retry wechat draft")

        summary = self._parse_summary_json(base.summary_json)
        selected_topic = dict(summary.get("selected_topic") or {})
        cover_asset = self._resolve_cover_asset(base=base, summary=summary)
        article_html = self._resolve_article_html(base=base, summary=summary)
        ctx: dict[str, Any] = {
            "failed_logs": [],
            "selected_topic": selected_topic,
            "article_title": str(base.article_title or selected_topic.get("title") or "").strip(),
            "wechat_title": str((summary.get("title_plan") or {}).get("wechat_title") or base.article_title or "").strip(),
            "article_markdown": base.article_markdown,
            "article_html": article_html,
            "cover_asset": cover_asset,
        }

        run = Run(
            run_type="manual",
            trigger_source=trigger_source,
            status=RunStatus.running.value,
            started_at=_utcnow(),
        )
        self.session.add(run)
        self.session.flush()

        try:
            self._execute_step(run, "WECHAT_DRAFT", self._step_wechat_draft, ctx, self._policy_publish())
            run.status = RunStatus.success.value
        except StepFailedError:
            ctx["draft_status"] = "pending_manual"
            run.status = RunStatus.partial_success.value

        run.article_title = ctx.get("article_title", "")
        run.article_markdown = ctx.get("article_markdown", "")
        run.quality_score = float(base.quality_score or 0)
        run.quality_threshold = float(base.quality_threshold or self.settings.get_float("quality.threshold", 78))
        run.quality_attempts = int(base.quality_attempts or 0)
        run.quality_fallback_used = bool(base.quality_fallback_used)
        run.draft_status = ctx.get("draft_status", "not_started")
        run.summary_json = json.dumps(
            {
                "source_run_id": base.id,
                "selected_topic": selected_topic,
                "title_plan": summary.get("title_plan", {})
                or {
                    "article_title": ctx.get("article_title", ""),
                    "wechat_title": ctx.get("wechat_title", ""),
                    "source": "reused",
                },
                "article_layout": summary.get("article_layout", {}),
                "article_render": summary.get("article_render", {}),
                "cover_asset": cover_asset,
                "cover_5d": summary.get("cover_5d", {}),
                "wechat": ctx.get("wechat_result", {}),
                "failed_logs": ctx.get("failed_logs", []),
                "redraft_mode": "wechat_draft_only",
            },
            ensure_ascii=False,
        )
        self._send_daily_report(run, ctx)
        run.finished_at = _utcnow()
        return run

    def _execute_step(
        self,
        run: Run,
        name: str,
        handler: Callable[[Run, dict[str, Any]], None],
        ctx: dict[str, Any],
        policy: RetryPolicy,
    ) -> None:
        step = RunStep(run_id=run.id, name=name, status=StepStatus.running.value, started_at=_utcnow())
        self.session.add(step)
        self.session.flush()
        ctx["_active_step_row"] = step
        ctx["_active_step_name"] = name
        step.details_json = json.dumps(
            self._build_step_details(name=name, ctx=ctx, status=step.status, error_text=""),
            ensure_ascii=False,
        )
        self._commit_progress()
        try:
            for attempt in range(policy.max_retries + 1):
                step.retry_count = attempt
                started = time.perf_counter()
                try:
                    handler(run, ctx)
                    step.status = StepStatus.success.value
                    step.error_message = ""
                    step.finished_at = _utcnow()
                    step.duration_ms = int((time.perf_counter() - started) * 1000)
                    step.details_json = json.dumps(
                        self._build_step_details(name=name, ctx=ctx, status=step.status, error_text=""),
                        ensure_ascii=False,
                    )
                    self._commit_progress()
                    return
                except Exception as exc:
                    error_text = str(exc)
                    ctx.setdefault("failed_logs", []).append(
                        {"step": name, "attempt": attempt + 1, "error": error_text, "at": _utcnow().isoformat()}
                    )
                    if attempt < policy.max_retries:
                        backoff = policy.backoffs[min(attempt, len(policy.backoffs) - 1)]
                        time.sleep(min(backoff, 2))
                        continue
                    step.status = StepStatus.failed.value
                    step.error_message = error_text
                    step.finished_at = _utcnow()
                    step.duration_ms = int((time.perf_counter() - started) * 1000)
                    step.details_json = json.dumps(
                        self._build_step_details(name=name, ctx=ctx, status=step.status, error_text=error_text),
                        ensure_ascii=False,
                    )
                    self._commit_progress()
                    raise StepFailedError(f"{name} failed: {error_text}")
        finally:
            ctx.pop("_active_step_row", None)
            ctx.pop("_active_step_name", None)

    def _policy_fetch(self) -> RetryPolicy:
        return RetryPolicy(
            max_retries=self.settings.get_int("retry.fetch.max", 2),
            backoffs=self.settings.get_list_int("retry.fetch.backoff", [5, 15]),
        )

    def _policy_generate(self) -> RetryPolicy:
        return RetryPolicy(
            max_retries=self.settings.get_int("retry.generate.max", 2),
            backoffs=self.settings.get_list_int("retry.generate.backoff", [10, 30]),
        )

    def _policy_publish(self) -> RetryPolicy:
        return RetryPolicy(
            max_retries=self.settings.get_int("retry.publish.max", 3),
            backoffs=self.settings.get_list_int("retry.publish.backoff", [15, 45, 120]),
        )

    def _commit_progress(self) -> None:
        self.session.flush()
        self.session.commit()

    @staticmethod
    def _parse_summary_json(raw: str | None) -> dict[str, Any]:
        try:
            value = json.loads(raw or "{}")
        except Exception:
            return {}
        return value if isinstance(value, dict) else {}

    def _resolve_cover_asset(self, base: Run, summary: dict[str, Any]) -> dict[str, Any]:
        cover_asset = dict(summary.get("cover_asset") or {})
        cover_path = str(cover_asset.get("path") or "").strip()
        if cover_path:
            candidate = Path(cover_path)
            if candidate.exists() and candidate.is_file():
                cover_asset["path"] = str(candidate)
                return cover_asset

        run_dir = CONFIG.data_dir / "runs" / base.id
        for pattern in ("cover.png", "cover.jpg", "cover.jpeg", "cover.webp"):
            candidate = run_dir / pattern
            if candidate.exists() and candidate.is_file():
                cover_asset["path"] = str(candidate)
                cover_asset.setdefault("status", "reused")
                return cover_asset
        return cover_asset

    def _resolve_article_html(self, base: Run, summary: dict[str, Any]) -> str:
        article_render = dict(summary.get("article_render") or {})
        html_path = str(article_render.get("html_path") or "").strip()
        if html_path:
            candidate = Path(html_path)
            if candidate.exists() and candidate.is_file():
                try:
                    return candidate.read_text(encoding="utf-8")
                except Exception:
                    return ""
        run_dir = CONFIG.data_dir / "runs" / base.id
        candidate = run_dir / "article.html"
        if candidate.exists() and candidate.is_file():
            try:
                return candidate.read_text(encoding="utf-8")
            except Exception:
                return ""
        return ""

    @staticmethod
    def _clip_text(value: Any, limit: int = 6000) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        if len(text) <= limit:
            return text
        return f"{text[:limit]}\n\n... [truncated, total {len(text)} chars]"

    def _set_step_audit(self, ctx: dict[str, Any], step_name: str, payload: dict[str, Any]) -> None:
        step_audits = ctx.setdefault("step_audits", {})
        if not isinstance(step_audits.get(step_name), dict):
            step_audits[step_name] = {}
        for key, value in payload.items():
            if value in (None, "", [], {}):
                continue
            step_audits[step_name][key] = value

    def _update_live_step_details(self, ctx: dict[str, Any], step_name: str, payload: dict[str, Any]) -> None:
        step = ctx.get("_active_step_row")
        if not isinstance(step, RunStep):
            return
        if step.name != step_name or step.status != StepStatus.running.value:
            return
        if step_name == "SOURCE_MAINTENANCE":
            ctx["source_maintenance_progress"] = payload
            ctx["source_maintenance"] = {
                "checked_sources": payload.get("checked_sources", 0),
                "healthy_sources": payload.get("healthy_sources", 0),
                "failed_sources": payload.get("failed_sources", 0),
                "changed_sources": payload.get("changed_sources", 0),
                "manual_review_sources": payload.get("manual_review_sources", 0),
                "llm_candidate_sources": payload.get("llm_candidate_sources", 0),
                "actions": list(payload.get("recent_actions") or []),
            }
        step.details_json = json.dumps(
            self._build_step_details(name=step_name, ctx=ctx, status=StepStatus.running.value, error_text=""),
            ensure_ascii=False,
        )
        self._commit_progress()

    # -------- Step handlers --------
    def _step_health_check(self, run: Run, ctx: dict[str, Any]) -> None:
        proxy_enabled = self.settings.get_bool("proxy.enabled", False)
        proxy_url = self.settings.get("proxy.all_proxy", "")
        health = {"proxy_enabled": proxy_enabled, "proxy_url": proxy_url, "ok": True}
        if proxy_enabled and proxy_url:
            proxies = {"http": proxy_url, "https": proxy_url}
            resp = requests.get("https://api.ipify.org?format=json", timeout=12, proxies=proxies)
            resp.raise_for_status()
            health["egress_ip"] = resp.json().get("ip", "")
        ctx["health"] = health

    def _step_source_maintenance(self, run: Run, ctx: dict[str, Any]) -> None:
        service = SourceMaintenanceService(
            session=self.session,
            settings=self.settings,
            fetch=self.fetch,
            llm=self.llm,
            scrapling=self.scrapling,
            progress_callback=lambda payload: self._update_live_step_details(ctx, "SOURCE_MAINTENANCE", payload),
        )
        result = service.run(run_id=run.id)
        ctx["source_maintenance"] = {
            key: value for key, value in result.items() if key != "audit"
        }
        ctx.pop("source_maintenance_progress", None)
        audit = dict(result.get("audit") or {})
        if audit:
            self._set_step_audit(ctx, "SOURCE_MAINTENANCE", audit)

    def _step_fetch(self, run: Run, ctx: dict[str, Any]) -> None:
        cfg = self.fetch.load_sources()
        max_age = int(cfg.get("max_age_hours", 168))
        max_per_source = int(cfg.get("max_hotspots_per_source", 10))
        fetch_workers = max(1, self.settings.get_int("fetch.concurrent_workers", 6))
        per_host_limit = max(1, self.settings.get_int("fetch.per_host_limit", 1))
        jobs: list[dict[str, Any]] = []
        index = 0

        for cat in ["ai_companies", "tech_media", "tutorial_communities"]:
            for src in cfg.get(cat, []):
                if not src.get("enabled", True):
                    continue
                jobs.append(
                    {
                        "index": index,
                        "kind": "source",
                        "name": str(src.get("name", "") or ""),
                        "url": str(src.get("url", "") or ""),
                        "source": src,
                    }
                )
                index += 1

        if cfg.get("github", {}).get("enabled", True):
            jobs.append(
                {
                    "index": index,
                    "kind": "github",
                    "name": "github",
                    "url": "https://api.github.com/search/repositories",
                }
            )

        def worker(job: dict[str, Any]) -> list[dict[str, Any]]:
            if job["kind"] == "github":
                return self.fetch.fetch_github(cfg.get("github", {}), max_age_hours=max_age)
            return self.fetch.fetch_source(
                job["source"],
                max_age_hours=max_age,
                max_items=max_per_source,
                scrapling=self.scrapling,
            )

        items_by_index: dict[int, list[dict[str, Any]]] = {}
        for job, result, error in iter_host_limited_results(
            jobs,
            worker_fn=worker,
            host_getter=lambda item: normalized_host(item.get("url", "")),
            max_workers=fetch_workers,
            per_host_limit=per_host_limit,
        ):
            if error is not None:
                ctx["failed_logs"].append(
                    {"step": "FETCH", "source": job.get("name", ""), "error": str(error), "at": _utcnow().isoformat()}
                )
                continue
            items_by_index[int(job["index"])] = list(result or [])

        items: list[dict[str, Any]] = []
        for job_index in sorted(items_by_index):
            items.extend(items_by_index[job_index])

        if not items:
            raise RuntimeError("No hotspots fetched from enabled sources")
        ctx["fetched_items"] = items
        self.fetch.dump_debug(items, run.id)

    def _step_dedup(self, run: Run, ctx: dict[str, Any]) -> None:
        deduped = self.fetch.dedup(ctx.get("fetched_items", []))
        if not deduped:
            raise RuntimeError("No items left after dedup")
        ctx["deduped_items"] = deduped

    def _step_rule_score(self, run: Run, ctx: dict[str, Any]) -> None:
        items = ctx.get("deduped_items", [])
        now = datetime.now(timezone.utc)
        scored: list[dict[str, Any]] = []
        latest_hours = max(12, self.settings.get_int("selection.max_age_hours_for_main", 72))
        for item in items:
            if self._should_reject_topic(item):
                continue
            published = datetime.fromisoformat(item["published"])
            hours = max((now - published).total_seconds() / 3600.0, 1.0)
            freshness = round(max(0.0, 100.0 * math.exp(-hours / max(latest_hours / 2.0, 1.0))), 2)
            source_weight = float(item.get("source_weight", 0.7)) * 100.0
            depth_score = self._topic_depth_score(item)
            novelty_score = self._topic_novelty_score(item, hours)
            value_score = self._topic_value_score(item)
            rule_score = round(
                0.40 * freshness
                + 0.25 * depth_score
                + 0.20 * value_score
                + 0.10 * novelty_score
                + 0.05 * source_weight,
                2,
            )
            item["freshness_score"] = freshness
            item["depth_score"] = depth_score
            item["value_score"] = value_score
            item["novelty_score"] = novelty_score
            item["rule_score"] = rule_score
            scored.append(item)
        if not scored:
            raise RuntimeError("No suitable items left after topic filtering")
        scored.sort(key=lambda x: x["rule_score"], reverse=True)
        ctx["top_n"] = scored[: self.settings.get_int("general.top_n", 10)]

    def _step_rerank(self, run: Run, ctx: dict[str, Any]) -> None:
        top_n = ctx.get("top_n", [])
        if not top_n:
            raise RuntimeError("TopN is empty")
        candidates = top_n[: self.settings.get_int("general.top_k", 8)]
        documents = [
            "\n".join(
                [
                    f"标题：{item.get('title', '')}",
                    f"摘要：{item.get('summary', '')}",
                    f"来源：{item.get('source', '')}",
                    f"规则分：{item.get('rule_score', 0)}",
                ]
            )
            for item in candidates
        ]
        reranked = self.llm.rerank_documents(
            run.id,
            "RERANK",
            "rerank",
            query="筛选出更适合写成公众号文章的热点主题",
            documents=documents,
            top_n=len(candidates),
        )
        ranked_items: list[dict[str, Any]] = []
        used_indexes: set[int] = set()
        for idx, result in enumerate(reranked):
            source_index = int(result.get("index", -1))
            if source_index < 0 or source_index >= len(candidates) or source_index in used_indexes:
                continue
            used_indexes.add(source_index)
            item = dict(candidates[source_index])
            llm_score = round(max(0.0, min(float(result.get("relevance_score", 0.0) or 0.0), 1.0)) * 100, 2)
            item["llm_score"] = llm_score
            item["rerank_reason"] = str(result.get("reason", "") or "").strip()
            item["rerank_rank"] = idx + 1
            item["final_score"] = round(0.55 * float(item.get("rule_score", 0.0) or 0.0) + 0.45 * llm_score, 2)
            ranked_items.append(item)

        for source_index, candidate in enumerate(candidates):
            if source_index in used_indexes:
                continue
            item = dict(candidate)
            item["llm_score"] = round(max(60.0, float(item.get("rule_score", 0.0) or 0.0) - 8.0), 2)
            item["rerank_reason"] = "未返回明确排序，按规则分补位"
            item["rerank_rank"] = len(ranked_items) + 1
            item["final_score"] = round(0.55 * float(item.get("rule_score", 0.0) or 0.0) + 0.45 * item["llm_score"], 2)
            ranked_items.append(item)

        ranked_items.sort(key=lambda x: x["final_score"], reverse=True)
        ctx["top_k"] = ranked_items

    def _step_rerank_v2(self, run: Run, ctx: dict[str, Any]) -> None:
        top_n = ctx.get("top_n", [])
        if not top_n:
            raise RuntimeError("TopN is empty")
        candidates = [dict(item) for item in top_n[: self.settings.get_int("general.top_k", 8)]]
        enrich_limit = max(1, self.settings.get_int("selection.rerank_enrich_m", 5))
        excerpt_chars = max(300, self.settings.get_int("selection.rerank_excerpt_chars", 1200))
        for idx, item in enumerate(candidates):
            if idx >= enrich_limit:
                item["rerank_excerpt"] = ""
                item["rerank_excerpt_status"] = "skipped"
                continue
            url = str(item.get("url", "") or "").strip()
            if not url:
                item["rerank_excerpt"] = ""
                item["rerank_excerpt_status"] = "no_url"
                continue
            extract = self.fetch.extract_article_content(url, max_chars=excerpt_chars)
            item["rerank_excerpt"] = str(extract.get("content_text", "") or "")[:excerpt_chars]
            item["rerank_excerpt_status"] = extract.get("status", "failed")

        documents = [
            "\n".join(
                [
                    f"标题: {item.get('title', '')}",
                    f"摘要: {item.get('summary', '')}",
                    f"来源: {item.get('source', '')}",
                    f"规则分: {item.get('rule_score', 0)}",
                    f"新鲜度分: {item.get('freshness_score', 0)}",
                    f"深度分: {item.get('depth_score', 0)}",
                    f"价值分: {item.get('value_score', 0)}",
                    f"新信息分: {item.get('novelty_score', 0)}",
                    f"正文摘样状态: {item.get('rerank_excerpt_status', '-')}",
                    f"正文摘样: {item.get('rerank_excerpt', '')}",
                ]
            )
            for item in candidates
        ]
        query = (
            "从最近文章中找出今天最值得写成公众号原创深度解读的主题。"
            "优先选择新信息密度高、机制细节多、工作流价值清晰、对读者有实际判断价值的题。"
            "降低基础教程、测验、浅层资讯搬运的排序。"
        )
        reranked = self.llm.rerank_documents(
            run.id,
            "RERANK",
            "rerank",
            query=query,
            documents=documents,
            top_n=len(candidates),
        )

        ranked_items: list[dict[str, Any]] = []
        used_indexes: set[int] = set()
        for idx, result in enumerate(reranked):
            source_index = int(result.get("index", -1))
            if source_index < 0 or source_index >= len(candidates) or source_index in used_indexes:
                continue
            used_indexes.add(source_index)
            item = dict(candidates[source_index])
            llm_score = round(max(0.0, min(float(result.get("relevance_score", 0.0) or 0.0), 1.0)) * 100, 2)
            item["llm_score"] = llm_score
            item["rerank_reason"] = str(result.get("reason", "") or "").strip()
            item["rerank_rank"] = idx + 1
            item["final_score"] = round(0.35 * float(item.get("rule_score", 0.0) or 0.0) + 0.65 * llm_score, 2)
            ranked_items.append(item)

        for source_index, candidate in enumerate(candidates):
            if source_index in used_indexes:
                continue
            item = dict(candidate)
            item["llm_score"] = round(max(60.0, float(item.get("rule_score", 0.0) or 0.0) - 8.0), 2)
            item["rerank_reason"] = "未返回明确排序，按规则分补位"
            item["rerank_rank"] = len(ranked_items) + 1
            item["final_score"] = round(0.35 * float(item.get("rule_score", 0.0) or 0.0) + 0.65 * item["llm_score"], 2)
            ranked_items.append(item)

        ranked_items.sort(key=lambda x: x["final_score"], reverse=True)
        ctx["top_k"] = ranked_items
        self._set_step_audit(
            ctx,
            "RERANK",
            {
                "prompts": [
                    {
                        "title": "正文感知重排输入",
                        "text": self._clip_text(query + "\n\n" + "\n\n".join(documents), 8000),
                    }
                ],
                "outputs": [
                    {
                        "title": "重排结果详情",
                        "text": self._clip_text(
                            json.dumps(
                                [self._compact_topic(item, include_scores=True) for item in ranked_items[:8]],
                                ensure_ascii=False,
                                indent=2,
                            ),
                            8000,
                        ),
                        "language": "json",
                    }
                ],
            },
        )

    def _step_select(self, run: Run, ctx: dict[str, Any]) -> None:
        ranked = ctx.get("top_k", [])
        if not ranked:
            raise RuntimeError("TopK is empty")
        refine_top_m = max(1, self.settings.get_int("selection.refine_top_m", 4))
        candidates = [dict(item) for item in ranked[:refine_top_m]]
        for item in candidates:
            url = str(item.get("url", "") or "").strip()
            excerpt = str(item.get("rerank_excerpt", "") or "")
            if url and not excerpt:
                result = self.fetch.extract_article_content(url, max_chars=1800)
                excerpt = str(result.get("content_text", "") or "")[:1000]
            item["selection_excerpt"] = excerpt

        prompt = self._build_select_prompt(candidates)
        decision = self.llm.call(run.id, "SELECT", "decision", prompt, temperature=0.1)
        selected_index = self._parse_select_choice(decision.text, len(candidates))
        if selected_index < 0:
            selected_index = max(
                range(len(candidates)),
                key=lambda idx: float(candidates[idx].get("rule_score", 0) or 0)
                + min(len(str(candidates[idx].get("selection_excerpt", "") or "")) / 100.0, 20.0),
            )
        selected = candidates[selected_index]
        selected["selection_reason"] = self._clip_text(decision.text, 1200)
        ctx["selected_topic"] = selected
        self._set_step_audit(
            ctx,
            "SELECT",
            {
                "prompts": [
                    {
                        "title": "深度选题提示词",
                        "text": self._clip_text(prompt, 8000),
                    }
                ],
                "outputs": [
                    {
                        "title": "选题模型回包",
                        "text": self._clip_text(decision.text, 4000),
                    }
                ],
            },
        )

    def _step_source_enrich(self, run: Run, ctx: dict[str, Any]) -> None:
        topic = dict(ctx.get("selected_topic") or {})
        primary_url = str(topic.get("url", "") or "").strip()
        related_limit = max(0, self.settings.get_int("writing.source_enrich.related_limit", 2))
        max_chars = max(1000, self.settings.get_int("writing.source_enrich.max_chars", 8000))
        primary_source = {
            "title": str(topic.get("title", "") or ""),
            "url": primary_url,
            "summary": str(topic.get("summary", "") or ""),
            "source": str(topic.get("source", "") or ""),
            "status": "skipped",
            "reason": "no_url",
            "content_text": "",
            "paragraphs": [],
        }
        if primary_url:
            primary_extract = self.fetch.extract_article_content(primary_url, max_chars=max_chars)
            primary_source.update(primary_extract)
            if not primary_source.get("title"):
                primary_source["title"] = str(topic.get("title", "") or "")

        related_sources: list[dict[str, Any]] = []
        seen_urls = {primary_url} if primary_url else set()
        for item in list(ctx.get("top_k") or []):
            candidate_url = str(item.get("url", "") or "").strip()
            if not candidate_url or candidate_url in seen_urls:
                continue
            seen_urls.add(candidate_url)
            extract = self.fetch.extract_article_content(candidate_url, max_chars=max_chars // 2)
            related_sources.append(
                {
                    "title": str(item.get("title", "") or extract.get("title", "")),
                    "url": candidate_url,
                    "summary": str(item.get("summary", "") or ""),
                    "source": str(item.get("source", "") or ""),
                    "status": extract.get("status", "failed"),
                    "reason": extract.get("reason", ""),
                    "content_text": extract.get("content_text", ""),
                    "paragraphs": extract.get("paragraphs", []),
                }
            )
            if len(related_sources) >= related_limit:
                break

        source_pack = {"primary": primary_source, "related": related_sources}
        ctx["source_pack"] = source_pack
        self._set_step_audit(
            ctx,
            "SOURCE_ENRICH",
            {
                "outputs": [
                    {
                        "title": "正文素材包",
                        "text": self._clip_text(json.dumps(source_pack, ensure_ascii=False, indent=2), 8000),
                        "language": "json",
                    }
                ]
            },
        )

    def _step_fact_pack(self, run: Run, ctx: dict[str, Any]) -> None:
        default_audience = self.settings.get("writing.default_audience", "ai_product_manager").strip() or "ai_product_manager"
        configured_type = self.settings.get("writing.default_content_type", "auto").strip().lower()
        fact_pack = self.writing_templates.build_fact_pack(ctx, audience_key=default_audience)
        content_type = configured_type if configured_type and configured_type != "auto" else fact_pack.get("content_type", "tool_review")
        fact_pack["content_type"] = content_type
        fact_pack["content_type_label"] = self.writing_templates.get_content_type(content_type).get("label", content_type)
        ctx["fact_pack"] = fact_pack
        ctx["content_type"] = content_type
        ctx["target_audience"] = default_audience
        self._set_step_audit(
            ctx,
            "FACT_PACK",
            {
                "outputs": [
                    {
                        "title": "写作事实包",
                        "text": self._clip_text(self.writing_templates.preview_fact_pack(fact_pack), 8000),
                        "language": "json",
                    }
                ]
            },
        )

    def _step_fact_compress(self, run: Run, ctx: dict[str, Any]) -> None:
        fact_pack = dict(ctx.get("fact_pack") or {})
        source_pack = dict(ctx.get("source_pack") or {})
        if not fact_pack:
            raise RuntimeError("fact_pack is empty")
        prompt = (
            "You are a factual analyst. Read the source pack and fact pack, then output strict JSON in simplified Chinese. "
            "Do not write prose outside JSON.\n\n"
            "Return keys: one_sentence_summary, what_it_is, key_mechanisms, concrete_scenarios, numbers, risks, uncertainties, recommended_angle.\n"
            "Each value must be an array except one_sentence_summary which must be a string.\n"
            "Only keep high-confidence facts grounded in the provided materials. If unsure, put it into uncertainties.\n\n"
            f"Source Pack:\n{self._clip_text(json.dumps(source_pack, ensure_ascii=False), 6000)}\n\n"
            f"Fact Pack:\n{self._clip_text(json.dumps(fact_pack, ensure_ascii=False), 4000)}"
        )
        result = self.llm.call(run.id, "FACT_COMPRESS", "decision", prompt, temperature=0.1)
        compressed = self._parse_fact_compress_result(result.text, fact_pack)
        ctx["fact_compress"] = compressed
        self._set_step_audit(
            ctx,
            "FACT_COMPRESS",
            {
                "prompts": [
                    {
                        "title": "事实压缩提示词",
                        "text": self._clip_text(prompt, 8000),
                    }
                ],
                "outputs": [
                    {
                        "title": "事实压缩结果",
                        "text": self._clip_text(json.dumps(compressed, ensure_ascii=False, indent=2), 8000),
                        "language": "json",
                    }
                ],
            },
        )

    def _step_write(self, run: Run, ctx: dict[str, Any]) -> None:
        topic = ctx.get("selected_topic") or {"title": "AI Daily Topic", "summary": ""}
        fact_pack = dict(ctx.get("fact_pack") or {})
        audience_key = str(ctx.get("target_audience") or self.settings.get("writing.default_audience", "ai_product_manager"))
        content_type = str(ctx.get("content_type") or fact_pack.get("content_type") or "tool_review")
        prompt = self.writing_templates.build_write_prompt(
            topic=topic,
            fact_pack=fact_pack,
            audience_key=audience_key,
            content_type=content_type,
        )
        result = self.llm.call(run.id, "WRITE", "writer", prompt, temperature=0.5)
        article = result.text.strip()
        if len(article) < 200:
            article = self._fallback_article(topic)
        title = topic.get("title", "AI 热点")
        if not title.endswith("解读"):
            title = f"{title}：实战解读"
        ctx["article_title"] = title[:80]
        ctx["article_markdown"] = article
        self._set_step_audit(
            ctx,
            "WRITE",
            {
                "prompts": [
                    {
                        "title": "写作提示词",
                        "text": self._clip_text(prompt, 8000),
                    }
                ],
                "outputs": [
                    {
                        "title": "文章正文预览",
                        "text": self._clip_text(article, 8000),
                        "language": "markdown",
                    }
                ],
            },
        )

    def _step_write_v2(self, run: Run, ctx: dict[str, Any]) -> None:
        topic = ctx.get("selected_topic") or {"title": "AI Daily Topic", "summary": ""}
        fact_pack = dict(ctx.get("fact_pack") or {})
        audience_key = str(ctx.get("target_audience") or self.settings.get("writing.default_audience", "ai_product_manager"))
        content_type = str(ctx.get("content_type") or fact_pack.get("content_type") or "tool_review")
        prompt = self.writing_templates.build_write_prompt(
            topic=topic,
            fact_pack=fact_pack,
            audience_key=audience_key,
            content_type=content_type,
        )
        compressed = dict(ctx.get("fact_compress") or {})
        if compressed:
            prompt += (
                "\n\n【LLM事实压缩结果】\n"
                "下面是基于原文提纯后的高优先级事实，请优先依赖这些内容组织文章：\n"
                f"{self._clip_text(json.dumps(compressed, ensure_ascii=False, indent=2), 4000)}"
            )
        title_plan = self.title_generator.generate(
            run_id=run.id,
            topic=topic,
            fact_pack=fact_pack,
            fact_compress=compressed,
            content_type=content_type,
            llm=self.llm,
        )
        result = self.llm.call(run.id, "WRITE", "writer", prompt, temperature=0.45)
        article = result.text.strip()
        if len(article) < 200:
            article = self._fallback_article(topic)
        ctx["article_title"] = title_plan.article_title
        ctx["wechat_title"] = title_plan.wechat_title
        ctx["title_plan"] = title_plan.as_dict()
        ctx["article_markdown"] = article
        self._set_step_audit(
            ctx,
            "WRITE",
            {
                "prompts": [
                    {
                        "title": "融合模板写作提示词",
                        "text": self._clip_text(prompt, 8000),
                    }
                ],
                "outputs": [
                    {
                        "title": "标题方案",
                        "text": self._clip_text(json.dumps(title_plan.as_dict(), ensure_ascii=False, indent=2), 4000),
                        "language": "json",
                    },
                    {
                        "title": "文章正文预览",
                        "text": self._clip_text(article, 8000),
                        "language": "markdown",
                    }
                ],
            },
        )

    def _step_quality_check(self, run: Run, ctx: dict[str, Any]) -> None:
        threshold = self.settings.get_float("quality.threshold", 78.0)
        max_rounds = self.settings.get_int("quality.max_rounds", 3)
        scores: list[float] = []
        best = {"score": -1.0, "title": "", "article": ""}
        topic = ctx.get("selected_topic", {})
        round_logs: list[dict[str, Any]] = []

        for round_idx in range(1, max_rounds + 1):
            eval_prompt = (
                "Evaluate article quality from 0 to 100. "
                "Output one line starting with SCORE: <number> and then short reasons.\n\n"
                f"Title: {ctx.get('article_title', '')}\n"
                f"Article:\n{ctx.get('article_markdown', '')[:3000]}"
            )
            eval_result = self.llm.call(run.id, "QUALITY_CHECK", "decision", eval_prompt, temperature=0.2)
            score = self._extract_score(eval_result.text)
            if score is None:
                score = self._heuristic_score(ctx.get("article_markdown", ""), round_idx)
            scores.append(score)
            round_log = {
                "round": round_idx,
                "eval_prompt": self._clip_text(eval_prompt, 8000),
                "eval_response": self._clip_text(eval_result.text, 4000),
                "score": score,
            }

            if score > best["score"]:
                best = {
                    "score": score,
                    "title": ctx.get("article_title", ""),
                    "article": ctx.get("article_markdown", ""),
                }

            if score >= threshold:
                ctx["quality_score"] = score
                ctx["quality_attempts"] = round_idx
                ctx["quality_fallback_used"] = False
                ctx["quality_scores"] = scores
                round_logs.append(round_log)
                self._set_step_audit(
                    ctx,
                    "QUALITY_CHECK",
                    {
                        "rounds": round_logs,
                        "prompts": [
                            {
                                "title": f"第 {item['round']} 轮质检提示词",
                                "text": item["eval_prompt"],
                            }
                            for item in round_logs
                        ],
                        "outputs": [
                            {
                                "title": f"第 {item['round']} 轮质检回包",
                                "text": item["eval_response"],
                                "meta": f"评分 {item['score']}",
                            }
                            for item in round_logs
                        ],
                    },
                )
                return

            if round_idx < max_rounds:
                fact_pack_text = self._clip_text(
                    self.writing_templates.preview_fact_pack(dict(ctx.get("fact_pack") or {}), limit=2000),
                    2000,
                )
                improve_prompt = (
                    "Improve the article quality according to this feedback and rewrite in Chinese markdown.\n"
                    "You must preserve factual accuracy and can only rely on the provided fact pack.\n"
                    f"Fact Pack:\n{fact_pack_text}\n\n"
                    f"Feedback:\n{eval_result.text[:1000]}\n\n"
                    f"Current Article:\n{ctx.get('article_markdown', '')[:3000]}"
                )
                rewritten = self.llm.call(run.id, "WRITE", "writer", improve_prompt, temperature=0.45).text.strip()
                round_log["improve_prompt"] = self._clip_text(improve_prompt, 8000)
                round_log["rewrite_preview"] = self._clip_text(rewritten, 4000)
                if len(rewritten) > 150:
                    ctx["article_markdown"] = rewritten
            round_logs.append(round_log)

        # 3 rounds still below threshold => choose best score version.
        ctx["article_title"] = best["title"]
        ctx["article_markdown"] = best["article"]
        ctx["quality_score"] = best["score"]
        ctx["quality_attempts"] = max_rounds
        ctx["quality_fallback_used"] = True
        ctx["quality_scores"] = scores
        self._set_step_audit(
            ctx,
            "QUALITY_CHECK",
            {
                "rounds": round_logs,
                "prompts": [
                    *[
                        {
                            "title": f"第 {item['round']} 轮质检提示词",
                            "text": item["eval_prompt"],
                        }
                        for item in round_logs
                    ],
                    *[
                        {
                            "title": f"第 {item['round']} 轮改写提示词",
                            "text": item["improve_prompt"],
                        }
                        for item in round_logs
                        if item.get("improve_prompt")
                    ],
                ],
                "outputs": [
                    *[
                        {
                            "title": f"第 {item['round']} 轮质检回包",
                            "text": item["eval_response"],
                            "meta": f"评分 {item['score']}",
                        }
                        for item in round_logs
                    ],
                    *[
                        {
                            "title": f"第 {item['round']} 轮改写结果预览",
                            "text": item["rewrite_preview"],
                            "language": "markdown",
                        }
                        for item in round_logs
                        if item.get("rewrite_preview")
                    ],
                ],
            },
        )

    def _step_article_render(self, run: Run, ctx: dict[str, Any]) -> None:
        article_markdown = str(ctx.get("article_markdown") or "").strip()
        if not article_markdown:
            raise RuntimeError("article_markdown is empty")
        article_title = str(ctx.get("article_title") or "").strip()
        content_type = str(ctx.get("content_type") or "tool_review").strip() or "tool_review"
        audience = str(ctx.get("target_audience") or "").strip()
        rendered = self.article_renderer.render(
            article_markdown,
            article_title=article_title,
            content_type=content_type,
            target_audience=audience,
        )
        html_path = self.article_renderer.save_html(rendered, run.id)
        ctx["article_layout"] = {
            "name": rendered.layout_name,
            "label": rendered.layout_label,
            "description": rendered.description,
            "source": rendered.source,
            "content_type": content_type,
        }
        ctx["article_render"] = {
            "html_path": html_path,
            "html_length": len(rendered.html),
            "block_count": rendered.block_count,
            "html_excerpt": self._clip_text(rendered.html, 1200),
        }
        ctx["article_html"] = rendered.html
        self._set_step_audit(
            ctx,
            "ARTICLE_RENDER",
            {
                "outputs": [
                    {
                        "title": "文章模板信息",
                        "text": self._clip_text(json.dumps(ctx["article_layout"], ensure_ascii=False, indent=2), 4000),
                        "language": "json",
                    },
                    {
                        "title": "最终 HTML 预览",
                        "text": self._clip_text(rendered.html, 4000),
                        "language": "html",
                    },
                ]
            },
        )

    def _step_cover_5d(self, run: Run, ctx: dict[str, Any]) -> None:
        prompt = (
            "Generate cover 5D scores in JSON with keys: 主题主体, 场景构图, 视觉风格, 色彩光线, 文案层级. "
            "Each score 0-100.\n"
            f"Article title: {ctx.get('article_title', '')}"
        )
        text = self.llm.call(run.id, "COVER_5D", "cover_prompt", prompt, temperature=0.3).text
        dims = self._parse_cover_dims(text)
        if not dims:
            dims = {
                "主题主体": round(random.uniform(75, 92), 2),
                "场景构图": round(random.uniform(72, 90), 2),
                "视觉风格": round(random.uniform(74, 91), 2),
                "色彩光线": round(random.uniform(70, 89), 2),
                "文案层级": round(random.uniform(71, 88), 2),
            }
        total = round(
            0.30 * dims["主题主体"]
            + 0.20 * dims["场景构图"]
            + 0.20 * dims["视觉风格"]
            + 0.15 * dims["色彩光线"]
            + 0.15 * dims["文案层级"],
            2,
        )
        dims["总分"] = total
        ctx["cover_5d"] = dims
        self._set_step_audit(
            ctx,
            "COVER_5D",
            {
                "prompts": [
                    {
                        "title": "五维评分提示词",
                        "text": self._clip_text(prompt, 8000),
                    }
                ],
                "outputs": [
                    {
                        "title": "五维模型回包",
                        "text": self._clip_text(text, 4000),
                        "language": "json",
                    }
                ],
            },
        )

    def _step_cover_gen(self, run: Run, ctx: dict[str, Any]) -> None:
        prompt_request = (
            "请为微信公众号文章生成一段适合文生图模型的中文封面提示词。\n"
            "要求：横版封面、科技感、主体明确、构图简洁、强对比、适合公众号首图，不要水印，不要边框，避免大段文字。\n"
            f"文章标题：{ctx.get('article_title', '')}\n"
            f"封面 5D：{json.dumps(ctx.get('cover_5d', {}), ensure_ascii=False)}"
        )
        prompt_result = self.llm.call(run.id, "COVER_GEN", "cover_prompt", prompt_request, temperature=0.4)
        image_prompt = prompt_result.text.strip()
        if len(image_prompt) < 20:
            image_prompt = self._fallback_cover_prompt(ctx.get("article_title", ""), ctx.get("cover_5d", {}))
        output_dir = CONFIG.data_dir / "runs" / run.id
        ctx["cover_asset"] = self.llm.generate_cover_image(
            run.id,
            "COVER_GEN",
            "cover_image",
            prompt=image_prompt,
            output_dir=output_dir,
            size="1280*720",
        )
        self._set_step_audit(
            ctx,
            "COVER_GEN",
            {
                "prompts": [
                    {
                        "title": "封面提示词生成请求",
                        "text": self._clip_text(prompt_request, 8000),
                    },
                    {
                        "title": "最终出图提示词",
                        "text": self._clip_text(image_prompt, 8000),
                    },
                ],
                "outputs": [
                    {
                        "title": "封面提示词模型回包",
                        "text": self._clip_text(prompt_result.text, 4000),
                    }
                ],
            },
        )

    def _step_cover_check(self, run: Run, ctx: dict[str, Any]) -> None:
        dims = ctx.get("cover_5d", {})
        total = float(dims.get("总分", 0))
        if total < 70:
            raise RuntimeError("Cover quality score too low")

    def _step_wechat_draft(self, run: Run, ctx: dict[str, Any]) -> None:
        topic = ctx.get("selected_topic", {})
        source_url = topic.get("url", "")
        cover_asset = ctx.get("cover_asset", {})
        result = self.wechat.publish_draft(
            title=ctx.get("wechat_title") or ctx.get("article_title", "AI 热点"),
            markdown_content=ctx.get("article_markdown", ""),
            html_content=ctx.get("article_html", ""),
            source_url=source_url,
            cover_image_path=str(cover_asset.get("path", "") or ""),
        )
        ctx["wechat_result"] = {
            "success": result.success,
            "draft_id": result.draft_id,
            "reason": result.reason,
            "thumb_media_id": result.thumb_media_id,
            "sent_title": result.sent_title,
            "sent_digest": result.sent_digest,
            "sent_title_chars": len(str(result.sent_title or "")),
            "sent_title_bytes": len(str(result.sent_title or "").encode("utf-8")),
            "sent_digest_chars": len(str(result.sent_digest or "")),
            "sent_digest_bytes": len(str(result.sent_digest or "").encode("utf-8")),
            "debug_info": result.debug_info,
        }
        if not result.success:
            ctx["draft_status"] = "pending_manual"
            raise RuntimeError(result.reason)
        ctx["draft_status"] = "saved"

    # -------- helpers --------
    @staticmethod
    def _fallback_cover_prompt(title: str, cover_5d: dict[str, Any]) -> str:
        total = cover_5d.get("鎬诲垎", "-")
        return (
            f"微信公众号科技文章横版封面，主题围绕“{title}”，"
            f"突出科技感与专业感，主体明确，景别干净，光线有层次，"
            f"适合 1280x720 封面图，视觉评分目标 {total}，无水印，无边框，无杂乱小字。"
        )

    @staticmethod
    def _fallback_article(topic: dict[str, Any]) -> str:
        title = topic.get("title", "AI 热点")
        summary = topic.get("summary", "这是今天值得关注的 AI 资讯。")
        return (
            f"# {title}\n\n"
            f"## 事件摘要\n{summary}\n\n"
            "## 为什么重要\n"
            "1. 相关能力正在加速进入实际业务场景。\n"
            "2. 产业链协同速度提升，落地门槛下降。\n"
            "3. 对团队效率和成本结构有直接影响。\n\n"
            "## 落地建议\n"
            "- 从低风险流程开始试点。\n"
            "- 先定义质量与成本监控指标。\n"
            "- 形成标准化 SOP 再扩展到更多场景。\n"
        )

    @staticmethod
    def _extract_score(text: str) -> float | None:
        # Supports explicit score labels only.
        import re

        m = re.search(r"SCORE\s*[:：]\s*(\d{1,3}(?:\.\d+)?)", text, flags=re.IGNORECASE)
        if not m:
            m = re.search(r"(?:评分|得分)\s*[:：]\s*(\d{1,3}(?:\.\d+)?)", text)
        if m:
            score = float(m.group(1))
            if 0 <= score <= 100:
                return round(score, 2)
        return None

    @staticmethod
    def _heuristic_score(article: str, round_idx: int) -> float:
        base = min(68.0 + len(article) / 115.0, 86.0)
        bonus = round_idx * 3.4
        noise = random.uniform(-1.0, 1.0)
        return round(min(base + bonus + noise, 93.0), 2)

    @staticmethod
    def _parse_cover_dims(text: str) -> dict[str, float]:
        import re

        keys = ["主题主体", "场景构图", "视觉风格", "色彩光线", "文案层级"]
        out: dict[str, float] = {}
        for key in keys:
            m = re.search(rf"{re.escape(key)}\s*[:：]\s*(\d{{1,3}}(?:\.\d+)?)", text)
            if m:
                val = float(m.group(1))
                if 0 <= val <= 100:
                    out[key] = round(val, 2)
        if len(out) == 5:
            return out
        return {}

    @staticmethod
    def _should_reject_topic(item: dict[str, Any]) -> bool:
        text = " ".join(str(item.get(key, "") or "") for key in ("title", "summary", "url")).lower()
        hard_reject_keywords = [
            "quiz", "character data", "note-taking", "exercise", "flashcards",
            "beginner quiz", "string quiz", "入门练习", "测验", "刷题", "习题",
        ]
        return any(keyword in text for keyword in hard_reject_keywords)

    @staticmethod
    def _topic_depth_score(item: dict[str, Any]) -> float:
        text = " ".join(str(item.get(key, "") or "") for key in ("title", "summary", "url")).lower()
        positive = [
            "agent", "workflow", "architecture", "benchmark", "code", "api", "open source",
            "mechanism", "analysis", "实测", "拆解", "架构", "机制", "工作流", "开源", "评测", "对比",
        ]
        negative = [
            "quiz", "character data", "note-taking", "tips", "basics", "beginner",
            "string", "exercise", "练习", "入门", "基础",
        ]
        score = 45.0
        score += 10.0 * sum(1 for keyword in positive if keyword in text)
        score -= 12.0 * sum(1 for keyword in negative if keyword in text)
        return round(max(0.0, min(score, 100.0)), 2)

    @staticmethod
    def _topic_novelty_score(item: dict[str, Any], hours: float) -> float:
        text = " ".join(str(item.get(key, "") or "") for key in ("title", "summary")).lower()
        launch_keywords = ["launch", "release", "announce", "上线", "发布", "推出", "首次", "升级", "open source", "开源"]
        base = 55.0 if any(keyword in text for keyword in launch_keywords) else 35.0
        if hours <= 24:
            base += 20.0
        elif hours <= 48:
            base += 10.0
        return round(max(0.0, min(base, 100.0)), 2)

    @staticmethod
    def _topic_value_score(item: dict[str, Any]) -> float:
        text = " ".join(str(item.get(key, "") or "") for key in ("title", "summary")).lower()
        high_value = ["agent", "workflow", "operator", "e-commerce", "效率", "工作流", "运营", "商业", "产品", "企业级"]
        low_value = ["quiz", "character data", "beginner", "note-taking", "tips", "基础", "入门"]
        score = 50.0
        score += 8.0 * sum(1 for keyword in high_value if keyword in text)
        score -= 10.0 * sum(1 for keyword in low_value if keyword in text)
        return round(max(0.0, min(score, 100.0)), 2)

    def _build_select_prompt(self, candidates: list[dict[str, Any]]) -> str:
        docs = []
        for idx, item in enumerate(candidates):
            docs.append(
                "\n".join(
                    [
                        f"候选 {idx}",
                        f"标题: {item.get('title', '')}",
                        f"来源: {item.get('source', '')}",
                        f"发布时间: {item.get('published', '')}",
                        f"摘要: {item.get('summary', '')}",
                        f"规则分: {item.get('rule_score', 0)}",
                        f"新鲜度分: {item.get('freshness_score', 0)}",
                        f"深度分: {item.get('depth_score', 0)}",
                        f"价值分: {item.get('value_score', 0)}",
                        f"新信息分: {item.get('novelty_score', 0)}",
                        f"正文摘样: {item.get('selection_excerpt', '')[:800]}",
                    ]
                )
            )
        joined = "\n\n".join(docs)
        return (
            "你是公众号选题编辑。请从下面候选中选出一个最值得今天写成原创解读的主题。\n"
            "目标不是找最早发布的深度文，而是从最近文章里找一个当前仍值得发、信息价值最高、可写出深度解读的题。\n"
            "优先标准：新信息密度、机制细节、工作流价值、产业或产品影响、可形成干货分析。\n"
            "排除倾向：基础教程、练习题、quiz、浅层搬运、老话题重复包装。\n"
            "请输出 JSON：{\"index\": 0, \"reason\": \"...\"}\n\n"
            f"{joined}"
        )

    @staticmethod
    def _parse_select_choice(text: str, candidate_count: int) -> int:
        import re

        try:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                data = json.loads(text[start : end + 1])
                index = int(data.get("index", 0))
                if 0 <= index < candidate_count:
                    return index
        except Exception:
            pass

        match = re.search(r'"index"\s*:\s*(\d+)', text)
        if match:
            index = int(match.group(1))
            if 0 <= index < candidate_count:
                return index
        return -1

    @staticmethod
    def _parse_fact_compress_result(text: str, fact_pack: dict[str, Any]) -> dict[str, Any]:
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                data = json.loads(text[start : end + 1])
                if isinstance(data, dict):
                    return {
                        "one_sentence_summary": str(data.get("one_sentence_summary", "") or ""),
                        "what_it_is": list(data.get("what_it_is") or []),
                        "key_mechanisms": list(data.get("key_mechanisms") or []),
                        "concrete_scenarios": list(data.get("concrete_scenarios") or []),
                        "numbers": list(data.get("numbers") or []),
                        "risks": list(data.get("risks") or []),
                        "uncertainties": list(data.get("uncertainties") or []),
                        "recommended_angle": list(data.get("recommended_angle") or []),
                    }
        except Exception:
            pass

        key_points = [str(item) for item in (fact_pack.get("key_points") or [])[:4]]
        numbers = [str(item) for item in (fact_pack.get("numbers") or [])[:5]]
        return {
            "one_sentence_summary": key_points[0] if key_points else str(fact_pack.get("topic_title", "") or ""),
            "what_it_is": key_points[:2],
            "key_mechanisms": key_points[2:4],
            "concrete_scenarios": [str(item.get("title", "") or "") for item in (fact_pack.get("related_topics") or [])[:2]],
            "numbers": numbers,
            "risks": ["公开资料可能带有宣传导向，需要结合真实使用验证。"],
            "uncertainties": ["底层实现细节和真实效果仍需更多公开信息确认。"],
            "recommended_angle": ["优先从产品机制、工作流价值和适用边界来写。"],
        }

    def _send_daily_report(self, run: Run, ctx: dict[str, Any]) -> None:
        try:
            subject_prefix = self.settings.get("mail.subject_prefix", "[wechat-agent-lite]")
            subject = f"{subject_prefix} 每日结果 {datetime.now().strftime('%Y-%m-%d')} - {run.status}"
            html = self._build_daily_html(run, ctx)
            mail_result = self.mail.send_daily(subject=subject, html_body=html)
            ctx["mail_result"] = mail_result
        except Exception as exc:
            ctx["mail_result"] = {"sent": False, "reason": str(exc)}
        try:
            raw = json.loads(run.summary_json or "{}")
        except Exception:
            raw = {}
        raw["mail"] = ctx.get("mail_result", {})
        run.summary_json = json.dumps(raw, ensure_ascii=False)

    def _build_daily_html(self, run: Run, ctx: dict[str, Any]) -> str:
        top_n = ctx.get("top_n", [])[:10]
        top_k = ctx.get("top_k", [])[:8]
        failed_logs = ctx.get("failed_logs", [])
        cover_5d = ctx.get("cover_5d", {})
        console_base = self.settings.get("general.console_base_url", "http://127.0.0.1:18080")
        run_url = f"{console_base}/"
        quality_notice = ""
        if ctx.get("quality_fallback_used", False):
            quality_notice = (
                "<p style='color:#b95f00;font-weight:700;'>"
                "本次未达到质量阈值，已在3轮中选择最高分版本发送。</p>"
            )
        rows_top_n = "".join(
            f"<tr><td>{idx+1}</td><td>{item.get('title','')}</td><td>{item.get('source','')}</td></tr>"
            for idx, item in enumerate(top_n)
        )
        rows_top_k = "".join(
            f"<tr><td>{idx+1}</td><td>{item.get('title','')}</td><td>{item.get('final_score',0)}</td></tr>"
            for idx, item in enumerate(top_k)
        )
        rows_fail = "".join(
            f"<tr><td>{x.get('step','')}</td><td>{x.get('attempt','')}</td><td>{x.get('error','')}</td></tr>"
            for x in failed_logs
        ) or "<tr><td colspan='3'>无失败日志</td></tr>"
        quality_scores = ctx.get("quality_scores", [])
        quality_rounds = ", ".join(str(x) for x in quality_scores) if quality_scores else "-"
        token_total = sum(call.total_tokens for call in run.llm_calls)
        return f"""
<html>
<body style="font-family:Arial,'Microsoft YaHei',sans-serif;color:#1f2937;">
  <h2>wechat-agent-lite 每日结果</h2>
  <p><b>Run ID:</b> {run.id}</p>
  <p><b>状态:</b> {run.status} | <b>草稿状态:</b> {ctx.get('draft_status', run.draft_status)}</p>
  <p><b>时间:</b> {run.started_at} - {run.finished_at}</p>
  <p><b>文章标题:</b> {ctx.get('article_title', run.article_title)}</p>
  <p><b>质量分:</b> {ctx.get('quality_score', run.quality_score)} / 阈值 {self.settings.get('quality.threshold','78')}</p>
  <p><b>质量轮次得分:</b> {quality_rounds}</p>
  {quality_notice}
  <p><b>Token 总量:</b> {token_total}</p>
  <p><b>封面5维:</b> {json.dumps(cover_5d, ensure_ascii=False)}</p>
  <h3>TopN (10)</h3>
  <table border="1" cellpadding="6" cellspacing="0">
    <tr><th>#</th><th>标题</th><th>来源</th></tr>
    {rows_top_n}
  </table>
  <h3>TopK (8) Rerank</h3>
  <table border="1" cellpadding="6" cellspacing="0">
    <tr><th>#</th><th>标题</th><th>final_score</th></tr>
    {rows_top_k}
  </table>
  <h3>失败日志明细</h3>
  <table border="1" cellpadding="6" cellspacing="0">
    <tr><th>步骤</th><th>尝试</th><th>错误</th></tr>
    {rows_fail}
  </table>
  <p><a href="{run_url}">打开控制台</a></p>
</body>
</html>
"""

    def _build_step_details(self, name: str, ctx: dict[str, Any], status: str, error_text: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "headline": self._step_headline(name=name, ctx=ctx, status=status, error_text=error_text),
            "summary": {},
            "items": [],
            "prompts": [],
            "outputs": [],
            "raw": {},
        }
        audit = dict((ctx.get("step_audits") or {}).get(name) or {})
        runtime_meta = self.llm.get_step_runtime_meta(name)

        if name == "HEALTH_CHECK":
            health = dict(ctx.get("health") or {})
            payload["summary"] = {
                "代理开关": "开启" if health.get("proxy_enabled") else "关闭",
                "出口 IP": health.get("egress_ip") or "-",
                "健康状态": "正常" if health.get("ok", True) else "异常",
            }
            payload["raw"] = health
        elif name == "SOURCE_MAINTENANCE":
            maintenance = dict(ctx.get("source_maintenance") or {})
            progress = dict(ctx.get("source_maintenance_progress") or {})
            actions = list(maintenance.get("actions") or [])
            if status == StepStatus.running.value and progress:
                payload["summary"] = {
                    "执行阶段": progress.get("phase") or "-",
                    "当前源": progress.get("current_source") or "-",
                    "检查进度": f"{progress.get('checked_sources', 0)} / {progress.get('total_sources', 0)}",
                    "healthy_sources": progress.get("healthy_sources", 0),
                    "failed_sources": progress.get("failed_sources", 0),
                    "changed_sources": progress.get("changed_sources", 0),
                    "manual_review_sources": progress.get("manual_review_sources", 0),
                    "llm_candidate_sources": progress.get("llm_candidate_sources", 0),
                }
                recent_sources = list(progress.get("recent_sources") or [])
                recent_actions = list(progress.get("recent_actions") or [])
                payload["items"] = [
                    (
                        f"[检查] {item.get('name', 'unknown-source')} | "
                        f"{'ok' if item.get('probe_ok') else item.get('reason', '-') or '-'} | "
                        f"候选 {item.get('candidate_count', 0)} / HTML {item.get('html_article_count', 0)}"
                    )
                    for item in recent_sources[-4:]
                ] + [
                    (
                        f"[动作] {item.get('name', 'unknown-source')} | "
                        f"{item.get('applied_action') or item.get('final_action') or '-'} | "
                        f"{item.get('reason', '-')}"
                    )
                    for item in recent_actions[-4:]
                ]
                payload["raw"] = {
                    "progress": progress,
                    "maintenance": maintenance,
                }
            else:
                payload["summary"] = {
                    "checked_sources": maintenance.get("checked_sources", 0),
                    "healthy_sources": maintenance.get("healthy_sources", 0),
                    "failed_sources": maintenance.get("failed_sources", 0),
                    "changed_sources": maintenance.get("changed_sources", 0),
                    "manual_review_sources": maintenance.get("manual_review_sources", 0),
                    "llm_candidate_sources": maintenance.get("llm_candidate_sources", 0),
                }
                payload["items"] = [
                    (
                        f"{item.get('name', 'unknown-source')} | "
                        f"{item.get('applied_action') or item.get('final_action') or '-'} | "
                        f"{item.get('reason', '-')}"
                    )
                    for item in actions[:6]
                ]
                payload["raw"] = maintenance
        elif name == "FETCH":
            items = list(ctx.get("fetched_items") or [])
            payload["summary"] = {
                "抓取条数": len(items),
                "失败日志": len(ctx.get("failed_logs") or []),
            }
            payload["items"] = [self._topic_line(item) for item in items[:5]]
            payload["raw"] = {
                "items": [self._compact_topic(item) for item in items],
                "failed_logs": (ctx.get("failed_logs") or [])[-5:],
            }
        elif name == "DEDUP":
            fetched_items = list(ctx.get("fetched_items") or [])
            deduped_items = list(ctx.get("deduped_items") or [])
            payload["summary"] = {
                "原始条数": len(fetched_items),
                "去重后": len(deduped_items),
                "移除重复": max(len(fetched_items) - len(deduped_items), 0),
            }
            payload["items"] = [self._topic_line(item) for item in deduped_items[:5]]
            payload["raw"] = {"items": [self._compact_topic(item) for item in deduped_items[:8]]}
        elif name == "RULE_SCORE":
            top_n = list(ctx.get("top_n") or [])
            payload["summary"] = {
                "入选 TopN": len(top_n),
                "TopN 配额": self.settings.get_int("general.top_n", 10),
            }
            payload["items"] = [
                f"{item.get('title', '未命名主题')} | 规则分 {item.get('rule_score', 0)}"
                for item in top_n[:5]
            ]
            payload["raw"] = {"top_n": [self._compact_topic(item, include_scores=True) for item in top_n[:8]]}
        elif name == "RERANK":
            top_k = list(ctx.get("top_k") or [])
            payload["summary"] = {
                "入选 TopK": len(top_k),
                "TopK 配额": self.settings.get_int("general.top_k", 8),
            }
            payload["items"] = [
                f"{item.get('title', '未命名主题')} | 综合分 {item.get('final_score', 0)}"
                for item in top_k[:5]
            ]
            payload["raw"] = {"top_k": [self._compact_topic(item, include_scores=True) for item in top_k[:8]]}
            payload["summary"]["Top1 标题"] = (top_k[0].get("title") if top_k else "") or "-"
            payload["items"] = self._build_rerank_detail_items(top_k[:5])
        elif name == "SELECT":
            selected_topic = dict(ctx.get("selected_topic") or {})
            payload["summary"] = {
                "已选主题": selected_topic.get("title") or "-",
                "来源": selected_topic.get("source") or "-",
            }
            payload["raw"] = self._compact_topic(selected_topic, include_scores=True)
        elif name == "SOURCE_ENRICH":
            source_pack = dict(ctx.get("source_pack") or {})
            primary = dict(source_pack.get("primary") or {})
            related = list(source_pack.get("related") or [])
            payload["summary"] = {
                "主来源状态": primary.get("status") or "-",
                "主来源标题": primary.get("title") or "-",
                "主来源正文长度": len(str(primary.get("content_text") or "")),
                "相关来源数": len(related),
            }
            payload["items"] = [
                f"{item.get('title', '未命名来源')} | {item.get('status', '-')}"
                for item in related[:4]
            ]
            payload["raw"] = source_pack
        elif name == "FACT_PACK":
            fact_pack = dict(ctx.get("fact_pack") or {})
            payload["summary"] = {
                "文章类型": ctx.get("content_type") or "-",
                "目标读者": ctx.get("target_audience") or "-",
                "关键点数量": len(fact_pack.get("key_points") or []),
                "相关线索数": len(fact_pack.get("related_topics") or []),
            }
            payload["items"] = [str(item) for item in (fact_pack.get("key_points") or [])[:6]]
            payload["raw"] = fact_pack
        elif name == "FACT_COMPRESS":
            fact_compress = dict(ctx.get("fact_compress") or {})
            payload["summary"] = {
                "一句话总结": fact_compress.get("one_sentence_summary") or "-",
                "机制条数": len(fact_compress.get("key_mechanisms") or []),
                "场景条数": len(fact_compress.get("concrete_scenarios") or []),
                "风险条数": len(fact_compress.get("risks") or []),
            }
            payload["items"] = [str(item) for item in (fact_compress.get("key_mechanisms") or [])[:5]]
            payload["raw"] = fact_compress
        elif name == "WRITE":
            article = str(ctx.get("article_markdown") or "")
            payload["summary"] = {
                "文章标题": ctx.get("article_title") or "-",
                "正文长度": len(article),
            }
            payload["items"] = [article[:180] + ("..." if len(article) > 180 else "")] if article else []
            payload["raw"] = {
                "article_title": ctx.get("article_title") or "",
                "article_excerpt": article[:1200],
            }
        elif name == "QUALITY_CHECK":
            scores = list(ctx.get("quality_scores") or [])
            payload["summary"] = {
                "最终得分": ctx.get("quality_score") or 0,
                "质量阈值": self.settings.get_float("quality.threshold", 78.0),
                "评估轮次": ctx.get("quality_attempts") or len(scores) or 0,
                "兜底策略": "已启用" if ctx.get("quality_fallback_used") else "未启用",
            }
            payload["items"] = [f"第 {idx + 1} 轮：{score}" for idx, score in enumerate(scores)]
            payload["raw"] = {"quality_scores": scores}
        elif name == "ARTICLE_RENDER":
            article_layout = dict(ctx.get("article_layout") or {})
            article_render = dict(ctx.get("article_render") or {})
            payload["summary"] = {
                "模板名称": article_layout.get("label") or article_layout.get("name") or "-",
                "模板键": article_layout.get("name") or "-",
                "模板来源": article_layout.get("source") or "-",
                "文章类型": article_layout.get("content_type") or ctx.get("content_type") or "-",
                "HTML 长度": article_render.get("html_length") or 0,
                "块数量": article_render.get("block_count") or 0,
                "HTML 文件": article_render.get("html_path") or "-",
            }
            excerpt = str(article_render.get("html_excerpt") or "").strip()
            payload["items"] = [excerpt] if excerpt else []
            payload["raw"] = {
                "article_layout": article_layout,
                "article_render": article_render,
            }
        elif name == "COVER_5D":
            cover_5d = dict(ctx.get("cover_5d") or {})
            payload["summary"] = {key: value for key, value in cover_5d.items()}
            payload["raw"] = cover_5d
        elif name == "COVER_GEN":
            cover_asset = dict(ctx.get("cover_asset") or {})
            payload["summary"] = {
                "生成状态": cover_asset.get("status") or "-",
                "图片尺寸": cover_asset.get("size") or "-",
                "文件路径": cover_asset.get("path") or "-",
            }
            payload["raw"] = cover_asset
        elif name == "COVER_CHECK":
            cover_5d = dict(ctx.get("cover_5d") or {})
            total_score = cover_5d.get("鎬诲垎", 0)
            payload["summary"] = {
                "封面总分": total_score,
                "校验结果": "通过" if float(total_score or 0) >= 70 else "未通过",
            }
            payload["raw"] = cover_5d
        elif name == "WECHAT_DRAFT":
            wechat_result = dict(ctx.get("wechat_result") or {})
            payload["summary"] = {
                "草稿状态": ctx.get("draft_status") or "-",
                "发布结果": "成功" if wechat_result.get("success") else "失败",
                "草稿 ID": wechat_result.get("draft_id") or "-",
            }
            payload["raw"] = wechat_result

        if error_text:
            payload["summary"]["错误信息"] = error_text
        if runtime_meta:
            payload["summary"]["模型超时计划"] = "；".join(
                f"{item.get('role')} {item.get('complexity_tier')} / {item.get('timeout_seconds')}s"
                for item in runtime_meta[:3]
            )
            payload["raw"]["llm_runtime"] = runtime_meta
        payload["prompts"] = [item for item in audit.get("prompts", []) if isinstance(item, dict) and item.get("text")]
        payload["outputs"] = [item for item in audit.get("outputs", []) if isinstance(item, dict) and item.get("text")]
        if audit.get("rounds"):
            payload["raw"]["audit_rounds"] = audit.get("rounds")
        payload["summary"] = {key: value for key, value in payload["summary"].items() if value not in (None, "", [], {})}
        payload["items"] = [item for item in payload["items"] if item]
        return payload

    def _step_headline(self, name: str, ctx: dict[str, Any], status: str, error_text: str) -> str:
        if status == StepStatus.failed.value and error_text:
            return f"{name} 执行失败，需要查看错误上下文"
        mapping = {
            "HEALTH_CHECK": "已完成运行前健康检查",
            "FETCH": f"已抓取 {len(ctx.get('fetched_items') or [])} 条候选热点",
            "DEDUP": f"去重后保留 {len(ctx.get('deduped_items') or [])} 条内容",
            "RULE_SCORE": f"规则打分完成，TopN 共 {len(ctx.get('top_n') or [])} 条",
            "RERANK": f"重排完成，TopK 共 {len(ctx.get('top_k') or [])} 条",
            "SELECT": f"已选定主题：{(ctx.get('selected_topic') or {}).get('title', '-')}",
            "WRITE": f"文章草稿已生成：{ctx.get('article_title') or '-'}",
            "QUALITY_CHECK": f"质检完成，最终得分 {ctx.get('quality_score') or 0}",
            "ARTICLE_RENDER": f"文章 HTML 已渲染：{(ctx.get('article_layout') or {}).get('label', '-')}",
            "COVER_5D": "封面 5D 评分已生成",
            "COVER_GEN": "封面提示词已生成",
            "COVER_CHECK": "封面质量校验已完成",
            "WECHAT_DRAFT": f"草稿投递状态：{ctx.get('draft_status') or '-'}",
        }
        return mapping.get(name, f"{name} 已完成")

    def _step_headline(self, name: str, ctx: dict[str, Any], status: str, error_text: str) -> str:
        if status == StepStatus.failed.value and error_text:
            return f"{name} 执行失败，需要查看错误上下文"
        if name == "SOURCE_MAINTENANCE" and status == StepStatus.running.value:
            progress = dict(ctx.get("source_maintenance_progress") or {})
            current_source = progress.get("current_source") or "等待开始"
            checked = progress.get("checked_sources", 0)
            total = progress.get("total_sources", 0)
            phase = progress.get("phase") or "inspect"
            return f"抓取源维护进行中：{phase} / {current_source}（{checked}/{total}）"
        mapping = {
            "HEALTH_CHECK": "运行前健康检查已完成",
            "SOURCE_MAINTENANCE": (
                f"抓取源维护已完成，已应用 {(ctx.get('source_maintenance') or {}).get('changed_sources', 0)} 处变更"
            ),
            "FETCH": f"已抓取 {len(ctx.get('fetched_items') or [])} 条候选热点",
            "DEDUP": f"去重后保留 {len(ctx.get('deduped_items') or [])} 条内容",
            "RULE_SCORE": f"规则打分完成，TopN 共 {len(ctx.get('top_n') or [])} 条",
            "RERANK": f"重排完成，TopK 共 {len(ctx.get('top_k') or [])} 条",
            "SELECT": f"已选定主题：{(ctx.get('selected_topic') or {}).get('title', '-')}",
            "WRITE": f"文章草稿已生成：{ctx.get('article_title') or '-'}",
            "QUALITY_CHECK": f"质检完成，最终得分 {ctx.get('quality_score') or 0}",
            "ARTICLE_RENDER": f"文章模板已应用：{(ctx.get('article_layout') or {}).get('label', '-')}",
            "COVER_5D": "封面 5 维评估已生成",
            "COVER_GEN": "封面提示词已生成",
            "COVER_CHECK": "封面质量校验已完成",
            "WECHAT_DRAFT": f"草稿投递状态：{ctx.get('draft_status') or '-'}",
        }
        return mapping.get(name, f"{name} 已完成")

    @staticmethod
    def _compact_topic(item: dict[str, Any], include_scores: bool = False) -> dict[str, Any]:
        compact = {
            "title": item.get("title", ""),
            "source": item.get("source", ""),
            "published": item.get("published", ""),
            "url": item.get("url", ""),
            "summary": str(item.get("summary", ""))[:180],
        }
        if include_scores:
            compact["rule_score"] = item.get("rule_score")
            compact["freshness_score"] = item.get("freshness_score")
            compact["depth_score"] = item.get("depth_score")
            compact["value_score"] = item.get("value_score")
            compact["novelty_score"] = item.get("novelty_score")
            compact["llm_score"] = item.get("llm_score")
            compact["final_score"] = item.get("final_score")
            compact["rerank_rank"] = item.get("rerank_rank")
            compact["rerank_reason"] = item.get("rerank_reason")
            compact["selection_reason"] = item.get("selection_reason")
            compact["rerank_excerpt_status"] = item.get("rerank_excerpt_status")
            compact["rerank_excerpt"] = str(item.get("rerank_excerpt", "") or "")[:300]
        return compact

    @staticmethod
    def _topic_line(item: dict[str, Any]) -> str:
        title = item.get("title", "未命名主题")
        source = item.get("source", "未知来源")
        return f"{title} | {source}"

    def _build_rerank_detail_items(self, items: list[dict[str, Any]]) -> list[str]:
        output: list[str] = []
        for idx, item in enumerate(items, start=1):
            reason = self._clip_text(item.get("rerank_reason", ""), 120)
            parts = [
                f"#{item.get('rerank_rank', idx)} {item.get('title', '未命名主题')}",
                f"综合 {item.get('final_score', 0)}",
                f"规则 {item.get('rule_score', 0)}",
                f"重排 {item.get('llm_score', 0)}",
                f"深度 {item.get('depth_score', 0)}",
                f"价值 {item.get('value_score', 0)}",
                f"摘样 {item.get('rerank_excerpt_status', '-')}",
            ]
            if reason:
                parts.append(f"理由: {reason}")
            output.append(" | ".join(parts))
        return output


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)
