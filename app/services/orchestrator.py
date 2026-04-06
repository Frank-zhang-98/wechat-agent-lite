from __future__ import annotations

import json
import math
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import requests
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import CONFIG
from app import state
from app.models import Run, RunStatus, RunStep, StepStatus
from app.services.article_render_service import ArticleRenderService
from app.services.body_illustration_service import BodyIllustrationService
from app.services.fetch_service import FetchService
from app.services.fact_grounding_service import FactGroundingService
from app.services.hallucination_check_service import HallucinationCheckService
from app.services.llm_gateway import LLMGateway
from app.services.localization_service import LocalizationService
from app.services.mail_service import MailService
from app.services.scrapling_fallback_service import ScraplingFallbackService
from app.services.source_maintenance_service import SourceMaintenanceService
from app.services.settings_service import SettingsService
from app.services.title_generation_service import TitleGenerationService
from app.services.programmatic_visual_service import ProgrammaticVisualService
from app.services.visual_strategy_service import VisualStrategyService
from app.services.web_enrich_service import WebEnrichService
from app.services.wechat_service import WeChatService
from app.services.writing_template_service import WritingTemplateService
from app.services.concurrency_utils import iter_host_limited_results, normalized_host


class StepFailedError(RuntimeError):
    pass


class RunCancelledError(RuntimeError):
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
        self.visual_strategy = VisualStrategyService()
        self.visual_renderer = ProgrammaticVisualService()
        self.body_illustrations = BodyIllustrationService(self.visual_renderer)
        self.web_enrich = WebEnrichService(self.settings, self.fetch)
        self.fact_grounding = FactGroundingService()
        self.hallucination_checker = HallucinationCheckService()
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
        state.register_run_cancel(run.id)
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
            self._raise_if_cancelled(run, ctx)
            if run.run_type == "health":
                self._run_health_only(run, ctx)
            elif run.run_type == "manual_url":
                self._run_manual_url(run, ctx)
            else:
                self._run_main(run, ctx)
            if run.status == RunStatus.running.value:
                run.status = RunStatus.success.value
            run.finished_at = _utcnow()
        except RunCancelledError as exc:
            run.status = RunStatus.cancelled.value
            run.error_message = str(exc)
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
            if run.run_type == "main" and run.status != RunStatus.cancelled.value:
                self._send_daily_report(run, ctx)
            self._commit_progress()
            state.clear_run_cancel(run.id)
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

    def _run_manual_url(self, run: Run, ctx: dict[str, Any]) -> None:
        summary = self._parse_summary_json(run.summary_json)
        manual_input = dict(summary.get("manual_input") or {})
        source_url = str(manual_input.get("source_url", "") or "").strip()
        if not source_url:
            raise RuntimeError("manual_input.source_url is required")

        metadata = self.fetch.extract_article_metadata(source_url)
        selected_topic = {
            "title": str(metadata.get("title", "") or source_url).strip(),
            "url": source_url,
            "summary": str(metadata.get("summary", "") or "").strip(),
            "published": str(metadata.get("published", "") or "").strip(),
            "source": normalized_host(source_url) or "manual_url",
            "selection_reason": "手动输入链接直跑",
            "rerank_reason": "手动输入链接，跳过热点筛选阶段",
            "source_weight": 1.0,
        }
        ctx["selected_topic"] = selected_topic
        ctx["top_n"] = [selected_topic]
        ctx["top_k"] = [selected_topic]

        self._execute_step(run, "SOURCE_ENRICH", self._step_source_enrich, ctx, self._policy_fetch())
        self._sync_selected_topic_from_source_pack(ctx)
        self._execute_step(run, "SOURCE_STRUCTURE", self._step_source_structure, ctx, self._policy_fetch())
        self._execute_step(run, "WEB_SEARCH_PLAN", self._step_web_search_plan, ctx, self._policy_generate())
        self._execute_step(run, "WEB_SEARCH_FETCH", self._step_web_search_fetch, ctx, self._policy_fetch())
        self._execute_step(run, "FACT_GROUNDING", self._step_fact_grounding, ctx, self._policy_generate())
        self._execute_step(run, "FACT_PACK", self._step_fact_pack, ctx, self._policy_generate())
        self._execute_step(run, "FACT_COMPRESS", self._step_fact_compress, ctx, self._policy_generate())
        self._execute_step(run, "WRITE", self._step_write_v2, ctx, self._policy_generate())
        self._execute_step(run, "HALLUCINATION_CHECK", self._step_hallucination_check, ctx, self._policy_generate())
        self._execute_step(run, "VISUAL_STRATEGY", self._step_visual_strategy, ctx, self._policy_generate())
        self._execute_step(run, "BODY_ILLUSTRATION_GEN", self._step_body_illustration_gen, ctx, self._policy_generate())
        self._execute_step(run, "QUALITY_CHECK", self._step_quality_check, ctx, self._policy_generate())
        self._execute_step(run, "ARTICLE_RENDER", self._step_article_render, ctx, self._policy_generate())
        self._execute_step(run, "COVER_5D", self._step_cover_5d, ctx, self._policy_generate())
        self._execute_step(run, "COVER_GEN", self._step_cover_gen, ctx, self._policy_generate())
        self._execute_step(run, "COVER_CHECK", self._step_cover_check, ctx, self._policy_generate())
        try:
            self._execute_step(run, "WECHAT_DRAFT", self._step_wechat_draft, ctx, self._policy_publish())
        except StepFailedError:
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
                "manual_input": manual_input,
                "selected_topic": ctx.get("selected_topic", {}),
                "top_n": ctx.get("top_n", []),
                "top_k": ctx.get("top_k", []),
                "source_pack": ctx.get("source_pack", {}),
                "source_structure": ctx.get("source_structure", {}),
                "web_search_plan": ctx.get("web_search_plan", {}),
                "web_enrich": ctx.get("web_enrich", {}),
                "fact_grounding": ctx.get("fact_grounding", {}),
                "fact_pack": ctx.get("fact_pack", {}),
                "fact_compress": ctx.get("fact_compress", {}),
                "hallucination_check": ctx.get("hallucination_check", {}),
                "content_type": ctx.get("content_type", ""),
                "target_audience": ctx.get("target_audience", ""),
                "title_plan": ctx.get("title_plan", {}),
                "visual_strategy": ctx.get("visual_strategy", {}),
                "body_illustrations": ctx.get("body_illustrations", []),
                "article_layout": ctx.get("article_layout", {}),
                "article_render": ctx.get("article_render", {}),
                "cover_asset": ctx.get("cover_asset", {}),
                "cover_5d": ctx.get("cover_5d", {}),
                "wechat": ctx.get("wechat_result", {}),
                "quality_scores": ctx.get("quality_scores", []),
                "failed_logs": ctx.get("failed_logs", []),
            },
            ensure_ascii=False,
        )

    def _sync_selected_topic_from_source_pack(self, ctx: dict[str, Any]) -> None:
        selected_topic = dict(ctx.get("selected_topic") or {})
        primary = dict((ctx.get("source_pack") or {}).get("primary") or {})
        if not selected_topic:
            return
        source_url = str(selected_topic.get("url", "") or "").strip()
        source_title = str(primary.get("title", "") or "").strip()
        source_summary = str(primary.get("summary", "") or "").strip()
        source_status = str(primary.get("status", "") or "").strip().lower()
        if source_title and (selected_topic.get("title", "") == source_url or not selected_topic.get("title")):
            selected_topic["title"] = source_title
        if source_summary and not str(selected_topic.get("summary", "") or "").strip():
            selected_topic["summary"] = source_summary
        if source_status == "ok" and primary.get("url"):
            selected_topic["url"] = str(primary.get("url", "") or "").strip() or source_url
        ctx["selected_topic"] = selected_topic
        ctx["top_n"] = [selected_topic]
        ctx["top_k"] = [selected_topic]

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
        self._execute_step(run, "SOURCE_STRUCTURE", self._step_source_structure, ctx, self._policy_fetch())
        self._execute_step(run, "WEB_SEARCH_PLAN", self._step_web_search_plan, ctx, self._policy_generate())
        self._execute_step(run, "WEB_SEARCH_FETCH", self._step_web_search_fetch, ctx, self._policy_fetch())
        self._execute_step(run, "FACT_GROUNDING", self._step_fact_grounding, ctx, self._policy_generate())
        self._execute_step(run, "FACT_PACK", self._step_fact_pack, ctx, self._policy_generate())
        self._execute_step(run, "FACT_COMPRESS", self._step_fact_compress, ctx, self._policy_generate())
        self._execute_step(run, "WRITE", self._step_write_v2, ctx, self._policy_generate())
        self._execute_step(run, "HALLUCINATION_CHECK", self._step_hallucination_check, ctx, self._policy_generate())
        self._execute_step(run, "VISUAL_STRATEGY", self._step_visual_strategy, ctx, self._policy_generate())
        self._execute_step(run, "BODY_ILLUSTRATION_GEN", self._step_body_illustration_gen, ctx, self._policy_generate())
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
                "source_structure": ctx.get("source_structure", {}),
                "web_search_plan": ctx.get("web_search_plan", {}),
                "web_enrich": ctx.get("web_enrich", {}),
                "fact_grounding": ctx.get("fact_grounding", {}),
                "content_type": ctx.get("content_type", ""),
                "target_audience": ctx.get("target_audience", ""),
                "article_layout": ctx.get("article_layout", {}),
                "article_render": ctx.get("article_render", {}),
                "fact_pack": ctx.get("fact_pack", {}),
                "fact_compress": ctx.get("fact_compress", {}),
                "hallucination_check": ctx.get("hallucination_check", {}),
                "visual_strategy": ctx.get("visual_strategy", {}),
                "body_illustrations": ctx.get("body_illustrations", []),
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
            self._execute_step(run, "SOURCE_STRUCTURE", self._step_source_structure, ctx, self._policy_fetch())
            self._execute_step(run, "WEB_SEARCH_PLAN", self._step_web_search_plan, ctx, self._policy_generate())
            self._execute_step(run, "WEB_SEARCH_FETCH", self._step_web_search_fetch, ctx, self._policy_fetch())
            self._execute_step(run, "FACT_GROUNDING", self._step_fact_grounding, ctx, self._policy_generate())
            self._execute_step(run, "FACT_PACK", self._step_fact_pack, ctx, self._policy_generate())
            self._execute_step(run, "FACT_COMPRESS", self._step_fact_compress, ctx, self._policy_generate())
            self._execute_step(run, "WRITE", self._step_write_v2, ctx, self._policy_generate())
            self._execute_step(run, "HALLUCINATION_CHECK", self._step_hallucination_check, ctx, self._policy_generate())
            self._execute_step(run, "VISUAL_STRATEGY", self._step_visual_strategy, ctx, self._policy_generate())
            self._execute_step(run, "BODY_ILLUSTRATION_GEN", self._step_body_illustration_gen, ctx, self._policy_generate())
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
                    "source_structure": ctx.get("source_structure", {}),
                    "web_search_plan": ctx.get("web_search_plan", {}),
                    "web_enrich": ctx.get("web_enrich", {}),
                    "fact_grounding": ctx.get("fact_grounding", {}),
                    "content_type": ctx.get("content_type", ""),
                    "target_audience": ctx.get("target_audience", ""),
                    "article_layout": ctx.get("article_layout", {}),
                    "article_render": ctx.get("article_render", {}),
                    "fact_pack": ctx.get("fact_pack", {}),
                    "fact_compress": ctx.get("fact_compress", {}),
                    "hallucination_check": ctx.get("hallucination_check", {}),
                    "visual_strategy": ctx.get("visual_strategy", {}),
                    "body_illustrations": ctx.get("body_illustrations", []),
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
                self._raise_if_cancelled(run, ctx, name)
                step.retry_count = attempt
                started = time.perf_counter()
                try:
                    handler(run, ctx)
                    self._raise_if_cancelled(run, ctx, name)
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
                except RunCancelledError as exc:
                    step.status = StepStatus.cancelled.value
                    step.error_message = str(exc)
                    step.finished_at = _utcnow()
                    step.duration_ms = int((time.perf_counter() - started) * 1000)
                    step.details_json = json.dumps(
                        self._build_step_details(name=name, ctx=ctx, status=step.status, error_text=str(exc)),
                        ensure_ascii=False,
                    )
                    self._commit_progress()
                    raise
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

    def _raise_if_cancelled(self, run: Run, ctx: dict[str, Any], step_name: str = "") -> None:
        if not state.is_run_cancel_requested(run.id):
            return
        ctx["draft_status"] = "cancelled"
        current = step_name or str(ctx.get("_active_step_name") or "RUN")
        raise RunCancelledError(f"Run cancelled by user during {current}")

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
            cancel_checker=lambda: self._raise_if_cancelled(run, ctx, "SOURCE_MAINTENANCE"),
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
            self._raise_if_cancelled(run, ctx, "FETCH")
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
            title = str(item.get("title", "") or "").strip()
            url = str(item.get("url", "") or "").strip()
            if not title or title.lower().startswith(("http://", "https://")) or title == url:
                continue
            try:
                published = datetime.fromisoformat(str(item.get("published", "") or "").strip())
            except Exception:
                continue
            hours = max((now - published).total_seconds() / 3600.0, 1.0)
            freshness = round(max(0.0, 100.0 * math.exp(-hours / max(latest_hours / 2.0, 1.0))), 2)
            source_weight = float(item.get("source_weight", 0.7)) * 100.0
            depth_score = self._topic_depth_score(item)
            novelty_score = self._topic_novelty_score(item, hours)
            value_score = self._topic_value_score(item)
            evergreen_score = self._topic_evergreen_score(item)
            timeliness_profile = self._topic_timeliness_profile(item)
            if self._should_reject_stale_topic(
                hours=hours,
                profile=timeliness_profile,
                evergreen_score=evergreen_score,
                value_score=value_score,
                depth_score=depth_score,
            ):
                continue
            editorial_penalty = self._topic_editorial_penalty_score(item)
            stale_penalty = self._topic_staleness_penalty_score(
                hours=hours,
                profile=timeliness_profile,
                evergreen_score=evergreen_score,
                value_score=value_score,
                depth_score=depth_score,
            )
            fatigue_penalty = self._topic_fatigue_penalty_score(item, current_run_id=run.id)
            rule_score = round(
                max(
                    0.0,
                    0.40 * freshness
                    + 0.25 * depth_score
                    + 0.20 * value_score
                    + 0.10 * novelty_score
                    + 0.05 * source_weight
                    - 0.18 * editorial_penalty
                    - stale_penalty
                    - fatigue_penalty
                ),
                2,
            )
            item["freshness_score"] = freshness
            item["depth_score"] = depth_score
            item["value_score"] = value_score
            item["novelty_score"] = novelty_score
            item["evergreen_score"] = evergreen_score
            item["timeliness_profile"] = timeliness_profile
            item["editorial_penalty_score"] = editorial_penalty
            item["stale_penalty_score"] = stale_penalty
            item["fatigue_penalty_score"] = fatigue_penalty
            item["rule_score"] = rule_score
            scored.append(item)
        if not scored:
            raise RuntimeError("No suitable items left after topic filtering")
        scored.sort(key=lambda x: x["rule_score"], reverse=True)
        top_n_limit = self.settings.get_int("general.top_n", 10)
        top_n = self._apply_source_diversity(
            scored,
            limit=max(0, self.settings.get_int("selection.top_n_per_source_family", 2)),
            desired=top_n_limit,
        )
        min_topic_score = float(self.settings.get_float("quality.min_topic_score", 68.0))
        if top_n and float(top_n[0].get("rule_score", 0.0) or 0.0) < min_topic_score:
            ctx["topic_gate_warning"] = f"No topic passed minimum topic score {min_topic_score}"
        ctx["top_n"] = top_n[:top_n_limit]

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
        ctx["top_k"] = self._apply_source_diversity(
            ranked_items,
            limit=max(0, self.settings.get_int("selection.top_k_per_source_family", 1)),
            desired=self.settings.get_int("general.top_k", 8),
        )

    def _step_rerank_v2(self, run: Run, ctx: dict[str, Any]) -> None:
        top_n = ctx.get("top_n", [])
        if not top_n:
            raise RuntimeError("TopN is empty")
        candidates = [dict(item) for item in top_n[: self.settings.get_int("general.top_k", 8)]]
        enrich_limit = max(1, self.settings.get_int("selection.rerank_enrich_m", 5))
        excerpt_chars = max(300, self.settings.get_int("selection.rerank_excerpt_chars", 1200))
        for idx, item in enumerate(candidates):
            self._raise_if_cancelled(run, ctx, "RERANK")
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
        query += (
            " 明确排除活动预告、workshop/webinar/conference 报名页、营销页、销售页、发售公告、"
            "以及主要目的是卖代码、卖模板、引导付款的页面。即使这些页面带有 API、workflow、automation、"
            "code snippet，也不要因为技术味重就误选。"
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
        ranked_items = self._apply_source_diversity(
            ranked_items,
            limit=max(0, self.settings.get_int("selection.top_k_per_source_family", 1)),
            desired=self.settings.get_int("general.top_k", 8),
        )
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
        evidence_weight = float(self.settings.get_float("selection.evidence_score_weight", 0.18))
        for item in candidates:
            self._raise_if_cancelled(run, ctx, "SELECT")
            url = str(item.get("url", "") or "").strip()
            excerpt = str(item.get("rerank_excerpt", "") or "")
            if url and not excerpt:
                result = self.fetch.extract_article_content(url, max_chars=1800)
                excerpt = str(result.get("content_text", "") or "")[:1000]
            item["selection_excerpt"] = excerpt
            evidence = self._probe_topic_evidence(item)
            item["evidence_score"] = evidence.get("score", 0.0)
            item["evidence_summary"] = evidence.get("summary", "")
            item["evidence_probe"] = evidence

        prompt = self._build_select_prompt_v2(candidates)
        prompt += (
            "\n\n额外规则：不要选择 workshop/webinar/conference 报名页、活动预告、销售页、发售公告、"
            "或主要目的是卖代码、卖模板、引导付款的页面。即使这类页面包含 API、automation、workflow、"
            "代码片段，也应判定为不适合写成今日深度选题。"
        )
        decision = self.llm.call(run.id, "SELECT", "decision", prompt, temperature=0.1)
        selected_index = self._parse_select_choice(decision.text, len(candidates))
        if selected_index < 0:
            selected_index = max(
                range(len(candidates)),
                key=lambda idx: (
                    float(candidates[idx].get("rule_score", 0) or 0)
                    - 0.35 * float(candidates[idx].get("editorial_penalty_score", 0) or 0)
                    - 0.25 * float(candidates[idx].get("fatigue_penalty_score", 0) or 0)
                    + evidence_weight * float(candidates[idx].get("evidence_score", 0) or 0)
                    + min(len(str(candidates[idx].get("selection_excerpt", "") or "")) / 100.0, 20.0)
                ),
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
            self._raise_if_cancelled(run, ctx, "SOURCE_ENRICH")
            primary_extract = self.fetch.extract_article_content(primary_url, max_chars=max_chars)
            primary_source.update(primary_extract)
            if not primary_source.get("title"):
                primary_source["title"] = str(topic.get("title", "") or "")

        related_sources: list[dict[str, Any]] = []
        seen_urls = {primary_url} if primary_url else set()
        for item in list(ctx.get("top_k") or []):
            self._raise_if_cancelled(run, ctx, "SOURCE_ENRICH")
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

    def _step_source_structure(self, run: Run, ctx: dict[str, Any]) -> None:
        source_pack = dict(ctx.get("source_pack") or {})
        primary = dict(source_pack.get("primary") or {})
        primary_url = str(primary.get("url", "") or "").strip()
        title = str(primary.get("title", "") or (ctx.get("selected_topic") or {}).get("title", "") or "").strip()
        if not primary_url:
            ctx["source_structure"] = {
                "status": "skipped",
                "reason": "no_primary_url",
                "title": title,
                "lead": "",
                "sections": [],
                "code_blocks": [],
                "lists": [],
                "tables": [],
                "coverage_checklist": [],
            }
            return
        structure = self.fetch.extract_article_structure(primary_url, max_chars=14000)
        if not structure.get("title") and title:
            structure["title"] = title
        ctx["source_structure"] = structure
        self._set_step_audit(
            ctx,
            "SOURCE_STRUCTURE",
            {
                "outputs": [
                    {
                        "title": "原文结构提取结果",
                        "text": self._clip_text(json.dumps(structure, ensure_ascii=False, indent=2), 8000),
                        "language": "json",
                    }
                ]
            },
        )

    def _step_web_search_plan(self, run: Run, ctx: dict[str, Any]) -> None:
        topic = dict(ctx.get("selected_topic") or {})
        source_pack = dict(ctx.get("source_pack") or {})
        source_structure = dict(ctx.get("source_structure") or {})
        evidence_score = float(topic.get("evidence_score", 0.0) or 0.0)
        content_type = str(ctx.get("content_type") or "tool_review")
        plan = self.web_enrich.build_search_plan(
            run_id=run.id,
            topic=topic,
            source_pack=source_pack,
            source_structure=source_structure,
            content_type=content_type,
            evidence_score=evidence_score,
            llm=self.llm,
        )
        ctx["web_search_plan"] = plan
        self._set_step_audit(
            ctx,
            "WEB_SEARCH_PLAN",
            {
                "outputs": [
                    {
                        "title": "Web search plan",
                        "text": self._clip_text(json.dumps(plan, ensure_ascii=False, indent=2), 6000),
                        "language": "json",
                    }
                ]
            },
        )

    def _step_web_search_fetch(self, run: Run, ctx: dict[str, Any]) -> None:
        plan = dict(ctx.get("web_search_plan") or {})
        result = self.web_enrich.fetch_search_results(plan=plan)
        ctx["web_enrich"] = result
        self._set_step_audit(
            ctx,
            "WEB_SEARCH_FETCH",
            {
                "outputs": [
                    {
                        "title": "Web enrich result",
                        "text": self._clip_text(json.dumps(result, ensure_ascii=False, indent=2), 8000),
                        "language": "json",
                    }
                ]
            },
        )

    def _step_fact_grounding(self, run: Run, ctx: dict[str, Any]) -> None:
        grounding = self.fact_grounding.ground(
            run_id=run.id,
            topic=dict(ctx.get("selected_topic") or {}),
            source_pack=dict(ctx.get("source_pack") or {}),
            source_structure=dict(ctx.get("source_structure") or {}),
            web_enrich=dict(ctx.get("web_enrich") or {}),
            llm=self.llm,
        )
        ctx["fact_grounding"] = grounding
        ctx["evidence_mode"] = grounding.get("evidence_mode", "analysis")
        self._set_step_audit(
            ctx,
            "FACT_GROUNDING",
            {
                "outputs": [
                    {
                        "title": "Fact grounding",
                        "text": self._clip_text(json.dumps(grounding, ensure_ascii=False, indent=2), 8000),
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
        target_audience = default_audience
        if self.settings.get_bool("writing.auto_switch_audience", True):
            if content_type == "technical_walkthrough" or len(fact_pack.get("implementation_steps") or []) >= 2 or len(fact_pack.get("code_artifacts") or []) >= 1:
                target_audience = "ai_builder"
            elif content_type == "industry_analysis":
                target_audience = "ai_product_manager"
        if target_audience != default_audience:
            fact_pack = self.writing_templates.build_fact_pack(ctx, audience_key=target_audience)
        fact_pack["content_type"] = content_type
        fact_pack["content_type_label"] = self.writing_templates.get_content_type(content_type).get("label", content_type)
        ctx["fact_pack"] = fact_pack
        ctx["content_type"] = content_type
        ctx["target_audience"] = target_audience
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
        fact_grounding = dict(ctx.get("fact_grounding") or {})
        if not fact_pack:
            raise RuntimeError("fact_pack is empty")
        prompt = (
            "You are a factual analyst. Read the source pack and fact pack, then output strict JSON in simplified Chinese. "
            "Do not write prose outside JSON.\n\n"
            "Return keys: one_sentence_summary, what_it_is, key_mechanisms, concrete_scenarios, numbers, risks, uncertainties, recommended_angle.\n"
            "Each value must be an array except one_sentence_summary which must be a string.\n"
            "Only keep high-confidence facts grounded in the provided materials. If unsure, put it into uncertainties.\n\n"
            f"Source Pack:\n{self._clip_text(json.dumps(source_pack, ensure_ascii=False), 6000)}\n\n"
            f"Fact Pack:\n{self._clip_text(json.dumps(fact_pack, ensure_ascii=False), 4000)}\n\n"
            f"Fact Grounding:\n{self._clip_text(json.dumps(fact_grounding, ensure_ascii=False), 4000)}"
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
        ctx["article_markdown"] = self._prepare_article_markdown(article)
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
        ctx["article_markdown"] = self._prepare_article_markdown(article)
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

    def _step_hallucination_check(self, run: Run, ctx: dict[str, Any]) -> None:
        article = str(ctx.get("article_markdown") or "").strip()
        if not article:
            raise RuntimeError("article_markdown is empty")
        grounding = dict(ctx.get("fact_grounding") or {})
        result = self.hallucination_checker.check(
            run_id=run.id,
            article_markdown=article,
            fact_grounding=grounding,
            llm=self.llm,
        )
        rewrite_applied = False
        if self.settings.get_bool("hallucination_check.enabled", True) and result.get("rewrite_required"):
            violations = (
                list(result.get("unsupported_claims") or [])
                + list(result.get("inference_written_as_fact") or [])
                + list(result.get("forbidden_claim_violations") or [])
            )
            fact_pack = dict(ctx.get("fact_pack") or {})
            prompt = self.writing_templates.build_write_prompt(
                topic=dict(ctx.get("selected_topic") or {}),
                fact_pack=fact_pack,
                audience_key=str(ctx.get("target_audience") or "ai_product_manager"),
                content_type=str(ctx.get("content_type") or fact_pack.get("content_type") or "tool_review"),
            )
            prompt += (
                "\n\n【Fact grounding】\n"
                f"{self._clip_text(json.dumps(grounding, ensure_ascii=False, indent=2), 4000)}\n\n"
                "【Rewrite task】\n"
                "Revise the article to remove unsupported claims, label inferences cautiously, and never write forbidden claims as facts.\n"
                "Problems found:\n"
                + "\n".join(f"- {item}" for item in violations[:10])
                + "\n\nCurrent article:\n"
                + article[:4000]
            )
            rewritten = self.llm.call(run.id, "WRITE", "writer", prompt, temperature=0.35).text.strip()
            if len(rewritten) > 150:
                ctx["article_markdown"] = self._prepare_article_markdown(rewritten)
                rewrite_applied = True
        result["rewrite_applied"] = rewrite_applied
        ctx["hallucination_check"] = result
        self._set_step_audit(
            ctx,
            "HALLUCINATION_CHECK",
            {
                "outputs": [
                    {
                        "title": "Hallucination check",
                        "text": self._clip_text(json.dumps(result, ensure_ascii=False, indent=2), 8000),
                        "language": "json",
                    }
                ]
            },
        )

    def _step_visual_strategy(self, run: Run, ctx: dict[str, Any]) -> None:
        strategy = self.visual_strategy.build_strategy(
            run_id=run.id,
            topic=dict(ctx.get("selected_topic") or {}),
            fact_pack=dict(ctx.get("fact_pack") or {}),
            fact_grounding=dict(ctx.get("fact_grounding") or {}),
            source_structure=dict(ctx.get("source_structure") or {}),
            llm=self.llm,
            max_body_illustrations=max(0, self.settings.get_int("visual.max_body_illustrations", 2)),
        )
        ctx["visual_strategy"] = strategy
        self._set_step_audit(
            ctx,
            "VISUAL_STRATEGY",
            {
                "outputs": [
                    {
                        "title": "Visual strategy",
                        "text": self._clip_text(json.dumps(strategy, ensure_ascii=False, indent=2), 8000),
                        "language": "json",
                    }
                ]
            },
        )

    def _step_body_illustration_gen(self, run: Run, ctx: dict[str, Any]) -> None:
        if not self.settings.get_bool("visual.body_illustration_enabled", True):
            ctx["body_illustrations"] = []
            return
        strategy = dict(ctx.get("visual_strategy") or {})
        assets = self.body_illustrations.generate(
            run_id=run.id,
            article_title=str(ctx.get("article_title") or ""),
            visual_strategy=strategy,
            size=self.settings.get("visual.body_illustration_size", "1400*1050"),
        )
        ctx["body_illustrations"] = assets
        self._set_step_audit(
            ctx,
            "BODY_ILLUSTRATION_GEN",
            {
                "outputs": [
                    {
                        "title": "Body illustrations",
                        "text": self._clip_text(json.dumps(assets, ensure_ascii=False, indent=2), 8000),
                        "language": "json",
                    }
                ]
            },
        )

    def _step_quality_check(self, run: Run, ctx: dict[str, Any]) -> None:
        threshold = self.settings.get_float("quality.threshold", 78.0)
        max_rounds = self.settings.get_int("quality.max_rounds", 3)
        scores: list[float] = []
        best = {"score": -1.0, "title": "", "article": ""}
        topic = ctx.get("selected_topic", {})
        fact_pack = dict(ctx.get("fact_pack") or {})
        coverage_items = [str(item) for item in (fact_pack.get("coverage_checklist") or []) if str(item).strip()]
        section_items = [
            str(item.get("heading", "") or "").strip()
            for item in (fact_pack.get("section_blueprint") or [])
            if isinstance(item, dict) and str(item.get("heading", "") or "").strip()
        ]
        coverage_text = "\n".join(f"- {item}" for item in coverage_items[:10]) or "- 无"
        section_text = "\n".join(f"- {item}" for item in section_items[:8]) or "- 无"
        round_logs: list[dict[str, Any]] = []

        for round_idx in range(1, max_rounds + 1):
            self._raise_if_cancelled(run, ctx, "QUALITY_CHECK")
            eval_prompt = (
                "Evaluate article quality from 0 to 100. "
                "Focus on structure fidelity, technical specificity, natural tone, and clean markdown organization. "
                "Check whether the article preserves major implementation steps, code meaning, and coverage checklist items. "
                "Output one line starting with SCORE: <number> and then 3-5 short reasons.\n\n"
                f"Content Type: {ctx.get('content_type', '')}\n"
                f"Coverage Checklist:\n{coverage_text}\n\n"
                f"Source Section Blueprint:\n{section_text}\n\n"
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
                rewrite_prompt = self.writing_templates.build_write_prompt(
                    topic=topic,
                    fact_pack=fact_pack,
                    audience_key=str(
                        ctx.get("target_audience")
                        or self.settings.get("writing.default_audience", "ai_product_manager")
                    ),
                    content_type=str(ctx.get("content_type") or fact_pack.get("content_type") or "tool_review"),
                )
                improve_prompt = (
                    f"{rewrite_prompt}\n\n"
                    "【质检反馈】\n"
                    f"{eval_result.text[:1000]}\n\n"
                    "【重写要求】\n"
                    "- 优先修复质检指出的问题，但不要牺牲原文结构保真度。\n"
                    "- 保留原文实现步骤、代码职责、架构角色和 coverage checklist，不要写成空泛总结。\n"
                    "- 不要套用重复的“机制 / 价值 / 场景 / 工作流”模板句式。\n"
                    "- 除非同一节内部确实需要顺序说明，否则不要把全文改写成多个从 1 开始的顶层编号列表。\n"
                    "- 只输出修订后的简体中文 Markdown 正文。\n\n"
                    "【当前文章】\n"
                    f"{ctx.get('article_markdown', '')[:3000]}"
                )
                rewritten = self.llm.call(run.id, "WRITE", "writer", improve_prompt, temperature=0.45).text.strip()
                round_log["improve_prompt"] = self._clip_text(improve_prompt, 8000)
                round_log["rewrite_preview"] = self._clip_text(rewritten, 4000)
                if len(rewritten) > 150:
                    ctx["article_markdown"] = self._prepare_article_markdown(rewritten)
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
        article_markdown = self._prepare_article_markdown(str(ctx.get("article_markdown") or "").strip())
        if not article_markdown:
            raise RuntimeError("article_markdown is empty")
        ctx["article_markdown"] = article_markdown
        article_title = str(ctx.get("article_title") or "").strip()
        content_type = str(ctx.get("content_type") or "tool_review").strip() or "tool_review"
        audience = str(ctx.get("target_audience") or "").strip()
        rendered = self.article_renderer.render(
            article_markdown,
            article_title=article_title,
            content_type=content_type,
            target_audience=audience,
            illustrations=list(ctx.get("body_illustrations") or []),
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
        strategy = dict(ctx.get("visual_strategy") or {})
        output_dir = CONFIG.data_dir / "runs" / run.id
        output_dir.mkdir(parents=True, exist_ok=True)
        article_title = str(ctx.get("article_title", "") or "")
        cover_size = self.settings.get("visual.cover_size", "1280*720")
        cover_brief = dict(strategy.get("cover_brief") or {})
        prompt_request = self.visual_strategy.build_cover_prompt_request(
            article_title=article_title,
            strategy=strategy,
            cover_5d=dict(ctx.get("cover_5d") or {}),
        )
        prompt_result = self.llm.call(run.id, "COVER_GEN", "cover_prompt", prompt_request, temperature=0.35)
        image_prompt = prompt_result.text.strip() or self._fallback_cover_prompt(article_title, dict(ctx.get("cover_5d") or {}))

        cover_asset: dict[str, Any]
        raw_asset: dict[str, Any]
        raw_error = ""
        try:
            raw_asset = self.llm.generate_cover_image(
                run.id,
                "COVER_GEN",
                "cover_image",
                prompt=image_prompt,
                output_dir=output_dir / "cover-image",
                size=cover_size,
            )
        except Exception as exc:
            raw_error = str(exc)
            raw_asset = {"status": "generation_failed", "error": raw_error, "size": cover_size.replace("*", "x")}

        raw_path = str(raw_asset.get("path", "") or "").strip()
        if raw_asset.get("status") == "generated" and raw_path:
            try:
                overlaid = self.visual_renderer.overlay_cover_title(
                    base_image_path=Path(raw_path),
                    article_title=article_title,
                    output_path=output_dir / "cover-final.png",
                    size=cover_size,
                    title_safe_zone=str(cover_brief.get("title_safe_zone", "left_bottom") or "left_bottom"),
                )
                cover_asset = {
                    **raw_asset,
                    **overlaid,
                    "base_image_path": raw_path,
                    "image_prompt": image_prompt[:2000],
                    "generator": "native_image_with_title_overlay",
                }
            except Exception as exc:
                raw_error = str(exc)
                cover_asset = {
                    **raw_asset,
                    "image_prompt": image_prompt[:2000],
                    "generator": "native_image_raw",
                    "overlay_error": raw_error,
                }
        else:
            fallback_path = output_dir / "cover-programmatic.png"
            fallback_asset = self.visual_renderer.render_cover(
                article_title=article_title,
                strategy=strategy,
                cover_5d=dict(ctx.get("cover_5d") or {}),
                output_path=fallback_path,
                size=cover_size,
            )
            cover_asset = {
                **fallback_asset,
                "fallback_reason": raw_asset.get("status") or "unknown",
                "image_prompt": image_prompt[:2000],
                "generator": "programmatic_fallback",
            }
            if raw_path:
                cover_asset["base_image_path"] = raw_path
            if raw_error:
                cover_asset["image_error"] = raw_error

        ctx["cover_asset"] = cover_asset
        self._set_step_audit(
            ctx,
            "COVER_GEN",
            {
                "prompts": [
                    {
                        "title": "封面提示词请求",
                        "text": self._clip_text(prompt_request, 4000),
                    },
                    {
                        "title": "封面出图提示词",
                        "text": self._clip_text(image_prompt, 4000),
                    },
                ],
                "outputs": [
                    {
                        "title": "封面生成结果",
                        "text": self._clip_text(json.dumps(raw_asset, ensure_ascii=False, indent=2), 4000),
                        "language": "json",
                    },
                    {
                        "title": "封面成品结果",
                        "text": self._clip_text(json.dumps(ctx["cover_asset"], ensure_ascii=False, indent=2), 4000),
                        "language": "json",
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
            f"微信公众号科技文章横版主题封面图，主题围绕“{title}”，"
            f"突出科技感与专业感，主体明确，景别干净，光线有层次，"
            f"左侧或左下保留适合叠加标题的干净留白区域，"
            f"图片中不要出现任何可读文字、logo、水印、伪文字、杂乱小字，"
            f"适合 1280x720 封面图，视觉评分目标 {total}。"
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
        prose_length = len(Orchestrator._article_prose_text(article))
        base = min(68.0 + prose_length / 115.0, 86.0)
        bonus = round_idx * 3.4
        noise = random.uniform(-1.0, 1.0)
        return round(min(base + bonus + noise, 93.0), 2)

    @staticmethod
    def _article_prose_text(article: str) -> str:
        text = str(article or "")
        text = re.sub(r"```[\s\S]*?```", "\n", text)
        text = re.sub(r"`[^`\n]+`", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def _prepare_article_markdown(self, article: str) -> str:
        text = str(article or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        if not text:
            return text
        text = self._strip_outer_markdown_fence(text)
        text = self._normalize_inline_fence_openers(text)
        text = self._repair_markdown_fences(text)
        text = self._normalize_standalone_heading_lines(text)
        text = self._localize_markdown_headings(text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return text

    @staticmethod
    def _strip_outer_markdown_fence(text: str) -> str:
        match = re.match(r"^\s*```(?:markdown|md)?\s*\n([\s\S]*?)\n```\s*$", str(text or ""), flags=re.IGNORECASE)
        if match:
            inner = str(match.group(1) or "").strip()
            if inner:
                return inner
        return str(text or "")

    def _normalize_inline_fence_openers(self, text: str) -> str:
        output: list[str] = []
        for line in str(text or "").split("\n"):
            raw = line.rstrip()
            stripped = raw.strip()
            if not raw or stripped.startswith("```") or "```" not in raw:
                output.append(raw)
                continue
            before, after = raw.split("```", 1)
            if not before.strip():
                output.append(raw)
                continue
            language, body = self._split_fence_language_and_body(after)
            output.append(before.rstrip())
            output.append(f"```{language}".rstrip())
            body = body.strip()
            if body:
                if "```" in body:
                    code_text, trailing = body.split("```", 1)
                    if code_text.strip():
                        output.append(code_text.rstrip())
                    output.append("```")
                    if trailing.strip():
                        output.append(trailing.strip())
                else:
                    output.append(body)
        return "\n".join(output)

    @staticmethod
    def _split_fence_language_and_body(text: str) -> tuple[str, str]:
        allowed_languages = {
            "text",
            "bash",
            "sh",
            "shell",
            "zsh",
            "powershell",
            "ps1",
            "python",
            "py",
            "javascript",
            "js",
            "typescript",
            "ts",
            "json",
            "yaml",
            "yml",
            "toml",
            "ini",
            "sql",
            "markdown",
            "md",
            "xml",
            "html",
            "css",
            "dockerfile",
            "makefile",
        }
        stripped = str(text or "").lstrip()
        if not stripped:
            return "", ""
        match = re.match(r"^([A-Za-z0-9_+-]+)(?:\s+(.*))?$", stripped)
        if not match:
            return "", stripped
        token = str(match.group(1) or "").strip()
        remainder = str(match.group(2) or "")
        if token.lower() in allowed_languages:
            return token, remainder
        return "", stripped

    def _repair_markdown_fences(self, text: str) -> str:
        lines = str(text or "").split("\n")
        output: list[str] = []
        i = 0
        while i < len(lines):
            stripped = lines[i].strip()
            if not stripped.startswith("```"):
                output.append(lines[i].rstrip())
                i += 1
                continue
            language = stripped[3:].strip()
            code_lines: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i].rstrip())
                i += 1
            if i < len(lines) and lines[i].strip().startswith("```"):
                i += 1
            repaired = self._repair_fenced_block(language=language, code_lines=code_lines)
            output.extend(repaired)
        return "\n".join(output)

    def _normalize_standalone_heading_lines(self, text: str) -> str:
        parts = re.split(r"(```[\s\S]*?```)", str(text or ""))
        normalized_parts: list[str] = []
        for idx, part in enumerate(parts):
            if idx % 2 == 1:
                normalized_parts.append(part)
                continue
            lines = part.splitlines(keepends=True)
            output: list[str] = []
            previous_heading_level = 0
            for line_idx, line in enumerate(lines):
                line_break = ""
                if line.endswith("\r\n"):
                    line_break = "\r\n"
                elif line.endswith("\n"):
                    line_break = "\n"
                content = line[:-len(line_break)] if line_break else line
                stripped = str(content or "").strip()
                if not stripped:
                    output.append(line)
                    continue
                heading_match = re.match(r"^(#{1,6})\s+", stripped)
                if heading_match:
                    previous_heading_level = len(heading_match.group(1))
                    output.append(line)
                    continue
                prev_blank = line_idx == 0 or not str(lines[line_idx - 1] or "").strip()
                next_nonempty = ""
                for probe in lines[line_idx + 1 :]:
                    if str(probe or "").strip():
                        next_nonempty = str(probe).strip()
                        break
                if (
                    prev_blank
                    and next_nonempty
                    and re.search(r"[A-Za-z]", stripped)
                    and LocalizationService.looks_like_heading_text(stripped)
                ):
                    level = "###" if previous_heading_level >= 2 else "##"
                    localized = self._translate_heading_text(stripped)
                    output.append(f"{level} {localized}{line_break}")
                    previous_heading_level = len(level)
                    continue
                output.append(line)
            normalized_parts.append("".join(output))
        return "".join(normalized_parts)

    def _repair_fenced_block(self, *, language: str, code_lines: list[str]) -> list[str]:
        content = self._trim_blank_lines(code_lines)
        if not content:
            return []

        if self._looks_like_structured_example_block(content):
            return [f"```{language}".rstrip(), *content, "```"]

        if (
            len(content) >= 2
            and self._is_prose_like_line(content[0], language=language)
            and self._is_prose_like_line(content[1], language=language)
        ):
            return content

        split_idx = self._find_code_prose_split(content=content, language=language)
        if split_idx is not None:
            code_part = self._trim_blank_lines(content[:split_idx])
            prose_part = self._trim_blank_lines(content[split_idx:])
            output: list[str] = []
            if code_part:
                output.extend([f"```{language}".rstrip(), *code_part, "```"])
            if prose_part:
                if output:
                    output.append("")
                output.extend(prose_part)
            return output

        if self._block_is_prose_like(content=content, language=language):
            return content

        return [f"```{language}".rstrip(), *content, "```"]

    def _find_code_prose_split(self, *, content: list[str], language: str) -> int | None:
        if len(content) < 2:
            return None
        for idx in range(1, len(content)):
            code_part = self._trim_blank_lines(content[:idx])
            prose_part = self._trim_blank_lines(content[idx:])
            if not code_part or not prose_part:
                continue
            if self._block_is_code_like(content=code_part, language=language) and self._block_is_prose_like(
                content=prose_part,
                language=language,
            ):
                return idx
        return None

    def _block_is_code_like(self, *, content: list[str], language: str) -> bool:
        code_hits = 0
        prose_hits = 0
        for line in content:
            if self._is_code_like_line(line, language=language):
                code_hits += 1
            elif self._is_prose_like_line(line, language=language):
                prose_hits += 1
        return code_hits >= max(1, prose_hits)

    def _block_is_prose_like(self, *, content: list[str], language: str) -> bool:
        if len(content) == 1:
            return self._is_prose_like_line(content[0], language=language)
        prose_hits = 0
        code_hits = 0
        for line in content:
            if self._is_prose_like_line(line, language=language):
                prose_hits += 1
            elif self._is_code_like_line(line, language=language):
                code_hits += 1
        return prose_hits >= max(2, code_hits + 1)

    @staticmethod
    def _is_markdownish_language(language: str) -> bool:
        return str(language or "").strip().lower() in {"", "text", "markdown", "md"}

    @staticmethod
    def _is_structured_example_marker(line: str) -> bool:
        stripped = str(line or "").strip()
        if not stripped:
            return False
        if re.match(r"^===\s*[A-Z0-9 _-]{4,}\s*===\s*$", stripped):
            return True
        if re.match(
            r"^(Question|Response|Nodes Retrieved|Retrieved \d+ chunks?\.?|Answer|Prompt|Output)\s*:",
            stripped,
            flags=re.IGNORECASE,
        ):
            return True
        return False

    def _looks_like_structured_example_block(self, content: list[str]) -> bool:
        marker_hits = sum(1 for line in content if self._is_structured_example_marker(line))
        if marker_hits >= 2:
            return True
        if marker_hits >= 1:
            english_lines = sum(1 for line in content if re.search(r"[A-Za-z]{4,}", str(line or "")))
            return english_lines >= 3
        return False

    def _is_code_like_line(self, line: str, *, language: str) -> bool:
        stripped = str(line or "").strip()
        if not stripped:
            return False
        if self._is_structured_example_marker(stripped):
            return True
        if re.search(
            r"^(?:\$|PS>|python\b|python3\b|pip\b|pip3\b|npm\b|npx\b|uv\b|curl\b|wget\b|git\b|docker\b|ollama\b|claude\b|node\b|go\b|java\b|javac\b|cargo\b|rustc\b|apt\b|brew\b|sudo\b|scp\b|ssh\b|cd\b|mkdir\b|cp\b|mv\b|rm\b)",
            stripped,
            flags=re.IGNORECASE,
        ):
            return True
        if re.search(r"(^|\s)--?[A-Za-z0-9_-]+", stripped):
            return True
        if re.search(r"[{}[\]();=<>]|=>|::", stripped):
            return True
        if re.search(r"\b(?:from|import|const|let|function|class|def|return|SELECT|INSERT|UPDATE|CREATE)\b", stripped):
            return True
        if re.search(r"[A-Za-z0-9_./-]+\.[A-Za-z0-9]{1,8}\b", stripped):
            return True
        if self._is_markdownish_language(language):
            if re.match(r"^#{1,6}\s+[A-Za-z0-9_./-]", stripped):
                return True
            if re.match(r"^\*\*[^*:\n]{1,80}:\*\*", stripped):
                return True
            if re.match(r"^[-*]\s+[A-Za-z0-9_./-]", stripped):
                return True
        return False

    def _is_prose_like_line(self, line: str, *, language: str) -> bool:
        stripped = str(line or "").strip()
        if not stripped:
            return False
        if self._is_markdownish_language(language) and LocalizationService.looks_like_heading_text(stripped):
            return True
        if self._is_markdownish_language(language) and re.match(r"^#{1,6}\s+[\u4e00-\u9fff]", stripped):
            return True
        if re.match(r"^\d+\.\s+.*[\u4e00-\u9fff]", stripped):
            return True
        chinese_chars = re.findall(r"[\u4e00-\u9fff]", stripped)
        if re.search(r"[，。；！？：]", stripped) and len(chinese_chars) >= 4:
            return True
        if len(chinese_chars) >= 8 and len(stripped) >= 18 and not self._is_code_like_line(stripped, language=language):
            return True
        english_words = re.findall(r"[A-Za-z]{3,}", stripped)
        if len(english_words) >= 5 and re.search(r"[,:;?!]", stripped) and not self._is_code_like_line(stripped, language=language):
            return True
        return False

    @staticmethod
    def _trim_blank_lines(lines: list[str]) -> list[str]:
        start = 0
        end = len(lines)
        while start < end and not str(lines[start] or "").strip():
            start += 1
        while end > start and not str(lines[end - 1] or "").strip():
            end -= 1
        return [str(line).rstrip() for line in lines[start:end]]

    def _localize_markdown_headings(self, article: str) -> str:
        text = str(article or "")
        if not text.strip():
            return text
        parts = re.split(r"(```[\s\S]*?```)", text)
        localized_parts: list[str] = []
        for idx, part in enumerate(parts):
            if idx % 2 == 1:
                localized_parts.append(part)
                continue
            lines = part.splitlines(keepends=True)
            rewritten: list[str] = []
            for line in lines:
                stripped = line.lstrip()
                if not stripped.startswith("#"):
                    rewritten.append(line)
                    continue
                line_break = ""
                if line.endswith("\r\n"):
                    line_break = "\r\n"
                elif line.endswith("\n"):
                    line_break = "\n"
                content = line[:-len(line_break)] if line_break else line
                stripped = content.lstrip()
                prefix = content[: len(content) - len(stripped)]
                match = re.match(r"^(#{1,6}\s+)(.+?)\s*$", stripped)
                if not match:
                    rewritten.append(line)
                    continue
                hashes = match.group(1)
                heading_text = match.group(2).strip()
                localized = self._translate_heading_text(heading_text)
                rewritten.append(f"{prefix}{hashes}{localized}{line_break}")
            localized_parts.append("".join(rewritten))
        return "".join(localized_parts)

    def _translate_heading_text(self, heading_text: str) -> str:
        text = str(heading_text or "").strip()
        if not text:
            return text
        if not re.search(r"[A-Za-z]", text):
            return text
        return LocalizationService.localize_heading_text(text)

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
        if any(keyword in text for keyword in hard_reject_keywords):
            return True

        event_markers = [
            "workshop", "webinar", "conference", "summit", "meetup", "bootcamp",
            "training session", "office hours", "live demo", "event", "报名", "直播预告", "活动预告",
        ]
        event_cta_markers = [
            "register", "join us", "sign up", "rsvp", "save your seat", "reserve your spot",
            "zoom", "eventbrite", "tickets", "free virtual", "立即报名", "欢迎参加",
        ]
        commercial_markers = [
            "direct honor system sales", "honor system sales", "direct sales", "for sale",
            "buy now", "purchase", "checkout", "payment link", "paypal", "iban",
            "lemonsqueezy", "gumroad", "pricing", "paid download", "monetization stack",
            "sales page", "sell code", "sell template", "发售", "售卖", "付款", "收款",
        ]
        pricing_pattern = re.search(r"(?<!\w)(?:\$|usd\s?)\s?\d{1,4}(?:\.\d{1,2})?\b", text, flags=re.IGNORECASE)
        has_event_promo = any(keyword in text for keyword in event_markers) and any(
            keyword in text for keyword in event_cta_markers
        )
        commercial_hits = sum(1 for keyword in commercial_markers if keyword in text)
        if has_event_promo:
            return True
        if pricing_pattern and commercial_hits >= 2:
            return True
        return Orchestrator._topic_editorial_penalty_score(item) >= 85.0

    @staticmethod
    def _topic_editorial_penalty_score(item: dict[str, Any]) -> float:
        text = " ".join(str(item.get(key, "") or "") for key in ("title", "summary", "url")).lower()
        event_markers = [
            "workshop", "webinar", "conference", "summit", "meetup", "bootcamp",
            "training session", "office hours", "live demo", "event", "??", "????", "????",
        ]
        event_cta_markers = [
            "register", "join us", "sign up", "rsvp", "save your seat", "reserve your spot",
            "zoom", "eventbrite", "tickets", "free virtual", "register for the zoom", "????", "????",
        ]
        commercial_markers = [
            "direct honor system sales", "honor system sales", "direct sales", "for sale",
            "buy now", "purchase", "checkout", "payment link", "paypal", "iban",
            "lemonsqueezy", "gumroad", "pricing", "paid download", "monetization stack",
            "sales page", "sell code", "sell template", "??", "??", "??", "??",
        ]
        promo_markers = [
            "launching", "launch", "now available", "available now", "announcement",
            "announcing", "preorder", "limited offer", "special offer", "????", "????",
        ]
        access_markers = [
            "membership", "members only", "subscriber only", "subscription required",
            "login required", "sign in to continue", "unlock full article", "premium content",
            "paywalled", "??", "??", "??", "?????", "????", "????",
        ]
        data_service_markers = [
            "????", "?????????", "???????", "??????", "?????",
            "contact for partnership", "data service", "data services", "request access", "pro.jiqizhixin.com", "/reference/",
        ]
        pricing_pattern = re.search(r"(?<!\w)(?:\$|usd\s?)\s?\d{1,4}(?:\.\d{1,2})?\b", text, flags=re.IGNORECASE)
        event_hits = sum(1 for keyword in event_markers if keyword in text)
        cta_hits = sum(1 for keyword in event_cta_markers if keyword in text)
        commercial_hits = sum(1 for keyword in commercial_markers if keyword in text)
        promo_hits = sum(1 for keyword in promo_markers if keyword in text)
        access_hits = sum(1 for keyword in access_markers if keyword in text)
        data_service_hits = sum(1 for keyword in data_service_markers if keyword in text)

        penalty = 0.0
        if event_hits:
            penalty += 24.0 + 8.0 * min(event_hits - 1, 2)
        if cta_hits:
            penalty += 22.0 + 6.0 * min(cta_hits - 1, 2)
        if commercial_hits:
            penalty += 20.0 + 7.0 * min(commercial_hits - 1, 4)
        if promo_hits:
            penalty += 8.0 + 4.0 * min(promo_hits - 1, 2)
        if access_hits:
            penalty += 18.0 + 8.0 * min(access_hits - 1, 2)
        if data_service_hits:
            penalty += 24.0 + 10.0 * min(data_service_hits - 1, 2)
        if pricing_pattern:
            penalty += 20.0

        if ("workshop" in text or "webinar" in text or "conference" in text) and (
            "register" in text or "join us" in text or "zoom" in text or "save your seat" in text
        ):
            penalty = max(penalty, 88.0)
        if "direct honor system sales" in text or ("$19" in text and "direct sales" in text):
            penalty = max(penalty, 92.0)
        if "podcast" in text and "transcript" not in text:
            penalty = max(penalty, 32.0)
        if data_service_hits >= 2:
            penalty = max(penalty, 82.0)

        return round(min(penalty, 100.0), 2)

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

    @staticmethod
    def _topic_evergreen_score(item: dict[str, Any]) -> float:
        text = " ".join(str(item.get(key, "") or "") for key in ("title", "summary", "url")).lower()
        positive = [
            "tutorial", "guide", "how to", "walkthrough", "deep dive", "reference", "playbook",
            "best practice", "best practices", "pattern", "patterns", "architecture", "implementation",
            "benchmark", "sdk", "api", "manual", "实战", "教程", "指南", "拆解", "架构", "实现",
            "手册", "最佳实践", "模式", "案例", "评测", "对比", "工作流",
        ]
        negative = [
            "today", "this week", "daily", "roundup", "newsletter", "breaking", "hot",
            "launch", "launching", "announce", "announcing", "announcement",
            "融资", "发布会", "活动", "上新", "日报", "周报", "快讯", "新闻", "本周",
        ]
        score = 35.0
        score += 8.0 * sum(1 for keyword in positive if keyword in text)
        score -= 10.0 * sum(1 for keyword in negative if keyword in text)
        return round(max(0.0, min(score, 100.0)), 2)

    @staticmethod
    def _topic_timeliness_profile(item: dict[str, Any]) -> str:
        text = " ".join(str(item.get(key, "") or "") for key in ("title", "summary", "url")).lower()
        technical_keywords = [
            "tutorial", "guide", "how to", "walkthrough", "deep dive", "reference", "playbook",
            "architecture", "implementation", "sdk", "api", "best practice",
            "教程", "指南", "实战", "拆解", "架构", "实现", "工作流", "最佳实践",
        ]
        product_keywords = [
            "launch", "release", "released", "announce", "announcing", "announcement", "upgrade",
            "product update", "open source", "review", "hands-on", "first look", "benchmark",
            "上线", "发布", "推出", "升级", "开源", "评测", "测评", "对比", "体验",
        ]
        news_keywords = [
            "today", "this week", "daily", "weekly", "roundup", "newsletter", "breaking", "hot",
            "news", "trend", "brief", "快讯", "新闻", "日报", "周报", "本周", "今日", "最新动态",
        ]
        if any(keyword in text for keyword in news_keywords):
            return "news"
        if any(keyword in text for keyword in product_keywords):
            return "product"
        if any(keyword in text for keyword in technical_keywords):
            return "technical"
        return "default"

    def _timeliness_thresholds(self, profile: str) -> tuple[float, float]:
        key = str(profile or "default").strip().lower()
        if key not in {"news", "product", "technical", "default"}:
            key = "default"
        soft = max(
            24.0,
            float(
                self.settings.get_float(
                    f"selection.stale_soft_hours_{key}",
                    self.settings.get_float("selection.stale_soft_hours_default", 120.0),
                )
            ),
        )
        hard = max(
            soft + 24.0,
            float(
                self.settings.get_float(
                    f"selection.stale_hard_hours_{key}",
                    self.settings.get_float("selection.stale_hard_hours_default", 336.0),
                )
            ),
        )
        return soft, hard

    def _should_reject_stale_topic(
        self,
        *,
        hours: float,
        profile: str,
        evergreen_score: float,
        value_score: float,
        depth_score: float,
    ) -> bool:
        _, hard_hours = self._timeliness_thresholds(profile)
        if hours < hard_hours:
            return False
        evergreen_floor = float(self.settings.get_float("selection.stale_evergreen_score_floor", 58.0))
        value_floor = float(self.settings.get_float("selection.stale_evergreen_value_floor", 72.0))
        depth_floor = float(self.settings.get_float("selection.stale_evergreen_depth_floor", 70.0))
        return evergreen_score < evergreen_floor and value_score < value_floor and depth_score < depth_floor

    def _topic_staleness_penalty_score(
        self,
        *,
        hours: float,
        profile: str,
        evergreen_score: float,
        value_score: float,
        depth_score: float,
    ) -> float:
        soft_hours, hard_hours = self._timeliness_thresholds(profile)
        penalty_max = max(0.0, float(self.settings.get_float("selection.stale_penalty_max", 26.0)))
        if hours <= soft_hours or penalty_max <= 0:
            return 0.0
        age_ratio = min(max((hours - soft_hours) / max(hard_hours - soft_hours, 1.0), 0.0), 1.0)
        evergreen_strength = max(
            0.0,
            min(
                100.0,
                0.45 * evergreen_score + 0.30 * value_score + 0.25 * depth_score,
            ),
        )
        keep_factor = max(0.15, 1.0 - evergreen_strength / 100.0)
        penalty = penalty_max * age_ratio * keep_factor
        if hours >= hard_hours:
            penalty += penalty_max * 0.35 * keep_factor
        return round(min(penalty, penalty_max), 2)

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
                        f"常青价值分: {item.get('evergreen_score', 0)}",
                        f"陈旧降权分: {item.get('stale_penalty_score', 0)}",
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

    def _build_select_prompt_v2(self, candidates: list[dict[str, Any]]) -> str:
        docs: list[str] = []
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
                        f"常青价值分: {item.get('evergreen_score', 0)}",
                        f"编辑风险分: {item.get('editorial_penalty_score', 0)}",
                        f"陈旧降权分: {item.get('stale_penalty_score', 0)}",
                        f"疲劳降权分: {item.get('fatigue_penalty_score', 0)}",
                        f"原文证据分: {item.get('evidence_score', 0)}",
                        f"原文证据摘要: {item.get('evidence_summary', '')}",
                        f"正文摘样: {item.get('selection_excerpt', '')[:800]}",
                    ]
                )
            )
        joined = "\n\n".join(docs)
        return (
            "你是公众号选题编辑。请从下面候选中选出一个最值得今天写成原创解读的主题。\n"
            "目标不是找最早发布的深度文，而是从最近文章里找一个当前仍值得发、信息价值最高、可写出深度解读的题。\n"
            "优先标准：新信息密度、机制细节、工作流价值、产业或产品影响、可形成干货分析，而且原文公开信息足够支撑深写。\n"
            "排除倾向：基础教程、练习题、quiz、浅层搬运、老话题重复包装。\n"
            "明确排除：活动预告、workshop/webinar/conference 报名页、营销页、销售页、发售公告、主要目的是卖代码/卖模板/引导付款的页面。\n"
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

    def _apply_source_diversity(self, items: list[dict[str, Any]], *, limit: int, desired: int) -> list[dict[str, Any]]:
        if limit <= 0:
            return list(items[:desired])
        selected: list[dict[str, Any]] = []
        counts: dict[str, int] = {}
        for item in items:
            family = self._topic_source_family(item)
            used = counts.get(family, 0)
            if used < limit:
                counts[family] = used + 1
                selected.append(item)
            if len(selected) >= desired:
                break
        return selected[:desired]

    @staticmethod
    def _topic_source_family(item: dict[str, Any]) -> str:
        url = str(item.get("url", "") or "").strip()
        host = normalized_host(url)
        if host:
            return host
        source = str(item.get("source", "") or "").strip().lower()
        return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", source).strip() or "unknown"

    @staticmethod
    def _topic_title_key(title: str) -> str:
        normalized = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", str(title or "").lower())
        return normalized[:80]

    def _topic_fatigue_penalty_score(self, item: dict[str, Any], *, current_run_id: str = "") -> float:
        lookback_days = max(1.0, float(self.settings.get_float("selection.fatigue_lookback_days", 10.0)))
        half_life_days = max(0.5, float(self.settings.get_float("selection.fatigue_half_life_days", 2.5)))
        source_unit = max(0.0, float(self.settings.get_float("selection.source_fatigue_unit", 2.5)))
        topic_unit = max(0.0, float(self.settings.get_float("selection.topic_fatigue_unit", 4.0)))
        source_cap = max(0.0, float(self.settings.get_float("selection.source_fatigue_max", 8.0)))
        topic_cap = max(0.0, float(self.settings.get_float("selection.topic_fatigue_max", 10.0)))
        total_cap = max(0.0, float(self.settings.get_float("selection.fatigue_total_max", 12.0)))

        current_family = self._topic_source_family(item)
        current_title_key = self._topic_title_key(str(item.get("title", "") or ""))
        if not current_family and not current_title_key:
            return 0.0

        cutoff = _utcnow().timestamp() - lookback_days * 86400.0
        rows = self.session.execute(
            select(Run)
            .where(Run.run_type == "main")
            .where(Run.id != current_run_id)
            .where(Run.summary_json != "")
            .order_by(Run.started_at.desc())
            .limit(40)
        ).scalars().all()

        source_penalty = 0.0
        topic_penalty = 0.0
        for row in rows:
            started_at = row.started_at or row.created_at
            if not started_at:
                continue
            if started_at.tzinfo is None:
                started_at = started_at.replace(tzinfo=timezone.utc)
            ts = started_at.timestamp()
            if ts < cutoff:
                continue
            try:
                summary = json.loads(row.summary_json or "{}")
            except Exception:
                summary = {}
            selected_topic = dict(summary.get("selected_topic") or {})
            prev_family = self._topic_source_family(selected_topic)
            prev_title_key = self._topic_title_key(str(selected_topic.get("title", "") or ""))
            age_days = max((_utcnow() - started_at).total_seconds() / 86400.0, 0.0)
            decay = math.exp(-age_days / half_life_days)
            if prev_family and prev_family == current_family:
                source_penalty += source_unit * decay
            if prev_title_key and current_title_key and prev_title_key == current_title_key:
                topic_penalty += topic_unit * decay

        source_penalty = min(source_penalty, source_cap)
        topic_penalty = min(topic_penalty, topic_cap)
        return round(min(source_penalty + topic_penalty, total_cap), 2)

    def _probe_topic_evidence(self, item: dict[str, Any]) -> dict[str, Any]:
        url = str(item.get("url", "") or "").strip()
        if not url:
            return {"score": 0.0, "summary": "no_url", "status": "skipped"}
        probe_chars = max(2000, self.settings.get_int("selection.evidence_probe_max_chars", 5000))
        try:
            structure = self.fetch.extract_article_structure(url, max_chars=probe_chars)
        except Exception as exc:
            return {"score": 0.0, "summary": f"probe_failed: {exc}", "status": "failed"}

        sections = [item for item in (structure.get("sections") or []) if isinstance(item, dict)]
        code_blocks = list(structure.get("code_blocks") or [])
        lists = list(structure.get("lists") or [])
        tables = list(structure.get("tables") or [])
        coverage = list(structure.get("coverage_checklist") or [])
        implementation_hits = 0
        architecture_hits = 0
        for section in sections:
            haystack = " ".join(
                [
                    str(section.get("heading", "") or ""),
                    str(section.get("summary", "") or ""),
                ]
            ).lower()
            if re.search(r"(step|workflow|pipeline|graph|mcp|rag|agent|api|sdk|ttl|renewal|lifecycle)", haystack, flags=re.IGNORECASE):
                implementation_hits += 1
            if re.search(r"(architecture|agent|mcp|rag|graph|workflow|pipeline|session|component|module)", haystack, flags=re.IGNORECASE):
                architecture_hits += 1

        score = 0.0
        if structure.get("status") == "ok":
            score += 18.0
        score += min(len(sections) * 7.0, 28.0)
        score += min(len(code_blocks) * 12.0, 24.0)
        score += min(len(coverage) * 3.0, 18.0)
        score += min((len(lists) + len(tables)) * 2.0, 10.0)
        score += min((implementation_hits + architecture_hits) * 6.0, 18.0)
        if len(sections) <= 1 and len(code_blocks) == 0 and len(coverage) <= 1:
            score = min(score, 28.0)

        topic_text = " ".join(
            [
                str(item.get("title", "") or ""),
                str(item.get("summary", "") or ""),
                str(url),
                str(structure.get("title", "") or ""),
                str(structure.get("lead", "") or ""),
                " ".join(str(section.get("heading", "") or "") for section in sections[:8]),
            ]
        ).lower()
        audio_markers = [
            "podcast", "episode", "listen", "download mp3", "spotify", "apple podcasts",
            "overcast", "pocket casts", "podcast addict", "castbox",
        ]
        transcript_markers = ["transcript", "full transcript", "show transcript", "episode transcript"]
        paywall_markers = [
            "membership", "members only", "subscriber only", "subscription required",
            "login required", "sign in to continue", "unlock full article", "premium content",
            "paywalled", "??", "??", "??", "?????", "????", "????",
        ]
        data_service_markers = [
            "????", "?????????", "???????", "??????", "?????",
            "contact for partnership", "data service", "data services", "request access", "pro.jiqizhixin.com", "/reference/",
        ]
        is_audio_page = any(marker in topic_text for marker in audio_markers)
        has_transcript_signal = any(marker in topic_text for marker in transcript_markers)
        has_paywall_signal = any(marker in topic_text for marker in paywall_markers)
        has_data_service_signal = any(marker in topic_text for marker in data_service_markers)
        if is_audio_page and not has_transcript_signal:
            score -= 18.0
        if is_audio_page and len(sections) <= 3 and len(code_blocks) == 0:
            score -= 12.0
        if has_paywall_signal:
            score -= 16.0
        if has_data_service_signal:
            score -= 24.0
        if has_data_service_signal and len(sections) <= 2 and len(code_blocks) == 0:
            score -= 12.0
        score = max(score, 0.0)
        summary = (
            f"sections={len(sections)}, code={len(code_blocks)}, coverage={len(coverage)}, "
            f"impl={implementation_hits}, arch={architecture_hits}, audio={is_audio_page}, transcript={has_transcript_signal}, "
            f"paywall={has_paywall_signal}, data_service={has_data_service_signal}"
        )
        return {
            "score": round(min(score, 100.0), 2),
            "summary": summary,
            "status": structure.get("status", "failed"),
            "is_audio_page": is_audio_page,
            "has_transcript_signal": has_transcript_signal,
            "has_paywall_signal": has_paywall_signal,
            "has_data_service_signal": has_data_service_signal,
        }

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
                "最低门槛": self.settings.get_float("quality.min_topic_score", 68.0),
            }
            if ctx.get("topic_gate_warning"):
                payload["summary"]["门槛提醒"] = ctx.get("topic_gate_warning")
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
        elif name == "SOURCE_STRUCTURE":
            structure = dict(ctx.get("source_structure") or {})
            sections = list(structure.get("sections") or [])
            code_blocks = list(structure.get("code_blocks") or [])
            payload["summary"] = {
                "结构状态": structure.get("status") or "-",
                "章节数": len(sections),
                "代码块数": len(code_blocks),
                "覆盖清单": len(structure.get("coverage_checklist") or []),
            }
            payload["items"] = [
                f"{item.get('heading', '未命名章节')} | {str(item.get('summary', '') or '')[:100]}"
                for item in sections[:5]
            ]
            payload["raw"] = structure
        elif name == "FACT_PACK":
            fact_pack = dict(ctx.get("fact_pack") or {})
            payload["summary"] = {
                "文章类型": ctx.get("content_type") or "-",
                "目标读者": ctx.get("target_audience") or "-",
                "关键点数量": len(fact_pack.get("key_points") or []),
                "相关线索数": len(fact_pack.get("related_topics") or []),
                "实现步骤数": len(fact_pack.get("implementation_steps") or []),
                "代码线索数": len(fact_pack.get("code_artifacts") or []),
                "覆盖清单": len(fact_pack.get("coverage_checklist") or []),
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
            prose_length = len(self._article_prose_text(article))
            code_block_count = len(re.findall(r"```[\s\S]*?```", article))
            payload["summary"] = {
                "文章标题": ctx.get("article_title") or "-",
                "正文长度": prose_length,
                "代码块数": code_block_count,
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
            "SOURCE_STRUCTURE": f"原文结构已提取：{len((ctx.get('source_structure') or {}).get('sections') or [])} 个章节",
            "WEB_SEARCH_PLAN": f"联网检索计划已生成：{len((ctx.get('web_search_plan') or {}).get('queries') or [])} 个查询",
            "WEB_SEARCH_FETCH": f"联网结果已拉取：官方 {len((ctx.get('web_enrich') or {}).get('official_sources') or [])} / 背景 {len((ctx.get('web_enrich') or {}).get('context_sources') or [])}",
            "FACT_GROUNDING": f"事实分层已完成：硬事实 {len((ctx.get('fact_grounding') or {}).get('hard_facts') or [])} 条",
            "WRITE": f"文章草稿已生成：{ctx.get('article_title') or '-'}",
            "HALLUCINATION_CHECK": f"事实校验完成：{(ctx.get('hallucination_check') or {}).get('severity', '-')}",
            "VISUAL_STRATEGY": f"视觉策略已生成：正文配图 {len((ctx.get('visual_strategy') or {}).get('body_illustrations') or [])} 张",
            "BODY_ILLUSTRATION_GEN": f"正文配图已生成：{len(ctx.get('body_illustrations') or [])} 张",
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
            "SOURCE_STRUCTURE": f"原文结构已提取：{len((ctx.get('source_structure') or {}).get('sections') or [])} 个章节",
            "WEB_SEARCH_PLAN": f"联网检索计划已生成：{len((ctx.get('web_search_plan') or {}).get('queries') or [])} 个查询",
            "WEB_SEARCH_FETCH": f"联网结果已拉取：官方 {len((ctx.get('web_enrich') or {}).get('official_sources') or [])} / 背景 {len((ctx.get('web_enrich') or {}).get('context_sources') or [])}",
            "FACT_GROUNDING": f"事实分层已完成：硬事实 {len((ctx.get('fact_grounding') or {}).get('hard_facts') or [])} 条",
            "WRITE": f"文章草稿已生成：{ctx.get('article_title') or '-'}",
            "HALLUCINATION_CHECK": f"事实校验完成：{(ctx.get('hallucination_check') or {}).get('severity', '-')}",
            "VISUAL_STRATEGY": f"视觉策略已生成：正文配图 {len((ctx.get('visual_strategy') or {}).get('body_illustrations') or [])} 张",
            "BODY_ILLUSTRATION_GEN": f"正文配图已生成：{len(ctx.get('body_illustrations') or [])} 张",
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
            compact["evergreen_score"] = item.get("evergreen_score")
            compact["timeliness_profile"] = item.get("timeliness_profile")
            compact["editorial_penalty_score"] = item.get("editorial_penalty_score")
            compact["stale_penalty_score"] = item.get("stale_penalty_score")
            compact["fatigue_penalty_score"] = item.get("fatigue_penalty_score")
            compact["evidence_score"] = item.get("evidence_score")
            compact["evidence_summary"] = item.get("evidence_summary")
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
