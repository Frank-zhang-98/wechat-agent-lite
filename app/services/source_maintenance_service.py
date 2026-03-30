from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from typing import Any, Callable
from urllib.parse import urljoin, urlparse

import feedparser
import requests
from sqlalchemy.orm import Session

from app.models import SourceHealthState
from app.services.concurrency_utils import iter_host_limited_results, normalized_host
from app.services.fetch_service import FetchService
from app.services.llm_gateway import LLMGateway
from app.services.scrapling_fallback_service import ScraplingFallbackService
from app.services.settings_service import SettingsService


class SourceMaintenanceService:
    CATEGORY_KEYS = ("ai_companies", "tech_media", "tutorial_communities")
    FEED_SUFFIXES = (
        "feed",
        "feed/",
        "feed.xml",
        "rss",
        "rss/",
        "rss.xml",
        "index.xml",
        "atom.xml",
        "feeds/posts/default",
    )

    def __init__(
        self,
        session: Session,
        settings: SettingsService,
        fetch: FetchService,
        llm: LLMGateway,
        scrapling: ScraplingFallbackService | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ):
        self.session = session
        self.settings = settings
        self.fetch = fetch
        self.llm = llm
        self.scrapling = scrapling
        self.progress_callback = progress_callback

    def run(self, run_id: str) -> dict[str, Any]:
        report: dict[str, Any] = {
            "enabled": self.settings.get_bool("source_maintenance.enabled", True),
            "checked_sources": 0,
            "healthy_sources": 0,
            "failed_sources": 0,
            "changed_sources": 0,
            "manual_review_sources": 0,
            "llm_used": False,
            "actions": [],
            "audit": {"prompts": [], "outputs": []},
        }
        if not report["enabled"]:
            return report

        cfg = self.fetch.load_sources() or {}
        source_defs: list[tuple[str, dict[str, Any]]] = []
        for category in self.CATEGORY_KEYS:
            for source in cfg.get(category, []) or []:
                if isinstance(source, dict):
                    source_defs.append((category, source))
        total_enabled_sources = sum(1 for _, source in source_defs if bool(source.get("enabled", True)))
        inspect_workers = max(1, self.settings.get_int("source_maintenance.inspect_workers", 3))
        per_host_limit = max(1, self.settings.get_int("source_maintenance.per_host_limit", 1))
        inspection_jobs: list[dict[str, Any]] = []
        states_by_key: dict[str, SourceHealthState] = {}
        for index, (category, source) in enumerate(source_defs):
            state = self._get_state(source=source, category=category)
            enabled = bool(source.get("enabled", True))
            inspection_jobs.append(
                {
                    "index": index,
                    "source_key": state.source_key,
                    "name": str(source.get("name", "") or ""),
                    "category": category,
                    "enabled": enabled,
                    "mode": str(source.get("mode", "rss") or "rss").strip().lower(),
                    "weight": float(source.get("weight", 0.7) or 0.7),
                    "source_ref": source,
                }
            )
            states_by_key[state.source_key] = state

        inspections_by_index: dict[int, dict[str, Any]] = {}
        recent_sources: list[dict[str, Any]] = []
        healthy_count = 0
        failed_count = 0
        checked_count = 0
        inflight_sources = [job["name"] for job in inspection_jobs if job.get("enabled", True)][:inspect_workers]
        self._emit_progress(
            {
                "phase": "inspect",
                "run_id": run_id,
                "current_source": " / ".join(inflight_sources),
                "total_sources": total_enabled_sources,
                "checked_sources": 0,
                "healthy_sources": 0,
                "failed_sources": 0,
                "changed_sources": report["changed_sources"],
                "manual_review_sources": report["manual_review_sources"],
                "llm_candidate_sources": 0,
                "recent_sources": [],
                "recent_actions": [],
            }
        )
        for job, result, error in iter_host_limited_results(
            inspection_jobs,
            worker_fn=self._inspect_source_network,
            host_getter=lambda item: normalized_host(str((item.get("source_ref") or {}).get("url", "") or "")),
            max_workers=inspect_workers,
            per_host_limit=per_host_limit,
        ):
            if error is not None or result is None:
                inspection = {
                    **job,
                    "state": states_by_key[job["source_key"]],
                    "probe": {"ok": False, "reason": f"inspect_error: {error}", "mode": job.get("mode", "rss")},
                    "candidates": [],
                    "html_fallback": {},
                }
            else:
                inspection = dict(result)
                inspection["state"] = states_by_key[inspection["source_key"]]
            inspections_by_index[int(job["index"])] = inspection
            if inspection.get("enabled", True):
                checked_count += 1
                if inspection["probe"].get("ok"):
                    healthy_count += 1
                else:
                    failed_count += 1
                recent_sources.append(self._compact_progress_source(inspection))
                self._emit_progress(
                    {
                        "phase": "inspect",
                        "run_id": run_id,
                        "current_source": inspection["name"],
                        "total_sources": total_enabled_sources,
                        "checked_sources": checked_count,
                        "healthy_sources": healthy_count,
                        "failed_sources": failed_count,
                        "changed_sources": report["changed_sources"],
                        "manual_review_sources": report["manual_review_sources"],
                        "llm_candidate_sources": 0,
                        "recent_sources": recent_sources[-8:],
                        "recent_actions": [],
                    }
                )
        inspections = [inspections_by_index[idx] for idx in sorted(inspections_by_index)]

        enabled_inspections = [item for item in inspections if item.get("enabled", True)]
        failed_inspections = [item for item in enabled_inspections if not item["probe"].get("ok")]
        report["checked_sources"] = len(enabled_inspections)
        report["healthy_sources"] = sum(1 for item in enabled_inspections if item["probe"].get("ok"))
        report["failed_sources"] = len(failed_inspections)

        decisions: dict[str, dict[str, Any]] = {}
        max_llm_cases = max(0, self.settings.get_int("source_maintenance.max_llm_cases", 0))
        llm_review_items = self._select_llm_review_items(failed_inspections)
        report["llm_candidate_sources"] = len(llm_review_items)
        if llm_review_items:
            self._emit_progress(
                {
                    "phase": "llm_decision",
                    "run_id": run_id,
                    "current_source": llm_review_items[0]["name"],
                    "total_sources": report["checked_sources"],
                    "checked_sources": report["checked_sources"],
                    "healthy_sources": report["healthy_sources"],
                    "failed_sources": report["failed_sources"],
                    "changed_sources": report["changed_sources"],
                    "manual_review_sources": report["manual_review_sources"],
                    "llm_candidate_sources": len(llm_review_items),
                    "recent_sources": recent_sources[-8:],
                    "recent_actions": [],
                }
            )
        if llm_review_items and max_llm_cases > 0:
            decisions, audit = self._decision_actions(run_id=run_id, failed_items=llm_review_items)
            report["llm_used"] = bool(audit.get("prompts"))
            report["audit"] = audit

        changed = False
        recent_actions: list[dict[str, Any]] = []
        for item in inspections:
            action_result = self._resolve_action(item=item, llm_decision=decisions.get(item["source_key"]))
            if action_result["applied"]:
                changed = True
                report["changed_sources"] += 1
            if action_result["final_action"] == "manual_review":
                report["manual_review_sources"] += 1
            report["actions"].append(action_result)
            self._update_state(item=item, action=action_result)
            recent_actions.append(self._compact_progress_action(action_result))
            self._emit_progress(
                {
                    "phase": "apply",
                    "run_id": run_id,
                    "current_source": item["name"],
                    "total_sources": report["checked_sources"],
                    "checked_sources": report["checked_sources"],
                    "healthy_sources": report["healthy_sources"],
                    "failed_sources": report["failed_sources"],
                    "changed_sources": report["changed_sources"],
                    "manual_review_sources": report["manual_review_sources"],
                    "llm_candidate_sources": report.get("llm_candidate_sources", 0),
                    "recent_sources": recent_sources[-8:],
                    "recent_actions": recent_actions[-8:],
                }
            )

        if changed:
            self.fetch.save_sources(cfg)
        self._emit_progress(
            {
                "phase": "completed",
                "run_id": run_id,
                "current_source": "",
                "total_sources": report["checked_sources"],
                "checked_sources": report["checked_sources"],
                "healthy_sources": report["healthy_sources"],
                "failed_sources": report["failed_sources"],
                "changed_sources": report["changed_sources"],
                "manual_review_sources": report["manual_review_sources"],
                "llm_candidate_sources": report.get("llm_candidate_sources", 0),
                "recent_sources": recent_sources[-8:],
                "recent_actions": recent_actions[-8:],
            }
        )
        return report

    def _inspect_source_network(self, job: dict[str, Any]) -> dict[str, Any]:
        source = job.get("source_ref") or {}
        enabled = bool(job.get("enabled", True))
        mode = str(job.get("mode", "rss") or "rss").strip().lower()
        budget_seconds = max(5, self.settings.get_int("source_maintenance.per_source_budget_seconds", 20))
        deadline = time.monotonic() + budget_seconds if enabled else None
        probe = self._probe_source(source, deadline=deadline) if enabled else {"ok": False, "reason": "disabled", "mode": mode}
        discovery = {"feed_candidates": [], "html_fallback": {}}
        if enabled and not probe.get("ok") and not self._deadline_exceeded(deadline):
            discovery = self._discover_candidates(
                source=source,
                probe=probe,
                deadline=deadline,
            )
        return {
            "index": int(job.get("index", 0) or 0),
            "source_key": str(job.get("source_key", "") or ""),
            "name": str(job.get("name", "") or ""),
            "category": str(job.get("category", "") or ""),
            "enabled": enabled,
            "mode": mode,
            "weight": float(job.get("weight", source.get("weight", 0.7)) or 0.7),
            "source_ref": source,
            "probe": probe,
            "candidates": list(discovery.get("feed_candidates") or []),
            "html_fallback": dict(discovery.get("html_fallback") or {}),
        }

    def _select_llm_review_items(self, failed_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        for item in failed_items:
            state: SourceHealthState = item["state"]
            failure_streak = int(state.consecutive_failures or 0) + 1
            heuristic = self._heuristic_decision(item=item, failure_streak=failure_streak)
            review_reason = self._llm_review_reason(heuristic)
            if not review_reason:
                continue
            llm_item = dict(item)
            llm_item["_heuristic"] = heuristic
            llm_item["_llm_review_reason"] = review_reason
            selected.append(llm_item)
        return selected

    def _llm_review_reason(self, heuristic: dict[str, Any]) -> str:
        action = str(heuristic.get("action", "") or "").strip().lower()
        confidence = self._safe_float(heuristic.get("confidence", 0.0))
        threshold = max(
            0.0,
            min(1.0, self.settings.get_float("source_maintenance.llm_low_confidence_threshold", 0.7)),
        )
        if action == "manual_review":
            return "manual_review"
        if confidence < threshold:
            return "low_confidence"
        return ""

    def _decision_actions(self, run_id: str, failed_items: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        max_cases = max(0, self.settings.get_int("source_maintenance.max_llm_cases", 0))
        if max_cases <= 0 or not failed_items:
            return {}, {"prompts": [], "outputs": []}
        cases = [self._llm_case_payload(item) for item in failed_items[:max_cases]]
        prompt = (
            "你是 RSS/Atom/HTML 抓取源维护决策器。请基于失败原因、连续失败次数、候选 feed 列表和 HTML 列表页可用性，为每个源给出最稳妥的动作。\n"
            "以下案例是启发式无法稳妥自动决策或置信度较低的项目，请只在必要时调整启发式结论。\n"
            "动作只允许：keep、update_url、switch_to_html_list、disable、lower_weight、manual_review。\n"
            "约束：\n"
            "1. 只有候选 URL 已被验证为可解析 feed 时，才允许 update_url。\n"
            "2. 只有同域 HTML 列表页成功发现文章链接时，才允许 switch_to_html_list。\n"
            "3. 403/429 优先 lower_weight 或 manual_review，不要直接 disable。\n"
            "4. 404/410 连续失败较多且没有候选 feed/HTML 列表页时，才允许 disable。\n"
            "5. 输出必须是 JSON 数组，不要输出任何额外解释。\n"
            '每个元素格式：{"source_key":"...","action":"...","candidate_url":"...","reason":"...","confidence":0.0}。\n'
            f"待决策案例：\n{json.dumps(cases, ensure_ascii=False, indent=2)}"
        )
        audit = {
            "prompts": [{"title": "抓取源维护决策提示词", "text": prompt[:12000], "language": "text"}],
            "outputs": [],
        }
        decisions: dict[str, dict[str, Any]] = {}
        try:
            result = self.llm.call(run_id, "SOURCE_MAINTENANCE", "decision", prompt, temperature=0.1)
            audit["outputs"].append(
                {
                    "title": "抓取源维护决策输出",
                    "text": (result.text or "")[:12000],
                    "language": "json",
                }
            )
            decisions = self._parse_llm_decisions(result.text)
        except Exception as exc:
            audit["outputs"].append(
                {
                    "title": "抓取源维护决策输出",
                    "text": f"decision llm failed: {exc}",
                    "language": "text",
                }
            )
        return decisions, audit

    def _resolve_action(self, item: dict[str, Any], llm_decision: dict[str, Any] | None) -> dict[str, Any]:
        source = item["source_ref"]
        probe = item["probe"]
        state: SourceHealthState = item["state"]
        source_key = item["source_key"]
        previous_url = str(source.get("url", "") or "")
        previous_mode = str(source.get("mode", "rss") or "rss").strip().lower()
        previous_weight = float(source.get("weight", 0.7) or 0.7)
        failure_streak = int(state.consecutive_failures or 0) + (0 if probe.get("ok") else 1)
        heuristic = self._heuristic_decision(item=item, failure_streak=failure_streak)
        decision = llm_decision or {}
        proposed_action = self._normalize_action_name(str(decision.get("action", "") or heuristic["action"]))
        original_action = proposed_action
        candidate_url = str(decision.get("candidate_url", "") or heuristic.get("candidate_url", "") or "").strip()
        reason = str(decision.get("reason", "") or heuristic.get("reason", "") or "").strip()
        decision_source = "llm" if llm_decision else "heuristic"
        confidence = self._safe_float(decision.get("confidence", heuristic.get("confidence", 0.0)))
        proposed_action = self._safeguard_action(
            item=item,
            proposed_action=proposed_action,
            heuristic_action=str(heuristic.get("action", "manual_review") or "manual_review"),
            failure_streak=failure_streak,
        )
        if proposed_action != original_action and llm_decision:
            decision_source = "heuristic_safeguard"

        current_url = previous_url
        current_mode = previous_mode
        current_weight = previous_weight
        applied = False
        applied_action = ""
        final_action = proposed_action

        if probe.get("ok"):
            final_action = "keep"
        elif proposed_action == "update_url":
            candidate = self._pick_candidate(item=item, candidate_url=candidate_url)
            if candidate:
                final_action = "update_url"
                applied = self.settings.get_bool("source_maintenance.auto_apply", True)
                applied_action = "update_url" if applied else ""
                current_url = candidate["url"] if applied else previous_url
                current_mode = "rss"
                if applied:
                    source["url"] = candidate["url"]
                    source["mode"] = "rss"
            else:
                fallback = self._heuristic_decision(item=item, failure_streak=failure_streak)
                final_action = str(fallback.get("action", "manual_review") or "manual_review")
                applied, applied_action, current_url, current_mode, current_weight = self._apply_non_url_action(
                    source=source,
                    previous_url=previous_url,
                    previous_mode=previous_mode,
                    previous_weight=previous_weight,
                    action=final_action,
                    html_fallback=dict(item.get("html_fallback") or {}),
                )
                reason = reason or str(fallback.get("reason", "") or "")
                decision_source = "heuristic"
                candidate_url = ""
                confidence = self._safe_float(fallback.get("confidence", 0.0))
        else:
            applied, applied_action, current_url, current_mode, current_weight = self._apply_non_url_action(
                source=source,
                previous_url=previous_url,
                previous_mode=previous_mode,
                previous_weight=previous_weight,
                action=final_action,
                html_fallback=dict(item.get("html_fallback") or {}),
            )

        if final_action == "manual_review" and not reason:
            reason = str(heuristic.get("reason", "") or "需要人工复核")

        return {
            "source_key": source_key,
            "name": item["name"],
            "category": item["category"],
            "mode": item["mode"],
            "probe": self._compact_probe(probe),
            "candidate_count": len(item["candidates"]),
            "candidates": [self._compact_candidate(candidate) for candidate in item["candidates"][:3]],
            "html_fallback_page": str((item.get("html_fallback") or {}).get("page_url", "") or ""),
            "html_fallback_article_count": int((item.get("html_fallback") or {}).get("article_count", 0) or 0),
            "failure_streak": failure_streak,
            "previous_url": previous_url,
            "current_url": current_url,
            "previous_mode": previous_mode,
            "current_mode": current_mode,
            "previous_weight": previous_weight,
            "current_weight": current_weight,
            "final_action": final_action,
            "applied_action": applied_action,
            "applied": applied,
            "candidate_url": candidate_url,
            "reason": reason or final_action,
            "decision_source": decision_source,
            "confidence": confidence,
        }

    def _apply_non_url_action(
        self,
        *,
        source: dict[str, Any],
        previous_url: str,
        previous_mode: str,
        previous_weight: float,
        action: str,
        html_fallback: dict[str, Any],
    ) -> tuple[bool, str, str, str, float]:
        auto_apply = self.settings.get_bool("source_maintenance.auto_apply", True)
        current_url = previous_url
        current_mode = previous_mode
        current_weight = previous_weight
        applied = False
        applied_action = ""

        if action == "disable" and auto_apply:
            source["enabled"] = False
            applied = True
            applied_action = "disable"
        elif action == "lower_weight" and auto_apply:
            factor = max(0.1, min(1.0, self.settings.get_float("source_maintenance.lower_weight_factor", 0.85)))
            min_weight = max(0.05, self.settings.get_float("source_maintenance.min_weight", 0.2))
            current_weight = round(max(min_weight, previous_weight * factor), 3)
            source["weight"] = current_weight
            applied = current_weight != previous_weight
            applied_action = "lower_weight" if applied else ""
        elif action == "switch_to_html_list":
            page_url = str(html_fallback.get("page_url", "") or "").strip()
            article_count = int(html_fallback.get("article_count", 0) or 0)
            if auto_apply and page_url and article_count > 0:
                current_url = page_url
                current_mode = "html_list"
                source["url"] = page_url
                source["mode"] = "html_list"
                applied = True
                applied_action = "switch_to_html_list"
            else:
                action = "manual_review"
        else:
            action = "manual_review" if action not in {"keep", "manual_review"} else action

        return applied, applied_action, current_url, current_mode, current_weight

    def _heuristic_decision(self, item: dict[str, Any], failure_streak: int) -> dict[str, Any]:
        probe = item["probe"]
        candidates = item["candidates"]
        html_fallback = dict(item.get("html_fallback") or {})
        reason = str(probe.get("reason", "") or "")
        if candidates:
            return {
                "action": "update_url",
                "candidate_url": candidates[0]["url"],
                "reason": "发现同域可解析 feed，优先切换",
                "confidence": 0.88,
            }
        if int(html_fallback.get("article_count", 0) or 0) > 0 and str(item.get("mode", "rss")) != "html_list":
            return {
                "action": "switch_to_html_list",
                "candidate_url": str(html_fallback.get("page_url", "") or ""),
                "reason": "未找到可用 feed，但列表页可稳定提取文章链接",
                "confidence": 0.78,
            }
        if reason in {"http_404", "http_410"} and self._can_disable(reason=reason, failure_streak=failure_streak):
            return {"action": "disable", "reason": "连续 404/410 且无替代抓取路径，自动停用", "confidence": 0.82}
        if reason in {"http_403", "http_429"}:
            return {"action": "lower_weight", "reason": "命中限制或反爬，先降权观察", "confidence": 0.73}
        if reason.startswith("http_5") and failure_streak >= 2:
            return {"action": "lower_weight", "reason": "服务端不稳定，先降权观察", "confidence": 0.68}
        if reason in {"timeout", "request_error", "parse_error"} and failure_streak >= 3:
            return {"action": "lower_weight", "reason": "长期不稳定，先降权避免影响抓取", "confidence": 0.64}
        return {"action": "manual_review", "reason": "暂无安全的自动修复动作", "confidence": 0.45}

    def _safeguard_action(
        self,
        *,
        item: dict[str, Any],
        proposed_action: str,
        heuristic_action: str,
        failure_streak: int,
    ) -> str:
        probe = item["probe"]
        reason = str(probe.get("reason", "") or "")
        html_fallback = dict(item.get("html_fallback") or {})
        if proposed_action == "update_url" and not item["candidates"]:
            return self._normalize_action_name(heuristic_action)
        if proposed_action == "switch_to_html_list" and int(html_fallback.get("article_count", 0) or 0) <= 0:
            return self._normalize_action_name(heuristic_action)
        if proposed_action == "disable" and not self._can_disable(reason=reason, failure_streak=failure_streak):
            return self._normalize_action_name(heuristic_action)
        return proposed_action

    def _discover_candidates(self, source: dict[str, Any], probe: dict[str, Any], deadline: float | None = None) -> dict[str, Any]:
        current_url = str(source.get("url", "") or "").strip()
        if not current_url:
            return {"feed_candidates": [], "html_fallback": {}}

        candidate_urls: list[tuple[str, str]] = []
        html_fallback: dict[str, Any] = {}
        max_page_candidates = max(1, self.settings.get_int("source_maintenance.max_page_candidates", 2))
        page_candidates = self._candidate_pages(current_url)[:max_page_candidates]
        for idx, page_url in enumerate(page_candidates):
            if self._deadline_exceeded(deadline):
                break
            origin = "html_link_current" if idx == 0 else "html_link_page"
            for link in self._extract_feed_links_from_page(page_url, deadline=deadline):
                candidate_urls.append((link, origin))

            if self.scrapling and idx == 0 and not self._deadline_exceeded(deadline):
                discovered = self.scrapling.discover_page(
                    page_url,
                    max_articles=12,
                    timeout_seconds=self._bounded_timeout(
                        self.settings.get_int("source_maintenance.scrapling_timeout_seconds", 20),
                        deadline,
                    ),
                )
                for link in discovered.get("feed_links", []):
                    candidate_urls.append((str(link or "").strip(), "scrapling_feed"))
                if not html_fallback and discovered.get("articles"):
                    html_fallback = {
                        "page_url": str(discovered.get("page_url", "") or page_url),
                        "article_count": len(discovered.get("articles", [])),
                        "articles": list(discovered.get("articles", []))[:6],
                        "discovery_method": "scrapling",
                        "error": str(discovered.get("error", "") or ""),
                    }

        for guessed in self._guess_candidate_urls(current_url):
            if self._deadline_exceeded(deadline):
                break
            candidate_urls.append((guessed, "guess"))

        max_candidates = max(1, self.settings.get_int("source_maintenance.max_candidates", 8))
        deduped: list[tuple[str, str]] = []
        seen: set[str] = {current_url}
        for url, origin in candidate_urls:
            normalized = url.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append((normalized, origin))
            if len(deduped) >= max_candidates:
                break

        results: list[dict[str, Any]] = []
        for url, origin in deduped:
            if self._deadline_exceeded(deadline):
                break
            candidate_probe = self._probe_feed(url, deadline=deadline)
            if not candidate_probe.get("ok"):
                continue
            if not self._hosts_related(current_url, url):
                continue
            results.append(
                {
                    "url": url,
                    "origin": origin,
                    "entry_count": int(candidate_probe.get("entry_count", 0) or 0),
                    "latest_published": str(candidate_probe.get("latest_published", "") or ""),
                    "title": str(candidate_probe.get("feed_title", "") or ""),
                    "content_type": str(candidate_probe.get("content_type", "") or ""),
                    "status_code": candidate_probe.get("status_code"),
                }
            )
        results.sort(key=self._candidate_sort_key)
        return {"feed_candidates": results, "html_fallback": html_fallback}

    def _probe_source(self, source: dict[str, Any], deadline: float | None = None) -> dict[str, Any]:
        mode = str(source.get("mode", "rss") or "rss").strip().lower()
        url = str(source.get("url", "") or "").strip()
        if mode == "html_list":
            return self._probe_html_list(url, deadline=deadline)
        probe = self._probe_feed(url, deadline=deadline)
        probe["mode"] = "rss"
        return probe

    def _probe_feed(self, url: str, deadline: float | None = None) -> dict[str, Any]:
        if self._deadline_exceeded(deadline):
            return {"ok": False, "url": url, "reason": "budget_exceeded", "mode": "rss"}
        timeout = self._bounded_timeout(self.settings.get_int("source_maintenance.probe_timeout_seconds", 12), deadline)
        try:
            response = self.fetch._request(url, timeout=timeout)
        except (requests.exceptions.ReadTimeout, requests.exceptions.Timeout):
            return {"ok": False, "url": url, "reason": "timeout", "mode": "rss"}
        except Exception:
            return {"ok": False, "url": url, "reason": "request_error", "mode": "rss"}

        status_code = int(response.status_code)
        content_type = str(response.headers.get("Content-Type", "") or "").lower()
        if status_code >= 400:
            reason = f"http_{status_code}" if status_code not in {500, 501, 502, 503, 504} else "http_5xx"
            return {
                "ok": False,
                "url": url,
                "status_code": status_code,
                "reason": reason,
                "content_type": content_type,
                "mode": "rss",
            }

        parsed = feedparser.parse(response.content)
        version = str(getattr(parsed, "version", "") or "")
        entries = list(getattr(parsed, "entries", []) or [])
        latest_published = self._latest_entry_time(entries)
        feed_title = str((getattr(parsed, "feed", {}) or {}).get("title", "") or "")
        looks_like_feed = bool(entries) or bool(version) or any(token in content_type for token in ("rss", "atom", "xml"))
        if not looks_like_feed:
            return {
                "ok": False,
                "url": url,
                "status_code": status_code,
                "reason": "parse_error",
                "content_type": content_type,
                "mode": "rss",
            }
        return {
            "ok": True,
            "url": url,
            "status_code": status_code,
            "reason": "",
            "content_type": content_type,
            "feed_title": feed_title,
            "entry_count": len(entries),
            "latest_published": latest_published,
            "mode": "rss",
        }

    def _probe_html_list(self, url: str, deadline: float | None = None) -> dict[str, Any]:
        if self._deadline_exceeded(deadline):
            return {"ok": False, "url": url, "reason": "budget_exceeded", "mode": "html_list"}
        timeout = self._bounded_timeout(self.settings.get_int("source_maintenance.probe_timeout_seconds", 12), deadline)
        try:
            response = self.fetch._request(url, timeout=timeout)
        except (requests.exceptions.ReadTimeout, requests.exceptions.Timeout):
            return {"ok": False, "url": url, "reason": "timeout", "mode": "html_list"}
        except Exception:
            return {"ok": False, "url": url, "reason": "request_error", "mode": "html_list"}

        status_code = int(response.status_code)
        content_type = str(response.headers.get("Content-Type", "") or "").lower()
        if status_code >= 400:
            reason = f"http_{status_code}" if status_code not in {500, 501, 502, 503, 504} else "http_5xx"
            return {
                "ok": False,
                "url": url,
                "status_code": status_code,
                "reason": reason,
                "content_type": content_type,
                "mode": "html_list",
            }

        if self.scrapling:
            discovered = self.scrapling.discover_page(
                url,
                max_articles=8,
                timeout_seconds=self._bounded_timeout(
                    self.settings.get_int("source_maintenance.scrapling_timeout_seconds", 20),
                    deadline,
                ),
            )
            article_count = len(discovered.get("articles", []))
            if article_count > 0:
                return {
                    "ok": True,
                    "url": str(discovered.get("page_url", "") or url),
                    "status_code": status_code,
                    "reason": "",
                    "content_type": content_type,
                    "article_count": article_count,
                    "mode": "html_list",
                }

        feed_links = self._extract_feed_links_from_page(url, deadline=deadline)
        if feed_links:
            return {
                "ok": True,
                "url": url,
                "status_code": status_code,
                "reason": "",
                "content_type": content_type,
                "article_count": len(feed_links),
                "mode": "html_list",
            }
        return {
            "ok": False,
            "url": url,
            "status_code": status_code,
            "reason": "parse_error",
            "content_type": content_type,
            "mode": "html_list",
        }

    def _extract_feed_links_from_page(self, page_url: str, deadline: float | None = None) -> list[str]:
        if self._deadline_exceeded(deadline):
            return []
        timeout = self._bounded_timeout(self.settings.get_int("source_maintenance.probe_timeout_seconds", 12), deadline)
        try:
            response = self.fetch._request(page_url, timeout=timeout)
        except Exception:
            return []
        if response.status_code >= 400:
            return []
        content_type = str(response.headers.get("Content-Type", "") or "").lower()
        if "html" not in content_type and "text" not in content_type:
            return []
        text = response.text or ""
        links: list[str] = []
        patterns = [
            r'<link[^>]+type=["\']application/(?:rss|atom)\+xml["\'][^>]+href=["\']([^"\']+)["\']',
            r'<a[^>]+href=["\']([^"\']*(?:rss|feed|atom)[^"\']*)["\']',
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                href = str(match.group(1) or "").strip()
                if href:
                    links.append(urljoin(page_url, href))
        return links

    def _candidate_pages(self, current_url: str) -> list[str]:
        parsed = urlparse(current_url)
        if not parsed.scheme or not parsed.netloc:
            return []
        pages: list[str] = [f"{parsed.scheme}://{parsed.netloc}/"]
        path_parts = [part for part in parsed.path.split("/") if part]
        if path_parts:
            while path_parts and path_parts[-1].lower() in {"rss", "feed", "atom", "index.xml", "rss.xml", "feed.xml", "atom.xml"}:
                path_parts.pop()
            if path_parts:
                page_path = "/" + "/".join(path_parts)
                pages.insert(0, f"{parsed.scheme}://{parsed.netloc}{page_path}")
        deduped: list[str] = []
        seen: set[str] = set()
        for page in pages:
            if page not in seen:
                seen.add(page)
                deduped.append(page)
        return deduped[:3]

    @staticmethod
    def _deadline_exceeded(deadline: float | None) -> bool:
        return deadline is not None and time.monotonic() >= deadline

    @staticmethod
    def _bounded_timeout(default_timeout: int | float, deadline: float | None) -> float:
        if deadline is None:
            return max(1.0, float(default_timeout))
        remaining = deadline - time.monotonic()
        return max(1.0, min(float(default_timeout), remaining))

    def _guess_candidate_urls(self, current_url: str) -> list[str]:
        parsed = urlparse(current_url)
        if not parsed.scheme or not parsed.netloc:
            return []
        base_paths = [""]
        path_parts = [part for part in parsed.path.split("/") if part]
        if path_parts:
            trimmed = path_parts[:]
            while trimmed and trimmed[-1].lower() in {"rss", "feed", "atom", "index.xml", "rss.xml", "feed.xml", "atom.xml"}:
                trimmed.pop()
            if trimmed:
                base_paths.append("/".join(trimmed))

        urls: list[str] = []
        for base_path in base_paths:
            prefix = f"{parsed.scheme}://{parsed.netloc}/"
            if base_path:
                prefix = f"{prefix}{base_path.strip('/')}/"
            for suffix in self.FEED_SUFFIXES:
                urls.append(urljoin(prefix, suffix))
        return urls

    def _pick_candidate(self, item: dict[str, Any], candidate_url: str) -> dict[str, Any] | None:
        candidates = item["candidates"]
        if candidate_url:
            for candidate in candidates:
                if candidate["url"] == candidate_url and self._hosts_related(item["probe"].get("url", ""), candidate["url"]):
                    return candidate
        return candidates[0] if candidates else None

    def _update_state(self, item: dict[str, Any], action: dict[str, Any]) -> None:
        state: SourceHealthState = item["state"]
        probe = item["probe"]
        now = datetime.now(timezone.utc)
        state.source_name = item["name"]
        state.category = item["category"]
        state.enabled = bool(item["source_ref"].get("enabled", True))
        state.weight = float(item["source_ref"].get("weight", item["weight"]) or item["weight"])
        state.current_url = str(item["source_ref"].get("url", "") or "")
        state.last_checked_at = now
        state.last_http_status = probe.get("status_code")
        state.last_error = str(probe.get("reason", "") or "")
        state.last_action = str(action.get("applied_action") or action.get("final_action") or "")
        state.last_action_reason = str(action.get("reason", "") or "")
        state.last_candidate_url = str(action.get("candidate_url", "") or "")

        if probe.get("ok"):
            state.consecutive_failures = 0
            state.total_successes = int(state.total_successes or 0) + 1
            state.last_status = "ok"
            state.last_success_at = now
            return

        state.total_failures = int(state.total_failures or 0) + 1
        if action.get("final_action") in {"update_url", "switch_to_html_list"} and action.get("applied"):
            state.consecutive_failures = 0
            if action.get("final_action") == "update_url":
                state.last_status = "updated_url"
            else:
                state.last_status = str(action.get("final_action") or "recovered")
            state.last_success_at = now
        else:
            state.consecutive_failures = int(state.consecutive_failures or 0) + 1
            state.last_status = str(action.get("final_action") or probe.get("reason") or "failed")
            state.last_failure_at = now

    def _get_state(self, source: dict[str, Any], category: str) -> SourceHealthState:
        source_key = self._source_key(category=category, name=str(source.get("name", "") or ""))
        for obj in self.session.new:
            if isinstance(obj, SourceHealthState) and obj.source_key == source_key:
                obj.source_name = str(source.get("name", "") or "")
                obj.category = category
                obj.current_url = str(source.get("url", "") or "")
                obj.enabled = bool(source.get("enabled", True))
                obj.weight = float(source.get("weight", 0.7) or 0.7)
                return obj
        state = self.session.get(SourceHealthState, source_key)
        if state:
            state.source_name = str(source.get("name", "") or "")
            state.category = category
            state.current_url = str(source.get("url", "") or "")
            state.enabled = bool(source.get("enabled", True))
            state.weight = float(source.get("weight", 0.7) or 0.7)
            return state
        state = SourceHealthState(
            source_key=source_key,
            source_name=str(source.get("name", "") or ""),
            category=category,
            current_url=str(source.get("url", "") or ""),
            enabled=bool(source.get("enabled", True)),
            weight=float(source.get("weight", 0.7) or 0.7),
        )
        self.session.add(state)
        return state

    def _can_disable(self, *, reason: str, failure_streak: int) -> bool:
        threshold = self.settings.get_int("source_maintenance.auto_disable_failures", 3)
        return reason in {"http_404", "http_410"} and failure_streak >= threshold

    @staticmethod
    def _source_key(*, category: str, name: str) -> str:
        cleaned = name.strip().lower()
        slug = re.sub(r"[^a-z0-9]+", "-", cleaned).strip("-")
        if not slug:
            digest = hashlib.sha1(name.strip().encode("utf-8")).hexdigest()[:10]
            slug = f"source-{digest}"
        return f"{category}:{slug}"

    @staticmethod
    def _latest_entry_time(entries: list[Any]) -> str:
        latest: datetime | None = None
        for entry in entries:
            published = getattr(entry, "published_parsed", None) or entry.get("published_parsed")
            if not published:
                continue
            candidate = datetime(*published[:6], tzinfo=timezone.utc)
            if latest is None or candidate > latest:
                latest = candidate
        return latest.isoformat() if latest else ""

    @classmethod
    def _candidate_sort_key(cls, candidate: dict[str, Any]) -> tuple[int, int, int]:
        origin_priority = {"html_link_current": 0, "html_link_page": 1, "scrapling_feed": 2, "guess": 3}
        published = str(candidate.get("latest_published", "") or "")
        published_score = int(datetime.fromisoformat(published).timestamp()) if published else 0
        return (
            origin_priority.get(str(candidate.get("origin", "") or ""), 9),
            -int(candidate.get("entry_count", 0) or 0),
            -published_score,
        )

    @staticmethod
    def _normalize_action_name(value: str) -> str:
        lowered = value.strip().lower()
        if lowered in {"keep", "update_url", "switch_to_html_list", "disable", "lower_weight", "manual_review"}:
            return lowered
        return "manual_review"

    def _emit_progress(self, payload: dict[str, Any]) -> None:
        if not self.progress_callback:
            return
        try:
            self.progress_callback(payload)
        except Exception:
            return

    @staticmethod
    def _compact_progress_source(item: dict[str, Any]) -> dict[str, Any]:
        probe = dict(item.get("probe") or {})
        html_fallback = dict(item.get("html_fallback") or {})
        return {
            "source_key": item.get("source_key", ""),
            "name": item.get("name", ""),
            "category": item.get("category", ""),
            "enabled": bool(item.get("enabled", True)),
            "probe_ok": bool(probe.get("ok")),
            "reason": probe.get("reason", ""),
            "status_code": probe.get("status_code"),
            "candidate_count": len(item.get("candidates") or []),
            "html_article_count": int(html_fallback.get("article_count", 0) or 0),
            "mode": item.get("mode", ""),
        }

    @staticmethod
    def _compact_progress_action(action: dict[str, Any]) -> dict[str, Any]:
        return {
            "source_key": action.get("source_key", ""),
            "name": action.get("name", ""),
            "final_action": action.get("final_action", ""),
            "applied_action": action.get("applied_action", ""),
            "decision_source": action.get("decision_source", ""),
            "reason": action.get("reason", ""),
            "confidence": action.get("confidence", 0.0),
        }

    @staticmethod
    def _parse_llm_decisions(text: str) -> dict[str, dict[str, Any]]:
        if not text.strip():
            return {}
        snippets = []
        array_match = re.search(r"\[[\s\S]*\]", text)
        object_match = re.search(r"\{[\s\S]*\}", text)
        if array_match:
            snippets.append(array_match.group(0))
        if object_match:
            snippets.append(object_match.group(0))
        for snippet in snippets:
            try:
                data = json.loads(snippet)
            except Exception:
                continue
            if isinstance(data, dict):
                data = data.get("actions", [])
            if not isinstance(data, list):
                continue
            decisions: dict[str, dict[str, Any]] = {}
            for item in data:
                if not isinstance(item, dict):
                    continue
                source_key = str(item.get("source_key", "") or "").strip()
                if not source_key:
                    continue
                decisions[source_key] = {
                    "action": str(item.get("action", "") or "").strip(),
                    "candidate_url": str(item.get("candidate_url", "") or "").strip(),
                    "reason": str(item.get("reason", "") or "").strip(),
                    "confidence": item.get("confidence", 0.0),
                }
            if decisions:
                return decisions
        return {}

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            return round(float(value), 3)
        except Exception:
            return 0.0

    @staticmethod
    def _normalize_host(url: str) -> str:
        host = urlparse(url).netloc.lower().split("@")[-1]
        if ":" in host:
            host = host.split(":", 1)[0]
        if host.startswith("www."):
            host = host[4:]
        return host

    @classmethod
    def _hosts_related(cls, current_url: str, candidate_url: str) -> bool:
        host_a = cls._normalize_host(current_url)
        host_b = cls._normalize_host(candidate_url)
        if not host_a or not host_b:
            return False
        return host_a == host_b or host_a.endswith(f".{host_b}") or host_b.endswith(f".{host_a}")

    @staticmethod
    def _compact_probe(probe: dict[str, Any]) -> dict[str, Any]:
        return {
            "ok": bool(probe.get("ok")),
            "status_code": probe.get("status_code"),
            "reason": probe.get("reason", ""),
            "content_type": probe.get("content_type", ""),
            "mode": probe.get("mode", ""),
        }

    @staticmethod
    def _compact_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
        return {
            "url": candidate.get("url", ""),
            "origin": candidate.get("origin", ""),
            "entry_count": candidate.get("entry_count", 0),
            "latest_published": candidate.get("latest_published", ""),
        }

    @staticmethod
    def _llm_case_payload(item: dict[str, Any]) -> dict[str, Any]:
        state: SourceHealthState = item["state"]
        probe = item["probe"]
        html_fallback = dict(item.get("html_fallback") or {})
        heuristic = dict(item.get("_heuristic") or {})
        return {
            "source_key": item["source_key"],
            "name": item["name"],
            "category": item["category"],
            "mode": item["mode"],
            "current_url": item["source_ref"].get("url", ""),
            "weight": item["weight"],
            "failure_reason": probe.get("reason", ""),
            "status_code": probe.get("status_code"),
            "consecutive_failures_if_keep_failing": int(state.consecutive_failures or 0) + 1,
            "heuristic_action": heuristic.get("action", ""),
            "heuristic_confidence": heuristic.get("confidence", 0.0),
            "llm_review_reason": item.get("_llm_review_reason", ""),
            "candidates": [SourceMaintenanceService._compact_candidate(candidate) for candidate in item["candidates"][:4]],
            "html_fallback": {
                "page_url": html_fallback.get("page_url", ""),
                "article_count": html_fallback.get("article_count", 0),
            },
        }
