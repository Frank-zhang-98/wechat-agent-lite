from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.exc import OperationalError

from app.db import get_read_session, get_session
from app.core.config import CONFIG
from app.models import Run, RunStep, SourceHealthState
from app.schemas import ConfigUpdatePayload, RunActionPayload, TriggerRunPayload
from app import state
from app.services.fetch_service import FetchService
from app.services.metrics_service import get_step_timing_metrics, get_storage_metrics, get_token_metrics, get_token_overview
from app.services.model_pricing_service import get_pricing_catalog, sync_pricing_catalog
from app.services.orchestrator import Orchestrator
from app.services.proxy_link_service import ProxyLinkService
from app.services.settings_service import SettingsService
from app.services.source_maintenance_service import SourceMaintenanceService

router = APIRouter(prefix="/api")


def _parse_json_field(raw: str | None) -> dict:
    try:
        value = json.loads(raw or "{}")
        return value if isinstance(value, dict) else {"value": value}
    except Exception:
        return {}


def _load_run_hotspots(run_id: str) -> list[dict]:
    path = CONFIG.data_dir / "runs" / run_id / "hotspots.json"
    if not path.exists():
        return []
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return value if isinstance(value, list) else []


def _find_run_cover_path(run: Run, summary: dict | None = None) -> Path | None:
    summary = summary or {}
    cover_asset = summary.get("cover_asset") if isinstance(summary, dict) else {}
    cover_path = ""
    if isinstance(cover_asset, dict):
        cover_path = str(cover_asset.get("path", "") or "").strip()
    if cover_path:
        candidate = Path(cover_path)
        if candidate.exists() and candidate.is_file():
            return candidate

    run_dir = CONFIG.data_dir / "runs" / run.id
    if not run_dir.exists():
        return None
    for pattern in ("cover.png", "cover.jpg", "cover.jpeg", "cover.webp"):
        candidate = run_dir / pattern
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _build_selection_reason_preview(summary: dict) -> str:
    if not isinstance(summary, dict):
        return ""
    selected_topic = summary.get("selected_topic") if isinstance(summary.get("selected_topic"), dict) else {}
    top_k = summary.get("top_k") if isinstance(summary.get("top_k"), list) else []
    fact_compress = summary.get("fact_compress") if isinstance(summary.get("fact_compress"), dict) else {}

    candidates = [
        str(selected_topic.get("selection_reason", "") or "").strip(),
        str(selected_topic.get("rerank_reason", "") or "").strip(),
        str((top_k[0] or {}).get("selection_reason", "") if top_k else "").strip(),
        str((top_k[0] or {}).get("rerank_reason", "") if top_k else "").strip(),
        str(fact_compress.get("one_sentence_summary", "") or "").strip(),
    ]
    for text in candidates:
        if text:
            return text[:160]
    return ""


def _execute_run_background(run_id: str) -> None:
    with get_session() as session:
        Orchestrator(session).execute_existing(run_id)


def _build_source_health_snapshot(session) -> dict:
    fetch = FetchService()
    cfg = fetch.load_sources() or {}
    try:
        states = {
            state.source_key: state
            for state in session.execute(
                select(SourceHealthState).order_by(SourceHealthState.category.asc(), SourceHealthState.source_name.asc())
            ).scalars().all()
        }
    except Exception:
        states = {}
    sources: list[dict] = []
    for category in SourceMaintenanceService.CATEGORY_KEYS:
        for source in cfg.get(category, []) or []:
            if not isinstance(source, dict):
                continue
            source_key = SourceMaintenanceService._source_key(
                category=category,
                name=str(source.get("name", "") or ""),
            )
            state = states.get(source_key)
            sources.append(
                {
                    "source_key": source_key,
                    "category": category,
                    "name": str(source.get("name", "") or ""),
                    "enabled": bool(source.get("enabled", True)),
                    "mode": str(source.get("mode", "rss") or "rss"),
                    "weight": float(source.get("weight", 0.7) or 0.7),
                    "current_url": str(source.get("url", "") or ""),
                    "last_status": state.last_status if state else "unknown",
                    "last_http_status": state.last_http_status if state else None,
                    "last_error": state.last_error if state else "",
                    "last_action": state.last_action if state else "",
                    "last_action_reason": state.last_action_reason if state else "",
                    "last_candidate_url": state.last_candidate_url if state else "",
                    "consecutive_failures": int(state.consecutive_failures or 0) if state else 0,
                    "total_successes": int(state.total_successes or 0) if state else 0,
                    "total_failures": int(state.total_failures or 0) if state else 0,
                    "last_checked_at": state.last_checked_at.isoformat() if state and state.last_checked_at else None,
                    "last_success_at": state.last_success_at.isoformat() if state and state.last_success_at else None,
                    "last_failure_at": state.last_failure_at.isoformat() if state and state.last_failure_at else None,
                }
            )

    active_step = session.execute(
        select(RunStep).where(RunStep.name == "SOURCE_MAINTENANCE", RunStep.status == "running").order_by(RunStep.started_at.desc())
    ).scalars().first()
    active_maintenance = None
    if active_step:
        run = session.get(Run, active_step.run_id)
        active_maintenance = {
            "run_id": active_step.run_id,
            "run_type": run.run_type if run else "",
            "started_at": active_step.started_at.isoformat() if active_step.started_at else None,
            "details": _parse_json_field(active_step.details_json),
        }

    summary = {
        "total_sources": len(sources),
        "enabled_sources": sum(1 for item in sources if item["enabled"]),
        "healthy_sources": sum(1 for item in sources if item["last_status"] == "ok"),
        "attention_sources": sum(
            1 for item in sources if item["last_status"] not in {"ok", "unknown"} or item["consecutive_failures"] > 0
        ),
    }
    return {"summary": summary, "sources": sources, "active_maintenance": active_maintenance}


@router.get("/health")
def health() -> dict:
    return {"ok": True, "time": datetime.now(timezone.utc).isoformat()}


@router.get("/source-health")
def source_health() -> dict:
    try:
        with get_read_session() as session:
            snapshot = _build_source_health_snapshot(session)
    except OperationalError:
        snapshot = state.source_health_snapshot or {"summary": {}, "sources": [], "active_maintenance": None}
    else:
        state.source_health_snapshot = snapshot
    return snapshot


@router.get("/settings")
def get_settings() -> dict:
    with get_session() as session:
        service = SettingsService(session)
        service.ensure_defaults()
        session.flush()
        # Console currently runs only via SSH tunnel, so we return full values
        # to avoid accidental "******" overwrites on save.
        return {"values": service.as_dict(include_secrets=True)}


@router.put("/settings")
def update_settings(payload: ConfigUpdatePayload) -> dict:
    auto_proxy: dict[str, str] = {}
    if "proxy.share_link" in payload.values:
        share_link = (payload.values.get("proxy.share_link", "") or "").strip()
        if share_link:
            current_all_proxy = (payload.values.get("proxy.all_proxy", "") or "").strip()
            try:
                auto_proxy = ProxyLinkService.derive_settings(share_link=share_link, current_all_proxy=current_all_proxy)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=f"invalid proxy share link: {exc}") from exc
    with get_session() as session:
        service = SettingsService(session)
        service.ensure_defaults()
        session.flush()
        service.update_many(payload.values)
        if auto_proxy:
            service.update_many(auto_proxy)
    if state.scheduler:
        state.scheduler.reload_jobs()
    return {"ok": True, "updated": len(payload.values) + len(auto_proxy), "auto_proxy": auto_proxy}


@router.post("/proxy/parse")
def parse_proxy_share_link(payload: ConfigUpdatePayload) -> dict:
    share_link = (payload.values.get("proxy.share_link", "") or "").strip()
    if not share_link:
        raise HTTPException(status_code=400, detail="proxy.share_link is required")
    current_all_proxy = (payload.values.get("proxy.all_proxy", "") or "").strip()
    try:
        auto_proxy = ProxyLinkService.derive_settings(share_link=share_link, current_all_proxy=current_all_proxy)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"invalid proxy share link: {exc}") from exc
    return {"ok": True, "auto_proxy": auto_proxy}


@router.post("/runs/trigger")
def trigger_run(payload: TriggerRunPayload, background_tasks: BackgroundTasks) -> dict:
    source_url = (payload.source_url or "").strip()
    run_type = payload.run_type if payload.run_type in {"main", "health"} else "main"
    if source_url:
        parsed = urlparse(source_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise HTTPException(status_code=400, detail="source_url must be a valid http(s) URL")
        run_type = "manual_url"
    with get_session() as session:
        orch = Orchestrator(session)
        run = orch.create_run(run_type=run_type, trigger_source=payload.trigger_source, status="pending")
        if source_url:
            run.summary_json = json.dumps(
                {
                    "manual_input": {
                        "source_url": source_url,
                    }
                },
                ensure_ascii=False,
            )
            session.flush()
        run_id = run.id
    background_tasks.add_task(_execute_run_background, run_id)
    return {"ok": True, "run_id": run_id, "status": "pending"}


@router.post("/runs/{run_id}/action")
def run_action(run_id: str, payload: RunActionPayload) -> dict:
    with get_session() as session:
        orch = Orchestrator(session)
        try:
            run = orch.rerun_from_action(run_id=run_id, action=payload.action)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "new_run_id": run.id, "status": run.status}


@router.post("/runs/{run_id}/cancel")
def cancel_run(run_id: str) -> dict:
    with get_session() as session:
        run = session.get(Run, run_id)
        if not run:
            raise HTTPException(status_code=404, detail="run not found")
        if run.status in {"success", "failed", "partial_success", "cancelled"}:
            return {"ok": True, "run_id": run.id, "status": run.status, "message": "run already finished"}
    state.request_run_cancel(run_id)
    return {"ok": True, "run_id": run_id, "status": "cancelling", "message": "cancel requested"}


@router.get("/runs")
def list_runs(limit: int = Query(default=20, ge=1, le=200)) -> dict:
    with get_session() as session:
        rows = session.execute(select(Run).order_by(Run.started_at.desc()).limit(limit)).scalars().all()
        output = []
        for run in rows:
            failed_step = next((s for s in run.steps if s.status == "failed"), None)
            summary = _parse_json_field(run.summary_json)
            output.append(
                {
                    "id": run.id,
                    "run_type": run.run_type,
                    "status": run.status,
                    "started_at": run.started_at.isoformat() if run.started_at else None,
                    "finished_at": run.finished_at.isoformat() if run.finished_at else None,
                    "quality_score": run.quality_score,
                    "quality_attempts": run.quality_attempts,
                    "quality_fallback_used": run.quality_fallback_used,
                    "article_title": run.article_title,
                    "draft_status": run.draft_status,
                    "failed_step": failed_step.name if failed_step else "",
                    "selection_reason_preview": _build_selection_reason_preview(summary),
                }
            )
        return {"runs": output}


@router.get("/runs/{run_id}")
def get_run_detail(run_id: str) -> dict:
    with get_session() as session:
        run = session.get(Run, run_id)
        if not run:
            raise HTTPException(status_code=404, detail="run not found")
        steps = (
            session.execute(select(RunStep).where(RunStep.run_id == run.id).order_by(RunStep.id.asc())).scalars().all()
        )
        summary = _parse_json_field(run.summary_json)
        if isinstance(summary, dict) and not summary.get("fetched_items"):
            summary["fetched_items"] = _load_run_hotspots(run.id)
        cover_path = _find_run_cover_path(run, summary)
        return {
            "run": {
                "id": run.id,
                "run_type": run.run_type,
                "status": run.status,
                "trigger_source": run.trigger_source,
                "started_at": run.started_at.isoformat() if run.started_at else None,
                "finished_at": run.finished_at.isoformat() if run.finished_at else None,
                "error_message": run.error_message,
                "quality_score": run.quality_score,
                "quality_threshold": run.quality_threshold,
                "quality_attempts": run.quality_attempts,
                "quality_fallback_used": run.quality_fallback_used,
                "article_title": run.article_title,
                "article_markdown": run.article_markdown,
                "cover_url": f"/api/runs/{run.id}/cover" if cover_path else "",
                "draft_status": run.draft_status,
                "summary": summary,
            },
            "steps": [
                {
                    "name": step.name,
                    "status": step.status,
                    "retry_count": step.retry_count,
                    "started_at": step.started_at.isoformat() if step.started_at else None,
                    "finished_at": step.finished_at.isoformat() if step.finished_at else None,
                    "duration_ms": step.duration_ms,
                    "error_message": step.error_message,
                    "details": _parse_json_field(step.details_json),
                }
                for step in steps
            ],
        }


@router.get("/runs/{run_id}/cover")
def get_run_cover(run_id: str):
    with get_session() as session:
        run = session.get(Run, run_id)
        if not run:
            raise HTTPException(status_code=404, detail="run not found")
        summary = _parse_json_field(run.summary_json)
        cover_path = _find_run_cover_path(run, summary)
        if not cover_path:
            raise HTTPException(status_code=404, detail="cover not found")
        media_type = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }.get(cover_path.suffix.lower(), "application/octet-stream")
        return FileResponse(cover_path, media_type=media_type)


@router.get("/metrics/storage")
def storage_metrics() -> dict:
    return get_storage_metrics()


@router.get("/metrics/tokens")
def token_metrics(days: int = Query(default=7, ge=1, le=365)) -> dict:
    with get_session() as session:
        return get_token_metrics(session, days=days)


@router.get("/metrics/tokens/overview")
def token_overview() -> dict:
    with get_session() as session:
        return get_token_overview(session)


@router.get("/metrics/steps")
def step_metrics(days: int = Query(default=7, ge=1, le=365)) -> dict:
    with get_session() as session:
        return get_step_timing_metrics(session, days=days)


@router.get("/pricing")
def pricing_status() -> dict:
    catalog = get_pricing_catalog(auto_sync=True)
    return {"meta": catalog.get("meta", {}), "rules": catalog.get("rules", {})}


@router.post("/pricing/sync")
def pricing_sync() -> dict:
    catalog = sync_pricing_catalog()
    active = catalog or get_pricing_catalog(auto_sync=False)
    return {"ok": catalog is not None, "meta": active.get("meta", {})}


@router.post("/mail/test")
def mail_test() -> dict:
    with get_session() as session:
        service = SettingsService(session)
        service.ensure_defaults()
        mail = Orchestrator(session).mail
        subject_prefix = service.get("mail.subject_prefix", "[wechat-agent-lite]")
        subject = f"{subject_prefix} SMTP 测试邮件"
        html_body = (
            "<html><body style='font-family:Arial,\"Microsoft YaHei\",sans-serif;'>"
            "<h3>SMTP 测试成功</h3>"
            "<p>这是一封来自 wechat-agent-lite 控制台的测试邮件。</p>"
            "<p>如果你看到这封邮件，说明当前 SMTP 配置至少可以完成一次发送。</p>"
            "</body></html>"
        )
        try:
            result = mail.send_test(subject=subject, html_body=html_body)
        except Exception as exc:
            result = {"sent": False, "reason": str(exc)}
        return {"ok": bool(result.get("sent")), "result": result}
