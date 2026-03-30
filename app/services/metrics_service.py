from __future__ import annotations

import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import CONFIG
from app.models import LLMCall, Run, RunStep
from app.services.model_pricing_service import estimate_call_cost, pricing_catalog_meta


def get_storage_metrics() -> dict:
    root = Path(CONFIG.data_dir).parents[0]
    usage = shutil.disk_usage(root)
    budget_bytes = 10 * 1024 * 1024 * 1024

    def dir_size(target: Path) -> int:
        if not target.exists():
            return 0
        total = 0
        for p in target.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
        return total

    data_dir = CONFIG.data_dir
    out_dir = root / "output"
    log_dir = root / "logs"
    runs_dir = data_dir / "runs"
    return {
        "disk_total_bytes": usage.total,
        "disk_used_bytes": usage.used,
        "disk_free_bytes": usage.free,
        "disk_used_percent": round((usage.used / usage.total) * 100, 2) if usage.total else 0,
        "project_budget_bytes": budget_bytes,
        "project_used_bytes": dir_size(data_dir) + dir_size(out_dir) + dir_size(log_dir),
        "project_breakdown": {
            "data_dir_bytes": dir_size(data_dir),
            "output_dir_bytes": dir_size(out_dir),
            "logs_bytes": dir_size(log_dir),
            "runs_bytes": dir_size(runs_dir),
        },
    }


def _token_bucket(
    session: Session,
    *,
    label: str,
    start: datetime | None = None,
    end: datetime | None = None,
    run_id: str | None = None,
    run: Run | None = None,
) -> dict:
    conditions = []
    if start is not None:
        conditions.append(LLMCall.created_at >= start)
    if end is not None:
        conditions.append(LLMCall.created_at < end)
    if run_id is not None:
        conditions.append(LLMCall.run_id == run_id)
    real_conditions = [*conditions, LLMCall.model != "mock-model"]
    mock_conditions = [*conditions, LLMCall.model == "mock-model"]

    total_stmt = select(
        func.count(LLMCall.id),
        func.sum(LLMCall.total_tokens),
        func.sum(LLMCall.prompt_tokens),
        func.sum(LLMCall.completion_tokens),
    ).where(*real_conditions)
    calls_count, total_tokens, prompt_tokens, completion_tokens = session.execute(total_stmt).one()

    role_stmt = select(
        LLMCall.role,
        func.count(LLMCall.id),
        func.sum(LLMCall.total_tokens),
        func.sum(LLMCall.prompt_tokens),
        func.sum(LLMCall.completion_tokens),
    ).where(*real_conditions)
    role_stmt = role_stmt.group_by(LLMCall.role).order_by(func.sum(LLMCall.total_tokens).desc(), LLMCall.role.asc())

    by_role = [
        {
            "role": role,
            "calls_count": int(count or 0),
            "total_tokens": int(total or 0),
            "prompt_tokens": int(prompt or 0),
            "completion_tokens": int(comp or 0),
        }
        for role, count, total, prompt, comp in session.execute(role_stmt).all()
    ]
    call_rows = session.execute(
        select(
            LLMCall.role,
            LLMCall.provider,
            LLMCall.model,
            LLMCall.prompt_tokens,
            LLMCall.completion_tokens,
            LLMCall.total_tokens,
        ).where(*real_conditions)
    ).all()
    mock_stmt = select(
        func.count(LLMCall.id),
        func.sum(LLMCall.total_tokens),
    ).where(*mock_conditions)
    mock_calls_count, mock_total_tokens = session.execute(mock_stmt).one()

    role_cost_map: dict[str, dict] = {}
    model_cost_map: dict[tuple[str, str], dict] = {}
    unsupported_models: set[str] = set()
    pricing_notes: set[str] = set()
    cost_totals = {
        "prompt_cost": 0.0,
        "completion_cost": 0.0,
        "image_cost": 0.0,
        "total_cost": 0.0,
        "supported_calls_count": 0,
        "unsupported_calls_count": 0,
    }
    for role, provider, model, prompt, completion, total in call_rows:
        estimate = estimate_call_cost(
            provider=provider or "",
            model=model or "",
            prompt_tokens=int(prompt or 0),
            completion_tokens=int(completion or 0),
            call_count=1,
        )
        key = (estimate.provider, model or "")
        model_entry = model_cost_map.setdefault(
            key,
            {
                "provider": estimate.provider,
                "model": model or "",
                "canonical_model": estimate.canonical_model,
                "billing_kind": estimate.billing_kind,
                "tier_label": estimate.tier_label,
                "calls_count": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "prompt_cost": 0.0,
                "completion_cost": 0.0,
                "image_cost": 0.0,
                "total_cost": 0.0,
                "supported": True,
                "note": estimate.note,
            },
        )
        model_entry["calls_count"] += 1
        model_entry["prompt_tokens"] += int(prompt or 0)
        model_entry["completion_tokens"] += int(completion or 0)
        model_entry["total_tokens"] += int(total or 0)
        model_entry["prompt_cost"] = round(model_entry["prompt_cost"] + estimate.prompt_cost, 6)
        model_entry["completion_cost"] = round(model_entry["completion_cost"] + estimate.completion_cost, 6)
        model_entry["image_cost"] = round(model_entry["image_cost"] + estimate.image_cost, 6)
        model_entry["total_cost"] = round(model_entry["total_cost"] + estimate.total_cost, 6)
        model_entry["supported"] = bool(model_entry["supported"] and estimate.supported)
        if estimate.note:
            model_entry["note"] = estimate.note

        role_entry = role_cost_map.setdefault(
            role,
            {
                "prompt_cost": 0.0,
                "completion_cost": 0.0,
                "image_cost": 0.0,
                "total_cost": 0.0,
            },
        )
        role_entry["prompt_cost"] = round(role_entry["prompt_cost"] + estimate.prompt_cost, 6)
        role_entry["completion_cost"] = round(role_entry["completion_cost"] + estimate.completion_cost, 6)
        role_entry["image_cost"] = round(role_entry["image_cost"] + estimate.image_cost, 6)
        role_entry["total_cost"] = round(role_entry["total_cost"] + estimate.total_cost, 6)

        cost_totals["prompt_cost"] = round(cost_totals["prompt_cost"] + estimate.prompt_cost, 6)
        cost_totals["completion_cost"] = round(cost_totals["completion_cost"] + estimate.completion_cost, 6)
        cost_totals["image_cost"] = round(cost_totals["image_cost"] + estimate.image_cost, 6)
        cost_totals["total_cost"] = round(cost_totals["total_cost"] + estimate.total_cost, 6)
        if estimate.supported:
            cost_totals["supported_calls_count"] += 1
        else:
            cost_totals["unsupported_calls_count"] += 1
            unsupported_models.add(model or estimate.canonical_model or "unknown")
        if estimate.note:
            pricing_notes.add(estimate.note)

    for item in by_role:
        role_costs = role_cost_map.get(item["role"], {})
        item["prompt_cost"] = round(role_costs.get("prompt_cost", 0.0), 6)
        item["completion_cost"] = round(role_costs.get("completion_cost", 0.0), 6)
        item["image_cost"] = round(role_costs.get("image_cost", 0.0), 6)
        item["total_cost"] = round(role_costs.get("total_cost", 0.0), 6)

    by_model = sorted(
        model_cost_map.values(),
        key=lambda item: (-float(item["total_cost"]), -int(item["total_tokens"]), str(item["model"])),
    )

    return {
        "label": label,
        "run_id": run_id,
        "run_status": run.status if run else None,
        "run_type": run.run_type if run else None,
        "started_at": run.started_at.isoformat() if run and run.started_at else None,
        "finished_at": run.finished_at.isoformat() if run and run.finished_at else None,
        "range_start": start.isoformat() if start else None,
        "range_end": end.isoformat() if end else None,
        "calls_count": int(calls_count or 0),
        "total_tokens": int(total_tokens or 0),
        "prompt_tokens": int(prompt_tokens or 0),
        "completion_tokens": int(completion_tokens or 0),
        "by_role": by_role,
        "by_model": by_model,
        "has_data": bool((calls_count or 0) > 0),
        "excluded_mock_calls_count": int(mock_calls_count or 0),
        "excluded_mock_total_tokens": int(mock_total_tokens or 0),
        "costs": {
            **cost_totals,
            "currency": pricing_catalog_meta()["currency"],
        },
        "pricing": {
            **pricing_catalog_meta(),
            "unsupported_models": sorted(x for x in unsupported_models if x),
            "notes": sorted(pricing_notes),
        },
    }


def get_token_metrics(session: Session, days: int = 7) -> dict:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    summary = _token_bucket(session, label=f"{days}d", start=cutoff)
    return {
        "days": days,
        "total_tokens": summary["total_tokens"],
        "prompt_tokens": summary["prompt_tokens"],
        "completion_tokens": summary["completion_tokens"],
        "calls_count": summary["calls_count"],
        "by_role": summary["by_role"],
        "by_model": summary["by_model"],
        "costs": summary["costs"],
        "pricing": summary["pricing"],
    }


def get_token_overview(session: Session) -> dict:
    tz = ZoneInfo(CONFIG.timezone)
    now_local = datetime.now(tz)
    now_utc = now_local.astimezone(timezone.utc)
    start_today_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    start_week_local = start_today_local - timedelta(days=start_today_local.weekday())
    start_month_local = start_today_local.replace(day=1)

    latest_run_id = session.execute(
        select(LLMCall.run_id)
        .where(LLMCall.model != "mock-model")
        .join(Run, Run.id == LLMCall.run_id)
        .group_by(LLMCall.run_id)
        .order_by(func.max(Run.started_at).desc())
        .limit(1)
    ).scalar_one_or_none()
    latest_run = session.get(Run, latest_run_id) if latest_run_id else None

    windows = {
        "current_run": _token_bucket(
            session,
            label="本轮任务",
            run_id=latest_run_id,
            run=latest_run,
        ),
        "today": _token_bucket(
            session,
            label="当天概览",
            start=start_today_local.astimezone(timezone.utc),
            end=now_utc,
        ),
        "week": _token_bucket(
            session,
            label="本周概览",
            start=start_week_local.astimezone(timezone.utc),
            end=now_utc,
        ),
        "month": _token_bucket(
            session,
            label="本月概览",
            start=start_month_local.astimezone(timezone.utc),
            end=now_utc,
        ),
    }
    return {
        "timezone": CONFIG.timezone,
        "generated_at": now_local.isoformat(),
        "windows": windows,
        "pricing": pricing_catalog_meta(),
    }


def get_step_timing_metrics(session: Session, days: int = 7) -> dict:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    rows = session.execute(
        select(
            RunStep.name,
            func.count(RunStep.id),
            func.avg(RunStep.duration_ms),
            func.max(RunStep.duration_ms),
        ).where(RunStep.started_at >= cutoff).group_by(RunStep.name)
    ).all()
    out = [
        {
            "step": name,
            "runs": int(count or 0),
            "avg_duration_ms": int(avg or 0),
            "max_duration_ms": int(max_d or 0),
        }
        for name, count, avg, max_d in rows
    ]
    return {"days": days, "steps": out}

