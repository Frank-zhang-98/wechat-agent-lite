from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class RunStatus(str, enum.Enum):
    pending = "pending"
    running = "running"
    success = "success"
    failed = "failed"
    cancelled = "cancelled"
    partial_success = "partial_success"


class StepStatus(str, enum.Enum):
    pending = "pending"
    running = "running"
    success = "success"
    failed = "failed"
    cancelled = "cancelled"
    skipped = "skipped"


class RunType(str, enum.Enum):
    health = "health"
    main = "main"
    manual = "manual"
    manual_url = "manual_url"


class ConfigEntry(Base):
    __tablename__ = "config_entries"

    key: Mapped[str] = mapped_column(String(200), primary_key=True)
    value: Mapped[str] = mapped_column(Text, default="")
    is_secret: Mapped[bool] = mapped_column(Boolean, default=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    run_type: Mapped[str] = mapped_column(String(20), default=RunType.main.value)
    status: Mapped[str] = mapped_column(String(20), default=RunStatus.pending.value)
    trigger_source: Mapped[str] = mapped_column(String(50), default="scheduler")
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error_message: Mapped[str] = mapped_column(Text, default="")
    summary_json: Mapped[str] = mapped_column(Text, default="{}")
    quality_score: Mapped[float] = mapped_column(Float, default=0.0)
    quality_threshold: Mapped[float] = mapped_column(Float, default=78.0)
    quality_attempts: Mapped[int] = mapped_column(Integer, default=1)
    quality_fallback_used: Mapped[bool] = mapped_column(Boolean, default=False)
    article_title: Mapped[str] = mapped_column(String(300), default="")
    article_markdown: Mapped[str] = mapped_column(Text, default="")
    draft_status: Mapped[str] = mapped_column(String(40), default="not_started")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    steps: Mapped[list["RunStep"]] = relationship(back_populates="run", cascade="all, delete-orphan")
    llm_calls: Mapped[list["LLMCall"]] = relationship(back_populates="run", cascade="all, delete-orphan")


class RunStep(Base):
    __tablename__ = "run_steps"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(36), ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(100), index=True)
    status: Mapped[str] = mapped_column(String(20), default=StepStatus.pending.value)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[str] = mapped_column(Text, default="")
    details_json: Mapped[str] = mapped_column(Text, default="{}")

    run: Mapped["Run"] = relationship(back_populates="steps")


class LLMCall(Base):
    __tablename__ = "llm_calls"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(36), ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    step_name: Mapped[str] = mapped_column(String(100), default="")
    role: Mapped[str] = mapped_column(String(80), index=True)
    provider: Mapped[str] = mapped_column(String(80), default="")
    model: Mapped[str] = mapped_column(String(160), default="")
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    estimated: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    run: Mapped["Run"] = relationship(back_populates="llm_calls")


class SourceHealthState(Base):
    __tablename__ = "source_health_states"

    source_key: Mapped[str] = mapped_column(String(200), primary_key=True)
    category: Mapped[str] = mapped_column(String(80), default="")
    source_name: Mapped[str] = mapped_column(String(200), default="")
    current_url: Mapped[str] = mapped_column(String(600), default="")
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    weight: Mapped[float] = mapped_column(Float, default=0.7)
    consecutive_failures: Mapped[int] = mapped_column(Integer, default=0)
    total_successes: Mapped[int] = mapped_column(Integer, default=0)
    total_failures: Mapped[int] = mapped_column(Integer, default=0)
    last_status: Mapped[str] = mapped_column(String(40), default="unknown")
    last_http_status: Mapped[int | None] = mapped_column(Integer, nullable=True)
    last_error: Mapped[str] = mapped_column(Text, default="")
    last_action: Mapped[str] = mapped_column(String(40), default="")
    last_action_reason: Mapped[str] = mapped_column(Text, default="")
    last_candidate_url: Mapped[str] = mapped_column(String(600), default="")
    last_checked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_success_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_failure_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
