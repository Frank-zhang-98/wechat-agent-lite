from __future__ import annotations

from pydantic import BaseModel, Field


class ConfigUpdatePayload(BaseModel):
    values: dict[str, str] = Field(default_factory=dict)


class TriggerRunPayload(BaseModel):
    run_type: str = "main"
    trigger_source: str = "manual"
    source_url: str = ""


class RunActionPayload(BaseModel):
    action: str
