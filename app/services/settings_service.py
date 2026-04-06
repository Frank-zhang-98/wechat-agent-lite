from __future__ import annotations

from collections.abc import Mapping

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.security import decrypt_text, encrypt_text
from app.models import ConfigEntry
from app.services.default_settings import DEFAULT_SETTINGS


class SettingsService:
    def __init__(self, session: Session):
        self.session = session

    def ensure_defaults(self) -> None:
        existing_keys = {
            row[0]
            for row in self.session.execute(select(ConfigEntry.key)).all()
        }
        pending_keys = {
            obj.key
            for obj in self.session.new
            if isinstance(obj, ConfigEntry)
        }
        for key, (value, is_secret) in DEFAULT_SETTINGS.items():
            if key in existing_keys or key in pending_keys:
                continue
            self.session.add(
                ConfigEntry(
                    key=key,
                    value=encrypt_text(value) if is_secret else value,
                    is_secret=is_secret,
                )
            )
            pending_keys.add(key)

    def get(self, key: str, default: str = "") -> str:
        pending = self._get_pending_entry(key)
        if pending:
            if pending.is_secret:
                return decrypt_text(pending.value)
            return pending.value
        row = self.session.get(ConfigEntry, key)
        if not row:
            return default
        if row.is_secret:
            return decrypt_text(row.value)
        return row.value

    def set(self, key: str, value: str) -> None:
        row = self.session.get(ConfigEntry, key)
        if not row:
            row = self._get_pending_entry(key)
        if row:
            row.value = encrypt_text(value) if row.is_secret else value
            return
        default_value, is_secret = DEFAULT_SETTINGS.get(key, ("", False))
        row = ConfigEntry(
            key=key,
            value=encrypt_text(value) if is_secret else value,
            is_secret=is_secret,
        )
        if value == "" and default_value != "":
            row.value = encrypt_text(default_value) if is_secret else default_value
        self.session.add(row)

    def _get_pending_entry(self, key: str) -> ConfigEntry | None:
        for obj in self.session.new:
            if isinstance(obj, ConfigEntry) and obj.key == key:
                return obj
        return None

    def as_dict(self, include_secrets: bool = False) -> dict[str, str]:
        rows = self.session.execute(select(ConfigEntry).order_by(ConfigEntry.key.asc())).scalars().all()
        output: dict[str, str] = {}
        for row in rows:
            if row.is_secret and not include_secrets:
                output[row.key] = "******" if row.value else ""
            elif row.is_secret:
                output[row.key] = decrypt_text(row.value)
            else:
                output[row.key] = row.value
        return output

    def update_many(self, values: Mapping[str, str]) -> None:
        for key, val in values.items():
            self.set(key, val)

    def get_int(self, key: str, default: int) -> int:
        try:
            return int(self.get(key, str(default)))
        except Exception:
            return default

    def get_float(self, key: str, default: float) -> float:
        try:
            return float(self.get(key, str(default)))
        except Exception:
            return default

    def get_bool(self, key: str, default: bool) -> bool:
        raw = self.get(key, "true" if default else "false").strip().lower()
        return raw in {"1", "true", "yes", "on"}

    def get_list_int(self, key: str, default: list[int]) -> list[int]:
        raw = self.get(key, ",".join(str(x) for x in default)).strip()
        try:
            return [int(x.strip()) for x in raw.split(",") if x.strip()]
        except Exception:
            return default
