from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    app_name: str
    timezone: str
    data_dir: Path
    db_path: Path
    encryption_key: str


def load_config() -> AppConfig:
    project_root = Path(__file__).resolve().parents[2]
    data_dir = Path(os.getenv("WAL_DATA_DIR", project_root / "data"))
    data_dir.mkdir(parents=True, exist_ok=True)
    db_path = Path(os.getenv("WAL_DB_PATH", data_dir / "wechat_agent_lite.db"))
    encryption_key = os.getenv("WAL_ENCRYPTION_KEY", os.getenv("APP_ENCRYPTION_KEY", ""))
    return AppConfig(
        app_name="wechat-agent-lite",
        timezone=os.getenv("WAL_TIMEZONE", "Asia/Shanghai"),
        data_dir=data_dir,
        db_path=db_path,
        encryption_key=encryption_key,
    )


CONFIG = load_config()

