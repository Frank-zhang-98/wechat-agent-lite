from __future__ import annotations

import base64
import hashlib

from cryptography.fernet import Fernet

from app.core.config import CONFIG


def _derive_key() -> bytes:
    """Derive a deterministic fernet key when WAL_ENCRYPTION_KEY is not set."""
    raw = CONFIG.encryption_key.strip()
    if raw:
        try:
            return raw.encode("utf-8")
        except Exception:
            pass
    digest = hashlib.sha256(b"wechat-agent-lite-dev-key").digest()
    return base64.urlsafe_b64encode(digest)


_FERNET = Fernet(_derive_key())


def encrypt_text(value: str) -> str:
    if value == "":
        return ""
    return _FERNET.encrypt(value.encode("utf-8")).decode("utf-8")


def decrypt_text(value: str) -> str:
    if value == "":
        return ""
    try:
        return _FERNET.decrypt(value.encode("utf-8")).decode("utf-8")
    except Exception:
        return ""

