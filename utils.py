"""Shared utilities: hashing, time, text cleaning, safe logging."""
from __future__ import annotations

import hashlib
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path

_KEY_PATTERN = re.compile(r"(?:gsk_|sk-or-v1-|sk-or-)[A-Za-z0-9_\-]{8,}")
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def redact(text: str, extra_secrets: tuple[str, ...] = ()) -> str:
    """Remove API-key-looking strings and known secrets from text."""
    out = _KEY_PATTERN.sub("[REDACTED]", text)
    for secret in (*extra_secrets, os.environ.get("GROQ_API_KEY", ""), os.environ.get("OPENROUTER_API_KEY", "")):
        if secret and len(secret) >= 8:
            out = out.replace(secret, "[REDACTED]")
    return out


class RedactingFilter(logging.Filter):
    """Logging filter that scrubs secrets from every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001
            return True
        red = redact(msg)
        if red != msg:
            record.msg, record.args = red, ()
        return True


def setup_logging(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("ventureiq")
    if not any(isinstance(f, RedactingFilter) for f in logger.filters):
        logger.addFilter(RedactingFilter())
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        handler.addFilter(RedactingFilter())
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level, logging.INFO))
    logger.propagate = False
    return logger


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def clean_text(value: str) -> str:
    """Strip control characters, normalise whitespace runs, trim."""
    value = _CTRL.sub("", value or "")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def optional_import(name: str):
    """Import a module if available, else return None."""
    try:
        return __import__(name)
    except Exception:  # noqa: BLE001
        return None