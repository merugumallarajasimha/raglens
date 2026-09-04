"""Structured logging configuration for RAGLens."""

from __future__ import annotations

import logging
import sys

from .config import Settings


class JsonFormatter(logging.Formatter):
    """Simple JSON formatter for structured logs."""

    def format(self, record: logging.LogRecord) -> str:
        import json

        from datetime import datetime, timezone

        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra_data") and isinstance(record.extra_data, dict):
            log_entry.update(record.extra_data)
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry, ensure_ascii=False)


def setup_logging(settings: Settings | None = None) -> logging.Logger:
    """Configure root logging and return the RAGLens logger."""
    if settings is None:
        settings = Settings()

    root = logging.getLogger()
    level = logging.DEBUG if settings.debug else logging.INFO
    root.setLevel(level)

    # Remove existing handlers to avoid duplicates on re-init
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = JsonFormatter() if not settings.debug else logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    root.addHandler(console)

    # Silence noisy third-party loggers
    for noisy in ("uvicorn.error",):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logger = logging.getLogger("raglens")
    logger.info("Logging initialised", extra={"extra_data": {"debug": settings.debug}})
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the raglens namespace."""
    if not name.startswith("raglens"):
        name = f"raglens.{name}"
    return logging.getLogger(name)
