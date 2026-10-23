"""Centralised logging configuration.

Call ``setup_logging()`` once at application start-up.  After that every
module can use the standard ``logging.getLogger(__name__)`` API and logs
are routed to:

* **stdout** — plain-text, picked up by ``podman logs / docker logs``
* **logs/app.log** — structured JSON, rotated at 10 MB × 5 files
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from datetime import datetime, timezone
from pathlib import Path

_LOG_DIR = Path("/app/logs")
_LOG_FILE = _LOG_DIR / "app.log"

# Fields emitted by LogRecord that we don't want repeated in the JSON body.
_SKIP = frozenset({
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "message", "taskName",
    "asctime",  # added by Formatter.format() — already present as "ts"
})

_configured = False  # idempotency guard


class _JSONFormatter(logging.Formatter):
    """Emit one JSON object per log line."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc)
                         .strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _SKIP and not key.startswith("_"):
                entry[key] = value
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


def setup_logging(log_level: str = "INFO") -> None:
    """Configure root logger: JSON file handler + plain-text stdout handler.

    Idempotent — safe to call multiple times; only configures once.
    """
    global _configured
    if _configured:
        return
    _configured = True

    level = getattr(logging, log_level.upper(), logging.INFO)

    plain_fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # --- stdout (plain text) ---
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(plain_fmt)
    stdout_handler.setLevel(level)

    handlers: list[logging.Handler] = [stdout_handler]

    # --- rotating JSON file (best-effort; skipped if path unwritable) ---
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            _LOG_FILE,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(_JSONFormatter())
        file_handler.setLevel(level)
        handlers.append(file_handler)
    except OSError:
        # Running in a read-only environment (e.g. unit tests) — skip file log.
        pass

    root = logging.getLogger()
    root.setLevel(level)
    for handler in handlers:
        root.addHandler(handler)

    # Silence noisy third-party loggers.
    for name in ("httpx", "httpcore", "urllib3", "qdrant_client", "hpack",
                 "watchfiles", "uvicorn.error"):
        logging.getLogger(name).setLevel(logging.WARNING)

    logging.getLogger("rag_platform").info(
        "Logging initialised",
        extra={"log_level": log_level, "log_file": str(_LOG_FILE)},
    )
