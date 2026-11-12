"""Structured logs: one JSON object per line (or plain text for a terminal).

What is logged: request id, caller name, route, status code, duration, and for /ask the outcome
(status, retried, token counts, number of citations). What is never logged: API keys, the question
text, answers, or document text (they may hold private business data). The question's length is
logged instead.
"""

from __future__ import annotations

import json
import logging
import sys
import time

LOGGER_NAME = "evidence_rag"
_HANDLER_FLAG = "_evidence_rag_handler"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname.lower(),
            "event": record.getMessage(),
        }
        data.update(getattr(record, "fields", {}) or {})
        if record.exc_info:
            data["exception"] = self.formatException(record.exc_info)
        return json.dumps(data, ensure_ascii=False, default=str)


class TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        fields = getattr(record, "fields", {}) or {}
        tail = " ".join(f"{k}={v}" for k, v in fields.items())
        line = f"{record.levelname:<7} {record.getMessage()} {tail}".rstrip()
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


def setup_logging(fmt: str = "json", level: str = "INFO") -> logging.Logger:
    """Attach one handler to the app logger (idempotent: safe to call on every app start)."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level.upper())
    for h in list(logger.handlers):
        if getattr(h, _HANDLER_FLAG, False):
            logger.removeHandler(h)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if fmt == "json" else TextFormatter())
    setattr(handler, _HANDLER_FLAG, True)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def log_event(logger: logging.Logger, event: str, level: int = logging.INFO, **fields) -> None:
    logger.log(level, event, extra={"fields": fields})
