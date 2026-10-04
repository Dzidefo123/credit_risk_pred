"""Application-owned JSON logging with no changes to the root logger."""

import json
import logging
import sys
from datetime import UTC, datetime


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "details"):
            payload["details"] = record.details
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, allow_nan=False)


def configure_logging(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("credit_risk")
    logger.setLevel(level)
    for handler in list(logger.handlers):
        if getattr(handler, "_credit_risk_owned", False):
            logger.removeHandler(handler)
            handler.close()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    handler._credit_risk_owned = True
    logger.addHandler(handler)
    logger.propagate = False
    return logger
