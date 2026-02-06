from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict


def log_event(event: str, payload: Dict[str, Any] | None = None) -> None:
    """Emit a structured JSON log line to stdout."""
    data = {
        "event": event,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    if payload:
        data.update(payload)
    print(json.dumps(data, ensure_ascii=False))
