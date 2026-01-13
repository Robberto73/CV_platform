from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional


def utc_ts() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


@dataclass
class ProgressReporter:
    progress_cb: Optional[Callable[[float, str], None]] = None

    def log(self, state: dict, msg: str) -> None:
        state.setdefault("processing_log", [])
        state["processing_log"].append(f"[{utc_ts()}] {msg}")

    def progress(self, value_0_1: float, msg: str) -> None:
        if self.progress_cb is None:
            return
        try:
            self.progress_cb(max(0.0, min(1.0, float(value_0_1))), msg)
        except Exception:
            # UI callbacks должны быть best-effort
            pass

