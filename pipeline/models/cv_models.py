from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class CVConfig:
    yolo_model_path: str = "yolov8x.pt"
    conf_threshold: float = 0.5
    batch_size: int = 8


class CVModels:
    """
    Тонкая обертка над CV-моделями из ТЗ.
    В этой базовой реализации допускается stub-режим (если зависимости/веса недоступны).
    """

    def __init__(self, cfg: Optional[CVConfig] = None):
        self.cfg = cfg or CVConfig()
        self._yolo = None
        self._yolo_ok = False

        try:
            from ultralytics import YOLO  # type: ignore

            self._yolo = YOLO(self.cfg.yolo_model_path)
            self._yolo_ok = True
        except Exception:
            self._yolo = None
            self._yolo_ok = False

    def describe_frame(
        self,
        frame_bgr: np.ndarray,
        required_models: list[str],
    ) -> list[str]:
        ctx: list[str] = []

        if any(m.startswith("yolov8") for m in required_models):
            ctx.extend(self._yolo_detect(frame_bgr))

        # ReID / zone-detector в этой версии оставлены как заглушки
        if "reid-tracker" in required_models:
            ctx.append("ReID: (stub) tracking not enabled in this build")
        if "zone-detector" in required_models:
            ctx.append("zone_tracker: (stub) zones not configured")

        return ctx

    def _yolo_detect(self, frame_bgr: np.ndarray) -> list[str]:
        if not self._yolo_ok or self._yolo is None:
            return ["YOLO: (stub) model not available"]

        try:
            results = self._yolo.predict(
                source=frame_bgr,
                conf=float(self.cfg.conf_threshold),
                verbose=False,
            )
            out: list[str] = []
            for r in results:
                if r.boxes is None:
                    continue
                names = getattr(r, "names", {}) or {}
                for b in r.boxes:
                    cls_id = int(getattr(b, "cls", [0])[0]) if hasattr(b, "cls") else 0
                    conf = float(getattr(b, "conf", [0.0])[0]) if hasattr(b, "conf") else 0.0
                    xyxy = (
                        getattr(b, "xyxy", None).tolist()[0]
                        if getattr(b, "xyxy", None) is not None
                        else [0, 0, 0, 0]
                    )
                    label = str(names.get(cls_id, f"class_{cls_id}"))
                    out.append(
                        f"YOLO: {label} (bbox {list(map(float, xyxy))}, conf {conf:.2f})"
                    )
            return out or ["YOLO: no detections"]
        except Exception as e:
            return [f"YOLO: error during inference: {type(e).__name__}: {e}"]


def make_cv_models() -> CVModels:
    return CVModels()

