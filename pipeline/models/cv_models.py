from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from pipeline.config import load_model_paths
from pipeline.models.person_reid_bytetrack import PersonReIDTrackerBT


@dataclass
class CVConfig:
    yolo_model_path: str = "yolov8x.pt"
    conf_threshold: float = 0.5
    batch_size: int = 8
    osnet_reid_model_path: str = "osnet_x1_0_imagenet.pt"


class CVModels:
    """
    Тонкая обертка над CV-моделями из ТЗ.
    В этой базовой реализации допускается stub-режим (если зависимости/веса недоступны).
    """

    def __init__(self, cfg: Optional[CVConfig] = None):
        if cfg is None:
            mp = load_model_paths()
            cfg = CVConfig(
                yolo_model_path=mp.yolo_model_path,
                osnet_reid_model_path=mp.osnet_reid_model,
            )
        self.cfg = cfg
        self._yolo = None
        self._yolo_ok = False
        self._reid_tracker: Optional[PersonReIDTrackerBT] = None
        self._track_persist = False

        try:
            from ultralytics import YOLO  # type: ignore

            self._yolo = YOLO(self.cfg.yolo_model_path)
            self._yolo_ok = True
        except Exception:
            self._yolo = None
            self._yolo_ok = False

    def begin_video(self) -> None:
        # reset ByteTrack state (ultralytics stores trackers inside predictor when persist=True)
        if self._yolo is not None and getattr(self._yolo, "predictor", None) is not None:
            try:
                if hasattr(self._yolo.predictor, "trackers"):
                    self._yolo.predictor.trackers = None
            except Exception:
                pass
        self._track_persist = False
        if self._reid_tracker is not None:
            self._reid_tracker.begin_video()

    def _ensure_reid(self) -> None:
        if self._reid_tracker is not None:
            return
        # Lazy init, чтобы не грузить torchreid/torch без необходимости
        device = "cuda"
        try:
            import torch  # type: ignore

            if not bool(getattr(torch, "cuda", None)) or not torch.cuda.is_available():
                device = "cpu"
        except Exception:
            device = "cpu"
        self._reid_tracker = PersonReIDTrackerBT(
            osnet_weights_path=self.cfg.osnet_reid_model_path,
            device=device,
        )

    def describe_frame(
        self,
        frame_bgr: np.ndarray,
        required_models: list[str],
        frame_id: int = 0,
        timestamp_sec: float = 0.0,
        video_path: Optional[str] = None,
    ) -> list[str]:
        ctx: list[str] = []

        if "reid-osnet" in required_models:
            # ReID режим: ByteTrack (track ids) + OSNet (глобальный person_id)
            self._ensure_reid()
            if self._reid_tracker is None:
                ctx.append("ReID: (stub) tracker init failed")
            else:
                tracked = self._yolo_track_persons(frame_bgr)
                if tracked:
                    ctx.extend(
                        self._reid_tracker.update(
                            frame_id=frame_id,
                            frame_bgr=frame_bgr,
                            tracked_dets=tracked,
                            timestamp_sec=float(timestamp_sec),
                            video_path=video_path,
                        )
                    )
                else:
                    # fallback: без track_id (хуже, но не падаем)
                    dets = self._yolo_detect_persons(frame_bgr)
                    ctx.append("ReID: tracking fallback (no track ids)")
                    # best-effort: превратим dets в local ids
                    tracked2 = [(i + 1, b, c) for i, (b, c) in enumerate(dets)]
                    ctx.extend(
                        self._reid_tracker.update(
                            frame_id=frame_id,
                            frame_bgr=frame_bgr,
                            tracked_dets=tracked2,
                            timestamp_sec=float(timestamp_sec),
                            video_path=video_path,
                        )
                    )
        elif "yolo-person" in required_models:
            # Быстрый режим: только люди (без bbox в тексте, чтобы не раздувать строки/IO)
            ctx.extend(self._yolo_detect(frame_bgr, allow_labels={"person"}))

        # ReID / zone-detector в этой версии оставлены как заглушки
        if "zone-detector" in required_models:
            ctx.append("zone_tracker: (stub) zones not configured")

        return ctx

    def describe_frames_yolo_person_batch(self, frames_bgr: list[np.ndarray]) -> list[list[str]]:
        """
        Быстрый путь: батч-инференс YOLO по списку кадров.
        Используем только когда нужны именно люди (yolo-person) без ReID/трекинга.
        """
        if not frames_bgr:
            return []
        if not self._yolo_ok or self._yolo is None:
            return [["YOLO: (stub) model not available"] for _ in frames_bgr]
        try:
            res = self._yolo.predict(
                source=frames_bgr,
                conf=float(self.cfg.conf_threshold),
                classes=[0],
                verbose=False,
            )
            out: list[list[str]] = []
            for r in res:
                if r.boxes is None or getattr(r.boxes, "conf", None) is None:
                    out.append(["YOLO: no detections"])
                    continue
                confs = r.boxes.conf.cpu().numpy().tolist()
                n = len(confs)
                if n <= 0:
                    out.append(["YOLO: no detections"])
                    continue
                cm = max(float(x) for x in confs) if confs else 0.0
                ca = (sum(float(x) for x in confs) / max(1, n)) if confs else 0.0
                out.append([f"YOLO: person count={n} conf_max={cm:.2f} conf_mean={ca:.2f}"])
            # safety: если SDK/ultralytics вернул меньше результатов, дополним
            while len(out) < len(frames_bgr):
                out.append(["YOLO: no detections"])
            return out[: len(frames_bgr)]
        except Exception as e:
            return [[f"YOLO: error during inference: {type(e).__name__}: {e}"] for _ in frames_bgr]

    def _yolo_detect(self, frame_bgr: np.ndarray, allow_labels: Optional[set[str]] = None) -> list[str]:
        if not self._yolo_ok or self._yolo is None:
            return ["YOLO: (stub) model not available"]

        try:
            results = self._yolo.predict(
                source=frame_bgr,
                conf=float(self.cfg.conf_threshold),
                classes=[0] if allow_labels == {"person"} else None,
                verbose=False,
            )
            counts: dict[str, int] = {}
            conf_max: dict[str, float] = {}
            conf_sum: dict[str, float] = {}
            for r in results:
                if r.boxes is None:
                    continue
                names = getattr(r, "names", {}) or {}
                for b in r.boxes:
                    cls_id = int(getattr(b, "cls", [0])[0]) if hasattr(b, "cls") else 0
                    conf = float(getattr(b, "conf", [0.0])[0]) if hasattr(b, "conf") else 0.0
                    label = str(names.get(cls_id, f"class_{cls_id}"))
                    if allow_labels is not None and label not in allow_labels:
                        continue
                    counts[label] = counts.get(label, 0) + 1
                    conf_sum[label] = conf_sum.get(label, 0.0) + float(conf)
                    conf_max[label] = max(conf_max.get(label, 0.0), float(conf))

            if not counts:
                return ["YOLO: no detections"]

            out: list[str] = []
            for label, n in sorted(counts.items(), key=lambda x: x[0]):
                cm = conf_max.get(label, 0.0)
                ca = (conf_sum.get(label, 0.0) / max(1, n)) if n else 0.0
                # важно: формат "YOLO: person count=N" используется в суммаризации (people_max)
                out.append(f"YOLO: {label} count={n} conf_max={cm:.2f} conf_mean={ca:.2f}")
            return out
        except Exception as e:
            return [f"YOLO: error during inference: {type(e).__name__}: {e}"]

    def _yolo_detect_persons(self, frame_bgr: np.ndarray) -> list[tuple[np.ndarray, float]]:
        """
        Возвращает список (bbox_xyxy, conf) только для класса person.
        """
        if not self._yolo_ok or self._yolo is None:
            return []
        try:
            results = self._yolo.predict(
                source=frame_bgr,
                conf=float(self.cfg.conf_threshold),
                classes=[0],
                verbose=False,
            )
            dets: list[tuple[np.ndarray, float]] = []
            for r in results:
                if r.boxes is None:
                    continue
                xyxy = r.boxes.xyxy.cpu().numpy().tolist()
                confs = r.boxes.conf.cpu().numpy().tolist()
                for bb, cf in zip(xyxy, confs):
                    dets.append((np.asarray(bb, dtype=np.float32), float(cf)))
            return dets
        except Exception:
            return []

    def _yolo_track_persons(self, frame_bgr: np.ndarray) -> list[tuple[int, np.ndarray, float]]:
        """
        ByteTrack через Ultralytics: возвращает (track_id, bbox_xyxy, conf) для person.
        """
        if not self._yolo_ok or self._yolo is None:
            return []
        try:
            # persist=True держит состояние трекера между кадрами
            res = self._yolo.track(
                source=frame_bgr,
                persist=True,
                classes=[0],  # person class in COCO
                conf=float(self.cfg.conf_threshold),
                verbose=False,
            )
            self._track_persist = True
            out: list[tuple[int, np.ndarray, float]] = []
            for r in res:
                if r.boxes is None:
                    continue
                ids = getattr(r.boxes, "id", None)
                if ids is None:
                    continue
                ids = ids.cpu().numpy().astype(int).tolist()
                xyxy = r.boxes.xyxy.cpu().numpy().tolist()
                confs = r.boxes.conf.cpu().numpy().tolist()
                for tid, bb, cf in zip(ids, xyxy, confs):
                    out.append((int(tid), np.asarray(bb, dtype=np.float32), float(cf)))
            return out
        except Exception:
            return []

    def get_person_routes(self) -> Optional[dict[str, Any]]:
        if self._reid_tracker is None:
            return None
        try:
            return self._reid_tracker.summarize_routes()
        except Exception:
            return None


_CV_SINGLETON: Optional[CVModels] = None


def make_cv_models() -> CVModels:
    """
    Кэшируем модели в процессе: повторные запуски через Streamlit не должны каждый раз
    заново грузить YOLO/torchreid, это очень дорого.
    """
    global _CV_SINGLETON
    if _CV_SINGLETON is None:
        _CV_SINGLETON = CVModels()
    return _CV_SINGLETON

