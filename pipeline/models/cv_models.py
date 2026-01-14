from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from pipeline.config import get_yolo_version_from_path, load_model_paths
from pipeline.models.person_pose_estimator import PersonPoseEstimator
from pipeline.models.person_reid_bytetrack import PersonReIDTrackerBT


@dataclass
class CVConfig:
    yolo_model_path: str = "yolov8n.pt"
    yolo_model_version: str = "auto"
    yolo_input_size: int = 640
    conf_threshold: float = 0.5
    batch_size: int = 8
    osnet_reid_model_path: str = "osnet_x1_0_imagenet.pt"
    # PRO settings overrides
    pro_yolo_input_size: Optional[int] = None
    pro_yolo_conf_threshold: Optional[float] = None


class CVModels:
    """
    Тонкая обертка над CV-моделями из ТЗ.
    В этой базовой реализации допускается stub-режим (если зависимости/веса недоступны).
    """

    def __init__(self, cfg: Optional[CVConfig] = None):
        if cfg is None:
            mp = load_model_paths()
            yolo_version = mp.yolo_model_version
            if yolo_version == "auto":
                yolo_version = get_yolo_version_from_path(mp.yolo_model_path)
            cfg = CVConfig(
                yolo_model_path=mp.yolo_model_path,
                yolo_model_version=yolo_version,
                yolo_input_size=mp.yolo_input_size,
                osnet_reid_model_path=mp.osnet_reid_model,
            )
        # Apply PRO overrides if available
        if cfg.pro_yolo_input_size is not None:
            cfg.yolo_input_size = cfg.pro_yolo_input_size
        if cfg.pro_yolo_conf_threshold is not None:
            cfg.conf_threshold = cfg.pro_yolo_conf_threshold

        self.cfg = cfg
        self._yolo = None
        self._yolo_ok = False
        self._reid_tracker: Optional[PersonReIDTrackerBT] = None
        self._pose_estimator: Optional[PersonPoseEstimator] = None
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

        # Pose estimation для анализа движений рук/ног
        if "pose-estimation" in required_models:
            pose_ctx = self._process_pose_estimation(frame_bgr, required_models)
            ctx.extend(pose_ctx)

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
                imgsz=self.cfg.yolo_input_size,
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
                imgsz=self.cfg.yolo_input_size,
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
                imgsz=self.cfg.yolo_input_size,
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
                imgsz=self.cfg.yolo_input_size,
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

    def _ensure_pose_estimator(self) -> None:
        if self._pose_estimator is not None:
            return
        # Lazy init pose estimator
        self._pose_estimator = PersonPoseEstimator()

    def _process_pose_estimation(self, frame_bgr: np.ndarray, required_models: list[str]) -> list[str]:
        """Обрабатывает pose estimation для найденных людей"""
        self._ensure_pose_estimator()
        if self._pose_estimator is None or not self._pose_estimator.is_available():
            return ["Pose: estimator not available"]

        ctx: list[str] = []

        # Получаем боксы людей из YOLO
        person_dets = []
        if "yolo-person" in required_models:
            person_dets = self._yolo_detect_persons(frame_bgr)

        if not person_dets:
            # Fallback: используем обычную детекцию
            dets = self._yolo_detect(frame_bgr, allow_labels={"person"})
            # Извлекаем боксы из описаний (простой парсинг)
            for desc in dets:
                if "bbox" in desc:
                    try:
                        # Парсим bbox из описания типа "YOLO: person (bbox [x1, y1, x2, y2], conf 0.85)"
                        import re
                        bbox_match = re.search(r'bbox \[([^\]]+)\]', desc)
                        if bbox_match:
                            bbox_str = bbox_match.group(1)
                            bbox_vals = [float(x.strip()) for x in bbox_str.split(',')]
                            if len(bbox_vals) == 4:
                                person_dets.append((np.array(bbox_vals, dtype=np.float32), 0.5))
                    except Exception:
                        continue

        # Оцениваем позу для каждого человека
        for i, (bbox, conf) in enumerate(person_dets):
            pose_result = self._pose_estimator.estimate_pose_in_bbox(frame_bgr, bbox)
            if pose_result:
                if pose_result.get('stub_mode'):
                    # Stub mode - provide basic info
                    ctx.append(f"Person_{i+1}: pose analysis (stub mode - MediaPipe not available)")
                    continue

                motion = pose_result.get('motion_analysis', {})

                # Формируем описание движений
                desc_parts = [f"Person_{i+1} pose (conf {pose_result.get('confidence', 0):.2f}):"]

                # Руки
                hands = motion.get('hand_movements', {})
                for side in ['left', 'right']:
                    if side in hands and hands[side]:
                        elbow_pos = hands[side].get('elbow_position', 'unknown')
                        desc_parts.append(f"{side} arm {elbow_pos}")

                # Ноги
                legs = motion.get('leg_movements', {})
                for side in ['left', 'right']:
                    if side in legs and legs[side]:
                        knee_pos = legs[side].get('knee_position', 'unknown')
                        desc_parts.append(f"{side} leg {knee_pos}")

                # Поза тела
                posture = motion.get('body_posture', 'unknown')
                if posture != 'unknown':
                    desc_parts.append(f"posture: {posture}")

                # Жесты
                gestures = motion.get('gesture_recognition', [])
                if gestures:
                    desc_parts.append(f"gestures: {', '.join(gestures)}")

                ctx.append(" | ".join(desc_parts))
            else:
                ctx.append(f"Person_{i+1}: pose estimation failed")

        return ctx if ctx else ["Pose: no persons detected for pose analysis"]


_CV_SINGLETON: Optional[CVModels] = None


def make_cv_models(pro_settings: Optional[dict] = None, yolo_batch_size: int = 16) -> CVModels:
    """
    Кэшируем модели в процессе: повторные запуски через Streamlit не должны каждый раз
    заново грузить YOLO/torchreid, это очень дорого.
    """
    global _CV_SINGLETON
    if _CV_SINGLETON is None:
        cfg = None
        if pro_settings:
            mp = load_model_paths()
            yolo_version = mp.yolo_model_version
            if yolo_version == "auto":
                yolo_version = get_yolo_version_from_path(mp.yolo_model_path)

            cfg = CVConfig(
                yolo_model_path=mp.yolo_model_path,
                yolo_model_version=yolo_version,
                yolo_input_size=mp.yolo_input_size,
                conf_threshold=pro_settings.get("yolo_conf_threshold", 0.5),
                batch_size=yolo_batch_size,  # Используем переданный batch_size
                osnet_reid_model_path=mp.osnet_reid_model,
                pro_yolo_input_size=pro_settings.get("yolo_input_size"),
                pro_yolo_conf_threshold=pro_settings.get("yolo_conf_threshold"),
            )
        _CV_SINGLETON = CVModels(cfg)
    return _CV_SINGLETON

