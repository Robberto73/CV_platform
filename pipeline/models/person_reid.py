from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v) + 1e-12)
    return v / n


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = _l2_normalize(a)
    b = _l2_normalize(b)
    return float(np.dot(a, b))


@dataclass
class Track:
    local_id: int
    bbox_xyxy: np.ndarray
    last_center: np.ndarray
    last_seen_frame_id: int
    gid: int
    emb: np.ndarray  # normalized centroid
    emb_count: int = 1


class OSNetExtractor:
    """
    Lazy wrapper around torchreid FeatureExtractor.
    """

    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self._extractor = None

    def _lazy(self):
        if self._extractor is not None:
            return
        from torchreid.reid.utils.feature_extractor import FeatureExtractor  # type: ignore

        # model_name должен совпадать с osnet семейством
        self._extractor = FeatureExtractor(
            model_name="osnet_x1_0",
            model_path=self.model_path,
            device=self.device,
            verbose=False,
        )

    def __call__(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Возвращает np.ndarray shape (D,) нормализованный.
        """
        self._lazy()
        assert self._extractor is not None
        # torchreid ожидает np.ndarray (H,W,C) в RGB; у нас BGR
        rgb = img_bgr[..., ::-1].copy()
        feat = self._extractor(rgb)  # torch.Tensor (1, D) или (D,)
        if hasattr(feat, "detach"):
            feat = feat.detach().cpu().numpy()
        feat = np.asarray(feat)
        if feat.ndim == 2:
            feat = feat[0]
        return _l2_normalize(feat.astype(np.float32))


class PersonReIDTracker:
    """
    Универсальный трекер людей + ReID (OSNet) без привязки к конкретным объектам сцены.

    - локальные track_id внутри одного видео строятся по IoU-сопоставлению (baseline)
    - глобальные person_id (gid) сопоставляются по cosine similarity между OSNet эмбеддингами (между видео тоже)
    """

    def __init__(
        self,
        osnet_weights_path: str,
        device: str = "cuda",
        iou_threshold: float = 0.3,
        gid_similarity_threshold: float = 0.75,
        movement_px_threshold: float = 20.0,
        stop_radius_px: float = 25.0,
        stop_min_frames: int = 3,
    ):
        self.extractor = OSNetExtractor(osnet_weights_path, device=device)
        self.iou_threshold = float(iou_threshold)
        self.gid_similarity_threshold = float(gid_similarity_threshold)
        self.movement_px_threshold = float(movement_px_threshold)
        self.stop_radius_px = float(stop_radius_px)
        self.stop_min_frames = int(stop_min_frames)

        self._next_local_id = 1
        self._next_gid = 1
        self._tracks: dict[int, Track] = {}
        self._global_db: dict[int, np.ndarray] = {}  # gid -> centroid emb (normalized)
        self._traj: dict[int, list[dict[str, Any]]] = {}  # gid -> [{t, x, y, frame_id, video_path?}]

    def begin_video(self) -> None:
        self._next_local_id = 1
        self._tracks = {}
        # траектории (global) не сбрасываем, чтобы работала реидентификация между видео

    def _assign_gid(self, emb: np.ndarray) -> int:
        if not self._global_db:
            gid = self._next_gid
            self._next_gid += 1
            self._global_db[gid] = emb
            return gid

        best_gid = None
        best_sim = -1.0
        for gid, cent in self._global_db.items():
            sim = _cosine(emb, cent)
            if sim > best_sim:
                best_sim = sim
                best_gid = gid

        if best_gid is not None and best_sim >= self.gid_similarity_threshold:
            # обновим centroid EMA (простая средняя)
            self._global_db[best_gid] = _l2_normalize((self._global_db[best_gid] + emb) / 2.0)
            return int(best_gid)

        gid = self._next_gid
        self._next_gid += 1
        self._global_db[gid] = emb
        return gid

    def update(
        self,
        frame_id: int,
        frame_bgr: np.ndarray,
        detections_xyxy_conf: list[tuple[np.ndarray, float]],
        timestamp_sec: Optional[float] = None,
        video_path: Optional[str] = None,
    ) -> list[str]:
        """
        Возвращает список строк контекста для кадра.
        """
        h, w = frame_bgr.shape[:2]
        dets = [(np.asarray(b, dtype=np.float32), float(c)) for b, c in detections_xyxy_conf]

        # match detections to existing tracks by IoU
        track_ids = list(self._tracks.keys())
        pairs: list[tuple[float, int, int]] = []  # (iou, tid, didx)
        for tid in track_ids:
            tb = self._tracks[tid].bbox_xyxy
            for di, (db, _) in enumerate(dets):
                pairs.append((_iou_xyxy(tb, db), tid, di))
        pairs.sort(key=lambda x: x[0], reverse=True)

        assigned_tracks: set[int] = set()
        assigned_dets: set[int] = set()
        matches: list[tuple[int, int]] = []  # (tid, didx)
        for iou, tid, di in pairs:
            if iou < self.iou_threshold:
                break
            if tid in assigned_tracks or di in assigned_dets:
                continue
            assigned_tracks.add(tid)
            assigned_dets.add(di)
            matches.append((tid, di))

        ctx: list[str] = []

        def crop(b: np.ndarray) -> np.ndarray:
            x1, y1, x2, y2 = [int(round(v)) for v in b.tolist()]
            x1 = max(0, min(w - 1, x1))
            x2 = max(0, min(w, x2))
            y1 = max(0, min(h - 1, y1))
            y2 = max(0, min(h, y2))
            if x2 <= x1 or y2 <= y1:
                return frame_bgr
            return frame_bgr[y1:y2, x1:x2]

        # update matched tracks
        for tid, di in matches:
            b, conf = dets[di]
            cxy = np.array([(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0], dtype=np.float32)
            tr = self._tracks[tid]
            dx, dy = (cxy - tr.last_center).tolist()

            emb = self.extractor(crop(b))
            # update track embedding centroid
            tr.emb = _l2_normalize((tr.emb * tr.emb_count + emb) / (tr.emb_count + 1))
            tr.emb_count += 1
            tr.bbox_xyxy = b
            tr.last_center = cxy
            tr.last_seen_frame_id = int(frame_id)

            moved = (abs(dx) + abs(dy)) >= self.movement_px_threshold
            move_txt = f" moved dx={dx:.1f},dy={dy:.1f}" if moved else ""
            ctx.append(
                f"ReID: person_{tr.gid} track_{tr.local_id}{move_txt} (bbox {b.tolist()}, conf {conf:.2f})"
            )
            self._traj.setdefault(tr.gid, []).append(
                {
                    "timestamp_sec": float(timestamp_sec) if timestamp_sec is not None else None,
                    "frame_id": int(frame_id),
                    "x": float(cxy[0]),
                    "y": float(cxy[1]),
                    "video_path": video_path,
                }
            )

        # create new tracks for unmatched detections
        for di, (b, conf) in enumerate(dets):
            if di in assigned_dets:
                continue
            cxy = np.array([(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0], dtype=np.float32)
            emb = self.extractor(crop(b))
            gid = self._assign_gid(emb)
            local_id = self._next_local_id
            self._next_local_id += 1
            self._tracks[local_id] = Track(
                local_id=local_id,
                bbox_xyxy=b,
                last_center=cxy,
                last_seen_frame_id=int(frame_id),
                gid=int(gid),
                emb=emb,
                emb_count=1,
            )
            ctx.append(
                f"ReID: person_{gid} track_{local_id} appeared (bbox {b.tolist()}, conf {conf:.2f})"
            )
            self._traj.setdefault(int(gid), []).append(
                {
                    "timestamp_sec": float(timestamp_sec) if timestamp_sec is not None else None,
                    "frame_id": int(frame_id),
                    "x": float(cxy[0]),
                    "y": float(cxy[1]),
                    "video_path": video_path,
                }
            )

        return ctx

    def summarize_routes(self) -> dict[str, Any]:
        """
        Универсальная сводка траекторий по каждому global person_id.
        Делает простые "стоп-сегменты" (когда человек находится в радиусе stop_radius_px
        минимум stop_min_frames подряд).
        """
        out: dict[str, Any] = {"schema": "person_routes_v1", "people": []}
        for gid, pts in self._traj.items():
            pts2 = [p for p in pts if p.get("x") is not None and p.get("y") is not None]
            pts2.sort(key=lambda p: (p.get("video_path") or "", p.get("timestamp_sec") or -1, p.get("frame_id") or 0))
            # стоп-сегменты
            segments: list[dict[str, Any]] = []
            cur: list[dict[str, Any]] = []
            cur_center = None
            for p in pts2:
                xy = np.array([float(p["x"]), float(p["y"])], dtype=np.float32)
                if cur_center is None:
                    cur = [p]
                    cur_center = xy
                    continue
                dist = float(np.linalg.norm(xy - cur_center))
                if dist <= self.stop_radius_px:
                    cur.append(p)
                    # обновляем центр средним
                    xs = [float(x["x"]) for x in cur]
                    ys = [float(x["y"]) for x in cur]
                    cur_center = np.array([sum(xs) / len(xs), sum(ys) / len(ys)], dtype=np.float32)
                else:
                    if len(cur) >= self.stop_min_frames:
                        segments.append(_segment_from_points(cur))
                    cur = [p]
                    cur_center = xy
            if cur and len(cur) >= self.stop_min_frames:
                segments.append(_segment_from_points(cur))

            out["people"].append(
                {
                    "person_id": int(gid),
                    "points": pts2,
                    "stop_segments": segments,
                }
            )
        return out


def _segment_from_points(points: list[dict[str, Any]]) -> dict[str, Any]:
    xs = [float(p["x"]) for p in points]
    ys = [float(p["y"]) for p in points]
    t0 = points[0].get("timestamp_sec")
    t1 = points[-1].get("timestamp_sec")
    return {
        "start_timestamp_sec": t0,
        "end_timestamp_sec": t1,
        "frames": [int(p.get("frame_id") or 0) for p in points],
        "center_x": float(sum(xs) / len(xs)),
        "center_y": float(sum(ys) / len(ys)),
        "count": len(points),
        "video_path": points[0].get("video_path"),
    }
