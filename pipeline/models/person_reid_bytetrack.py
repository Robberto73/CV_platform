from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v) + 1e-12)
    return v / n


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = _l2_normalize(a)
    b = _l2_normalize(b)
    return float(np.dot(a, b))


@dataclass
class Track:
    local_track_id: int
    bbox_xyxy: np.ndarray
    last_center: np.ndarray
    last_seen_frame_id: int
    gid: int
    emb: np.ndarray
    emb_count: int = 1


class OSNetExtractor:
    """
    Lazy wrapper around torchreid FeatureExtractor.
    Compatible with torchreid 0.2.x and newer layouts.
    """

    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self._extractor = None

    def _lazy(self) -> None:
        if self._extractor is not None:
            return

        FeatureExtractor = None
        try:
            from torchreid.reid.utils.feature_extractor import FeatureExtractor as FE  # type: ignore

            FeatureExtractor = FE
        except Exception:
            try:
                from torchreid.utils import FeatureExtractor as FE  # type: ignore

                FeatureExtractor = FE
            except Exception as e:
                raise RuntimeError(
                    "torchreid FeatureExtractor not found. Please verify torchreid installation/version."
                ) from e

        self._extractor = FeatureExtractor(
            model_name="osnet_x1_0",
            model_path=self.model_path,
            device=self.device,
            verbose=False,
        )

    def __call__(self, img_bgr: np.ndarray) -> np.ndarray:
        self._lazy()
        assert self._extractor is not None
        rgb = img_bgr[..., ::-1].copy()
        feat = self._extractor(rgb)
        if hasattr(feat, "detach"):
            feat = feat.detach().cpu().numpy()
        feat = np.asarray(feat)
        if feat.ndim == 2:
            feat = feat[0]
        return _l2_normalize(feat.astype(np.float32))


class PersonReIDTrackerBT:
    """
    ReID поверх внешнего трекера (ByteTrack из Ultralytics).
    Внутри видео используем стабильный local_track_id от ByteTrack.
    Между видео присваиваем глобальный person_id (gid) по cosine similarity OSNet эмбеддингов.
    """

    def __init__(
        self,
        osnet_weights_path: str,
        device: str = "cuda",
        gid_similarity_threshold: float = 0.75,
        movement_px_threshold: float = 20.0,
        stop_radius_px: float = 25.0,
        stop_min_points: int = 4,
    ):
        self.extractor = OSNetExtractor(osnet_weights_path, device=device)
        self.gid_similarity_threshold = float(gid_similarity_threshold)
        self.movement_px_threshold = float(movement_px_threshold)
        self.stop_radius_px = float(stop_radius_px)
        self.stop_min_points = int(stop_min_points)

        self._tracks: dict[int, Track] = {}
        self._global_db: dict[int, np.ndarray] = {}
        self._next_gid = 1
        self._traj: dict[int, list[dict[str, Any]]] = {}

    def begin_video(self) -> None:
        self._tracks = {}

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
        tracked_dets: list[tuple[int, np.ndarray, float]],
        timestamp_sec: Optional[float] = None,
        video_path: Optional[str] = None,
    ) -> list[str]:
        h, w = frame_bgr.shape[:2]
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

        for local_tid, b, conf in tracked_dets:
            b = np.asarray(b, dtype=np.float32)
            cxy = np.array([(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0], dtype=np.float32)
            emb = self.extractor(crop(b))

            tr = self._tracks.get(int(local_tid))
            if tr is None:
                gid = self._assign_gid(emb)
                tr = Track(
                    local_track_id=int(local_tid),
                    bbox_xyxy=b,
                    last_center=cxy,
                    last_seen_frame_id=int(frame_id),
                    gid=int(gid),
                    emb=emb,
                    emb_count=1,
                )
                self._tracks[int(local_tid)] = tr
                ctx.append(f"ReID: person_{tr.gid} track_{tr.local_track_id} appeared conf={conf:.2f}")
            else:
                dx, dy = (cxy - tr.last_center).tolist()
                moved = (abs(dx) + abs(dy)) >= self.movement_px_threshold
                move_txt = f" moved dx={dx:.1f},dy={dy:.1f}" if moved else ""
                tr.emb = _l2_normalize((tr.emb * tr.emb_count + emb) / (tr.emb_count + 1))
                tr.emb_count += 1
                tr.bbox_xyxy = b
                tr.last_center = cxy
                tr.last_seen_frame_id = int(frame_id)
                ctx.append(f"ReID: person_{tr.gid} track_{tr.local_track_id}{move_txt} conf={conf:.2f}")

            self._traj.setdefault(tr.gid, []).append(
                {
                    "timestamp_sec": float(timestamp_sec) if timestamp_sec is not None else None,
                    "frame_id": int(frame_id),
                    "x": float(cxy[0]),
                    "y": float(cxy[1]),
                    "video_path": video_path,
                    "local_track_id": int(local_tid),
                    "bbox_xyxy": [float(x) for x in b.tolist()],
                    "conf": float(conf),
                }
            )

        return ctx

    def summarize_routes(self) -> dict[str, Any]:
        out: dict[str, Any] = {"schema": "person_routes_v1", "people": []}
        for gid, pts in self._traj.items():
            pts2 = [p for p in pts if p.get("x") is not None and p.get("y") is not None]
            pts2.sort(key=lambda p: (p.get("video_path") or "", p.get("timestamp_sec") or -1, p.get("frame_id") or 0))

            # stop segments (универсальные "обслуживание началось/закончилось" как стоп)
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
                    xs = [float(x["x"]) for x in cur]
                    ys = [float(x["y"]) for x in cur]
                    cur_center = np.array([sum(xs) / len(xs), sum(ys) / len(ys)], dtype=np.float32)
                else:
                    if len(cur) >= self.stop_min_points:
                        segments.append(_segment_from_points(cur))
                    cur = [p]
                    cur_center = xy
            if cur and len(cur) >= self.stop_min_points:
                segments.append(_segment_from_points(cur))

            out["people"].append({"person_id": int(gid), "points": pts2, "stop_segments": segments})
        return out


def _segment_from_points(points: list[dict[str, Any]]) -> dict[str, Any]:
    xs = [float(p["x"]) for p in points]
    ys = [float(p["y"]) for p in points]
    return {
        "start_timestamp_sec": points[0].get("timestamp_sec"),
        "end_timestamp_sec": points[-1].get("timestamp_sec"),
        "count": len(points),
        "center_x": float(sum(xs) / len(xs)),
        "center_y": float(sum(ys) / len(ys)),
        "video_path": points[0].get("video_path"),
        "start_frame_id": points[0].get("frame_id"),
        "end_frame_id": points[-1].get("frame_id"),
    }

