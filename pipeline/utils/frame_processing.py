from __future__ import annotations

import hashlib
import io
import os
import pickle
from dataclasses import dataclass
from typing import Any, Callable, Optional

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim


def md5_bytes(data: bytes) -> str:
    h = hashlib.md5()
    h.update(data)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    h = hashlib.sha256()
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def frame_to_jpeg_bytes(frame_bgr: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Failed to encode frame to JPEG")
    return buf.tobytes()


def resize_for_llava(frame_bgr: np.ndarray, size: int = 336) -> Image.Image:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    return pil.resize((size, size))


def compute_ssim(prev_bgr: np.ndarray, curr_bgr: np.ndarray) -> float:
    prev = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    curr = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
    return float(ssim(prev, curr))


@dataclass
class FrameSelectorConfig:
    frame_sampling_rate: float = 1.0
    ssim_threshold: float = 0.9
    skip_static_frames: bool = True
    cache_frames: bool = True
    custom_preprocessing: str = "None"


def standard_frame_selector() -> Callable[[dict[str, Any]], bool]:
    def _keep(_: dict[str, Any]) -> bool:
        return True

    return _keep


def pro_frame_selector(cfg: FrameSelectorConfig) -> Callable[[dict[str, Any]], bool]:
    def _keep(frame_meta: dict[str, Any]) -> bool:
        # Решение уже включает SSIM/кэш на уровне decode_and_select_frames;
        # тут оставляем hook под кастомную предобработку (заглушка).
        _ = frame_meta
        return True

    return _keep


class FrameHashCache:
    def __init__(self, maxsize: int = 1000):
        self.maxsize = int(maxsize)
        self._order: list[str] = []
        self._set: set[str] = set()

    def __contains__(self, key: str) -> bool:
        return key in self._set

    def add(self, key: str) -> None:
        if key in self._set:
            return
        self._set.add(key)
        self._order.append(key)
        if len(self._order) > self.maxsize:
            oldest = self._order.pop(0)
            self._set.discard(oldest)

    def dump(self) -> dict[str, Any]:
        return {"maxsize": self.maxsize, "items": list(self._order)}

    @classmethod
    def load(cls, payload: dict[str, Any]) -> "FrameHashCache":
        obj = cls(maxsize=int(payload.get("maxsize", 1000)))
        for k in payload.get("items", []):
            obj.add(str(k))
        return obj


def save_frame_hashes(cache_path: str, cache: FrameHashCache) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(cache.dump(), f)


def load_frame_hashes(cache_path: str) -> Optional[FrameHashCache]:
    if not os.path.exists(cache_path):
        return None
    with open(cache_path, "rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict):
        return FrameHashCache.load(payload)
    return None


def decode_and_select_frames(
    video_path: str,
    frame_sampling_rate: float,
    mode: str,
    ssim_threshold: float = 0.9,
    skip_static_frames: bool = True,
    cache_frames: bool = True,
    hash_cache: Optional[FrameHashCache] = None,
    keep_hook: Optional[Callable[[dict[str, Any]], bool]] = None,
    max_frames: Optional[int] = None,
) -> list[dict[str, Any]]:
    """
    Возвращает список кадров:
    {
      "frame_id": int,
      "timestamp_sec": float,
      "frame_bgr": np.ndarray,
      "frame_hash": str
    }
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 25.0

    step = max(1, int(round(fps / max(0.1, float(frame_sampling_rate)))))
    frames: list[dict[str, Any]] = []
    prev_frame: Optional[np.ndarray] = None

    idx = 0
    kept = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if idx % step != 0:
                idx += 1
                continue

            timestamp_sec = float(idx / fps)
            jpeg = frame_to_jpeg_bytes(frame)
            frame_hash = md5_bytes(jpeg)

            if mode == "PRO" and cache_frames and hash_cache is not None:
                if frame_hash in hash_cache:
                    idx += 1
                    continue

            if mode == "PRO" and skip_static_frames and prev_frame is not None:
                score = compute_ssim(prev_frame, frame)
                if score >= float(ssim_threshold):
                    idx += 1
                    continue

            meta = {
                "frame_id": int(idx),
                "timestamp_sec": timestamp_sec,
                "frame_bgr": frame,
                "frame_hash": frame_hash,
            }
            if keep_hook is not None and not keep_hook(meta):
                idx += 1
                continue

            frames.append(meta)
            kept += 1
            prev_frame = frame

            if mode == "PRO" and cache_frames and hash_cache is not None:
                hash_cache.add(frame_hash)

            if max_frames is not None and kept >= int(max_frames):
                break

            idx += 1
    finally:
        cap.release()

    return frames


def pil_to_base64_jpeg(pil: Image.Image, quality: int = 85) -> str:
    import base64

    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

