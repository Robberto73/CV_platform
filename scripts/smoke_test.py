from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.run_pipeline import run_pipeline


def make_test_video(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fps = 10
    w, h = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    if not out.isOpened():
        raise RuntimeError("cv2.VideoWriter failed to open (mp4v). Check ffmpeg/codecs.")
    for i in range(20):  # 2 seconds
        img = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(
            img,
            f"frame {i}",
            (30, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.rectangle(img, (10 + i * 3, 50), (80 + i * 3, 150), (0, 255, 0), 2)
        out.write(img)
    out.release()


def assert_out_has(out_dir: str, filenames: list[str]) -> None:
    missing = [f for f in filenames if not os.path.exists(os.path.join(out_dir, f))]
    if missing:
        raise AssertionError(f"Missing in {out_dir}: {missing}")


def main() -> int:
    video_path = Path("tmp_test") / "test.mp4"
    make_test_video(video_path)

    # 1) PRO + require_visual_analysis=False (через текст) — без LLaVA/GigaChat
    state = {
        "video_paths": [str(video_path)],
        "user_query": "без визуального анализа. верни JSON: сколько людей в кадре?",
        "ui_mode": "PRO",
        "pro_settings": {
            "frame_sampling_rate": 1.0,
            "ssim_threshold": 0.9,
            "skip_static_frames": True,
            "cache_frames": True,
            "custom_preprocessing": "None",
        },
        "require_json": True,
        "llm_type": "local",
        "gigachat_api_key": None,
        "analyze_people": True,
        "processing_log": [],
    }
    res = run_pipeline(state)
    out_dir = res.get("result_path") or res.get("output_dir")
    if not out_dir:
        raise AssertionError("No result_path/output_dir in pipeline result")
    assert_out_has(out_dir, ["events.parquet", "metadata.yaml", "processing_log.log"])
    if res.get("require_json"):
        assert_out_has(out_dir, ["answer.json"])
    else:
        assert_out_has(out_dir, ["answer.txt"])

    print("SMOKE OK")
    print("out_dir:", out_dir)
    print("error:", res.get("error"))
    print("events:", len(res.get("events") or []))
    print("final_answer_type:", type(res.get("final_answer")).__name__)
    return 0


if __name__ == "__main__":
    sys.exit(main())

