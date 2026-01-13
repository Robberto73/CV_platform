from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.config import load_gigachat_ca_bundle_file, load_gigachat_default_key
from pipeline.run_pipeline import run_pipeline


def make_test_video(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fps = 5
    w, h = 320, 240
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not out.isOpened():
        raise RuntimeError("VideoWriter failed; check codecs.")
    for i in range(10):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(img, f"t{i}", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        out.write(img)
    out.release()


def main() -> int:
    vp = Path("tmp_test") / "gigachat_test.mp4"
    make_test_video(vp)

    state = {
        "video_paths": [str(vp)],
        "user_query": "Опиши, что на кадре. Коротко.",
        "ui_mode": "STANDARD",
        "pro_settings": {},
        "require_json": False,
        "vision_llm": "gigachat_api",
        "final_llm": "gigachat_api",
        "gigachat_api_key": load_gigachat_default_key(),
        "gigachat_ca_cert_path": load_gigachat_ca_bundle_file(),
        "force_no_visual_analysis": False,
        "processing_log": [],
    }
    res = run_pipeline(state)
    out_dir = res.get("result_path") or res.get("output_dir")
    print("out_dir:", out_dir)
    print("error:", res.get("error"), res.get("error_message"))
    print("events:", len(res.get("events") or []))
    print("final_answer:", str(res.get("final_answer"))[:200])
    if out_dir:
        print("files:", sorted(os.listdir(out_dir)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

