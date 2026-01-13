from __future__ import annotations

import argparse
from pathlib import Path

from pipeline.run_pipeline import run_pipeline
from pipeline.state import PipelineState


def main() -> int:
    ap = argparse.ArgumentParser(description="Local benchmark for CV_platform pipeline")
    ap.add_argument("--video", type=str, default=r"tmp_test\1000012546.mp4", help="Path to video")
    ap.add_argument("--mode", type=str, default="STANDARD", choices=["STANDARD", "PRO"])
    ap.add_argument("--fps", type=float, default=1.0, help="frame_sampling_rate for PRO")
    args = ap.parse_args()

    video = Path(args.video)
    if not video.exists():
        raise SystemExit(f"Video not found: {video}")

    pro_settings = {}
    if args.mode == "PRO":
        pro_settings = {
            "frame_sampling_rate": float(args.fps),
            "ssim_threshold": 0.9,
            "skip_static_frames": True,
            "cache_frames": False,
            "custom_preprocessing": "None",
        }

    # Benchmark in CV-only mode (no LLM calls), to measure video decode + CV + IO.
    s: PipelineState = {
        "video_paths": [str(video)],
        "user_query": "опиши видео",
        "ui_mode": args.mode,
        "pro_settings": pro_settings,
        "require_json": False,
        "vision_llm": "off",
        "final_llm": "gigachat_api",
        "gigachat_api_key": None,
        "gigachat_ca_cert_path": None,
        "analyze_people": True,
        "force_no_visual_analysis": True,
        "processing_log": [],
    }

    out = run_pipeline(s)
    print("out_dir:", out.get("result_path"))
    print("error:", out.get("error"), out.get("error_code"))
    print("events:", len(out.get("events") or []))
    print("")
    print("=== timings ===")
    for line in out.get("processing_log") or []:
        if "selected_frames=" in line or "took " in line:
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

