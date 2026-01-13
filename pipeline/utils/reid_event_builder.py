from __future__ import annotations

from typing import Any


def build_reid_events(
    person_routes: dict[str, Any],
    mode: str,
    video_path_hint: str | None = None,
    frames_min_dt_sec: float = 0.5,
    frames_max_points_per_person: int = 2000,
    frames_max_total_events: int = 20000,
) -> list[dict[str, Any]]:
    """
    Превращает person_routes.json (schema person_routes_v1) в события для events.parquet/финального ответа.

    mode:
      - "segments": компактные stop_segments (best practice для длинных видео)
      - "frames": детальные события по точкам траектории
    """
    people = person_routes.get("people") or []
    out: list[dict[str, Any]] = []

    if mode not in ("segments", "frames"):
        mode = "segments"

    for p in people:
        if len(out) >= int(frames_max_total_events) and mode == "frames":
            break
        pid = int(p.get("person_id", 0))
        pts = p.get("points") or []
        stops = p.get("stop_segments") or []

        if mode == "segments":
            for s in stops:
                start_ts = s.get("start_timestamp_sec")
                end_ts = s.get("end_timestamp_sec")
                dur = None
                try:
                    if start_ts is not None and end_ts is not None:
                        dur = float(end_ts) - float(start_ts)
                except Exception:
                    dur = None

                vp = s.get("video_path") or video_path_hint or ""
                out.append(
                    {
                        "event_id": f"{_basename(vp)}_p{pid}_stop_{_safe_t(start_ts)}",
                        "video_path": vp,
                        "timestamp_sec": float(start_ts) if start_ts is not None else 0.0,
                        "frame_id": int(s.get("start_frame_id") or 0),
                        "event_type": "person_stop_segment",
                        "entities": [
                            {
                                "id": f"person_{pid}",
                                "type": "person",
                                "center_x": s.get("center_x"),
                                "center_y": s.get("center_y"),
                                "duration_sec": dur,
                            }
                        ],
                        "llava_analysis": f"person_{pid} stayed near ({s.get('center_x'):.1f},{s.get('center_y'):.1f}) for ~{dur:.1f}s"
                        if dur is not None and s.get("center_x") is not None
                        else f"person_{pid} stop_segment",
                        "confidence": 0.6,
                        "source_frames": [int(s.get("start_frame_id") or 0), int(s.get("end_frame_id") or 0)],
                    }
                )

            # переходы между стопами как "moved_between_stops"
            for a, b in zip(stops, stops[1:]):
                try:
                    ax, ay = float(a.get("center_x")), float(a.get("center_y"))
                    bx, by = float(b.get("center_x")), float(b.get("center_y"))
                    dist = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
                except Exception:
                    dist = None
                t = b.get("start_timestamp_sec")
                vp = b.get("video_path") or a.get("video_path") or video_path_hint or ""
                out.append(
                    {
                        "event_id": f"{_basename(vp)}_p{pid}_move_{_safe_t(t)}",
                        "video_path": vp,
                        "timestamp_sec": float(t) if t is not None else 0.0,
                        "frame_id": int(b.get("start_frame_id") or 0),
                        "event_type": "person_moved_between_stops",
                        "entities": [{"id": f"person_{pid}", "type": "person", "distance_px": dist}],
                        "llava_analysis": f"person_{pid} moved between two stop areas (distance≈{dist:.1f}px)"
                        if dist is not None
                        else f"person_{pid} moved between stop areas",
                        "confidence": 0.55,
                        "source_frames": [int(a.get("end_frame_id") or 0), int(b.get("start_frame_id") or 0)],
                    }
                )

        else:  # frames
            # Downsample по времени: сохраняем не чаще, чем frames_min_dt_sec.
            # Это резко уменьшает нагрузку и размер events.parquet на длинных видео.
            min_dt = max(0.0, float(frames_min_dt_sec))
            max_pts = max(1, int(frames_max_points_per_person))
            last_t = None
            kept = 0
            for pt in pts:
                if len(out) >= int(frames_max_total_events):
                    break
                if kept >= max_pts:
                    break
                vp = pt.get("video_path") or video_path_hint or ""
                ts = pt.get("timestamp_sec")
                fid = pt.get("frame_id") or 0
                if ts is not None and min_dt > 0:
                    try:
                        t = float(ts)
                        if last_t is not None and (t - last_t) < min_dt:
                            continue
                        last_t = t
                    except Exception:
                        pass
                out.append(
                    {
                        "event_id": f"{_basename(vp)}_p{pid}_pos_{fid}",
                        "video_path": vp,
                        "timestamp_sec": float(ts) if ts is not None else 0.0,
                        "frame_id": int(fid),
                        "event_type": "person_position",
                        "entities": [
                            {
                                "id": f"person_{pid}",
                                "type": "person",
                                "x": pt.get("x"),
                                "y": pt.get("y"),
                                "local_track_id": pt.get("local_track_id"),
                                "bbox_xyxy": pt.get("bbox_xyxy"),
                                "confidence": pt.get("conf"),
                            }
                        ],
                        "llava_analysis": f"person_{pid} position update",
                        "confidence": float(pt.get("conf") or 0.5),
                        "source_frames": [int(fid)],
                    }
                )
                kept += 1

    return out


def _basename(path: str) -> str:
    try:
        import os

        return os.path.basename(path) or "video"
    except Exception:
        return "video"


def _safe_t(t: Any) -> str:
    try:
        if t is None:
            return "na"
        return str(round(float(t), 1)).replace(".", "_")
    except Exception:
        return "na"

