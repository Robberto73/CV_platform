from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, Optional


_YOLO_PERSON_RE = re.compile(r"\\bYOLO:\\s*person\\b", re.IGNORECASE)
_YOLO_PERSON_COUNT_RE = re.compile(r"YOLO:\\s*person\\s+count\\s*=\\s*(\\d+)", re.IGNORECASE)


def _stable_hash(s: str) -> str:
    h = hashlib.sha256()
    h.update(s.encode("utf-8", errors="ignore"))
    return h.hexdigest()[:12]


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _sec_to_hhmmss(sec: float) -> str:
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


@dataclass(frozen=True)
class SummarizationConfig:
    chunk_sec: float = 30.0
    max_chunks: int = 240  # 2 часа при 30с чанках
    max_events_per_chunk: int = 6
    max_evidence_events: int = 60


def chunk_events(events: list[dict[str, Any]], chunk_sec: float) -> dict[int, list[dict[str, Any]]]:
    """
    Группировка событий в чанки по времени.
    chunk_id = floor(timestamp_sec / chunk_sec)
    """
    out: dict[int, list[dict[str, Any]]] = {}
    cs = max(1.0, float(chunk_sec))
    for ev in events or []:
        t = _safe_float(ev.get("timestamp_sec"), 0.0)
        cid = int(math.floor(t / cs))
        out.setdefault(cid, []).append(ev)
    return out


def _count_people_from_cv_text(text: str) -> int:
    if not text:
        return 0
    # Новый формат: "YOLO: person count=N ..."
    m = _YOLO_PERSON_COUNT_RE.search(text)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    # Старый формат: "YOLO: person ...; YOLO: person ..."
    return len(_YOLO_PERSON_RE.findall(text))


def summarize_events_algorithmic(
    events: list[dict[str, Any]],
    cfg: SummarizationConfig,
) -> dict[str, Any]:
    """
    Универсальная (без LLM) сводка:
    - чанки по времени
    - пики по людям (из cv_detection текста)
    - топ событий по confidence
    - ограничение размера
    """
    if not events:
        return {
            "schema": "events_summary_v1",
            "chunk_sec": cfg.chunk_sec,
            "total_events": 0,
            "chunks": [],
            "global": {"people_max_seen": 0, "duration_sec_est": 0.0},
        }

    # порядок по времени
    events_sorted = sorted(events, key=lambda e: _safe_float(e.get("timestamp_sec"), 0.0))
    duration = _safe_float(events_sorted[-1].get("timestamp_sec"), 0.0)

    chunks_map = chunk_events(events_sorted, cfg.chunk_sec)
    chunk_ids = sorted(chunks_map.keys())[: int(cfg.max_chunks)]

    chunks_out: list[dict[str, Any]] = []
    global_people_max = 0

    for cid in chunk_ids:
        items = chunks_map.get(cid, [])
        start = cid * cfg.chunk_sec
        end = start + cfg.chunk_sec

        # people max in this chunk (best-effort)
        people_max = 0
        for ev in items:
            if str(ev.get("event_type", "")) == "cv_detection":
                people_max = max(people_max, _count_people_from_cv_text(str(ev.get("llava_analysis", ""))))

        global_people_max = max(global_people_max, people_max)

        # notable events: top by confidence with dedupe
        scored: list[tuple[float, dict[str, Any]]] = []
        for ev in items:
            conf = _safe_float(ev.get("confidence"), 0.0)
            scored.append((conf, ev))
        scored.sort(key=lambda x: x[0], reverse=True)

        seen: set[str] = set()
        notable: list[dict[str, Any]] = []
        for conf, ev in scored:
            desc = str(ev.get("llava_analysis", "") or ev.get("event_type", ""))
            key = _stable_hash(f"{ev.get('event_type')}|{desc[:200]}")
            if key in seen:
                continue
            seen.add(key)
            notable.append(
                {
                    "t": _sec_to_hhmmss(_safe_float(ev.get("timestamp_sec"), 0.0)),
                    "timestamp_sec": _safe_float(ev.get("timestamp_sec"), 0.0),
                    "event_id": str(ev.get("event_id", "")),
                    "event_type": str(ev.get("event_type", "")),
                    "confidence": conf,
                    "summary": desc[:220],
                }
            )
            if len(notable) >= int(cfg.max_events_per_chunk):
                break

        chunks_out.append(
            {
                "chunk_id": cid,
                "start_sec": float(start),
                "end_sec": float(end),
                "start": _sec_to_hhmmss(start),
                "end": _sec_to_hhmmss(end),
                "people_max_seen": int(people_max),
                "notable_events": notable,
                "events_count": len(items),
            }
        )

    return {
        "schema": "events_summary_v1",
        "chunk_sec": cfg.chunk_sec,
        "total_events": len(events),
        "chunks": chunks_out,
        "global": {
            "people_max_seen": int(global_people_max),
            "duration_sec_est": float(duration),
        },
    }


def format_summary_for_prompt(summary: dict[str, Any], max_chars: int = 6000) -> str:
    """
    Компактное текстовое представление, чтобы LLM не утонула в токенах.
    """
    chunks = summary.get("chunks") or []
    glob = summary.get("global") or {}
    lines: list[str] = []
    lines.append(f"SUMMARY: duration≈{float(glob.get('duration_sec_est', 0.0)):.1f}s, people_max≈{int(glob.get('people_max_seen', 0))}")
    lines.append(f"chunk_sec={summary.get('chunk_sec')}, chunks={len(chunks)}, total_events={summary.get('total_events')}")
    lines.append("")
    for ch in chunks:
        header = f"[chunk {ch.get('chunk_id')}] {ch.get('start')}–{ch.get('end')} | people_max={ch.get('people_max_seen')} | events={ch.get('events_count')}"
        lines.append(header)
        for ev in ch.get("notable_events") or []:
            lines.append(
                f"- {ev.get('t')} {ev.get('event_type')} conf={float(ev.get('confidence', 0.0)):.2f} id={ev.get('event_id')}: {ev.get('summary')}"
            )
        lines.append("")

        if sum(len(x) + 1 for x in lines) > max_chars:
            lines.append("[TRUNCATED]")
            break

    text = "\n".join(lines)
    return text[:max_chars]


def build_query_evidence(
    events: list[dict[str, Any]],
    summary: dict[str, Any],
    max_evidence_events: int = 60,
) -> dict[str, Any]:
    """
    Универсальный набор “доказательств” без привязки к конкретным объектам:
    берём верхние по confidence события, чтобы финальная LLM могла ссылаться на event_id.
    """
    ev_sorted = sorted(events or [], key=lambda e: _safe_float(e.get("confidence"), 0.0), reverse=True)
    evidence = []
    for ev in ev_sorted[: int(max_evidence_events)]:
        evidence.append(
            {
                "timestamp_sec": _safe_float(ev.get("timestamp_sec"), 0.0),
                "time": _sec_to_hhmmss(_safe_float(ev.get("timestamp_sec"), 0.0)),
                "event_id": str(ev.get("event_id", "")),
                "event_type": str(ev.get("event_type", "")),
                "confidence": _safe_float(ev.get("confidence"), 0.0),
                "text": str(ev.get("llava_analysis", ""))[:240],
            }
        )
    return {
        "schema": "query_evidence_v1",
        "evidence": evidence,
        "summary_stats": summary.get("global", {}),
    }

