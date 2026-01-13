from __future__ import annotations

import io
import json
import os
import zipfile
from datetime import datetime
from typing import Any, Optional

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from pipeline.utils.frame_processing import sha256_text


def make_output_dir(base_out: str, user_query: str) -> str:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    h = sha256_text(user_query)[:10]
    out_dir = os.path.join(base_out, f"{ts}_{h}")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "cache"), exist_ok=True)
    return out_dir


def save_answer(out_dir: str, require_json: bool, final_answer: Any) -> str:
    if require_json:
        path = os.path.join(out_dir, "answer.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(final_answer, f, ensure_ascii=False, indent=2)
        return path
    path = os.path.join(out_dir, "answer.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(str(final_answer))
    return path


def save_events_parquet(out_dir: str, events_rows: list[dict[str, Any]]) -> str:
    path = os.path.join(out_dir, "events.parquet")
    table = pa.Table.from_pylist(events_rows)
    pq.write_table(table, path)
    return path


def save_metadata(out_dir: str, metadata: dict[str, Any]) -> str:
    path = os.path.join(out_dir, "metadata.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(metadata, f, allow_unicode=True, sort_keys=False)
    return path


def save_processing_log(out_dir: str, lines: list[str]) -> str:
    path = os.path.join(out_dir, "processing_log.log")
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line.rstrip("\n") + "\n")
    return path


def zip_dir_to_bytes(folder: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(folder):
            for fn in files:
                full = os.path.join(root, fn)
                rel = os.path.relpath(full, folder)
                zf.write(full, arcname=rel)
    return buf.getvalue()


def safe_relpath(path: str, start: str) -> str:
    try:
        return os.path.relpath(path, start)
    except Exception:
        return path

