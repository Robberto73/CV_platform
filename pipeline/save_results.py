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
import cv2
import numpy as np
from typing import Dict, List, Tuple, Any
import os


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


def save_unique_people_photos(
    out_dir: str,
    reid_trajectories: Dict[int, List[Dict[str, Any]]],
    video_paths: List[str],
    min_faces: int = 3,
    quality_threshold: float = 0.7
) -> Dict[str, Any]:
    """
    Сохраняет уникальных людей из ReID данных.
    Возвращает информацию о сохраненных фото и пересечениях.
    """
    if not reid_trajectories:
        return {"saved_photos": [], "cross_video_overlaps": {}}

    unique_people_dir = os.path.join(out_dir, "unique_people")
    os.makedirs(unique_people_dir, exist_ok=True)

    saved_photos = []
    person_video_map = {}  # person_id -> set of video_paths

    # Группируем по person_id и выбираем лучшие фото
    for person_id, trajectory in reid_trajectories.items():
        if len(trajectory) < min_faces:
            continue

        # Группируем по видео
        video_groups = {}
        for point in trajectory:
            video_path = point.get("video_path", "")
            if video_path not in video_groups:
                video_groups[video_path] = []
            video_groups[video_path].append(point)

        # Для каждого видео сохраняем лучшее фото этого человека
        for video_path, points in video_groups.items():
            if not points:
                continue

            # Выбираем лучшее фото
            best_point = select_best_photo_point(points, quality_threshold)
            if best_point:
                # Сохраняем фото
                photo_path = save_person_photo(
                    unique_people_dir, video_path, person_id, best_point
                )
                if photo_path:
                    saved_photos.append({
                        "person_id": person_id,
                        "video_path": video_path,
                        "photo_path": photo_path,
                        "timestamp_sec": best_point.get("timestamp_sec", 0),
                        "frame_id": best_point.get("frame_id", 0),
                        "confidence": best_point.get("conf", 0)
                    })

                    # Запоминаем пересечения
                    if person_id not in person_video_map:
                        person_video_map[person_id] = set()
                    person_video_map[person_id].add(video_path)

    # Создаем отчет о пересечениях
    cross_video_overlaps = {}
    for person_id, video_set in person_video_map.items():
        if len(video_set) > 1:
            cross_video_overlaps[str(person_id)] = {
                "video_count": len(video_set),
                "videos": sorted(list(video_set))
            }

    # Сохраняем отчет о пересечениях
    if cross_video_overlaps:
        overlaps_file = os.path.join(unique_people_dir, "cross_video_overlaps.json")
        with open(overlaps_file, "w", encoding="utf-8") as f:
            json.dump({
                "cross_video_people": cross_video_overlaps,
                "total_unique_people": len(reid_trajectories),
                "people_with_multiple_videos": len(cross_video_overlaps)
            }, f, ensure_ascii=False, indent=2)

    return {
        "saved_photos": saved_photos,
        "cross_video_overlaps": cross_video_overlaps,
        "unique_people_dir": unique_people_dir
    }


def select_best_photo_point(points: List[Dict[str, Any]], quality_threshold: float) -> Dict[str, Any]:
    """
    Выбирает лучшее фото из траектории человека.
    Критерии: уверенность + расположение в кадре + четкость.
    """
    if not points:
        return None

    scored_points = []
    for point in points:
        conf = point.get("conf", 0)
        if conf < quality_threshold:
            continue

        # Оцениваем расположение в кадре (предполагаем стандартный размер кадра)
        bbox = point.get("bbox_xyxy", [])
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # Штраф за края кадра (предполагаем 1920x1080)
            edge_penalty = 0
            if center_x < 200 or center_x > 1720:  # близко к левому/правому краю
                edge_penalty += 0.2
            if center_y < 150 or center_y > 930:  # близко к верхнему/нижнему краю
                edge_penalty += 0.2

            # Бонус за размер лица
            size_bonus = min(width * height / 50000, 0.3)  # нормализуем

            score = conf + size_bonus - edge_penalty
        else:
            score = conf

        scored_points.append((score, point))

    if not scored_points:
        return None

    # Возвращаем точку с максимальным скором
    scored_points.sort(key=lambda x: x[0], reverse=True)
    return scored_points[0][1]


def save_person_photo(
    unique_people_dir: str,
    video_path: str,
    person_id: int,
    point: Dict[str, Any]
) -> str:
    """
    Сохраняет фото человека из видео по заданной точке.
    """
    try:
        # Открываем видео
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        frame_id = point.get("frame_id", 0)

        # Переходим к нужному кадру
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            return None

        # Вырезаем лицо по bbox
        bbox = point.get("bbox_xyxy", [])
        if len(bbox) == 4:
            x1, y1, x2, y2 = [int(round(v)) for v in bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

            if x2 > x1 and y2 > y1:
                face_crop = frame[y1:y2, x1:x2]
            else:
                face_crop = frame
        else:
            face_crop = frame

        # Создаем имя файла: video_name_person_id_timestamp.jpg
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        timestamp_sec = point.get("timestamp_sec", 0)
        timestamp_str = f"{int(timestamp_sec):03d}"
        filename = f"{video_name}_person_{person_id}_{timestamp_str}.jpg"
        filepath = os.path.join(unique_people_dir, filename)

        # Сохраняем фото
        success = cv2.imwrite(filepath, face_crop)
        if success:
            return filepath
        else:
            return None

    except Exception as e:
        print(f"Error saving person photo: {e}")
        return None

