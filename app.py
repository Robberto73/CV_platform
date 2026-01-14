from __future__ import annotations

import os
import time
import uuid
from typing import Any

import pandas as pd
import streamlit as st
import hashlib
try:
    from streamlit_agraph import Config, Edge, Node, agraph

    _AGRAPH_AVAILABLE = True
except Exception:
    Config = Edge = Node = agraph = None  # type: ignore
    _AGRAPH_AVAILABLE = False

from pipeline.config import (
    describe_loaded_config_for_ui,
    load_app_settings,
    load_gigachat_ca_bundle_file,
    load_gigachat_default_key,
    save_app_settings,
)
from pipeline.run_pipeline import run_pipeline
from pipeline.state import PipelineState


st.set_page_config(page_title="CV Platform — Video Analytics", layout="wide")

_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def _normalize_path(p: str) -> str:
    p = (p or "").strip().strip('"').strip("'")
    if not p:
        return ""
    return os.path.abspath(os.path.expandvars(os.path.expanduser(p)))


def _validate_video_file(path: str) -> str:
    p = _normalize_path(path)
    if not p or not os.path.isfile(p):
        return ""
    ext = os.path.splitext(p)[1].lower()
    return p if ext in _VIDEO_EXTS else ""


def _collect_videos_from_folder(folder: str, max_files: int = 10) -> list[str]:
    folder = _normalize_path(folder)
    if not folder or not os.path.isdir(folder):
        return []
    out: list[str] = []
    try:
        for root, _, files in os.walk(folder):
            for fn in files:
                ext = os.path.splitext(fn)[1].lower()
                if ext in _VIDEO_EXTS:
                    out.append(os.path.join(root, fn))
            # детерминируем порядок и ограничиваем
            out.sort()
            if len(out) >= int(max_files):
                return out[: int(max_files)]
    except Exception:
        return []
    return out[: int(max_files)]


def _save_uploads(files: list[Any]) -> list[str]:
    session_id = str(uuid.uuid4())
    base = os.path.join(".tmp_uploads", session_id)
    os.makedirs(base, exist_ok=True)
    out_paths: list[str] = []
    for f in files:
        path = os.path.join(base, f.name)
        with open(path, "wb") as w:
            w.write(f.getbuffer())
        out_paths.append(path)
    return out_paths


def _render_timeline(output_dir: str, events_df: pd.DataFrame) -> None:
    if not _AGRAPH_AVAILABLE:
        st.warning("streamlit-agraph не доступен в окружении — визуализация отключена.")
        return
    if events_df.empty:
        st.info("Нет событий для визуализации.")
        return

    nodes = []
    edges = []
    last_id = None
    safe_to_event_id: dict[str, str] = {}

    def _safe_node_id(raw: str) -> str:
        # streamlit_agraph иногда пытается загрузить Node.id как статический ресурс;
        # для event_id с точками/слешами это может приводить к FileNotFoundError.
        # Делаем безопасный детерминированный id.
        return "ev_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    for _, row in events_df.sort_values("timestamp_sec").iterrows():
        ev_id = str(row.get("event_id", ""))
        label = f"{row.get('timestamp_sec', 0):.1f}s • {row.get('event_type','')}"
        sid = _safe_node_id(ev_id)
        safe_to_event_id[sid] = ev_id
        nodes.append(Node(id=sid, label=label, size=18, title=ev_id))
        if last_id is not None:
            edges.append(Edge(source=last_id, target=sid))
        last_id = sid

    cfg = Config(
        width="100%",
        height=450,
        directed=True,
        physics=True,
        hierarchical=False,
    )
    sel = agraph(nodes=nodes, edges=edges, config=cfg)

    # best-effort: показать кадр, если он был закэширован
    if isinstance(sel, dict) and sel.get("selected"):
        selected_safe_id = str(sel["selected"])
        selected_event_id = safe_to_event_id.get(selected_safe_id, selected_safe_id)
        st.caption(f"Выбрано событие: {selected_event_id}")
        frames_dir = os.path.join(output_dir, "cache", "frames")
        if os.path.isdir(frames_dir):
            # пробуем найти jpg по basename+frame_id — в этой версии это best-effort
            import glob

            matches = glob.glob(os.path.join(frames_dir, "*.jpg"))
            if matches:
                st.image(matches[0], caption="Кадр (пример из cache/frames)", use_column_width=True)


st.title("Платформа анализа видео (LangGraph + Multimodal LLM)")

with st.sidebar:
    st.header("Настройки")

    cfg_info = describe_loaded_config_for_ui()
    with st.expander("Config (debug)", expanded=False):
        st.write(
            {
                "llava_model_id": cfg_info["llava_model_id"],
                "yolo_model_path": cfg_info["yolo_model_path"],
                "osnet_reid_model": cfg_info["osnet_reid_model"],
                "gigachat_key_present": cfg_info["gigachat_key_present"],
            }
        )

    ui_mode = st.radio("Режим", ["STANDARD", "PRO"], index=0, horizontal=True)
    force_no_visual_analysis = st.checkbox(
        "CV-only (без LLaVA/GigaChat)",
        value=False,
        help="Полезно для слабого ПК: пропускает мультимодальную LLM, но оставляет CV обработку и сохранение результатов.",
    )

    app_settings = load_app_settings()
    analyze_people = st.checkbox(
        "Анализ людей (YOLOv8)",
        value=bool(app_settings.get("analyze_people_default", True)),
        help="Если включено, детекция людей будет активна по умолчанию (yolov8n.pt).",
    )
    if st.button("Сохранить настройки по умолчанию"):
        app_settings["analyze_people_default"] = bool(analyze_people)
        save_app_settings(app_settings)
        st.success("Сохранено в config/app_settings.yaml")

    pro_settings = {
        "frame_sampling_rate": 1.0,
        "ssim_threshold": 0.9,
        "skip_static_frames": True,
        # По умолчанию выключаем: запись JPG на диск и упаковка в zip сильно замедляют обработку.
        "cache_frames": False,
        "custom_preprocessing": "None",
    }
    if ui_mode == "PRO":
        pro_settings["frame_sampling_rate"] = st.slider(
            "frame_sampling_rate (кадр/сек)", min_value=0.5, max_value=10.0, value=1.0, step=0.1
        )
        pro_settings["ssim_threshold"] = st.slider(
            "ssim_threshold", min_value=0.8, max_value=0.99, value=0.9, step=0.01
        )
        pro_settings["skip_static_frames"] = st.checkbox("skip_static_frames", value=True)
        pro_settings["cache_frames"] = st.checkbox("cache_frames", value=True)
        pro_settings["custom_preprocessing"] = st.selectbox(
            "custom_preprocessing",
            ["None", "Blur Detection (stub)", "Motion Emphasis (stub)"],
            index=0,
        )
        pro_settings["reid_event_mode"] = st.selectbox(
            "ReID события: режим детализации",
            ["segments (best practice)", "frames (detailed)"],
            index=0,
            help="Segments — компактные стоп-сегменты/переходы (лучше для длинных видео). Frames — события по кадрам/точкам (детально, но тяжелее).",
        )
        # нормализуем в внутреннее значение
        pro_settings["reid_event_mode"] = "segments" if "segments" in pro_settings["reid_event_mode"] else "frames"
        if pro_settings["reid_event_mode"] == "frames":
            pro_settings["reid_frames_min_dt_sec"] = st.slider(
                "ReID frames: min_dt_sec (downsample)",
                min_value=0.0,
                max_value=5.0,
                value=0.5,
                step=0.1,
                help="Минимальный интервал времени между position-событиями одной персоны. Повышайте для ускорения и уменьшения events.parquet.",
            )
            pro_settings["reid_frames_max_points_per_person"] = st.number_input(
                "ReID frames: max_points_per_person",
                min_value=50,
                max_value=20000,
                value=2000,
                step=50,
                help="Жёсткий лимит событий person_position на одну персону.",
            )
            pro_settings["reid_frames_max_total_events"] = st.number_input(
                "ReID frames: max_total_events",
                min_value=500,
                max_value=200000,
                value=20000,
                step=500,
                help="Жёсткий лимит всех ReID frame-событий на анализ (защита от раздувания).",
            )

    st.subheader("Модели")
    vision_llm_ui = st.radio(
        "Анализ кадров (куда уходит трафик изображений)",
        ["Local LLaVA", "GigaChat API", "Off (только CV)"],
        index=1,
    )
    final_llm_ui = st.radio(
        "Финальный ответ (текст/агрегация событий)",
        ["GigaChat API", "Local LLaVA (не рекомендуется)"],
        index=0,
    )

    gigachat_api_key = None
    gigachat_ca_cert_path = None
    need_gigachat = (vision_llm_ui == "GigaChat API") or (final_llm_ui == "GigaChat API")
    if need_gigachat:
        # По умолчанию берём ключ из config/gigachat_keys.json или env,
        # ручной ввод — только как override.
        default_key = load_gigachat_default_key()
        if default_key:
            st.caption("GigaChat key: найден в config/env (ввод не требуется)")
            override = st.checkbox("Переопределить ключ вручную", value=False)
            if override:
                gigachat_api_key = st.text_input("GigaChat API key (override)", type="password")
            else:
                gigachat_api_key = default_key
        else:
            st.warning("GigaChat key не найден в config/env — введите вручную")
        gigachat_api_key = st.text_input("GigaChat API key", type="password")

        default_ca = load_gigachat_ca_bundle_file()
        if default_ca:
            st.caption(f"CA bundle: найден в config/env ({default_ca})")
        gigachat_ca_cert_path = st.text_input(
            "CA bundle path (optional)",
            value=default_ca or "",
            help="Путь к файлу корневых сертификатов Минцифры. Если лежит в config/, подхватится автоматически.",
        ).strip() or None


st.subheader("Источник видео")
input_mode = st.radio(
    "Выберите, как задать видео",
    ["Загрузка (Upload)", "Путь к файлу", "Путь к папке"],
    index=0,
    horizontal=True,
    help="Upload удобен для разовых тестов. Путь к файлу/папке удобен на Linux-сервере (видео уже на диске).",
)

path_video_file = ""
path_video_folder = ""
selected_video_paths: list[str] = []

if input_mode == "Загрузка (Upload)":
    uploaded_files = st.file_uploader(
        "Загрузите видео (макс. 2 ГБ на файл)",
        type=["mp4", "avi", "mov"],
        accept_multiple_files=True,
    )
    if uploaded_files and len(uploaded_files) > 10:
        st.error("Можно загрузить максимум 10 файлов за раз.")
        uploaded_files = uploaded_files[:10]

    if uploaded_files:
        st.subheader("Превью")
        cols = st.columns(min(3, len(uploaded_files)))
        for i, f in enumerate(uploaded_files):
            with cols[i % len(cols)]:
                st.caption(f.name)
                st.video(f)
else:
    # В других режимах явно сбрасываем upload, чтобы ниже не было неоднозначности.
    uploaded_files = []

if input_mode == "Путь к файлу":
    path_video_file = st.text_input(
        "Путь к видео файлу (mp4/avi/mov/mkv)",
        value="",
        placeholder=r"Например: D:\videos\test.mp4 или /data/videos/test.mp4",
    )
    p = _validate_video_file(path_video_file)
    if path_video_file and not p:
        st.warning("Файл не найден или расширение не поддерживается.")
    if p:
        selected_video_paths = [p]
        st.caption(f"Файл: {p}")

elif input_mode == "Путь к папке":
    path_video_folder = st.text_input(
        "Путь к папке с видео",
        value="",
        placeholder=r"Например: D:\videos\ или /data/videos/",
    )
    p = _normalize_path(path_video_folder)
    if path_video_folder and not os.path.isdir(p):
        st.warning("Папка не найдена.")
    if os.path.isdir(p):
        selected_video_paths = _collect_videos_from_folder(p, max_files=10)
        st.caption(f"Найдено видео: {len(selected_video_paths)} (лимит 10)")

user_query = st.text_area(
    "Ваш вопрос по видео",
    placeholder="Например: 'Что делают люди в кадре и как они перемещаются во времени?'",
)
require_json = st.checkbox("Требуется JSON-ответ", help="Строгий формат JSON для интеграций")


has_any_videos = bool(uploaded_files) or bool(selected_video_paths)
run_btn = st.button("Запустить анализ", type="primary", disabled=not has_any_videos or not user_query)

if run_btn:
    # НЕ меняем схему state: всегда формируем `video_paths`.
    if input_mode == "Загрузка (Upload)":
        video_paths = _save_uploads(list(uploaded_files or []))
    else:
        video_paths = list(selected_video_paths)

    progress = st.progress(0)
    stage = st.empty()

    def _progress_cb(v: float, msg: str) -> None:
        progress.progress(int(v * 100))
        stage.info(msg)

    vision_llm = "llava_local" if vision_llm_ui == "Local LLaVA" else ("gigachat_api" if vision_llm_ui == "GigaChat API" else "off")
    final_llm = "gigachat_api" if final_llm_ui == "GigaChat API" else "llava_local"
    state: PipelineState = {
        "video_paths": video_paths,
        "user_query": user_query,
        "ui_mode": ui_mode,
        "pro_settings": pro_settings if ui_mode == "PRO" else {},
        "require_json": bool(require_json),
        "vision_llm": vision_llm,
        "final_llm": final_llm,
        "gigachat_api_key": gigachat_api_key,
        "gigachat_ca_cert_path": gigachat_ca_cert_path,
        "analyze_people": bool(analyze_people),
        "force_no_visual_analysis": bool(force_no_visual_analysis or vision_llm == "off"),
        "processing_log": [],
    }

    t0 = time.time()
    with st.spinner("Выполняется анализ..."):
        out = run_pipeline(state, progress_cb=_progress_cb)
    elapsed = time.time() - t0

    st.success(f"Готово. Output: {out.get('result_path','')}")

    # Load events table
    out_dir = out.get("result_path") or out.get("output_dir")
    events_df = pd.DataFrame(out.get("events") or [])
    if out_dir and os.path.exists(os.path.join(out_dir, "events.parquet")):
        try:
            events_df = pd.read_parquet(os.path.join(out_dir, "events.parquet"))
        except Exception:
            pass

    tabs = st.tabs(["Ответ LLM", "События", "Логи обработки"])

    with tabs[0]:
        final_answer = out.get("final_answer")
        if require_json and isinstance(final_answer, dict):
            st.json(final_answer)
        else:
            st.markdown(str(final_answer or ""))
        models_used = out.get("models_used") or []
        st.caption(f"Обработано за {elapsed:.1f} сек. Использовано моделей: {', '.join(models_used) or 'n/a'}")

    with tabs[1]:
        if events_df.empty:
            st.info("События отсутствуют.")
        else:
            st.dataframe(
                events_df,
                width="stretch",
                hide_index=True,
            )

    with tabs[2]:
        log_text = "\n".join(out.get("processing_log") or [])
        if out_dir and os.path.exists(os.path.join(out_dir, "processing_log.log")):
            try:
                with open(os.path.join(out_dir, "processing_log.log"), "r", encoding="utf-8") as f:
                    log_text = f.read()
            except Exception:
                pass
        st.text_area("processing_log.log", value=log_text, height=320)

    if out.get("result_zip_bytes"):
        st.download_button(
            "Скачать результаты (ZIP)",
            data=out["result_zip_bytes"],
            file_name=os.path.basename(out_dir or "results") + ".zip",
            mime="application/zip",
        )

    if ui_mode == "PRO":
        st.subheader("Визуализация (PRO)")
        if out_dir:
            _render_timeline(out_dir, events_df if isinstance(events_df, pd.DataFrame) else pd.DataFrame())

