from __future__ import annotations

import os
import time
import uuid
from typing import Any

import pandas as pd
import streamlit as st
from streamlit_agraph import Config, Edge, Node, agraph

from pipeline.run_pipeline import run_pipeline
from pipeline.state import PipelineState


st.set_page_config(page_title="CV Platform — Video Analytics", layout="wide")


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
    if events_df.empty:
        st.info("Нет событий для визуализации.")
        return

    nodes = []
    edges = []
    last_id = None
    for _, row in events_df.sort_values("timestamp_sec").iterrows():
        ev_id = str(row.get("event_id", ""))
        label = f"{row.get('timestamp_sec', 0):.1f}s • {row.get('event_type','')}"
        nodes.append(Node(id=ev_id, label=label, size=18))
        if last_id is not None:
            edges.append(Edge(source=last_id, target=ev_id))
        last_id = ev_id

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
        selected_id = sel["selected"]
        st.caption(f"Выбрано событие: {selected_id}")
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

    ui_mode = st.radio("Режим", ["STANDARD", "PRO"], index=0, horizontal=True)

    pro_settings = {
        "frame_sampling_rate": 1.0,
        "ssim_threshold": 0.9,
        "skip_static_frames": True,
        "cache_frames": True,
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

    llm_choice = st.radio(
        "LLM",
        ["Local (llava-v1.6-mistral-7b-hf)", "GigaChat API"],
        index=0,
    )
    gigachat_api_key = None
    if llm_choice == "GigaChat API":
        gigachat_api_key = st.text_input("GigaChat API key", type="password")


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

user_query = st.text_area(
    "Ваш вопрос по видео",
    placeholder="Например: 'Сколько раз посетители брали товар с полки №3 между 14:02 и 14:05?'",
)
require_json = st.checkbox("Требуется JSON-ответ", help="Строгий формат JSON для интеграций")


run_btn = st.button("Запустить анализ", type="primary", disabled=not uploaded_files or not user_query)

if run_btn:
    video_paths = _save_uploads(list(uploaded_files or []))

    progress = st.progress(0)
    stage = st.empty()

    def _progress_cb(v: float, msg: str) -> None:
        progress.progress(int(v * 100))
        stage.info(msg)

    llm_type = "local" if llm_choice.startswith("Local") else "gigachat"
    state: PipelineState = {
        "video_paths": video_paths,
        "user_query": user_query,
        "ui_mode": ui_mode,
        "pro_settings": pro_settings if ui_mode == "PRO" else {},
        "require_json": bool(require_json),
        "llm_type": llm_type,
        "gigachat_api_key": gigachat_api_key,
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
                use_container_width=True,
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

