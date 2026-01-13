from __future__ import annotations

import os
from typing import Any, Callable, Dict, List

from pipeline.models.cv_models import make_cv_models
from pipeline.models.gigachat_client import GigaChatError, maybe_make_gigachat_client
from pipeline.models.llava_handler import LocalLLaVAUnavailable, maybe_make_local_llava
from pipeline.save_results import (
    make_output_dir,
    save_answer,
    save_events_parquet,
    save_metadata,
    save_processing_log,
    zip_dir_to_bytes,
)
from pipeline.state import PipelineState
from pipeline.utils.data_schemas import Event, FinalAnswerJSON, FrameLLMAnswer, ParseUserRequestResult
from pipeline.utils.frame_processing import (
    FrameHashCache,
    FrameSelectorConfig,
    decode_and_select_frames,
    load_frame_hashes,
    pil_to_base64_jpeg,
    pro_frame_selector,
    resize_for_llava,
    save_frame_hashes,
    standard_frame_selector,
)
from pipeline.utils.json_utils import extract_first_json
from pipeline.utils.logging_utils import ProgressReporter


def _heuristic_parse(user_query: str, require_json_flag: bool) -> ParseUserRequestResult:
    q = (user_query or "").lower()
    required: list[str] = []
    if any(k in q for k in ["человек", "людей", "персона", "посетител", "сотрудник"]):
        required.append("yolov8-person")
    if any(k in q for k in ["товар", "полк", "product", "shelf"]):
        required.append("yolov8-product")
        if any(k in q for k in ["кто", "id", "трек", "тот же", "повторно"]):
            required.append("reid-tracker")
    if any(k in q for k in ["зона", "касса", "вход", "выход", "aisle"]):
        required.append("zone-detector")

    json_required = require_json_flag or any(
        k in q for k in ["json", "api", "структурированный ответ"]
    )
    require_visual = True
    if any(k in q for k in ["текст", "лог", "без визуального"]):
        require_visual = False

    return ParseUserRequestResult(
        require_visual_analysis=require_visual,
        required_models=sorted(set(required)),
        json_required=bool(json_required),
    )


def parse_user_request(state: PipelineState) -> PipelineState:
    pr = ProgressReporter(state.get("progress_cb"))
    pr.progress(0.05, "parse_user_request: анализ запроса")
    pr.log(state, "parse_user_request: start")

    if not state.get("output_dir"):
        state["output_dir"] = make_output_dir("out", state.get("user_query", ""))
        os.makedirs(os.path.join(state["output_dir"], "cache"), exist_ok=True)

    # Attempt LLM parse if gigachat selected
    require_json_flag = bool(state.get("require_json", False))
    parsed = _heuristic_parse(state.get("user_query", ""), require_json_flag)

    if state.get("llm_type") == "gigachat":
        client = maybe_make_gigachat_client(state.get("gigachat_api_key"))
        if client is not None:
            prompt = (
                "Проанализируй пользовательский запрос к системе анализа видео. Определи:\n"
                "1. Требуется ли визуальный анализ кадров (т.е. нужно ли показать и анализировать изображения)?\n"
                "2. Какие CV-модели необходимы из списка: [yolov8-person, yolov8-product, reid-tracker, zone-detector]?\n"
                "3. Если запрос содержит ключевые слова 'JSON', 'структурированный ответ', 'API' или активирован флаг require_json, "
                "установи флаг json_required=True.\n\n"
                f"Запрос: '{state.get('user_query','')}'\n\n"
                "Верни ответ в формате JSON:\n"
                "{\n"
                "  \"require_visual_analysis\": boolean,\n"
                "  \"required_models\": [\"model_name1\", \"model_name2\"],\n"
                "  \"json_required\": boolean\n"
                "}\n"
            )
            try:
                txt = client.chat_text(prompt)
                data = extract_first_json(txt) or {}
                parsed = ParseUserRequestResult.model_validate(data)
                pr.log(state, "parse_user_request: used gigachat for parsing")
            except (GigaChatError, Exception) as e:
                pr.log(state, f"parse_user_request: gigachat parse failed, fallback heuristic: {e}")

    state["required_models"] = list(parsed.required_models)
    state["require_visual_analysis"] = bool(parsed.require_visual_analysis)
    # combine flags: UI checkbox must win
    state["require_json"] = bool(require_json_flag or parsed.json_required)
    pr.log(state, f"parse_user_request: required_models={state['required_models']}")
    pr.log(state, f"parse_user_request: require_visual_analysis={state['require_visual_analysis']}")
    pr.log(state, f"parse_user_request: require_json={state['require_json']}")
    pr.progress(0.15, "parse_user_request: готово")
    return state


def select_video_analysis_mode(state: PipelineState) -> PipelineState:
    pr = ProgressReporter(state.get("progress_cb"))
    pr.progress(0.2, "select_video_analysis_mode: выбор режима анализа")
    pr.log(state, "select_video_analysis_mode: start")

    if state.get("ui_mode") == "PRO":
        ps = state.get("pro_settings", {}) or {}
        cfg = FrameSelectorConfig(
            frame_sampling_rate=float(ps.get("frame_sampling_rate", 1.0)),
            ssim_threshold=float(ps.get("ssim_threshold", 0.9)),
            skip_static_frames=bool(ps.get("skip_static_frames", True)),
            cache_frames=bool(ps.get("cache_frames", True)),
            custom_preprocessing=str(ps.get("custom_preprocessing", "None")),
        )
        state["selected_frame_selector_name"] = "PRO"
        state["_frame_selector_cfg"] = cfg  # runtime helper
        state["_frame_keep_hook"] = pro_frame_selector(cfg)  # runtime helper
    else:
        state["selected_frame_selector_name"] = "STANDARD"
        state["_frame_selector_cfg"] = FrameSelectorConfig(frame_sampling_rate=1.0)
        state["_frame_keep_hook"] = standard_frame_selector()

    pr.log(state, f"select_video_analysis_mode: {state['selected_frame_selector_name']}")
    pr.progress(0.25, "select_video_analysis_mode: готово")
    return state


def prepare_key_frames(state: PipelineState) -> PipelineState:
    pr = ProgressReporter(state.get("progress_cb"))
    pr.progress(0.3, "prepare_key_frames: выборка кадров и CV-контекст")
    pr.log(state, "prepare_key_frames: start")

    cv = make_cv_models()
    required_models = state.get("required_models", []) or []
    keep_hook: Callable[[dict[str, Any]], bool] = state.get("_frame_keep_hook")  # type: ignore
    cfg: FrameSelectorConfig = state.get("_frame_selector_cfg")  # type: ignore

    # hash cache
    cache_path = os.path.join(state["output_dir"], "cache", "frame_hashes.pkl")
    hash_cache = load_frame_hashes(cache_path) or FrameHashCache(maxsize=1000)

    key_frames: Dict[str, List[Dict[str, Any]]] = {}
    for i, vp in enumerate(state.get("video_paths", []) or []):
        pr.progress(0.3 + 0.4 * (i / max(1, len(state.get("video_paths", []) or []))), f"prepare_key_frames: {os.path.basename(vp)}")
        mode = state.get("ui_mode", "STANDARD")
        frames = decode_and_select_frames(
            video_path=vp,
            frame_sampling_rate=float(cfg.frame_sampling_rate),
            mode=mode,
            ssim_threshold=float(cfg.ssim_threshold),
            skip_static_frames=bool(cfg.skip_static_frames),
            cache_frames=bool(cfg.cache_frames),
            hash_cache=hash_cache if mode == "PRO" and cfg.cache_frames else None,
            keep_hook=keep_hook,
            max_frames=None if state.get("require_visual_analysis", True) else 5,
        )

        out_frames: List[Dict[str, Any]] = []
        for f in frames:
            ctx = cv.describe_frame(f["frame_bgr"], required_models=required_models)
            # сохраняем кадр на диск в out/cache/frames (только если PRO+cache_frames)
            frame_file = None
            if state.get("ui_mode") == "PRO" and cfg.cache_frames:
                frames_dir = os.path.join(state["output_dir"], "cache", "frames")
                os.makedirs(frames_dir, exist_ok=True)
                frame_file = os.path.join(frames_dir, f"{os.path.basename(vp)}_{f['frame_id']}.jpg")
                try:
                    import cv2

                    cv2.imwrite(frame_file, f["frame_bgr"])
                except Exception:
                    frame_file = None

            out_frames.append(
                {
                    "frame_id": int(f["frame_id"]),
                    "timestamp_sec": float(f["timestamp_sec"]),
                    "cv_context": ctx,
                    "frame_hash": str(f["frame_hash"]),
                    "frame_path": frame_file,
                    "frame_bgr": f["frame_bgr"],  # runtime-only for immediate LLM call
                }
            )
        key_frames[vp] = out_frames

    state["key_frames"] = key_frames
    if state.get("ui_mode") == "PRO" and cfg.cache_frames:
        save_frame_hashes(cache_path, hash_cache)

    pr.log(state, f"prepare_key_frames: total_videos={len(key_frames)}")
    pr.progress(0.55, "prepare_key_frames: готово")
    return state


def run_llava_analysis(state: PipelineState) -> PipelineState:
    pr = ProgressReporter(state.get("progress_cb"))
    pr.progress(0.6, "run_llava_analysis: мультимодальный анализ кадров")
    pr.log(state, "run_llava_analysis: start")

    llm_type = state.get("llm_type", "local")
    require_json = bool(state.get("require_json", False))
    user_query = state.get("user_query", "")
    results: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    models_used: list[str] = []

    local_llava = maybe_make_local_llava()
    gigachat = maybe_make_gigachat_client(state.get("gigachat_api_key"))

    def analyze_one(frame_meta: dict[str, Any]) -> str:
        cv_ctx = frame_meta.get("cv_context", [])
        prompt = (
            "Ты — эксперт по анализу видео. Проанализируй прикрепленный кадр и ответь на вопрос пользователя.\n"
            f"Контекст от вспомогательных CV-моделей:\n{cv_ctx}\n\n"
            f"Вопрос пользователя:\n'{user_query}'\n\n"
            "Инструкции:\n"
            "- Если в вопросе требуется ответ в формате JSON или активирован флаг json_required, верни ТОЛЬКО JSON без дополнительного текста.\n"
            "- Иначе ответь развернуто на русском языке.\n"
            "- Укажи уровень уверенности в ответе (0.0-1.0).\n"
            "- Если кадр не содержит релевантной информации, ответь 'no_relevant_info'.\n"
        )

        image = resize_for_llava(frame_meta["frame_bgr"])

        if llm_type == "local":
            try:
                models_used.append("llava-v1.6")
                return local_llava.chat_with_image(prompt, image)
            except LocalLLaVAUnavailable as e:
                pr.log(state, f"run_llava_analysis: local llava unavailable: {e}")
                if gigachat is not None:
                    models_used.append("gigachat")
                    b64 = pil_to_base64_jpeg(image)
                    return gigachat.chat_with_image(prompt, b64)
                return "no_relevant_info"
        else:
            if gigachat is None:
                raise GigaChatError("GigaChat selected but api_key is empty")
            models_used.append("gigachat")
            b64 = pil_to_base64_jpeg(image)
            return gigachat.chat_with_image(prompt, b64)

    # iterate frames
    all_frames: list[tuple[str, dict[str, Any]]] = []
    for vp, frames in (state.get("key_frames") or {}).items():
        for f in frames:
            all_frames.append((vp, f))

    total = max(1, len(all_frames))
    for i, (vp, f) in enumerate(all_frames):
        pr.progress(0.6 + 0.25 * (i / total), f"run_llava_analysis: кадр {i+1}/{total}")
        raw = analyze_one(f)

        parsed_frame: FrameLLMAnswer
        if require_json:
            obj = extract_first_json(raw) or {"answer": "no_relevant_info", "confidence": 0.0, "event_type": None}
            try:
                parsed_frame = FrameLLMAnswer.model_validate(obj)
            except Exception:
                parsed_frame = FrameLLMAnswer(answer="no_relevant_info", confidence=0.0, event_type=None)
        else:
            parsed_frame = FrameLLMAnswer(answer=str(raw).strip(), confidence=0.5, event_type=None)

        results.append(
            {
                "video_path": vp,
                "frame_id": int(f["frame_id"]),
                "timestamp_sec": float(f["timestamp_sec"]),
                "answer_raw": raw,
                "answer_parsed": parsed_frame.model_dump(),
            }
        )

        # events
        if parsed_frame.answer and parsed_frame.answer.strip() != "no_relevant_info":
            event_type = parsed_frame.event_type or "llm_observation"
            ev = Event(
                event_id=f"{os.path.basename(vp)}_{float(f['timestamp_sec']):.1f}_{event_type}",
                video_path=vp,
                timestamp_sec=float(f["timestamp_sec"]),
                frame_id=int(f["frame_id"]),
                event_type=str(event_type),
                entities=[],
                llava_analysis=parsed_frame.answer,
                confidence=float(parsed_frame.confidence),
                source_frames=[
                    max(0, int(f["frame_id"]) - 1),
                    int(f["frame_id"]),
                    int(f["frame_id"]) + 1,
                ],
            )
            events.append(ev.model_dump(by_alias=True))

    state["llava_results"] = results
    state["events"] = events
    state["models_used"] = sorted(set(models_used))
    pr.log(state, f"run_llava_analysis: events={len(events)}")
    pr.progress(0.85, "run_llava_analysis: готово")
    return state


def generate_final_answer(state: PipelineState) -> PipelineState:
    pr = ProgressReporter(state.get("progress_cb"))
    pr.progress(0.88, "generate_final_answer: агрегирование и финальный ответ")
    pr.log(state, "generate_final_answer: start")

    require_json = bool(state.get("require_json", False))
    events = state.get("events", []) or []
    user_query = state.get("user_query", "")
    models_used = state.get("models_used", []) or []

    # Best-effort summarization using gigachat (text) if selected, otherwise heuristic.
    final: Any
    if require_json:
        final = FinalAnswerJSON(
            answer="",
            details=[],
            confidence=0.5,
            processing_time_sec=0.0,
            models_used=models_used,
        ).model_dump()
    else:
        final = ""

    if state.get("llm_type") == "gigachat":
        client = maybe_make_gigachat_client(state.get("gigachat_api_key"))
        if client is not None:
            json_schema = {
                "answer": "Основной ответ на вопрос",
                "details": [
                    {
                        "time": "ЧЧ:ММ:СС",
                        "description": "Описание события",
                        "confidence": 0.0,
                        "source_event_id": "event_id",
                    }
                ],
                "confidence": 0.0,
                "processing_time_sec": 0.0,
                "models_used": ["llava-v1.6", "yolov8-person", "reid-osnet"],
            }
            prompt = (
                "На основе событий из видео ответь на исходный вопрос пользователя.\n"
                f"События:\n{events}\n\n"
                f"Вопрос:\n'{user_query}'\n\n"
                "Требования:\n"
                f"- Если требуется JSON-ответ: строго соблюдай схему {json_schema}.\n"
                "- Иначе: структурируй ответ с временными метками, используй маркированные списки.\n"
                "- Укажи источники событий через event_id.\n"
                "- В конце добавь раздел 'confidence_analysis' с оценкой надежности данных.\n"
            )
            try:
                txt = client.chat_text(prompt)
                if require_json:
                    obj = extract_first_json(txt)
                    if obj:
                        obj["models_used"] = models_used
                        final = obj
                    else:
                        final["answer"] = txt
                else:
                    final = txt
            except Exception as e:
                pr.log(state, f"generate_final_answer: gigachat failed: {e}")

    # Heuristic fallback
    if require_json and (not final or not isinstance(final, dict) or not final.get("answer")):
        final["answer"] = (
            "Не удалось получить структурированный ответ от LLM. "
            f"Найдено событий: {len(events)}. См. events.parquet для деталей."
        )
        final["details"] = [
            {
                "time": _sec_to_hhmmss(float(ev.get("timestamp_sec", 0.0))),
                "description": str(ev.get("llava_analysis", ""))[:200],
                "confidence": float(ev.get("confidence", 0.5)),
                "source_event_id": str(ev.get("event_id", "")),
            }
            for ev in events[:50]
        ]
        final["confidence"] = float(_avg_conf(events))
        final["models_used"] = models_used
    elif not require_json and not final:
        lines = [f"Вопрос: {user_query}", "", "События:"]
        for ev in events[:100]:
            lines.append(
                f"- [{_sec_to_hhmmss(float(ev.get('timestamp_sec', 0.0)))}] "
                f"{ev.get('event_type','')}: {ev.get('llava_analysis','')} (event_id={ev.get('event_id','')})"
            )
        lines.append("")
        lines.append("confidence_analysis:")
        lines.append(f"- avg_confidence: {_avg_conf(events):.2f}")
        final = "\n".join(lines)

    state["final_answer"] = final
    pr.progress(0.93, "generate_final_answer: готово")
    return state


def save_results(state: PipelineState) -> PipelineState:
    pr = ProgressReporter(state.get("progress_cb"))
    pr.progress(0.95, "save_results: сохранение артефактов")
    pr.log(state, "save_results: start")

    out_dir = state["output_dir"]
    require_json = bool(state.get("require_json", False))

    # events parquet
    events_rows: list[dict[str, Any]] = []
    for ev in state.get("events", []) or []:
        try:
            e = Event.model_validate(ev)
            events_rows.append(e.to_parquet_row())
        except Exception:
            # best-effort
            events_rows.append(
                {
                    "timestamp_sec": float(ev.get("timestamp_sec", 0.0)),
                    "video_path": str(ev.get("video_path", "")),
                    "event_type": str(ev.get("event_type", "")),
                    "entities_json": ev.get("entities", []),
                    "confidence": float(ev.get("confidence", 0.0)),
                    "event_id": str(ev.get("event_id", "")),
                    "frame_id": int(ev.get("frame_id", 0)),
                    "llava_analysis": str(ev.get("llava_analysis", "")),
                    "source_frames": ev.get("source_frames", []),
                }
            )

    save_events_parquet(out_dir, events_rows)
    save_answer(out_dir, require_json=require_json, final_answer=state.get("final_answer"))
    save_processing_log(out_dir, state.get("processing_log", []) or [])
    if bool(state.get("error", False)):
        try:
            with open(os.path.join(out_dir, "errors.log"), "w", encoding="utf-8") as f:
                f.write(f"error_code: {state.get('error_code')}\n")
                f.write(f"error_message: {state.get('error_message')}\n")
        except Exception:
            pass

    metadata = {
        "video_paths": state.get("video_paths", []),
        "user_query": state.get("user_query"),
        "ui_mode": state.get("ui_mode"),
        "pro_settings": state.get("pro_settings"),
        "require_json": state.get("require_json"),
        "llm_type": state.get("llm_type"),
        "required_models": state.get("required_models"),
        "require_visual_analysis": state.get("require_visual_analysis"),
        "models_used": state.get("models_used"),
        "error": bool(state.get("error", False)),
        "error_code": state.get("error_code"),
        "error_message": state.get("error_message"),
    }
    save_metadata(out_dir, metadata)

    state["result_path"] = out_dir
    state["result_zip_bytes"] = zip_dir_to_bytes(out_dir)
    pr.progress(1.0, "Готово")
    pr.log(state, f"save_results: done -> {out_dir}")
    return state


def error_handler(state: PipelineState) -> PipelineState:
    pr = ProgressReporter(state.get("progress_cb"))
    pr.log(state, "error_handler: start")
    state["error"] = True
    if not state.get("error_code"):
        state["error_code"] = "PIPELINE_ERROR"
    if not state.get("error_message"):
        state["error_message"] = "Pipeline failed"
    pr.log(state, f"error_handler: {state.get('error_code')}: {state.get('error_message')}")
    return state


def _avg_conf(events: list[dict[str, Any]]) -> float:
    if not events:
        return 0.0
    vals = []
    for ev in events:
        try:
            vals.append(float(ev.get("confidence", 0.0)))
        except Exception:
            pass
    return float(sum(vals) / max(1, len(vals)))


def _sec_to_hhmmss(sec: float) -> str:
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

