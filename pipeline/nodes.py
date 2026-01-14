from __future__ import annotations

import os
import time
from typing import Any, Callable, Dict, List

from pipeline.config import (
    get_available_cv_models,
    load_gigachat_ca_bundle_file,
    load_gigachat_default_key,
    load_model_paths,
)
from pipeline.models.cv_models import make_cv_models
from pipeline.models.gigachat_client import (
    GigaChatError,
    GigaChatPaymentRequired,
    maybe_make_gigachat_client,
)
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
from pipeline.utils.event_summarization import (
    SummarizationConfig,
    build_query_evidence,
    format_summary_for_prompt,
    summarize_events_algorithmic,
)
from pipeline.utils.reid_event_builder import build_reid_events


def _heuristic_parse(user_query: str, require_json_flag: bool) -> ParseUserRequestResult:
    q = (user_query or "").lower()
    required: list[str] = []
    if any(k in q for k in ["человек", "людей", "персона", "посетител", "сотрудник"]):
        required.append("yolo-person")
    if any(k in q for k in ["кто", "id", "трек", "тот же", "повторно"]):
        required.append("reid-osnet")

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

    # Backward compatibility: если кто-то запускает старые скрипты со `llm_type`,
    # маппим его на новые поля.
    if not state.get("vision_llm") or not state.get("final_llm"):
        legacy = state.get("llm_type")
        if legacy == "gigachat":
            state.setdefault("vision_llm", "gigachat_api")
            state.setdefault("final_llm", "gigachat_api")
        elif legacy == "local":
            state.setdefault("vision_llm", "llava_local")
            state.setdefault("final_llm", "gigachat_api")

    if not state.get("output_dir"):
        state["output_dir"] = make_output_dir("out", state.get("user_query", ""))
        os.makedirs(os.path.join(state["output_dir"], "cache"), exist_ok=True)

    available_cv = get_available_cv_models()

    # Attempt LLM parse if финальная LLM = GigaChat (т.к. это текстовый вызов)
    require_json_flag = bool(state.get("require_json", False))
    parsed = _heuristic_parse(state.get("user_query", ""), require_json_flag)

    if state.get("final_llm") == "gigachat_api":
        api_key = state.get("gigachat_api_key") or load_gigachat_default_key()
        state["gigachat_api_key"] = api_key
        ca_bundle = state.get("gigachat_ca_cert_path") or load_gigachat_ca_bundle_file()
        client = maybe_make_gigachat_client(api_key, ca_bundle_file=ca_bundle)
        if client is not None:
            prompt = (
                "Проанализируй пользовательский запрос к системе анализа видео. Определи:\n"
                "1. Требуется ли визуальный анализ кадров (т.е. нужно ли показать и анализировать изображения)?\n"
                f"2. Какие CV-модели необходимы из списка реально доступных: {available_cv}?\n"
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

    # Фильтруем только реально доступные CV модели, чтобы не было "вымышленных"
    filtered = [m for m in (parsed.required_models or []) if m in available_cv]
    state["required_models"] = list(dict.fromkeys(filtered))

    # ВАЖНО: решение "нужен ли визуальный анализ" НЕ должно зависеть от LLM-парсера,
    # иначе он может ошибочно отключить vision (как в вашем кейсе "Опиши видео").
    # Источник истины — выбор пользователя в UI: vision_llm = off|gigachat_api|llava_local.
    state["require_visual_analysis"] = str(state.get("vision_llm", "off")) != "off"
    if bool(state.get("force_no_visual_analysis", False)):
        state["require_visual_analysis"] = False
        pr.log(state, "parse_user_request: force_no_visual_analysis=True -> require_visual_analysis=False")

    # combine flags
    state["require_json"] = bool(require_json_flag or parsed.json_required)
    # CV defaults from UI
    if bool(state.get("analyze_people", False)) and "yolo-person" in available_cv:
        if "yolo-person" not in state["required_models"]:
            state["required_models"].insert(0, "yolo-person")
    pr.log(state, f"parse_user_request: available_cv_models={available_cv}")
    pr.log(state, f"parse_user_request: required_models={state['required_models']}")
    pr.log(state, f"parse_user_request: require_visual_analysis={state['require_visual_analysis']}")
    pr.log(state, f"parse_user_request: require_json={state['require_json']}")
    pr.progress(0.15, "parse_user_request: готово")
    return state


def select_video_analysis_mode(state: PipelineState) -> PipelineState:
    pr = ProgressReporter(state.get("progress_cb"))
    pr.progress(0.2, "select_video_analysis_mode: выбор режима анализа")
    pr.log(state, "select_video_analysis_mode: start")

    # ВАЖНО: LangGraph сохраняет только поля, которые присутствуют в схеме State.
    # Поэтому сюда пишем только сериализуемые данные; функции/объекты будем собирать
    # заново в `prepare_key_frames`.
    state["selected_frame_selector_name"] = (
        "PRO" if state.get("ui_mode") == "PRO" else "STANDARD"
    )

    pr.log(state, f"select_video_analysis_mode: {state['selected_frame_selector_name']}")
    pr.progress(0.25, "select_video_analysis_mode: готово")
    return state


def prepare_key_frames(state: PipelineState) -> PipelineState:
    pr = ProgressReporter(state.get("progress_cb"))
    pr.progress(0.3, "prepare_key_frames: выборка кадров и CV-контекст")
    pr.log(state, "prepare_key_frames: start")

    cv = make_cv_models()
    required_models = state.get("required_models", []) or []

    if state.get("ui_mode") == "PRO":
        ps = state.get("pro_settings", {}) or {}
        cfg = FrameSelectorConfig(
            frame_sampling_rate=float(ps.get("frame_sampling_rate", 1.0)),
            ssim_threshold=float(ps.get("ssim_threshold", 0.9)),
            skip_static_frames=bool(ps.get("skip_static_frames", True)),
            cache_frames=bool(ps.get("cache_frames", True)),
            custom_preprocessing=str(ps.get("custom_preprocessing", "None")),
        )
        keep_hook: Callable[[dict[str, Any]], bool] = pro_frame_selector(cfg)
    else:
        cfg = FrameSelectorConfig(frame_sampling_rate=1.0)
        keep_hook = standard_frame_selector()

    # hash cache
    cache_path = os.path.join(state["output_dir"], "cache", "frame_hashes.pkl")
    hash_cache = load_frame_hashes(cache_path) or FrameHashCache(maxsize=1000)

    key_frames: Dict[str, List[Dict[str, Any]]] = {}
    for i, vp in enumerate(state.get("video_paths", []) or []):
        cv.begin_video()
        pr.progress(0.3 + 0.4 * (i / max(1, len(state.get("video_paths", []) or []))), f"prepare_key_frames: {os.path.basename(vp)}")
        mode = state.get("ui_mode", "STANDARD")
        t_decode0 = time.perf_counter()
        frames = decode_and_select_frames(
            video_path=vp,
            frame_sampling_rate=float(cfg.frame_sampling_rate),
            mode=mode,
            ssim_threshold=float(cfg.ssim_threshold),
            skip_static_frames=bool(cfg.skip_static_frames),
            cache_frames=bool(cfg.cache_frames),
            hash_cache=hash_cache if mode == "PRO" and cfg.cache_frames else None,
            keep_hook=keep_hook,
            # CV-only не должен "обрубать" видео до 4–5 секунд.
            # 60 кадров при 1 кадр/сек покрывают короткие видео и защищают от runaway на очень длинных.
            max_frames=None if state.get("require_visual_analysis", True) else 60,
        )
        t_decode = time.perf_counter() - t_decode0
        pr.log(
            state,
            f"prepare_key_frames: selected_frames={len(frames)} mode={mode} "
            f"fps_sample={float(cfg.frame_sampling_rate):.2f} require_visual={bool(state.get('require_visual_analysis', True))}",
        )
        pr.log(state, f"prepare_key_frames: decode_and_select_frames took {t_decode:.3f}s")

        out_frames: List[Dict[str, Any]] = []
        t_cv0 = time.perf_counter()
        # Быстрый путь: если нужен только yolo-person (без ReID), батчим YOLO по всем выбранным кадрам.
        only_yolo_person = ("yolo-person" in required_models) and ("reid-osnet" not in required_models) and (
            len([m for m in required_models if m not in ("yolo-person",)]) == 0
        )
        batch_ctx: list[list[str]] = []
        if only_yolo_person:
            try:
                batch_ctx = cv.describe_frames_yolo_person_batch([f["frame_bgr"] for f in frames])
            except Exception:
                batch_ctx = []

        for j, f in enumerate(frames):
            if batch_ctx:
                ctx = batch_ctx[j] if j < len(batch_ctx) else ["YOLO: no detections"]
            else:
                ctx = cv.describe_frame(
                    f["frame_bgr"],
                    required_models=required_models,
                    frame_id=int(f["frame_id"]),
                    timestamp_sec=float(f["timestamp_sec"]),
                    video_path=vp,
                )
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
        t_cv = time.perf_counter() - t_cv0
        pr.log(state, f"prepare_key_frames: cv.describe_frame loop took {t_cv:.3f}s")

        # Снимем summary маршрутов после обработки видео (если включён reid-osnet)
        if "reid-osnet" in required_models:
            routes = cv.get_person_routes()
            if routes:
                # аккуратно сливаем в state: это глобальная база gid, поддерживает несколько видео
                state["person_routes"] = routes

    state["key_frames"] = key_frames
    if state.get("ui_mode") == "PRO" and cfg.cache_frames:
        save_frame_hashes(cache_path, hash_cache)

    # Всегда формируем CV-события из cv_context (это даёт полезные events даже если LLM не сработал/без vision)
    cv_events: list[dict[str, Any]] = []
    for vp, frames in key_frames.items():
        for f in frames:
            ctx = f.get("cv_context") or []
            if not ctx:
                continue
            cv_events.append(
                Event(
                    event_id=f"{os.path.basename(vp)}_{float(f['timestamp_sec']):.1f}_cv_detection",
                    video_path=vp,
                    timestamp_sec=float(f["timestamp_sec"]),
                    frame_id=int(f["frame_id"]),
                    event_type="cv_detection",
                    entities=[],
                    llava_analysis="; ".join(map(str, ctx[:10])),
                    confidence=0.5,
                    source_frames=[int(f["frame_id"])],
                ).model_dump(by_alias=True)
            )

    # ReID события (по выбору в PRO): segments vs frames
    reid_mode = "segments"
    try:
        ps = state.get("pro_settings", {}) or {}
        reid_mode = str(ps.get("reid_event_mode") or "segments")
    except Exception:
        reid_mode = "segments"

    reid_events: list[dict[str, Any]] = []
    if state.get("person_routes") and "reid-osnet" in required_models:
        try:
            frames_min_dt_sec = float(ps.get("reid_frames_min_dt_sec", 0.5))
            frames_max_points_per_person = int(ps.get("reid_frames_max_points_per_person", 2000))
            frames_max_total_events = int(ps.get("reid_frames_max_total_events", 20000))
            reid_events = build_reid_events(
                person_routes=state.get("person_routes") or {},
                mode=reid_mode,
                frames_min_dt_sec=frames_min_dt_sec,
                frames_max_points_per_person=frames_max_points_per_person,
                frames_max_total_events=frames_max_total_events,
            )
        except Exception as e:
            pr.log(state, f"prepare_key_frames: build_reid_events failed: {e}")

    state["events"] = cv_events + reid_events
    state["llava_results"] = []
    state["models_used"] = sorted(
        set((state.get("models_used") or []) + (state.get("required_models") or []))
    )

    # CV-only: ничего больше не делаем, LLM этап пропустит сам себя
    if not bool(state.get("require_visual_analysis", True)):
        pr.log(state, "prepare_key_frames: CV-only mode -> events from CV only")

    pr.log(state, f"prepare_key_frames: total_videos={len(key_frames)}")
    pr.progress(0.55, "prepare_key_frames: готово")
    return state


def run_llava_analysis(state: PipelineState) -> PipelineState:
    pr = ProgressReporter(state.get("progress_cb"))
    pr.progress(0.6, "run_llava_analysis: мультимодальный анализ кадров")
    pr.log(state, "run_llava_analysis: start")

    if bool(state.get("force_no_visual_analysis", False)) or not bool(
        state.get("require_visual_analysis", True)
    ):
        pr.log(state, "run_llava_analysis: skipped (CV-only mode)")
        return state

    vision_llm = state.get("vision_llm", "llava_local")
    require_json = bool(state.get("require_json", False))
    user_query = state.get("user_query", "")
    results: list[dict[str, Any]] = list(state.get("llava_results") or [])
    events: list[dict[str, Any]] = list(state.get("events") or [])
    models_used: list[str] = []

    mp = load_model_paths()
    local_llava = maybe_make_local_llava(mp.llava_model_id)
    api_key = state.get("gigachat_api_key") or load_gigachat_default_key()
    state["gigachat_api_key"] = api_key
    ca_bundle = state.get("gigachat_ca_cert_path") or load_gigachat_ca_bundle_file()
    gigachat = maybe_make_gigachat_client(api_key, ca_bundle_file=ca_bundle)

    def analyze_one(frame_meta: dict[str, Any]) -> str:
        cv_ctx = frame_meta.get("cv_context", [])
        prompt = (
            "Ты — эксперт по анализу видео. Проанализируй прикрепленный кадр и ответь на вопрос пользователя.\n"
            f"Контекст от вспомогательных CV-моделей:\n{cv_ctx}\n\n"
            f"Вопрос пользователя:\n'{user_query}'\n\n"
            "Инструкции:\n"
            "- Всегда отвечай на русском языке.\n"
            "- Если в вопросе требуется ответ в формате JSON или активирован флаг json_required, верни ТОЛЬКО JSON без дополнительного текста.\n"
            "- Иначе ответь кратко и по делу.\n"
            "- Укажи уровень уверенности в ответе (0.0-1.0).\n"
            "- Если кадр не содержит релевантной информации, ответь 'no_relevant_info'.\n"
        )

        image = resize_for_llava(frame_meta["frame_bgr"])

        if vision_llm == "llava_local":
            try:
                models_used.append("llava-v1.6")
                return local_llava.chat_with_image(prompt, image)
            except LocalLLaVAUnavailable as e:
                pr.log(state, f"run_llava_analysis: local llava unavailable: {e}")
                return "no_relevant_info"
        elif vision_llm == "gigachat_api":
            if gigachat is None:
                raise GigaChatError("GigaChat selected for vision but api_key is empty")
            models_used.append("gigachat")
            b64 = pil_to_base64_jpeg(image)
            try:
                return gigachat.chat_with_image(prompt, b64)
            except GigaChatPaymentRequired as ge:
                # Прекращаем дальнейшие попытки: это не transient и повторять бессмысленно.
                pr.log(state, f"run_llava_analysis: gigachat payment required (402), stopping vision calls: {ge}")
                state["error"] = True
                state["error_code"] = "GIGACHAT_PAYMENT_REQUIRED"
                state["error_message"] = str(ge)
                # маркер для внешнего цикла (через state)
                state["_gigachat_blocked"] = True  # runtime-only
                return "no_relevant_info"
            except Exception as ge:
                pr.log(state, f"run_llava_analysis: gigachat image call failed: {ge}")
                return "no_relevant_info"
        else:
            return "no_relevant_info"

    # iterate frames
    all_frames: list[tuple[str, dict[str, Any]]] = []
    for vp, frames in (state.get("key_frames") or {}).items():
        for f in frames:
            all_frames.append((vp, f))

    total = max(1, len(all_frames))
    gigachat_blocked = False
    for i, (vp, f) in enumerate(all_frames):
        if gigachat_blocked or bool(state.get("_gigachat_blocked", False)):
            break
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
    state["models_used"] = sorted(set((state.get("models_used") or []) + models_used))
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

    # Универсальная предсуммаризация для длинных данных (без привязки к объектам)
    sum_cfg = SummarizationConfig()
    summary = summarize_events_algorithmic(events, sum_cfg)
    state["events_summary"] = summary
    evidence = build_query_evidence(events, summary, max_evidence_events=sum_cfg.max_evidence_events)
    state["query_evidence"] = evidence
    # Для LLM не отправляем покадровые/посекундные списки — это провоцирует такой же ответ.
    # Дадим компактную агрегированную сводку по чанкам + несколько доказательств.
    def _compact_summary_for_llm(s: dict[str, Any], max_chunks: int = 8) -> str:
        chunks = (s.get("chunks") or []) if isinstance(s, dict) else []
        lines: list[str] = []
        # НЕ передаём длительность/FPS/разрешение и прочие "характеристики видео" в LLM:
        # это часто попадает в ответ даже если вопрос не про это.
        lines.append(f"chunks={len(chunks)}")
        for c in chunks[: int(max_chunks)]:
            try:
                lines.append(
                    f"- {c.get('start','?')}–{c.get('end','?')}: events={int(c.get('events_count', 0))}"
                )
            except Exception:
                continue
        return "\n".join(lines)

    def _format_evidence_for_llm(ev: dict[str, Any], max_items: int = 5, max_text_chars: int = 200) -> str:
        items = (ev.get("evidence") or []) if isinstance(ev, dict) else []
        out: list[str] = []
        for it in items[: int(max_items)]:
            try:
                t = str(it.get("time", ""))
                eid = str(it.get("event_id", ""))
                et = str(it.get("event_type", ""))
                txt = str(it.get("text", "")).replace("\n", " ").strip()
                if len(txt) > int(max_text_chars):
                    txt = txt[: int(max_text_chars)].rstrip() + "…"
                out.append(f"- {t} {et} id={eid}: {txt}")
            except Exception:
                continue
        return "\n".join(out) if out else "(нет)"

    summary_text = _compact_summary_for_llm(summary)
    evidence_text = _format_evidence_for_llm(evidence)

    # Финальный ответ: предпочитаем GigaChat (текст), т.к. Local LLaVA в проекте используется в основном как vision.
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

    if state.get("final_llm") == "gigachat_api":
        api_key = state.get("gigachat_api_key") or load_gigachat_default_key()
        state["gigachat_api_key"] = api_key
        ca_bundle = state.get("gigachat_ca_cert_path") or load_gigachat_ca_bundle_file()
        client = maybe_make_gigachat_client(api_key, ca_bundle_file=ca_bundle)
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
                "Ниже дана КОМПАКТНАЯ агрегированная сводка (без покадрового списка) и небольшой набор доказательств.\n"
                "Твоя задача — дать ИТОГОВЫЙ ответ по смыслу (универсально), НЕ переписывая входные строки и НЕ перечисляя события по секундам/кадрам.\n"
                "Фокус: люди, действия, перемещения и взаимодействия с окружением — только если это уместно для вопроса.\n\n"
                f"Сводка (агрегировано):\n{summary_text}\n\n"
                f"Доказательства (можно ссылаться на event_id):\n{evidence_text}\n\n"
                f"Вопрос:\n'{user_query}'\n\n"
                "Требования:\n"
                f"- Если требуется JSON-ответ: строго соблюдай схему {json_schema}.\n"
                "- Иначе: дай короткое итоговое описание (5–10 предложений) и 3–7 пунктов 'Ключевые наблюдения'.\n"
                "- НЕ делай покадрового/посекундного списка. Используй диапазоны времени только для 1–5 ключевых эпизодов (если они видны).\n"
                "- НЕ добавляй характеристики видео (длительность, FPS, разрешение, размер файла), если вопрос напрямую не про это.\n"
                "- Если данных недостаточно — явно скажи что именно не видно/чего не хватает.\n"
                "- Где уместно, укажи источники через event_id (не больше 5 ссылок).\n"
                "- В конце добавь раздел 'confidence_analysis' с оценкой надежности данных.\n"
                "- Если часть описаний событий на английском — переведи на русский.\n"
            )
            state["final_prompt"] = prompt
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
            except GigaChatPaymentRequired as e:
                pr.log(state, f"generate_final_answer: gigachat payment required (402): {e}")
                state["error"] = True
                state["error_code"] = "GIGACHAT_PAYMENT_REQUIRED"
                state["error_message"] = str(e)
            except Exception as e:
                pr.log(state, f"generate_final_answer: gigachat failed: {e}")

    # Safety: если LLM всё равно вернул покадровый список, ужимаем до итоговой сводки.
    if not require_json and isinstance(final, str) and final:
        # эвристика: слишком много строк с таймкодами вида **00:00:01**
        time_lines = sum(1 for ln in final.splitlines() if "00:" in ln and ("**" in ln or ln.strip().startswith("-")))
        if time_lines >= 10:
            pr.log(state, f"generate_final_answer: detected timecoded list ({time_lines} lines) -> compressing")
            note = ""
            if state.get("error_code") == "GIGACHAT_PAYMENT_REQUIRED":
                note = " (GigaChat недоступен: 402 Payment Required)"
            final = (
                "Итоговое описание (сжато, без покадрового перечисления)" + note + ":\n\n"
                f"{summary_text}\n\n"
                "confidence_analysis:\n"
                f"- avg_event_confidence: {_avg_conf(events):.2f}\n"
                f"- total_events: {len(events)}"
            )

    # Heuristic fallback
    if require_json and (not final or not isinstance(final, dict) or not final.get("answer")):
        note = ""
        if state.get("error_code") == "GIGACHAT_PAYMENT_REQUIRED":
            note = " (GigaChat недоступен: 402 Payment Required)"
        final["answer"] = (
            "Итоговый ответ сформирован без LLM" + note + ". "
            "См. events_summary.json и events.parquet в архиве для деталей."
        )
        ev_list = (state.get("query_evidence") or {}).get("evidence") or []
        final["details"] = [
            {
                "time": str(ev.get("time", "")),
                "description": str(ev.get("text", ""))[:200],
                "confidence": float(ev.get("confidence", 0.5)),
                "source_event_id": str(ev.get("event_id", "")),
            }
            for ev in ev_list[:15]
        ]
        final["confidence"] = float(_avg_conf(events))
        final["models_used"] = models_used
    elif not require_json and not final:
        # ВАЖНО: не выводим пользователю покадровые списки cv_detection/описания.
        # Отдаём только итоговую, компактную сводку.
        note = ""
        if state.get("error_code") == "GIGACHAT_PAYMENT_REQUIRED":
            note = " (GigaChat недоступен: 402 Payment Required)"
        lines = []
        lines.append(f"Ответ (fallback без LLM){note}:")
        lines.append("")
        lines.append(f"Вопрос: {user_query}")
        lines.append("")
        lines.append("Краткая сводка по видео:")
        lines.append(summary_text)
        lines.append("")
        lines.append("confidence_analysis:")
        lines.append(f"- avg_event_confidence: {_avg_conf(events):.2f}")
        lines.append(f"- total_events: {len(events)}")
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

    t0 = time.perf_counter()
    save_events_parquet(out_dir, events_rows)
    pr.log(state, f"save_results: save_events_parquet took {(time.perf_counter()-t0):.3f}s (rows={len(events_rows)})")
    save_answer(out_dir, require_json=require_json, final_answer=state.get("final_answer"))
    save_processing_log(out_dir, state.get("processing_log", []) or [])
    # сохраняем описания/ответы по кадрам (не показываем в UI, но кладём в zip)
    try:
        import json

        with open(os.path.join(out_dir, "frame_descriptions.json"), "w", encoding="utf-8") as f:
            json.dump(state.get("llava_results") or [], f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    # сохраняем агрегированную сводку событий и evidence для отладки длинных видео
    try:
        import json

        with open(os.path.join(out_dir, "events_summary.json"), "w", encoding="utf-8") as f:
            json.dump(state.get("events_summary") or {}, f, ensure_ascii=False, indent=2)
        with open(os.path.join(out_dir, "query_evidence.json"), "w", encoding="utf-8") as f:
            json.dump(state.get("query_evidence") or {}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    # сохраняем маршруты людей (ReID/трекинг), если есть
    try:
        import json

        if state.get("person_routes"):
            with open(os.path.join(out_dir, "person_routes.json"), "w", encoding="utf-8") as f:
                json.dump(state.get("person_routes"), f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    # сохраняем финальный промпт (чтобы понимать, что ушло в LLM)
    try:
        if state.get("final_prompt"):
            with open(os.path.join(out_dir, "prompt_final.txt"), "w", encoding="utf-8") as f:
                f.write(str(state.get("final_prompt")))
    except Exception:
        pass
    # сохраняем метаданные key_frames (без самих пикселей)
    try:
        import json

        kf = {}
        for vp, frames in (state.get("key_frames") or {}).items():
            kf[vp] = [
                {
                    "frame_id": fr.get("frame_id"),
                    "timestamp_sec": fr.get("timestamp_sec"),
                    "cv_context": fr.get("cv_context"),
                    "frame_hash": fr.get("frame_hash"),
                    "frame_path": fr.get("frame_path"),
                }
                for fr in frames
            ]
        with open(os.path.join(out_dir, "key_frames.json"), "w", encoding="utf-8") as f:
            json.dump(kf, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
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
        "vision_llm": state.get("vision_llm"),
        "final_llm": state.get("final_llm"),
        "required_models": state.get("required_models"),
        "require_visual_analysis": state.get("require_visual_analysis"),
        "models_used": state.get("models_used"),
        "person_routes_present": bool(state.get("person_routes")),
        "error": bool(state.get("error", False)),
        "error_code": state.get("error_code"),
        "error_message": state.get("error_message"),
    }
    save_metadata(out_dir, metadata)

    state["result_path"] = out_dir
    tzip0 = time.perf_counter()
    state["result_zip_bytes"] = zip_dir_to_bytes(out_dir)
    pr.log(state, f"save_results: zip_dir_to_bytes took {(time.perf_counter()-tzip0):.3f}s")
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

