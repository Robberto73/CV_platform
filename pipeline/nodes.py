from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Dict, List

from pipeline.config import (
    get_available_cv_models,
    load_gigachat_ca_bundle_file,
    load_gigachat_default_key,
    load_model_paths,
)
from pipeline.models.cv_models import CVConfig, make_cv_models
from pipeline.models.gigachat_client import (
    GigaChatError,
    GigaChatPaymentRequired,
    maybe_make_gigachat_client,
)
from pipeline.models.llava_handler import LocalLLaVAUnavailable, maybe_make_local_llava
from pipeline.models.openai_local_client import maybe_make_openai_local_client
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
    # Pose estimation для анализа движений рук/ног
    if any(k in q for k in ["движени", "поза", "жест", "рука", "нога", "танец", "спорт", "действи"]):
        required.append("pose-estimation")

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
    pr = ProgressReporter(
        progress_cb=state.get("progress_cb"),
        enable_detailed_logging=bool(state.get("enable_detailed_logging", False))
    )
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
            # Автоматически подключаем ReID для точности на Linux
            import platform
            if platform.system().lower() == "linux" and "reid-osnet" in available_cv:
                if "reid-osnet" not in state["required_models"]:
                    state["required_models"].append("reid-osnet")
                    pr.log(state, "parse_user_request: auto-enabled ReID for Linux + yolo-person")

    # ReID для сохранения уникальных людей (даже в CV-only режиме)
    if bool(state.get("save_unique_people", False)) and "reid-osnet" in available_cv:
        if "reid-osnet" not in state["required_models"]:
            state["required_models"].append("reid-osnet")
            pr.log(state, "parse_user_request: enabled ReID for unique people saving")

    # Pose estimation
    if bool(state.get("analyze_pose", False)) and "pose-estimation" in available_cv:
        if "pose-estimation" not in state["required_models"]:
            state["required_models"].append("pose-estimation")
            pr.log(state, "parse_user_request: enabled pose estimation")
    pr.log(state, f"parse_user_request: available_cv_models={available_cv}")
    pr.log(state, f"parse_user_request: required_models={state['required_models']}")
    pr.log(state, f"parse_user_request: require_visual_analysis={state['require_visual_analysis']}")
    pr.log(state, f"parse_user_request: require_json={state['require_json']}")
    pr.progress(0.15, "parse_user_request: готово")
    return state


def select_video_analysis_mode(state: PipelineState) -> PipelineState:
    pr = ProgressReporter(
        progress_cb=state.get("progress_cb"),
        enable_detailed_logging=bool(state.get("enable_detailed_logging", False))
    )
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
    if state.get("is_image_mode", False):
        return prepare_key_images(state)

    pr = ProgressReporter(
        progress_cb=state.get("progress_cb"),
        enable_detailed_logging=bool(state.get("enable_detailed_logging", False))
    )
    pr.progress(0.3, "prepare_key_frames: выборка кадров и CV-контекст")
    pr.log(state, "prepare_key_frames: start")

    yolo_batch_size = int(state.get("yolo_batch_size", 16))
    cv = make_cv_models(
        pro_settings=state.get("pro_settings") if state.get("ui_mode") == "PRO" else None,
        yolo_batch_size=yolo_batch_size
    )
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

    # Проверка на умную кластеризацию
    smart_clustering_enabled = (
        state.get("ui_mode") == "PRO" and
        state.get("pro_settings", {}).get("enable_smart_clustering", False)
    )

    if smart_clustering_enabled:
        pr.log(state, "prepare_key_frames: включена умная кластеризация видео")
        from pipeline.utils.video_clustering import smart_video_clustering, VideoClusterConfig

        ps = state.get("pro_settings", {})
        cluster_config = VideoClusterConfig(
            ssi_threshold=float(ps.get("clustering_ssi_threshold", 0.85)),
            window_duration_sec=int(ps.get("clustering_window_duration", 300)),
            base_sample_rate=0.5  # Фиксированная частота 0.5 (каждые 2 сек)
        )

    key_frames: Dict[str, List[Dict[str, Any]]] = {}
    for i, vp in enumerate(state.get("video_paths", []) or []):
        cv.begin_video()
        pr.progress(0.3 + 0.4 * (i / max(1, len(state.get("video_paths", []) or []))), f"prepare_key_frames: {os.path.basename(vp)}")
        mode = state.get("ui_mode", "STANDARD")

        if smart_clustering_enabled:
            # Умная кластеризация вместо стандартной выборки
            pr.log(state, f"prepare_key_frames: запускаем умную кластеризацию для {os.path.basename(vp)}")
            t_cluster0 = time.perf_counter()
            clustering_result = smart_video_clustering(
                video_path=vp,
                output_dir=state["output_dir"],
                ssi_threshold=cluster_config.ssi_threshold,
                window_duration=cluster_config.window_duration_sec
            )
            t_cluster = time.perf_counter() - t_cluster0

            # Конвертация результатов кластеризации в формат key_frames
            selected_frames = []
            for kf in clustering_result['key_frames']:
                # Создаем frame_meta как в стандартной обработке
                frame_meta = {
                    "frame_id": int(kf['timestamp'] * 30),  # Примерный frame_id
                    "timestamp_sec": kf['timestamp'],
                    "frame_bgr": kf['frame'],
                    "cv_context": [],  # CV контекст будет добавлен ниже
                    "cluster_info": {
                        "size": kf['cluster_size'],
                        "duration": kf['cluster_duration']
                    }
                }
                selected_frames.append(frame_meta)

            frames = selected_frames
            pr.log(state, f"prepare_key_frames: умная кластеризация выбрала {len(frames)} ключевых кадров за {t_cluster:.2f}s")
            pr.log(state, f"prepare_key_frames: статистика кластеризации: {clustering_result['processing_stats']}")

        else:
            # Стандартная выборка кадров
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
            # сохраняем кадр на диск в out/cache/frames (только если PRO+cache_frames и НЕ сохраняем уникальных людей)
            frame_file = None
            save_general_frames = cfg.cache_frames and not state.get("save_unique_people", False)
            if state.get("ui_mode") == "PRO" and save_general_frames:
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
    save_general_frames = cfg.cache_frames and not state.get("save_unique_people", False)
    if state.get("ui_mode") == "PRO" and save_general_frames:
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

    # Сохраняем полные ReID данные для сохранения уникальных людей
    if cv._reid_tracker is not None:
        state["reid_trajectories"] = cv._reid_tracker._traj.copy()
        state["reid_global_db"] = cv._reid_tracker._global_db.copy()

    # CV-only: ничего больше не делаем, LLM этап пропустит сам себя
    if not bool(state.get("require_visual_analysis", True)):
        pr.log(state, "prepare_key_frames: CV-only mode -> events from CV only")

    pr.log(state, f"prepare_key_frames: total_videos={len(key_frames)}")
    pr.progress(0.55, "prepare_key_frames: готово")
    return state


def prepare_key_images(state: PipelineState) -> PipelineState:
    """Обработка изображений: детекция объектов и анализ"""
    pr = ProgressReporter(
        progress_cb=state.get("progress_cb"),
        enable_detailed_logging=bool(state.get("enable_detailed_logging", False))
    )
    pr.progress(0.3, "prepare_key_images: анализ изображений и CV-контекст")
    pr.log(state, "prepare_key_images: start")

    image_paths = state.get("image_paths", []) or []
    if not image_paths:
        pr.log(state, "prepare_key_images: no images provided")
        return state

    # Загружаем настройки
    pro_settings = state.get("pro_settings", {})
    cfg = CVConfig(
        yolo_model_path=state.get("yolo_model_path", "yolov8n.pt"),
        yolo_model_version=state.get("yolo_model_version", "auto"),
        yolo_input_size=state.get("yolo_input_size", 640),
        conf_threshold=float(state.get("conf_threshold", 0.5)),
        batch_size=int(pro_settings.get("yolo_batch_size", 8)),
        osnet_reid_model_path=state.get("osnet_reid_model_path", "osnet_x1_0_imagenet.pt"),
        pro_yolo_input_size=pro_settings.get("yolo_input_size"),
        pro_yolo_conf_threshold=pro_settings.get("yolo_conf_threshold"),
    )

    # Создаем CV модели
    cv = make_cv_models(cfg, pro_settings, yolo_batch_size=int(pro_settings.get("yolo_batch_size", 8)))

    key_images: Dict[str, List[Dict[str, Any]]] = {}
    events = []

    for i, img_path in enumerate(image_paths):
        pr.progress(0.3 + 0.4 * (i / max(1, len(image_paths))), f"prepare_key_images: {os.path.basename(img_path)}")

        # Загружаем изображение
        frame_bgr = cv2.imread(img_path)
        if frame_bgr is None:
            pr.log(state, f"prepare_key_images: failed to load image {img_path}")
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        height, width = frame_rgb.shape[:2]

        # Детекция объектов на изображении
        detections = cv.detect_objects(frame_rgb)

        # Создаем "кадр" для каждого изображения
        image_frame = {
            "frame_id": 0,  # Для изображений всегда frame_id = 0
            "timestamp": 0.0,  # Для изображений timestamp = 0.0
            "image_path": img_path,
            "width": width,
            "height": height,
            "detections": detections,
        }

        key_images[img_path] = [image_frame]

        # Создаем события для каждого обнаруженного объекта
        for det in detections:
            event = {
                "video_path": img_path,  # Используем image_path как video_path для совместимости
                "frame_id": 0,
                "timestamp": 0.0,
                "class": det.get("class", "unknown"),
                "confidence": det.get("confidence", 0.0),
                "bbox": det.get("bbox", []),
                "bbox_xyxy": det.get("bbox_xyxy", []),
                "area": det.get("area", 0),
                "event_type": "object_detection",
                "description": f"Обнаружен объект '{det.get('class', 'unknown')}' с уверенностью {det.get('confidence', 0.0):.2f}",
            }
            events.append(event)

        # Если включен анализ людей, добавляем дополнительные события
        if state.get("analyze_people", False) and cv._reid_tracker:
            # Для изображений ReID работает как обычная детекция людей
            person_detections = [d for d in detections if d.get("class") == "person"]
            for person_det in person_detections:
                person_event = {
                    "video_path": img_path,
                    "frame_id": 0,
                    "timestamp": 0.0,
                    "class": "person",
                    "confidence": person_det.get("confidence", 0.0),
                    "bbox": person_det.get("bbox", []),
                    "bbox_xyxy": person_det.get("bbox_xyxy", []),
                    "area": person_det.get("area", 0),
                    "person_id": f"person_{len(events)}",  # Простая нумерация для изображений
                    "event_type": "person_detection",
                    "description": f"Обнаружен человек с уверенностью {person_det.get('confidence', 0.0):.2f}",
                }
                events.append(person_event)

    state["key_frames"] = key_images  # Для совместимости используем key_frames
    state["key_images"] = key_images   # И добавляем key_images для ясности
    state["events"] = events

    pr.log(state, f"prepare_key_images: total_images={len(key_images)} total_events={len(events)}")
    pr.progress(0.55, "prepare_key_images: готово")
    return state


def run_llava_analysis(state: PipelineState) -> PipelineState:
    import time
    analysis_start = time.time()

    pr = ProgressReporter(
        progress_cb=state.get("progress_cb"),
        enable_detailed_logging=bool(state.get("enable_detailed_logging", False))
    )
    pr.progress(0.6, "run_llava_analysis: мультимодальный анализ кадров")
    pr.log(state, "run_llava_analysis: start")

    if bool(state.get("force_no_visual_analysis", False)) or not bool(
        state.get("require_visual_analysis", True)
    ):
        pr.log(state, "run_llava_analysis: skipped (CV-only mode)")
        return state

    vision_llm = state.get("vision_llm", "llava_local")

    # Автоматическое переключение на Local LLaVA для изображений если выбрана GigaChat
    # GigaChat не поддерживает изображения, поэтому переключаемся на Local LLaVA
    if vision_llm == "gigachat_api" and state.get("is_image_mode", False):
        pr.log(state, "run_llava_analysis: switching from GigaChat to Local LLaVA for image analysis (GigaChat doesn't support images)")
        vision_llm = "llava_local"

    require_json = bool(state.get("require_json", False))
    user_query = state.get("user_query", "")
    openai_batch_size = int(state.get("openai_batch_size", 8))
    llava_image_size = int(state.get("llava_image_size", 336))  # Размер изображений для LLaVA из PRO настроек
    results: list[dict[str, Any]] = list(state.get("llava_results") or [])
    events: list[dict[str, Any]] = list(state.get("events") or [])
    models_used: list[str] = []

    mp = load_model_paths()
    local_llava = maybe_make_local_llava(mp.llava_model_id)
    api_key = state.get("gigachat_api_key") or load_gigachat_default_key()
    state["gigachat_api_key"] = api_key
    ca_bundle = state.get("gigachat_ca_cert_path") or load_gigachat_ca_bundle_file()
    gigachat = maybe_make_gigachat_client(api_key, ca_bundle_file=ca_bundle)
    openai_local = maybe_make_openai_local_client()

    def analyze_one(frame_meta: dict[str, Any]) -> str:
        cv_ctx = frame_meta.get("cv_context", [])
            prompt = (
                "You are a video analysis expert. Analyze the attached frame and answer the user's question.\n"
                f"Context from auxiliary CV models:\n{cv_ctx}\n\n"
                f"User question:\n'{user_query}'\n\n"
                "Instructions:\n"
                "- Always answer in Russian language.\n"
                "- If the question requires a JSON format answer or json_required flag is activated, return ONLY JSON without additional text.\n"
                "- Otherwise, answer briefly and to the point.\n"
                "- Indicate the confidence level in the answer (0.0-1.0).\n"
                "- If the frame contains no relevant information, answer 'no_relevant_info'.\n"
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
        elif vision_llm == "openai_local":
            if openai_local is None:
                pr.log(state, "run_llava_analysis: openai local unavailable")
                return "no_relevant_info"
            models_used.append("openai-local")
            b64 = pil_to_base64_jpeg(image)
            try:
                return openai_local.analyze_image(b64, prompt)
            except Exception as oe:
                pr.log(state, f"run_llava_analysis: openai local image call failed: {oe}")
                return "no_relevant_info"
        else:
            return "no_relevant_info"

    # iterate frames
    all_frames: list[tuple[str, dict[str, Any]]] = []
    for vp, frames in (state.get("key_frames") or {}).items():
        for f in frames:
            all_frames.append((vp, f))

    total = max(1, len(all_frames))

    if vision_llm == "openai_local" and openai_local is not None:
        # Батчевая обработка для OpenAI Local
        pr.progress(0.6, f"run_llava_analysis: подготовка батчей (batch_size={openai_batch_size})")
        batch_size = openai_batch_size  # используем настройку из UI

        # Подготовка данных для батчинга
        image_prompts = []
        frame_indices = []

        for i, (vp, f) in enumerate(all_frames):
            cv_ctx = f.get("cv_context", [])
            # Сначала проверим работу модели простым английским prompt
            if user_query.lower().strip() == "test":
                prompt = f"What do you see in this image? Context: {cv_ctx}"
            else:
                prompt = (
                    "You are a video analysis expert. Analyze the attached frame and answer the user's question.\n"
                    f"Context from auxiliary CV models:\n{cv_ctx}\n\n"
                    f"User question:\n'{user_query}'\n\n"
                    "Instructions:\n"
                    "- Always answer in Russian language.\n"
                    "- If the question requires a JSON format answer or json_required flag is activated, return ONLY JSON without additional text.\n"
                    "- Otherwise, answer briefly and to the point.\n"
                    "- Indicate the confidence level in the answer (0.0-1.0).\n"
                    "- If the frame contains no relevant information, answer 'no_relevant_info'.\n"
                )

            image = resize_for_llava(f["frame_bgr"], size=llava_image_size)
            b64 = pil_to_base64_jpeg(image)
            image_prompts.append({"image_base64": b64, "prompt": prompt})
            frame_indices.append(i)

        # Батчевая обработка
        pr.progress(0.65, f"run_llava_analysis: обработка батчей (размер: {openai_batch_size})")
        import time
        vllm_start = time.time()
        try:
            batch_results = openai_local.batch_analyze_images(image_prompts, batch_size=openai_batch_size)
            vllm_time = time.time() - vllm_start
            pr.log(state, f"run_llava_analysis: vLLM processed {len(batch_results)} images in {vllm_time:.2f}s ({vllm_time/max(1, len(batch_results)):.2f}s per image)")
            models_used.append("openai-local")
        except Exception as e:
            pr.log(state, f"run_llava_analysis: batch processing failed: {e}")
            batch_results = ["no_relevant_info"] * len(image_prompts)

        # Обработка результатов батча
        for i, raw in enumerate(batch_results):
            frame_idx = frame_indices[i]
            vp, f = all_frames[frame_idx]

            pr.progress(0.7 + 0.2 * (i / len(batch_results)), f"run_llava_analysis: кадр {i+1}/{len(batch_results)}")

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
                    event_type=event_type,
                    entities=[],
                    llava_analysis=parsed_frame.answer,
                    confidence=parsed_frame.confidence,
                    source_frames=[int(f["frame_id"])],
                )
                events.append(ev.model_dump(by_alias=True))

    else:
        # Обычная обработка по кадрам (для других LLM)
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

    total_analysis_time = time.time() - analysis_start
    pr.log(state, f"run_llava_analysis: events={len(events)}, total_time={total_analysis_time:.2f}s")
    pr.progress(0.85, "run_llava_analysis: готово")
    return state


def generate_final_answer(state: PipelineState) -> PipelineState:
    pr = ProgressReporter(
        progress_cb=state.get("progress_cb"),
        enable_detailed_logging=bool(state.get("enable_detailed_logging", False))
    )
    pr.progress(0.88, "generate_final_answer: агрегирование и финальный ответ")
    pr.log(state, "generate_final_answer: start")

    require_json = bool(state.get("require_json", False))
    events = state.get("events", []) or []
    user_query = state.get("user_query", "")
    models_used = state.get("models_used", []) or []
    video_paths = state.get("video_paths", []) or []
    summarization_mode = state.get("summarization_mode", "Баланс (умолчание)")
    custom_max_chunks = state.get("custom_max_chunks", 8)
    custom_max_evidence = state.get("custom_max_evidence", 5)

    # Универсальная предсуммаризация для длинных данных (без привязки к объектам)
    import time
    start_time = time.time()
    sum_cfg = SummarizationConfig()
    summary = summarize_events_algorithmic(events, sum_cfg)
    summarization_time = time.time() - start_time
    pr.log(state, f"generate_final_answer: summarization took {summarization_time:.2f}s for {len(events)} events")
    state["events_summary"] = summary
    evidence = build_query_evidence(events, summary, max_evidence_events=sum_cfg.max_evidence_events)
    state["query_evidence"] = evidence
    # Для LLM не отправляем покадровые/посекундные списки — это провоцирует такой же ответ.
    # Дадим компактную агрегированную сводку по чанкам + несколько доказательств.
    def _adaptive_summary_for_llm(
        s: dict[str, Any],
        video_duration_sec: float,
        summarization_mode: str = "Баланс (умолчание)",
        custom_max_chunks: int = 8,
        custom_max_evidence: int = 5
    ) -> str:
        """Адаптивная суммаризация в зависимости от режима и длины видео"""
        chunks = (s.get("chunks") or []) if isinstance(s, dict) else []

        # Определение лимитов в зависимости от режима и длины видео
        if summarization_mode == "Баланс (умолчание)":
            # Адаптивные лимиты по умолчанию
            if video_duration_sec < 60:  # < 1 мин
                max_chunks = 6
            elif video_duration_sec < 300:  # 1-5 мин
                max_chunks = 8
            elif video_duration_sec < 900:  # 5-15 мин
                max_chunks = 10
            else:  # > 15 мин
                max_chunks = 6
        elif summarization_mode == "Детальный (больше чанков)":
            max_chunks = max(custom_max_chunks, 12)
        elif summarization_mode == "Компактный (меньше чанков)":
            max_chunks = min(custom_max_chunks, 6)  # Изменено с 4 на 6 для менее агрессивного сокращения
        else:  # Стандартный (адаптивный)
            max_chunks = custom_max_chunks

        lines: list[str] = []
        # НЕ передаём длительность/FPS/разрешение и прочие "характеристики видео" в LLM:
        # это часто попадает в ответ даже если вопрос не про это.
        lines.append(f"chunks={len(chunks)} (показано {min(len(chunks), max_chunks)})")

        # Группировка чанков по временным окнам для лучшей компактности
        time_windows = {}
        for c in chunks[:max_chunks]:
            try:
                start_time = float(c.get('start', 0))
                window_key = int(start_time // 30) * 30  # 30-секундные окна
                if window_key not in time_windows:
                    time_windows[window_key] = []
                time_windows[window_key].append(c)
            except Exception:
                continue

        # Суммирование по окнам
        for window_start in sorted(time_windows.keys()):
            window_chunks = time_windows[window_start]
            total_events = sum(int(c.get('events_count', 0)) for c in window_chunks)
            start_times = [c.get('start', '?') for c in window_chunks]
            end_times = [c.get('end', '?') for c in window_chunks]

            lines.append(
                f"- {min(start_times)}–{max(end_times)}: events={total_events} (чанков: {len(window_chunks)})"
            )

        return "\n".join(lines)

    def _adaptive_evidence_for_llm(
        ev: dict[str, Any],
        video_duration_sec: float,
        summarization_mode: str = "Баланс (умолчание)",
        custom_max_evidence: int = 5
    ) -> str:
        """Адаптивное форматирование доказательств"""
        items = (ev.get("evidence") or []) if isinstance(ev, dict) else []

        # Определение лимитов в зависимости от режима
        if summarization_mode == "Баланс (умолчание)":
            if video_duration_sec < 60:  # < 1 мин
                max_items = 4
                max_text_chars = 150
            elif video_duration_sec < 300:  # 1-5 мин
                max_items = 5
                max_text_chars = 200
            elif video_duration_sec < 900:  # 5-15 мин
                max_items = 6
                max_text_chars = 180
            else:  # > 15 мин
                max_items = 4
                max_text_chars = 150
        elif summarization_mode == "Детальный (больше чанков)":
            max_items = max(custom_max_evidence, 8)
            max_text_chars = 250
        elif summarization_mode == "Компактный (меньше чанков)":
            max_items = min(custom_max_evidence, 3)
            max_text_chars = 120
        else:  # Стандартный (адаптивный)
            max_items = custom_max_evidence
            max_text_chars = 200

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
        return "\n".join(out) if out else "(нет доказательств)"

    # Определение общей длительности видео для адаптивной суммаризации
    video_duration_sec = 0
    if events:
        max_timestamp = max((ev.get("timestamp_sec", 0) for ev in events), default=0)
        video_duration_sec = max_timestamp

    summary_text = _adaptive_summary_for_llm(
        summary, video_duration_sec, summarization_mode, custom_max_chunks, custom_max_evidence
    )
    evidence_text = _adaptive_evidence_for_llm(
        evidence, video_duration_sec, summarization_mode, custom_max_evidence
    )

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

    elif state.get("final_llm") == "openai_local":
        client = maybe_make_openai_local_client()
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
                "Формат ответа (если требуется JSON) — строго следовать этой схеме:\n"
                f"{json.dumps(json_schema, ensure_ascii=False, indent=2)}\n\n"
                "Инструкции:\n"
                "- Всегда отвечай на русском языке.\n"
                "- Если требуется JSON — верни ТОЛЬКО валидный JSON без дополнительного текста.\n"
                "- Иначе ответь кратко и по делу.\n"
                "- Укажи уровень уверенности в ответе (0.0-1.0).\n"
                "- Фокус на ключевых моментах, а не на перечислении каждого события.\n"
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
            except Exception as e:
                pr.log(state, f"generate_final_answer: openai local failed: {e}")

    elif state.get("final_llm") == "llava_local":
        client = maybe_make_local_llava(mp.llava_model_id)
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
                "Формат ответа (если требуется JSON) — строго следовать этой схеме:\n"
                f"{json.dumps(json_schema, ensure_ascii=False, indent=2)}\n\n"
                "Инструкции:\n"
                "- Всегда отвечай на русском языке.\n"
                "- Если требуется JSON — верни ТОЛЬКО валидный JSON без дополнительного текста.\n"
                "- Иначе ответь кратко и по делу.\n"
                "- Укажи уровень уверенности в ответе (0.0-1.0).\n"
                "- Фокус на ключевых моментах, а не на перечислении каждого события.\n"
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
            except Exception as e:
                pr.log(state, f"generate_final_answer: llava local failed: {e}")

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
        image_paths = state.get("image_paths", []) or []
        if len(image_paths) > 1:
            # Множественные изображения — создаем структурированный ответ
            images_data = {}
            for img_path in image_paths:
                image_name = os.path.basename(img_path)
                image_events = [ev for ev in events if ev.get("video_path") == img_path]  # video_path используется для совместимости

                # Группируем объекты по типам
                objects_found = {}
                for ev in image_events:
                    obj_class = ev.get("class", "unknown")
                    if obj_class not in objects_found:
                        objects_found[obj_class] = []
                    objects_found[obj_class].append({
                        "bbox": ev.get("bbox", []),
                        "confidence": ev.get("confidence", 0.0),
                        "area": ev.get("area", 0),
                        "description": ev.get("description", ""),
                    })

                images_data[image_name] = {
                    "image_path": img_path,
                    "objects_found": objects_found,
                    "total_objects": len(image_events),
                    "objects_by_type": {obj_type: len(objs) for obj_type, objs in objects_found.items()},
                    "answer": f"Найдено {len(image_events)} объектов на изображении {image_name}",
                }

            final = {
                "analysis_type": "multiple_images",
                "question": user_query,
                "overall_answer": f"Анализ {len(image_paths)} изображений завершен. Детали по каждому изображению ниже.",
                "images": images_data,
                "total_images": len(image_paths),
                "total_objects": len(events),
                "models_used": models_used,
            }
        elif len(image_paths) == 1:
            # Одно изображение — детальный анализ объектов
            img_path = image_paths[0]
            image_name = os.path.basename(img_path)
            image_events = [ev for ev in events if ev.get("video_path") == img_path]

            # Группируем объекты по типам
            objects_found = {}
            for ev in image_events:
                obj_class = ev.get("class", "unknown")
                if obj_class not in objects_found:
                    objects_found[obj_class] = []
                objects_found[obj_class].append({
                    "bbox": ev.get("bbox", []),
                    "bbox_xyxy": ev.get("bbox_xyxy", []),
                    "confidence": ev.get("confidence", 0.0),
                    "area": ev.get("area", 0),
                    "description": ev.get("description", ""),
                })

            final = {
                "analysis_type": "single_image",
                "question": user_query,
                "image_path": img_path,
                "image_name": image_name,
                "objects_found": objects_found,
                "total_objects": len(image_events),
                "objects_by_type": {obj_type: len(objs) for obj_type, objs in objects_found.items()},
                "answer": f"На изображении {image_name} найдено {len(image_events)} объектов",
                "models_used": models_used,
            }
        elif len(video_paths) > 1:

            # Добавляем информацию о сохраненных уникальных людях
            if state.get("save_unique_people", False):
                final["unique_people_info"] = {
                    "photos_saved": state.get("unique_people_saved", 0),
                    "cross_video_overlaps": state.get("cross_video_overlaps", {}),
                    "unique_people_dir": "unique_people/"
                }
        else:
            # Одно видео — обычный формат
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
    pr = ProgressReporter(
        progress_cb=state.get("progress_cb"),
        enable_detailed_logging=bool(state.get("enable_detailed_logging", False))
    )
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

    # Сохраняем уникальных людей, если включено
    if state.get("save_unique_people", False) and state.get("reid_trajectories"):
        try:
            from pipeline.save_results import save_unique_people_photos
            unique_people_result = save_unique_people_photos(
                out_dir=out_dir,
                reid_trajectories=state.get("reid_trajectories", {}),
                video_paths=state.get("video_paths", []),
                min_faces=state.get("unique_people_min_faces", 3),
                quality_threshold=state.get("unique_people_quality_threshold", 0.7)
            )
            pr.log(state, f"save_results: saved {len(unique_people_result['saved_photos'])} unique people photos")

            # Добавляем информацию в metadata
            metadata["unique_people_saved"] = len(unique_people_result["saved_photos"])
            metadata["cross_video_overlaps"] = unique_people_result["cross_video_overlaps"]

            # Пересохраняем metadata с новой информацией
            save_metadata(out_dir, metadata)

        except Exception as e:
            pr.log(state, f"save_results: failed to save unique people: {e}")

    state["result_path"] = out_dir
    tzip0 = time.perf_counter()
    state["result_zip_bytes"] = zip_dir_to_bytes(out_dir)
    pr.log(state, f"save_results: zip_dir_to_bytes took {(time.perf_counter()-tzip0):.3f}s")
    pr.progress(1.0, "Готово")
    pr.log(state, f"save_results: done -> {out_dir}")
    return state


def error_handler(state: PipelineState) -> PipelineState:
    pr = ProgressReporter(
        progress_cb=state.get("progress_cb"),
        enable_detailed_logging=bool(state.get("enable_detailed_logging", False))
    )
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

