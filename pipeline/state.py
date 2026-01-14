from __future__ import annotations

from typing import Any, Callable, Dict, List, Literal, Optional, TypedDict


class ProSettings(TypedDict, total=False):
    frame_sampling_rate: float
    ssim_threshold: float
    skip_static_frames: bool
    cache_frames: bool
    custom_preprocessing: str
    llava_image_size: int  # Размер изображений для LLaVA (224, 336, 448, 560, 672) - PRO настройка
    enable_detailed_logging: bool  # Детальное логирование - PRO настройка

    # Умная кластеризация видео - PRO настройка
    enable_smart_clustering: bool
    clustering_ssi_threshold: float
    clustering_window_duration: int


class PipelineState(TypedDict, total=False):
    # inputs
    video_paths: List[str]
    image_paths: List[str]  # Для режима анализа изображений
    user_query: str
    ui_mode: Literal["STANDARD", "PRO"]
    pro_settings: ProSettings
    require_json: bool
    platform_mode: str  # "Видео анализ" или "Анализ изображений"
    is_image_mode: bool
    # routing: отдельно выбираем модель для анализа кадров и отдельно для финального ответа
    vision_llm: Literal["llava_local", "gigachat_api", "off"]
    final_llm: Literal["gigachat_api", "llava_local", "openai_local"]
    gigachat_api_key: Optional[str]
    gigachat_ca_cert_path: Optional[str]
    analyze_people: bool
    force_no_visual_analysis: bool

    # derived by parse_user_request
    required_models: List[str]
    require_visual_analysis: bool

    # frame selection
    selected_frame_selector_name: str

    # key frames
    key_frames: Dict[str, List[Dict[str, Any]]]

    # llava stage
    llava_results: List[Dict[str, Any]]
    events: List[Dict[str, Any]]

    # final
    final_answer: Any
    models_used: List[str]
    events_summary: Dict[str, Any]
    query_evidence: Dict[str, Any]
    final_prompt: str
    person_routes: Dict[str, Any]

    # output & logs
    output_dir: str
    result_path: str
    result_zip_bytes: bytes
    processing_log: List[str]

    # error handling
    error: bool
    error_code: str
    error_message: str

    # runtime-only
    progress_cb: Optional[Callable[[float, str], None]]
    enable_detailed_logging: bool

