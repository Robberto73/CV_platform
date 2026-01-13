from __future__ import annotations

from typing import Any, Callable, Dict, List, Literal, Optional, TypedDict


class ProSettings(TypedDict, total=False):
    frame_sampling_rate: float
    ssim_threshold: float
    skip_static_frames: bool
    cache_frames: bool
    custom_preprocessing: str


class PipelineState(TypedDict, total=False):
    # inputs
    video_paths: List[str]
    user_query: str
    ui_mode: Literal["STANDARD", "PRO"]
    pro_settings: ProSettings
    require_json: bool
    llm_type: Literal["local", "gigachat"]
    gigachat_api_key: Optional[str]

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

