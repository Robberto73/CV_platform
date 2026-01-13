from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Optional

import yaml


@dataclass(frozen=True)
class ModelPaths:
    llava_model_id: str = "llava-v1.6-mistral-7b-hf"
    yolo_model_path: str = "yolov8n.pt"
    osnet_reid_model: str = "osnet_x1_0_imagenet.pt"


def _read_yaml(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def _read_json(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f) or {}
    return data if isinstance(data, dict) else {}


def _write_yaml(path: str, data: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


@lru_cache(maxsize=1)
def load_model_paths(config_path: str = os.path.join("config", "model_paths.yaml")) -> ModelPaths:
    data = _read_yaml(config_path)
    # env overrides (если надо)
    llava = os.getenv("CV_LAVA_MODEL_ID") or data.get("llava_model_id") or ModelPaths.llava_model_id
    yolo = os.getenv("CV_YOLO_MODEL_PATH") or data.get("yolo_model_path") or ModelPaths.yolo_model_path
    osnet = os.getenv("CV_OSNET_REID_MODEL") or data.get("osnet_reid_model") or ModelPaths.osnet_reid_model
    return ModelPaths(
        llava_model_id=str(llava),
        yolo_model_path=str(yolo),
        osnet_reid_model=str(osnet),
    )


def get_available_cv_models() -> list[str]:
    """
    Возвращает только реально доступные модели/компоненты в текущей установке.
    Имена — внутренние ключи пайплайна (то, что пишем в required_models).
    """
    mp = load_model_paths()
    available: list[str] = []
    if mp.yolo_model_path and (os.path.exists(mp.yolo_model_path) or os.path.exists(os.path.join(os.getcwd(), mp.yolo_model_path))):
        available.append("yolo-person")
    # reid/osnet
    if mp.osnet_reid_model and os.path.exists(mp.osnet_reid_model):
        try:
            import torchreid  # type: ignore  # noqa: F401

            available.append("reid-osnet")
        except Exception:
            pass
    # zone detector пока только как будущий модуль (не считаем доступным без явной конфигурации)
    return available


@lru_cache(maxsize=1)
def load_app_settings(path: str = os.path.join("config", "app_settings.yaml")) -> dict[str, Any]:
    if not os.path.exists(path):
        return {"analyze_people_default": True}
    data = _read_yaml(path)
    if "analyze_people_default" not in data:
        data["analyze_people_default"] = True
    return data


def save_app_settings(settings: dict[str, Any], path: str = os.path.join("config", "app_settings.yaml")) -> None:
    _write_yaml(path, settings)


def _mask_secret(s: str) -> str:
    if not s:
        return ""
    if len(s) <= 6:
        return "***"
    return s[:3] + "***" + s[-3:]


@lru_cache(maxsize=1)
def load_gigachat_default_key(
    keys_path: str = os.path.join("config", "gigachat_keys.json"),
) -> Optional[str]:
    # В приоритете — env (не хранить ключи в репо)
    env_key = os.getenv("GIGACHAT_API_KEY")
    if env_key:
        return env_key

    data = _read_json(keys_path)
    keys = data.get("keys")
    if isinstance(keys, list) and keys:
        first = keys[0]
        if isinstance(first, dict):
            k = first.get("api_key")
            if isinstance(k, str) and k.strip():
                return k.strip()
    return None


@lru_cache(maxsize=1)
def load_gigachat_ca_bundle_file(
    config_dir: str = "config",
) -> Optional[str]:
    """
    Автопоиск файла корневых сертификатов рядом с gigachat_keys.json.
    Поддерживаем распространенные имена из документации.
    """
    env_path = os.getenv("GIGACHAT_CA_BUNDLE_FILE") or os.getenv("REQUESTS_CA_BUNDLE")
    if env_path and env_path.strip():
        p = os.path.expandvars(os.path.expanduser(env_path.strip()))
        if not os.path.isabs(p):
            p = os.path.abspath(p)
        return p if os.path.exists(p) else None

    candidates = [
        "russian_trusted_root_ca.cer",
        "russian_trusted_root_ca.crt",
        "russian_trusted_root_ca_pem.crt",
    ]
    for name in candidates:
        p = os.path.join(config_dir, name)
        if os.path.exists(p):
            return os.path.abspath(p)

    # fallback: любой *.cer/*.crt в config/
    try:
        for fn in os.listdir(config_dir):
            low = fn.lower()
            if low.endswith(".cer") or low.endswith(".crt"):
                p = os.path.join(config_dir, fn)
                if os.path.isfile(p):
                    return os.path.abspath(p)
    except Exception:
        pass
    return None


def describe_loaded_config_for_ui() -> dict[str, Any]:
    mp = load_model_paths()
    key = load_gigachat_default_key()
    ca_bundle = load_gigachat_ca_bundle_file()
    app = load_app_settings()
    return {
        "llava_model_id": mp.llava_model_id,
        "yolo_model_path": mp.yolo_model_path,
        "osnet_reid_model": mp.osnet_reid_model,
        "gigachat_key_present": bool(key),
        "gigachat_key_masked": _mask_secret(key or ""),
        "gigachat_ca_bundle_file": ca_bundle,
        "analyze_people_default": bool(app.get("analyze_people_default", True)),
        "available_cv_models": get_available_cv_models(),
    }

