from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from PIL import Image


class LocalLLaVAUnavailable(RuntimeError):
    pass


@dataclass
class LocalLLaVAConfig:
    model_id: str = "llava-v1.6-mistral-7b-hf"
    torch_dtype: torch.dtype = torch.float16
    device_map: str = "auto"
    max_new_tokens: int = 256


class LocalLLaVA:
    def __init__(self, cfg: LocalLLaVAConfig):
        self.cfg = cfg
        self._model = None
        self._processor = None

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor
        except Exception as e:
            raise LocalLLaVAUnavailable(
                "transformers не установлен или не поддерживается в окружении"
            ) from e

        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.cfg.model_id,
                torch_dtype=self.cfg.torch_dtype,
                low_cpu_mem_usage=True,
                device_map=self.cfg.device_map,
            )
            self._processor = AutoProcessor.from_pretrained(self.cfg.model_id)
        except Exception as e:
            raise LocalLLaVAUnavailable(
                f"Не удалось загрузить локальную LLaVA модель: {self.cfg.model_id}"
            ) from e

    def chat_with_image(self, prompt: str, image: Image.Image) -> str:
        self._lazy_load()
        assert self._model is not None
        assert self._processor is not None

        inputs = self._processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs, max_new_tokens=int(self.cfg.max_new_tokens)
            )
        text = self._processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        return text

    def chat_text(self, prompt: str) -> str:
        # Для совместимости: текстовый вызов без изображения (можно расширить).
        self._lazy_load()
        raise LocalLLaVAUnavailable(
            "Текстовый режим для Local LLaVA не реализован в этой версии (нужен multimodal prompt/template)."
        )


def maybe_make_local_llava(model_id: Optional[str] = None) -> LocalLLaVA:
    cfg = LocalLLaVAConfig(model_id=model_id or LocalLLaVAConfig.model_id)
    return LocalLLaVA(cfg)

