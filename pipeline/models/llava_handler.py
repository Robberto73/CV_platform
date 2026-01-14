from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from PIL import Image

from pipeline.config import load_model_paths

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore


class LocalLLaVAUnavailable(RuntimeError):
    pass


@dataclass
class LocalLLaVAConfig:
    model_id: str = "llava-v1.6-mistral-7b-hf"
    torch_dtype: str = "float16"
    device_map: str = "auto"
    max_new_tokens: int = 256
    temperature: float = 0.0


class LocalLLaVA:
    def __init__(self, cfg: LocalLLaVAConfig):
        self.cfg = cfg
        self._model = None
        self._processor = None

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        try:
            # transformers==4.37.0 содержит LlavaForConditionalGeneration + LlavaProcessor
            from transformers import LlavaForConditionalGeneration, LlavaProcessor  # type: ignore
        except Exception as e:
            raise LocalLLaVAUnavailable(
                "Не найдены LlavaForConditionalGeneration/LlavaProcessor. "
                "Проверьте версию transformers и что модель LLaVA совместима."
            ) from e

        if torch is None:
            raise LocalLLaVAUnavailable("PyTorch (torch) не установлен — локальная LLaVA недоступна")

        try:
            # dtype: fp16 имеет смысл на GPU; на CPU часто нужен fp32
            want_fp16 = str(self.cfg.torch_dtype).lower() in ("float16", "fp16")
            use_cuda = bool(torch.cuda.is_available())
            dtype = torch.float16 if (want_fp16 and use_cuda) else torch.float32

            model_kwargs = {
                "torch_dtype": dtype,
                "low_cpu_mem_usage": True,
            }

            # device_map требует accelerate; включаем только если просим auto и есть CUDA
            # Также отключаем для избежания конфликтов с CUDA
            if use_cuda and self.cfg.device_map and self.cfg.device_map != "auto":
                model_kwargs["device_map"] = self.cfg.device_map

            # Устанавливаем device вручную для избежания проблем
            device = "cuda" if use_cuda else "cpu"
            if use_cuda:
                model_kwargs.pop("device_map", None)  # Убираем device_map если используем CUDA

            self._model = LlavaForConditionalGeneration.from_pretrained(
                self.cfg.model_id,
                **model_kwargs,
            )

            # Перемещаем модель на устройство
            self._model = self._model.to(device)
            self._processor = LlavaProcessor.from_pretrained(
                self.cfg.model_id,
                use_fast=True  # Используем быстрый процессор изображений
            )
        except Exception as e:
            raise LocalLLaVAUnavailable(
                f"Не удалось загрузить локальную LLaVA модель: {self.cfg.model_id}"
            ) from e

    def _format_prompt(self, prompt: str, with_image: bool) -> str:
        """
        LLaVA ожидает специальный формат с <image> токеном. Если у tokenizer есть chat_template,
        используем его; иначе fallback на классический LLaVA формат.
        """
        self._lazy_load()
        assert self._processor is not None
        tok = getattr(self._processor, "tokenizer", None)

        if tok is not None and getattr(tok, "chat_template", None):
            try:
                if with_image:
                    messages = [{"role": "user", "content": "<image>\n" + prompt}]
                else:
                    messages = [{"role": "user", "content": prompt}]
                return tok.apply_chat_template(  # type: ignore[attr-defined]
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass

        if with_image:
            return f"USER: <image>\n{prompt}\nASSISTANT:"
        return f"USER: {prompt}\nASSISTANT:"

    def chat_with_image(self, prompt: str, image: Image.Image) -> str:
        self._lazy_load()
        assert self._model is not None
        assert self._processor is not None

        # Валидация изображения
        if image is None:
            return "Ошибка: изображение не предоставлено"

        if not hasattr(image, 'size') or image.size[0] == 0 or image.size[1] == 0:
            return "Ошибка: некорректное изображение (нулевой размер)"

        # Убедимся, что изображение в RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Ограничим размер изображения для стабильности
        max_size = 1024
        if image.size[0] > max_size or image.size[1] > max_size:
            image.thumbnail((max_size, max_size), Image.LANCZOS)

        formatted = self._format_prompt(prompt, with_image=True)

        try:
            inputs = self._processor(text=formatted, images=image, return_tensors="pt")

            # Проверяем, что все тензоры корректные
            for k, v in inputs.items():
                if hasattr(v, 'shape') and len(v.shape) > 0:
                    if any(dim <= 0 for dim in v.shape):
                        return f"Ошибка: некорректная размерность тензора {k}: {v.shape}"

        except Exception as e:
            # Попробуем fallback без изображения
            try:
                inputs = self._processor(text=formatted, return_tensors="pt")
                print(f"Предупреждение: изображение пропущено из-за ошибки: {str(e)}")
            except Exception as e2:
                return f"Ошибка обработки текста процессором: {str(e2)}"
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        assert torch is not None
        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=int(self.cfg.max_new_tokens),
                do_sample=bool(self.cfg.temperature and self.cfg.temperature > 0),
                temperature=float(self.cfg.temperature) if self.cfg.temperature else None,
            )
        text = self._processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        # best-effort: убрать префикс prompt
        if "ASSISTANT:" in text:
            text = text.split("ASSISTANT:")[-1].strip()
        return text.strip()

    def chat_text(self, prompt: str) -> str:
        self._lazy_load()
        assert self._model is not None
        assert self._processor is not None

        formatted = self._format_prompt(prompt, with_image=False)
        inputs = self._processor(text=formatted, images=None, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        assert torch is not None
        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=int(self.cfg.max_new_tokens),
                do_sample=bool(self.cfg.temperature and self.cfg.temperature > 0),
                temperature=float(self.cfg.temperature) if self.cfg.temperature else None,
            )
        text = self._processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        if "ASSISTANT:" in text:
            text = text.split("ASSISTANT:")[-1].strip()
        return text.strip()


def maybe_make_local_llava(model_id: Optional[str] = None) -> LocalLLaVA:
    if model_id is None:
        mp = load_model_paths()
        model_id = mp.llava_model_id
    cfg = LocalLLaVAConfig(model_id=model_id or LocalLLaVAConfig.model_id)
    return LocalLLaVA(cfg)

