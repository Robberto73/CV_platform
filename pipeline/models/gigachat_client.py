from __future__ import annotations

import json
from typing import Any, Optional

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed


class GigaChatError(RuntimeError):
    pass


class GigaChatClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions",
        model: str = "GigaChat-Max",
        timeout_sec: float = 60.0,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.timeout_sec = float(timeout_sec)

    def _headers(self) -> dict[str, str]:
        # В реальной интеграции может понадобиться OAuth токенизация.
        # Здесь — минимальный заглушечный вариант для ТЗ.
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    @retry(
        retry=retry_if_exception_type((requests.HTTPError, requests.RequestException)),
        stop=stop_after_attempt(3),
        wait=wait_fixed(1),
        reraise=True,
    )
    def chat_with_image(self, prompt: str, base64_image: str) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
        }
        resp = requests.post(
            self.base_url,
            headers=self._headers(),
            data=json.dumps(payload),
            timeout=self.timeout_sec,
        )
        if 500 <= resp.status_code < 600:
            resp.raise_for_status()
        if resp.status_code == 401:
            raise GigaChatError("Unauthorized (401): invalid API key or token flow")
        if resp.status_code >= 400:
            raise GigaChatError(f"HTTP {resp.status_code}: {resp.text[:500]}")

        data = resp.json()
        # Ожидаем OpenAI-like структуру
        try:
            return str(data["choices"][0]["message"]["content"])
        except Exception as e:
            raise GigaChatError(f"Unexpected response format: {data}") from e

    @retry(
        retry=retry_if_exception_type((requests.HTTPError, requests.RequestException)),
        stop=stop_after_attempt(3),
        wait=wait_fixed(1),
        reraise=True,
    )
    def chat_text(self, prompt: str) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        }
        resp = requests.post(
            self.base_url,
            headers=self._headers(),
            data=json.dumps(payload),
            timeout=self.timeout_sec,
        )
        if 500 <= resp.status_code < 600:
            resp.raise_for_status()
        if resp.status_code == 401:
            raise GigaChatError("Unauthorized (401): invalid API key or token flow")
        if resp.status_code >= 400:
            raise GigaChatError(f"HTTP {resp.status_code}: {resp.text[:500]}")

        data = resp.json()
        try:
            return str(data["choices"][0]["message"]["content"])
        except Exception as e:
            raise GigaChatError(f"Unexpected response format: {data}") from e


def maybe_make_gigachat_client(api_key: Optional[str]) -> Optional[GigaChatClient]:
    if not api_key:
        return None
    return GigaChatClient(api_key=api_key)

