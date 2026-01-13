from __future__ import annotations

import json
import os
import base64
from typing import Any, Optional

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from gigachat import GigaChat
from gigachat.models.messages import Messages
from gigachat.models.chat import Chat
from gigachat.exceptions import ResponseError, AuthenticationError


class GigaChatError(RuntimeError):
    pass


class GigaChatPaymentRequired(GigaChatError):
    """HTTP 402 from API (quota/billing/plan)."""


class GigaChatRetryableError(GigaChatError):
    """Temporary / retryable errors."""


class GigaChatClient:
    def __init__(
        self,
        credentials: str,
        scope: str = "GIGACHAT_API_PERS",
        base_url: Optional[str] = None,
        auth_url: Optional[str] = None,
        model: str = "GigaChat",
        timeout_sec: float = 60.0,
        ca_bundle_file: Optional[str] = None,
        max_tokens_text: int = 2048,
        max_tokens_vision: int = 512,
    ):
        """
        Реализация через официальный SDK `gigachat` (developers.sber.ru).

        Важно: для SSL может потребоваться корневой сертификат Минцифры России.
        В SDK это задаётся через `ca_bundle_file=...`.
        """
        self.credentials = credentials
        self.scope = scope
        self.base_url = base_url
        self.auth_url = auth_url
        self.model = model
        self.timeout_sec = float(timeout_sec)
        self.ca_bundle_file = ca_bundle_file
        self.ca_bundle_file = _normalize_ca_bundle_file(self.ca_bundle_file)
        self.max_tokens_text = int(max_tokens_text)
        self.max_tokens_vision = int(max_tokens_vision)

        try:
            self._sdk = GigaChat(
                credentials=self.credentials,
                scope=self.scope,
                model=self.model,
                timeout=self.timeout_sec,
                base_url=self.base_url,
                auth_url=self.auth_url,
                ca_bundle_file=self.ca_bundle_file,
                verify_ssl_certs=True,
            )
        except Exception as e:
            raise GigaChatError(f"Failed to init GigaChat SDK: {type(e).__name__}: {e}") from e

    @retry(
        retry=retry_if_exception_type((GigaChatRetryableError,)),
        stop=stop_after_attempt(3),
        wait=wait_fixed(1),
        reraise=True,
    )
    def chat_with_image(self, prompt: str, base64_image: str) -> str:
        try:
            # Официальный путь для передачи изображения в SDK:
            # 1) загрузить файл через /files
            # 2) сослаться на file_id через messages.attachments
            img_bytes = base64.b64decode(base64_image)
            uploaded = self._sdk.upload_file(
                file=("frame.jpg", img_bytes, "image/jpeg"),
                purpose="general",
            )
            file_id = getattr(uploaded, "id_", None)
            if not file_id:
                raise GigaChatError("upload_file did not return file id")

            chat = Chat(
                model=self.model,
                messages=[
                    Messages(role="user", content=prompt, attachments=[file_id]),
                ],
                max_tokens=self.max_tokens_vision,
            )
            resp = self._sdk.chat(chat)
            text = str(resp.choices[0].message.content)

            # best-effort cleanup
            try:
                self._sdk.delete_file(file_id)
            except Exception:
                pass

            return text
        except (AuthenticationError, ResponseError) as e:
            _raise_translated_sdk_error(e, where="chat_with_image")
            raise  # pragma: no cover
        except Exception as e:
            raise GigaChatError(f"GigaChat chat_with_image failed: {type(e).__name__}: {e}") from e

    @retry(
        retry=retry_if_exception_type((GigaChatRetryableError,)),
        stop=stop_after_attempt(3),
        wait=wait_fixed(1),
        reraise=True,
    )
    def chat_text(self, prompt: str) -> str:
        try:
            chat = Chat(
                model=self.model,
                messages=[Messages(role="user", content=prompt)],
                max_tokens=self.max_tokens_text,
            )
            resp = self._sdk.chat(chat)
            return str(resp.choices[0].message.content)
        except (AuthenticationError, ResponseError) as e:
            _raise_translated_sdk_error(e, where="chat_text")
            raise  # pragma: no cover
        except Exception as e:
            raise GigaChatError(f"GigaChat chat_text failed: {type(e).__name__}: {e}") from e


def maybe_make_gigachat_client(
    credentials: Optional[str],
    ca_bundle_file: Optional[str] = None,
    scope: str = "GIGACHAT_API_PERS",
) -> Optional[GigaChatClient]:
    if not credentials:
        return None
    return GigaChatClient(
        credentials=credentials,
        scope=scope,
        ca_bundle_file=ca_bundle_file,
    )


def _normalize_ca_bundle_file(path: Optional[str]) -> Optional[str]:
    """
    - Если путь относительный, считаем его относительным к `./config/` (как вы храните gigachat_keys.json),
      а если там нет — к текущей рабочей директории.
    - Если файла нет, возвращаем None (SDK пойдёт по системному store и может упасть с SSL error —
      но уже без "invalid path").
    """
    if not path:
        # Попробуем автопоиск в config/
        for name in ("russian_trusted_root_ca.cer", "russian_trusted_root_ca.crt", "russian_trusted_root_ca_pem.crt"):
            p = os.path.abspath(os.path.join("config", name))
            if os.path.exists(p):
                return p
        return None

    p = os.path.expandvars(os.path.expanduser(str(path).strip()))
    if not p:
        return None
    if not os.path.isabs(p):
        cand = os.path.abspath(os.path.join("config", p))
        if os.path.exists(cand):
            return cand
        p = os.path.abspath(p)
    return p if os.path.exists(p) else None


def _extract_status_code(err: Exception) -> Optional[int]:
    """
    gigachat.exceptions.ResponseError/AuthenticationError обычно хранят (url, status_code, body, headers) в args.
    """
    try:
        if getattr(err, "args", None) and len(err.args) >= 2:
            return int(err.args[1])
    except Exception:
        return None
    return None


def _extract_body(err: Exception) -> str:
    try:
        if getattr(err, "args", None) and len(err.args) >= 3:
            b = err.args[2]
            if isinstance(b, (bytes, bytearray)):
                return b.decode("utf-8", errors="ignore")
            return str(b)
    except Exception:
        pass
    return str(err)


def _raise_translated_sdk_error(e: Exception, where: str) -> None:
    code = _extract_status_code(e)
    body = _extract_body(e)
    if code == 402:
        raise GigaChatPaymentRequired(f"{where}: HTTP 402 Payment Required: {body[:300]}")
    if code is not None and 500 <= code < 600:
        raise GigaChatRetryableError(f"{where}: HTTP {code}: {body[:300]}")
    raise GigaChatError(f"{where}: HTTP {code}: {body[:300]}")

