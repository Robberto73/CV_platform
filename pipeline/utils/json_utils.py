from __future__ import annotations

import json
import re
from typing import Any, Optional


_JSON_RE = re.compile(r"\{[\s\S]*\}")


def extract_first_json(text: str) -> Optional[dict[str, Any]]:
    """
    Best-effort: вытаскивает первый JSON-объект из текста.
    """
    if not text:
        return None
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        try:
            return json.loads(text)
        except Exception:
            pass

    m = _JSON_RE.search(text)
    if not m:
        return None
    snippet = m.group(0)
    try:
        return json.loads(snippet)
    except Exception:
        return None

