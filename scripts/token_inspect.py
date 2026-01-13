from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gigachat import GigaChat

from pipeline.config import load_gigachat_ca_bundle_file, load_gigachat_default_key


def main() -> int:
    key = load_gigachat_default_key()
    ca = load_gigachat_ca_bundle_file()
    print("key_present:", bool(key))
    print("ca:", ca)
    g = GigaChat(credentials=key, scope="GIGACHAT_API_PERS", ca_bundle_file=ca, verify_ssl_certs=True)
    r = g.chat("ping")
    print("reply:", r.choices[0].message.content)
    print("token_present:", bool(getattr(g, "token", None)))
    at = getattr(g, "_access_token", None)
    print("_access_token_present:", bool(at))
    if at is not None:
        print("_access_token_type:", type(at).__name__)
        # не печатаем сам токен; только структуру
        if hasattr(at, "__dict__"):
            print("_access_token_keys:", sorted(list(at.__dict__.keys()))[:50])
        if isinstance(at, dict):
            print("_access_token_dict_keys:", sorted(list(at.keys()))[:50])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

