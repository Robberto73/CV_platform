from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.config import load_gigachat_ca_bundle_file, load_gigachat_default_key
from pipeline.models.gigachat_client import maybe_make_gigachat_client


def main() -> int:
    key = load_gigachat_default_key()
    ca = load_gigachat_ca_bundle_file()
    print("key_present:", bool(key))
    print("ca_bundle_file:", ca)
    client = maybe_make_gigachat_client(key, ca_bundle_file=ca)
    if not client:
        print("No client (missing key)")
        return 2
    try:
        txt = client.chat_text("ping")
        print("OK chat_text:", txt[:200])
        return 0
    except Exception as e:
        print("ERR:", type(e).__name__, str(e)[:500])
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

