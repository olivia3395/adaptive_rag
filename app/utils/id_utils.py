from __future__ import annotations

import hashlib


def stable_id(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()
