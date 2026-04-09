from __future__ import annotations

from pathlib import Path
from typing import List


def list_text_files(data_dir: str) -> List[Path]:
    root = Path(data_dir)
    exts = {".txt", ".md"}
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]
