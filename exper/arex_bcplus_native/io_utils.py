"""实验文件的原子写入和数据读取。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def atomic_write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, target)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "query_id" not in row:
                raise ValueError(f"{path}:{line_number}: missing query_id")
            rows.append(row)
    return rows


def response_record(answer: str, confidence: str) -> str:
    answer = str(answer).strip()
    confidence = str(confidence).strip()
    return f"{answer}\n\nConfidence: {confidence}" if confidence else answer

