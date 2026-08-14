#!/usr/bin/env python3
"""把一条 AREX 轨迹压缩成便于人工检查的时间线。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from io_utils import atomic_write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--include-tool-output", action="store_true")
    args = parser.parse_args()
    source = json.loads(args.trajectory.read_text(encoding="utf-8"))
    timeline = []
    for turn in source.get("turns", []):
        item = {
            "turn": turn.get("turn"),
            "reasoning": turn.get("reasoning"),
            "tool_call": turn.get("tool_call"),
            "elapsed_seconds": turn.get("elapsed_seconds"),
            "active_context_reset": turn.get("active_context_reset", False),
            "error": turn.get("error"),
        }
        if args.include_tool_output:
            item["tool_result"] = turn.get("tool_result")
        timeline.append(item)
    result = {
        "query_id": source.get("query_id"),
        "question": source.get("question"),
        "status": source.get("status"),
        "final": source.get("final"),
        "retrieved_docids": source.get("retrieved_docids", []),
        "visited_docids": source.get("visited_docids", []),
        "turns": timeline,
    }
    if args.output:
        atomic_write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

