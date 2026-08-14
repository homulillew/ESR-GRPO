"""从训练日志提取 OOM、Traceback、非有限数值和关键指标。"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path


ERROR_PATTERNS = {
    "cuda_oom": re.compile(r"(?:CUDA out of memory|OutOfMemoryError)", re.IGNORECASE),
    "traceback": re.compile(r"Traceback \(most recent call last\)"),
    "nccl": re.compile(r"NCCL.*(?:error|failed|timeout)", re.IGNORECASE),
    "ray_worker": re.compile(r"(?:RayTaskError|WorkerCrashedError|worker.*died)", re.IGNORECASE),
}
METRIC_PATTERN = re.compile(
    r"(?P<name>(?:policy[_/ ]loss|actor[_/ ]loss|reward|kl|clip[_/ ]fraction|"
    r"grad[_/ ]norm|mask[_/ ]density|legal[_/ ]submit[_/ ]rate))\s*[=:]\s*"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--context", type=int, default=2, help="错误前后保留的行数")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    lines = args.log.read_text(encoding="utf-8", errors="replace").splitlines()
    errors = Counter()
    excerpts = []
    metrics: dict[str, list[dict[str, float | int]]] = {}
    non_finite = []
    for index, line in enumerate(lines):
        matched_codes = [code for code, pattern in ERROR_PATTERNS.items() if pattern.search(line)]
        for code in matched_codes:
            errors[code] += 1
        if matched_codes:
            start = max(0, index - args.context)
            end = min(len(lines), index + args.context + 1)
            excerpts.append({"line": index + 1, "codes": matched_codes, "text": lines[start:end]})
        for match in METRIC_PATTERN.finditer(line):
            name = re.sub(r"[_/ ]+", "_", match.group("name").lower())
            value = float(match.group("value"))
            metrics.setdefault(name, []).append({"line": index + 1, "value": value})
            if not math.isfinite(value):
                non_finite.append({"line": index + 1, "name": name, "value": match.group("value")})
        if re.search(r"\b(?:nan|[+-]?inf)\b", line, re.IGNORECASE):
            non_finite.append({"line": index + 1, "text": line[:500]})

    summary = {
        name: {"count": len(values), "first": values[0]["value"], "last": values[-1]["value"],
               "min": min(item["value"] for item in values), "max": max(item["value"] for item in values)}
        for name, values in metrics.items()
    }
    report = {
        "valid": not errors and not non_finite,
        "lines": len(lines),
        "error_counts": dict(errors),
        "non_finite": non_finite,
        "metric_summary": summary,
        "error_excerpts": excerpts,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)
    if args.strict and not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
