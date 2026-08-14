"""离线检查 ECHO rollout 逐段导出的 ESR mask、最终奖励和提交标记。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from esr_grpo.diagnostics import audit_rollout_metadata, load_json_or_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON 数组或 JSONL")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    report = audit_rollout_metadata(load_json_or_jsonl(str(args.input)))
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)
    if args.strict and not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
