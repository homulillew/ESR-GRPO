"""显示单条轨迹的结构信用选择、原因、token mask 和动作时间线。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from esr_grpo.diagnostics import build_credit_debug_report
from esr_grpo.store import EpisodeStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("store", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--selected-only", action="store_true", help="时间线只保留被分配信用的动作")
    args = parser.parse_args()

    with EpisodeStore(args.store) as store:
        report = build_credit_debug_report(store)
    if args.selected_only:
        report["timeline"] = [item for item in report["timeline"] if item["selected"]]
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
