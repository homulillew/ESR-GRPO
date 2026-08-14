"""批量检查实验二轨迹是否遵守 TaskState、Evidence 和动作账本约束。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from esr_grpo.diagnostics import audit_episode_store
from esr_grpo.store import EpisodeStore


def discover_stores(root: Path, recursive: bool) -> list[Path]:
    if root.is_file():
        return [root]
    pattern = "**/*.sqlite" if recursive else "*.sqlite"
    return sorted(root.glob(pattern))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stores", required=True, type=Path, help="单个 SQLite 文件或轨迹目录")
    parser.add_argument("--recursive", action="store_true", help="递归查找 SQLite 轨迹")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--strict", action="store_true", help="存在协议错误时返回非零退出码")
    args = parser.parse_args()

    paths = discover_stores(args.stores, args.recursive)
    reports = []
    open_errors = []
    for path in paths:
        try:
            with EpisodeStore(path) as store:
                report = audit_episode_store(store)
            report["path"] = str(path.resolve())
            reports.append(report)
        except Exception as exc:  # 单个坏文件不应阻止发现其他坏轨迹。
            open_errors.append({"path": str(path.resolve()), "error": f"{type(exc).__name__}: {exc}"})

    codes = Counter(
        issue["code"]
        for report in reports
        for issue in report["issues"]
        if issue["severity"] == "error"
    )
    report = {
        "valid": bool(paths) and not open_errors and all(item["valid"] for item in reports),
        "stores": len(paths),
        "valid_stores": sum(item["valid"] for item in reports),
        "invalid_stores": sum(not item["valid"] for item in reports) + len(open_errors),
        "error_codes": dict(codes.most_common()),
        "open_errors": open_errors,
        "per_episode": reports,
    }
    if not paths:
        report["message"] = "没有找到 .sqlite 轨迹"
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)
    if args.strict and not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
