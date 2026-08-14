"""在启动 GPU 任务前检查本项目与指定 ECHO 快照的接口。"""

from __future__ import annotations

import argparse
from pathlib import Path


CHECKS = {
    "verl/trainer/ppo/core_algos.py": ("def register_adv_est", "def compute_supo_advantage"),
    "verl/trainer/ppo/ray_trainer.py": ("echo_response_selection_mask", "rollout_id"),
    "verl/experimental/agent_loop/tool_agent_loop.py": ("class AgentData", "is_final=True"),
    "verl/experimental/reward_loop/reward_manager/naive.py": ("tool_extra_fields", "overlong"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--echo-root", required=True)
    args = parser.parse_args()
    root = Path(args.echo_root).resolve()
    failed: list[str] = []
    for relative, markers in CHECKS.items():
        path = root / relative
        if not path.exists():
            failed.append(f"missing: {path}")
            continue
        text = path.read_text(encoding="utf-8")
        for marker in markers:
            if marker not in text:
                failed.append(f"{relative}: missing marker {marker!r}")
    if failed:
        raise SystemExit("ECHO compatibility check failed:\n" + "\n".join(failed))
    print(f"ECHO compatibility markers passed: {root}")


if __name__ == "__main__":
    main()
