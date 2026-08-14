"""逐动作查看一个实验一 SQLite episode，并执行协议与信用检查。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from esr_grpo.diagnostics import audit_episode_store, build_credit_debug_report
from esr_grpo.store import EpisodeStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("store")
    parser.add_argument("--output")
    parser.add_argument("--show-evidence-content", action="store_true")
    args = parser.parse_args()
    with EpisodeStore(args.store) as store:
        report = {
            "protocol": audit_episode_store(store),
            "credit": build_credit_debug_report(store),
            "task_states": [
                {
                    "version": state.version,
                    "answer": state.answer,
                    "supporting_evidence": list(state.supporting_evidence),
                    "gaps": [gap.description for gap in state.gaps],
                    "verification_status": state.verification_status.value,
                    "created_by_action_id": state.created_by_action_id,
                }
                for state in store.list_states()
            ],
            "evidence": [
                {
                    "evidence_id": item.evidence_id,
                    "docid": item.source.docid,
                    "title": item.source.title,
                    "content_chars": len(item.content),
                    **({"content": item.content} if args.show_evidence_content else {}),
                }
                for item in store.list_evidence()
            ],
        }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
