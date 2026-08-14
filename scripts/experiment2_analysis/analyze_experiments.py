"""实验二：汇总状态使用、非法动作、Evidence 回读和动作信用密度。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from esr_grpo.analysis import analyze_episode_stores, compare_credit_labels, write_json
from esr_grpo.credit import CreditRouter
from esr_grpo.store import EpisodeStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stores", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--labels", help="JSONL: episode_id + selected_action_ids")
    args = parser.parse_args()
    paths = sorted(Path(args.stores).glob("*.sqlite"))
    report = analyze_episode_stores(paths)
    if args.labels:
        labels = {
            str(row["episode_id"]): row["selected_action_ids"]
            for row in (
                json.loads(line)
                for line in Path(args.labels).read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        }
        comparisons = []
        for path in paths:
            with EpisodeStore(path) as store:
                episode_id = str(store.get_metadata("episode_id"))
                if episode_id in labels:
                    predicted = CreditRouter().structural_trace(store).selected_action_ids
                    comparisons.append(compare_credit_labels(predicted, labels[episode_id]))
        if comparisons:
            report["credit_label_comparison"] = {
                key: sum(float(item[key]) for item in comparisons) / len(comparisons)
                for key in ("precision", "recall", "f1")
            }
    write_json(args.output, report)
    print(json.dumps({key: value for key, value in report.items() if key != "per_episode"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
