"""检查人工动作信用标签，并与结构信用路由结果比较。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from esr_grpo.analysis import compare_credit_labels
from esr_grpo.credit import CreditRouter
from esr_grpo.store import EpisodeStore


def load_labels(path: Path) -> tuple[dict[str, list[str]], list[str]]:
    labels: dict[str, list[str]] = {}
    errors: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            episode_id = str(row["episode_id"])
            selected = [str(item) for item in row["selected_action_ids"]]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            errors.append(f"line {line_number}: {exc}")
            continue
        if episode_id in labels:
            errors.append(f"line {line_number}: duplicate episode_id {episode_id}")
        if len(selected) != len(set(selected)):
            errors.append(f"line {line_number}: duplicate action ID")
        labels[episode_id] = selected
    return labels, errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stores", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    labels, errors = load_labels(args.labels)
    store_paths = sorted(args.stores.glob("*.sqlite")) if args.stores.is_dir() else [args.stores]
    seen_episodes: set[str] = set()
    comparisons = []
    per_episode = []
    for path in store_paths:
        with EpisodeStore(path) as store:
            episode_id = str(store.get_metadata("episode_id"))
            seen_episodes.add(episode_id)
            if episode_id not in labels:
                continue
            actions = {item.action_id: item for item in store.list_actions()}
            unknown = sorted(set(labels[episode_id]) - set(actions))
            illegal = sorted(action_id for action_id in labels[episode_id] if action_id in actions and not actions[action_id].legal)
            if unknown:
                errors.append(f"{episode_id}: unknown actions {unknown}")
            if illegal:
                errors.append(f"{episode_id}: illegal labeled actions {illegal}")
            predicted = CreditRouter().structural_trace(store).selected_action_ids
            metrics = compare_credit_labels(predicted, labels[episode_id])
            comparisons.append(metrics)
            per_episode.append({"episode_id": episode_id, "path": str(path), **metrics})

    missing_stores = sorted(set(labels) - seen_episodes)
    if missing_stores:
        errors.append(f"labels without stores: {missing_stores}")
    averages = {
        key: sum(float(item[key]) for item in comparisons) / len(comparisons)
        for key in ("precision", "recall", "f1")
    } if comparisons else {}
    report = {
        "valid": not errors,
        "errors": errors,
        "labeled_episodes": len(labels),
        "matched_episodes": len(comparisons),
        "average": averages,
        "per_episode": per_episode,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)
    if args.strict and errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
