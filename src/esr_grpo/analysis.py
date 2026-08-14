"""实验二需要的状态、协议和信用分配诊断。"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from .credit import CreditRouter
from .models import ActionKind
from .store import EpisodeStore


def analyze_episode_stores(paths: Iterable[str | Path]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    action_counts: Counter[str] = Counter()
    illegal_counts: Counter[str] = Counter()
    for raw_path in paths:
        path = Path(raw_path)
        with EpisodeStore(path) as store:
            actions = store.list_actions()
            evidence = store.list_evidence()
            states = store.list_states()
            trace = CreditRouter().structural_trace(store)
            for action in actions:
                action_counts[action.kind.value] += 1
                if not action.legal:
                    illegal_counts[action.kind.value] += 1
            token_total = sum(span.end - span.start for action in actions for span in action.token_spans)
            selected = set(trace.selected_action_ids)
            credited = sum(
                span.end - span.start
                for action in actions
                if action.action_id in selected
                for span in action.token_spans
            )
            rows.append(
                {
                    "path": str(path),
                    "submitted": store.get_metadata("submit_action_id") is not None,
                    "evidence_count": len(evidence),
                    "state_versions": len(states),
                    "action_count": len(actions),
                    "illegal_actions": sum(not item.legal for item in actions),
                    "read_evidence_calls": sum(item.kind is ActionKind.READ_EVIDENCE for item in actions),
                    "credit_action_count": len(selected),
                    "credit_token_density": credited / token_total if token_total else 0.0,
                }
            )
    return {
        "episodes": len(rows),
        "legal_submit_rate": mean(item["submitted"] for item in rows) if rows else 0.0,
        "avg_evidence_count": mean(item["evidence_count"] for item in rows) if rows else 0.0,
        "avg_state_versions": mean(item["state_versions"] for item in rows) if rows else 0.0,
        "avg_read_evidence_calls": mean(item["read_evidence_calls"] for item in rows) if rows else 0.0,
        "avg_credit_token_density": mean(item["credit_token_density"] for item in rows) if rows else 0.0,
        "action_counts": dict(action_counts),
        "illegal_action_counts": dict(illegal_counts),
        "per_episode": rows,
    }


def compare_credit_labels(predicted_ids: Iterable[str], labelled_ids: Iterable[str]) -> dict[str, float | int]:
    predicted = set(predicted_ids)
    labelled = set(labelled_ids)
    true_positive = len(predicted.intersection(labelled))
    precision = true_positive / len(predicted) if predicted else (1.0 if not labelled else 0.0)
    recall = true_positive / len(labelled) if labelled else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": true_positive,
        "predicted": len(predicted),
        "labelled": len(labelled),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive": len(predicted - labelled),
        "false_negative": len(labelled - predicted),
    }


def write_json(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
