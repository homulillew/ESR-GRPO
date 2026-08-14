"""BrowseComp-Plus 数据、run 文件和可重复评测辅助函数。"""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class BrowseCompExample:
    query_id: str
    question: str
    answer: str


def load_examples(path: str | Path) -> list[BrowseCompExample]:
    examples: list[BrowseCompExample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            examples.append(
                BrowseCompExample(
                    query_id=str(row["query_id"]),
                    question=str(row.get("question", row.get("query", ""))),
                    answer=str(row.get("answer", "")),
                )
            )
    if len({item.query_id for item in examples}) != len(examples):
        raise ValueError("query_id values are not unique")
    return examples


def read_qrels(path: str | Path) -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for row in csv.reader(handle, delimiter=" "):
            fields = [field for field in row if field]
            if len(fields) < 4:
                continue
            query_id, _, docid, relevance = fields[:4]
            if float(relevance) > 0:
                qrels.setdefault(query_id, set()).add(docid)
    return qrels


def make_split_manifest(
    query_ids: Sequence[str],
    *,
    seed: int,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
) -> dict[str, object]:
    """创建显式自定义切分；不能称为 BrowseComp-Plus 官方切分。"""

    if not 0 < train_ratio < 1 or not 0 <= validation_ratio < 1:
        raise ValueError("invalid split ratios")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("train_ratio + validation_ratio must be below 1")
    ids = sorted(set(str(item) for item in query_ids))
    random.Random(seed).shuffle(ids)
    train_end = int(len(ids) * train_ratio)
    validation_end = train_end + int(len(ids) * validation_ratio)
    return {
        "schema_version": 1,
        "source": "custom split of the official BrowseComp-Plus test release",
        "seed": seed,
        "train": ids[:train_end],
        "validation": ids[train_end:validation_end],
        "test": ids[validation_end:],
    }


def validate_split_manifest(manifest: Mapping[str, object], all_query_ids: Iterable[str]) -> None:
    partitions = [list(manifest.get(name, [])) for name in ("train", "validation", "test")]
    flat = [str(item) for partition in partitions for item in partition]
    if len(flat) != len(set(flat)):
        raise ValueError("split manifest contains overlapping query IDs")
    expected = {str(item) for item in all_query_ids}
    if set(flat) != expected:
        missing = sorted(expected - set(flat))
        extra = sorted(set(flat) - expected)
        raise ValueError(f"split coverage mismatch; missing={missing[:10]}, extra={extra[:10]}")


def evidence_recall(retrieved_docids: Iterable[str], relevant_docids: Iterable[str]) -> float:
    relevant = set(str(item) for item in relevant_docids)
    if not relevant:
        return float("nan")
    return len(set(str(item) for item in retrieved_docids).intersection(relevant)) / len(relevant)


def calibration_error(
    confidences: Sequence[float],
    correctness: Sequence[bool | int],
    *,
    target_bin_size: int = 100,
    p: int = 2,
) -> float:
    """修复 BrowseComp-Plus 脚本遗漏最后一个自适应分箱的问题。"""

    if len(confidences) != len(correctness):
        raise ValueError("confidence and correctness lengths differ")
    if not confidences:
        return float("nan")
    if target_bin_size <= 0 or p not in {1, 2}:
        raise ValueError("invalid calibration parameters")
    ordered = sorted(zip(confidences, correctness), key=lambda item: float(item[0]))
    total = len(ordered)
    error = 0.0
    for start in range(0, total, target_bin_size):
        rows = ordered[start : min(start + target_bin_size, total)]
        confidence_mean = sum(float(item[0]) for item in rows) / len(rows)
        correct_mean = sum(float(bool(item[1])) for item in rows) / len(rows)
        difference = abs(confidence_mean - correct_mean)
        error += len(rows) / total * (difference**p)
    return math.sqrt(error) if p == 2 else error


def official_run_record(
    query_id: str,
    answer: str,
    actions: Sequence[Mapping[str, object]],
    *,
    completed: bool,
) -> dict[str, object]:
    tool_counts: dict[str, int] = {}
    retrieved: set[str] = set()
    for action in actions:
        kind = str(action.get("kind", ""))
        tool_counts[kind] = tool_counts.get(kind, 0) + 1
        metadata = dict(action.get("metadata", {}))
        if kind == "search":
            retrieved.update(str(item.get("docid")) for item in metadata.get("hits", []))
    return {
        "query_id": str(query_id),
        "tool_call_counts": tool_counts,
        "status": "completed" if completed else "incomplete",
        "retrieved_docids": sorted(item for item in retrieved if item),
        "result": [{"type": "output_text", "output": answer}] if completed else [],
    }
