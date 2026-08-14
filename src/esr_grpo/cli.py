"""项目统一命令行入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import analyze_episode_stores, write_json
from .browsecomp import (
    evidence_recall,
    load_examples,
    make_split_manifest,
    read_qrels,
    validate_split_manifest,
)


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="esr-grpo")
    sub = parser.add_subparsers(dest="command", required=True)

    split = sub.add_parser("make-split", help="创建并冻结自定义 BrowseComp-Plus 切分")
    split.add_argument("--dataset", required=True)
    split.add_argument("--output", required=True)
    split.add_argument("--seed", type=int, default=20260806)
    split.add_argument("--train-ratio", type=float, default=0.7)
    split.add_argument("--validation-ratio", type=float, default=0.15)

    validate = sub.add_parser("validate-split", help="检查切分是否重叠并覆盖全部 query ID")
    validate.add_argument("--dataset", required=True)
    validate.add_argument("--manifest", required=True)

    analyze = sub.add_parser("analyze", help="统计实验二所需的状态和动作指标")
    analyze.add_argument("stores", nargs="+")
    analyze.add_argument("--output")

    recall = sub.add_parser("retrieval-recall", help="根据官方 evidence qrels 计算检索召回率")
    recall.add_argument("--qrels", required=True)
    recall.add_argument("--runs", required=True, help="官方 run JSON 文件所在目录")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _make_parser().parse_args(argv)
    if args.command == "make-split":
        examples = load_examples(args.dataset)
        manifest = make_split_manifest(
            [item.query_id for item in examples],
            seed=args.seed,
            train_ratio=args.train_ratio,
            validation_ratio=args.validation_ratio,
        )
        validate_split_manifest(manifest, [item.query_id for item in examples])
        write_json(args.output, manifest)
        print(json.dumps({key: len(value) for key, value in manifest.items() if isinstance(value, list)}))
        return
    if args.command == "validate-split":
        examples = load_examples(args.dataset)
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        validate_split_manifest(manifest, [item.query_id for item in examples])
        print("split manifest is valid")
        return
    if args.command == "analyze":
        result = analyze_episode_stores(args.stores)
        if args.output:
            write_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.command == "retrieval-recall":
        qrels = read_qrels(args.qrels)
        scores: list[float] = []
        for path in sorted(Path(args.runs).glob("*.json")):
            row = json.loads(path.read_text(encoding="utf-8"))
            query_id = str(row["query_id"])
            if query_id in qrels:
                scores.append(evidence_recall(row.get("retrieved_docids", []), qrels[query_id]))
        average = sum(scores) / len(scores) if scores else float("nan")
        print(json.dumps({"queries": len(scores), "evidence_recall": average}, indent=2))
        return


if __name__ == "__main__":
    main()
