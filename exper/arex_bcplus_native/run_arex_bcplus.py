#!/usr/bin/env python3
"""在自定义 BC-Plus split 上逐题运行原生 AREX-Turbo。"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from clients import AREXChatClient, BCPlusRetrievalClient, CorpusURLIndex
from io_utils import atomic_write_json, load_jsonl
from runner import AREXRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="validation")
    parser.add_argument("--limit", type=int, default=10, help="正整数；按 split manifest 顺序取前 N 题")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--query-id", help="只运行指定 query_id；设置后忽略 limit/offset")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-base-url", default=os.getenv("AREX_BASE_URL", "http://127.0.0.1:8002/v1"))
    parser.add_argument("--model", default=os.getenv("AREX_MODEL", "AREX-Turbo"))
    parser.add_argument("--api-key", default=os.getenv("AREX_API_KEY", "EMPTY"))
    parser.add_argument("--retrieval-url", default=os.getenv("AREX_RETRIEVAL_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--corpus-file", type=Path, help="可选；只读取 docid/url，使轨迹保留真实 URL")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--snippet-chars", type=int, default=4000)
    parser.add_argument("--visit-chars", type=int, default=24000)
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--presence-penalty", type=float, default=1.5)
    parser.add_argument("--top-k-sampling", type=int, default=20)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def select_examples(args: argparse.Namespace) -> list[dict]:
    examples = {str(row["query_id"]): row for row in load_jsonl(args.dataset)}
    manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    split_ids = [str(item) for item in manifest[args.split]]
    all_manifest_ids = [
        str(item) for name in ("train", "validation", "test") for item in manifest[name]
    ]
    if len(all_manifest_ids) != len(set(all_manifest_ids)) or set(all_manifest_ids) != set(examples):
        raise ValueError("split manifest must cover every dataset query exactly once")
    if args.query_id:
        query_id = str(args.query_id)
        if query_id not in examples:
            raise ValueError(f"unknown query_id: {query_id}")
        return [examples[query_id]]
    if args.limit <= 0:
        raise ValueError("--limit must be positive; use the split size for a full run")
    return [examples[item] for item in split_ids[args.offset : args.offset + args.limit]]


def main() -> None:
    args = parse_args()
    selected = select_examples(args)
    trajectories = args.output_dir / "trajectories"
    runs = args.output_dir / "runs"
    failed = args.output_dir / "failed"
    for directory in (trajectories, runs, failed):
        directory.mkdir(parents=True, exist_ok=True)

    corpus_urls = CorpusURLIndex.from_parquet(args.corpus_file)
    summary = {"selected": len(selected), "completed": 0, "failed": 0, "skipped": 0}
    for index, example in enumerate(selected, start=1):
        query_id = str(example["query_id"])
        question = str(example.get("question", example.get("query", "")))
        trajectory_path = trajectories / f"{query_id}.json"
        run_path = runs / f"{query_id}.json"
        failed_path = failed / f"{query_id}.json"

        existing = None
        if trajectory_path.is_file():
            try:
                existing = json.loads(trajectory_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                existing = {"status": "corrupt"}
        if args.overwrite:
            for path in (trajectory_path, run_path, failed_path):
                if path.exists():
                    path.unlink()
        elif existing and existing.get("status") == "completed" and run_path.is_file():
            print(f"[{index}/{len(selected)}] skip completed {query_id}", flush=True)
            summary["skipped"] += 1
            continue
        elif existing and not args.retry_failed:
            print(
                f"[{index}/{len(selected)}] skip {existing.get('status')} {query_id}; "
                "use --retry-failed or --overwrite",
                flush=True,
            )
            summary["skipped"] += 1
            continue
        else:
            for path in (trajectory_path, run_path, failed_path):
                if path.exists():
                    backup = path.with_suffix(path.suffix + ".previous")
                    if backup.exists():
                        backup.unlink()
                    shutil.move(path, backup)

        retrieval = BCPlusRetrievalClient(
            base_url=args.retrieval_url,
            corpus_urls=corpus_urls,
            top_k=args.top_k,
            snippet_chars=args.snippet_chars,
            visit_chars=args.visit_chars,
        )
        model = AREXChatClient(
            base_url=args.model_base_url,
            model=args.model,
            api_key=args.api_key,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            presence_penalty=args.presence_penalty,
            top_k=args.top_k_sampling,
        )
        runner = AREXRunner(
            model=model,
            retrieval=retrieval,
            max_turns=args.max_turns,
            on_checkpoint=lambda value, path=trajectory_path: atomic_write_json(path, value),
        )
        trajectory, run = runner.run(query_id, question)
        atomic_write_json(trajectory_path, trajectory)
        atomic_write_json(run_path, run)
        if trajectory["status"] == "completed":
            if failed_path.exists():
                failed_path.unlink()
            summary["completed"] += 1
        else:
            atomic_write_json(
                failed_path,
                {"query_id": query_id, "status": trajectory["status"], "error": trajectory.get("error")},
            )
            summary["failed"] += 1
        print(
            f"[{index}/{len(selected)}] {query_id}: status={trajectory['status']} "
            f"turns={len(trajectory['turns'])}",
            flush=True,
        )
        if args.fail_fast and trajectory["status"] != "completed":
            raise SystemExit(1)

    atomic_write_json(args.output_dir / "run_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.fail_fast and summary["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

