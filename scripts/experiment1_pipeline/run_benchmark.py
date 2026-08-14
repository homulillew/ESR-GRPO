"""实验一：运行真实模型 Agent Loop 并生成 BrowseComp-Plus 官方 run JSON。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from esr_grpo.browsecomp import load_examples, official_run_record, validate_split_manifest
from esr_grpo.environment import ESREnvironment
from esr_grpo.retrieval import EchoRetrievalClient, InMemoryRetriever
from esr_grpo.rollout import AgentRunner, OpenAIChatPolicy
from esr_grpo.verification import OpenAICompatibleVerifier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="validation")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--corpus-jsonl")
    parser.add_argument("--retrieval-url")
    parser.add_argument("--model-base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--verifier-base-url", required=True)
    parser.add_argument("--verifier-model", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    examples = load_examples(args.dataset)
    manifest = json.loads(Path(args.split_manifest).read_text(encoding="utf-8"))
    validate_split_manifest(manifest, [item.query_id for item in examples])
    selected_ids = set(str(item) for item in manifest[args.split])
    selected = [item for item in examples if item.query_id in selected_ids][: args.limit]
    if bool(args.corpus_jsonl) == bool(args.retrieval_url):
        raise SystemExit("set exactly one of --corpus-jsonl and --retrieval-url")
    retriever = (
        InMemoryRetriever.from_jsonl(args.corpus_jsonl)
        if args.corpus_jsonl
        else EchoRetrievalClient(args.retrieval_url)
    )
    output = Path(args.output_dir)
    stores = output / "stores"
    runs = output / "runs"
    stores.mkdir(parents=True, exist_ok=True)
    runs.mkdir(parents=True, exist_ok=True)

    for index, example in enumerate(selected, start=1):
        database = stores / f"{example.query_id}.sqlite"
        if database.exists():
            print(f"skip existing {database}")
            continue
        environment = ESREnvironment(
            example.question,
            retriever,
            OpenAICompatibleVerifier(args.verifier_base_url, args.verifier_model),
            store_path=database,
            episode_id=example.query_id,
        )
        runner = AgentRunner(
            environment,
            OpenAIChatPolicy(args.model_base_url, args.model),
            max_turns=50,
        )
        result = runner.run()
        (output / f"trajectory_{example.query_id}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        run = official_run_record(
            example.query_id,
            str(result.get("answer") or ""),
            environment.snapshot()["actions"],
            completed=bool(result["submitted"]),
        )
        (runs / f"{example.query_id}.json").write_text(
            json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[{index}/{len(selected)}] {example.query_id}: submitted={result['submitted']}")


if __name__ == "__main__":
    main()
