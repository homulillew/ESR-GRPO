"""把解密后的 BrowseComp-Plus JSONL 和固定 manifest 转换为 ECHO parquet。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from esr_grpo.browsecomp import load_examples, validate_split_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--system-prompt", default="prompts/esr_system.txt")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("install the browsecomp extra: pip install -e '.[browsecomp]'") from exc

    examples = load_examples(args.dataset)
    manifest = json.loads(Path(args.split_manifest).read_text(encoding="utf-8"))
    validate_split_manifest(manifest, [item.query_id for item in examples])
    prompt = Path(args.system_prompt).read_text(encoding="utf-8")
    by_id = {item.query_id: item for item in examples}
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        rows = []
        for index, query_id in enumerate(manifest[split]):
            example = by_id[str(query_id)]
            rows.append(
                {
                    "data_source": "browsecomp_plus",
                    "prompt": [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": example.question},
                    ],
                    "ability": "agent_search",
                    "reward_model": {"style": "llm_judge", "ground_truth": example.answer},
                    "extra_info": {
                        "index": index,
                        "query_id": example.query_id,
                        "question": example.question,
                        "split": split,
                    },
                }
            )
        path = output / f"{split}.esr.parquet"
        pd.DataFrame(rows).to_parquet(path, index=False)
        print(f"wrote {path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
