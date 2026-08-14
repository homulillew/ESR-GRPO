#!/usr/bin/env python3
"""检查 AREX 原生 BC-Plus 实验的文件和服务。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from clients import AREXChatClient, BCPlusRetrievalClient, CorpusURLIndex
from io_utils import atomic_write_json, load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--qrels", required=True, type=Path)
    parser.add_argument("--corpus-file", type=Path)
    parser.add_argument("--model-base-url", default=os.getenv("AREX_BASE_URL", "http://127.0.0.1:8002/v1"))
    parser.add_argument("--model", default=os.getenv("AREX_MODEL", "AREX-Turbo"))
    parser.add_argument("--api-key", default=os.getenv("AREX_API_KEY", "EMPTY"))
    parser.add_argument("--retrieval-url", default=os.getenv("AREX_RETRIEVAL_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    errors: list[str] = []
    files = {}
    for name, path in (
        ("dataset", args.dataset),
        ("split_manifest", args.split_manifest),
        ("qrels", args.qrels),
        ("corpus", args.corpus_file),
    ):
        files[name] = None if path is None else {"path": str(path.resolve()), "exists": path.is_file()}
        if path is not None and not path.is_file():
            errors.append(f"missing {name}: {path}")

    dataset_rows = 0
    split_sizes = {}
    probe_query = "BrowseComp Plus retrieval preflight"
    if args.dataset.is_file() and args.split_manifest.is_file():
        try:
            rows = load_jsonl(args.dataset)
            ids = {str(row["query_id"]) for row in rows}
            manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
            flat = [str(item) for name in ("train", "validation", "test") for item in manifest[name]]
            dataset_rows = len(rows)
            if rows:
                probe_query = str(rows[0].get("question", rows[0].get("query", probe_query)))
            split_sizes = {name: len(manifest[name]) for name in ("train", "validation", "test")}
            if len(flat) != len(set(flat)) or set(flat) != ids:
                errors.append("split manifest does not cover dataset exactly once")
        except Exception as exc:
            errors.append(f"data validation failed: {type(exc).__name__}: {exc}")

    services = {}
    try:
        model_response = AREXChatClient(
            args.model_base_url, args.model, args.api_key
        ).models()
        model_ids = [str(item.get("id")) for item in model_response.get("data", [])]
        services["model"] = {"ok": args.model in model_ids, "model_ids": model_ids}
        if args.model not in model_ids:
            errors.append(f"served model name {args.model!r} not found: {model_ids}")
    except Exception as exc:
        services["model"] = {"ok": False, "error": str(exc)}
        errors.append("AREX model endpoint is unavailable")
    try:
        retrieval = BCPlusRetrievalClient(args.retrieval_url, CorpusURLIndex({}), top_k=1)
        health = retrieval.health()
        probe = retrieval.search([probe_query])
        probe_hits = probe[0].get("results", []) if probe else []
        nonempty_snippets = sum(bool(item.get("snippet")) for item in probe_hits)
        services["retrieval"] = {
            "health": health,
            "probe_groups": len(probe),
            "probe_hits": len(probe_hits),
            "nonempty_snippets": nonempty_snippets,
        }
        if not probe_hits or not nonempty_snippets:
            errors.append("retrieval probe returned no usable snippets")
    except Exception as exc:
        services["retrieval"] = {"ok": False, "error": str(exc)}
        errors.append("BC-Plus retrieval endpoint is unavailable")

    report = {
        "valid": not errors,
        "errors": errors,
        "files": files,
        "dataset_rows": dataset_rows,
        "split_sizes": split_sizes,
        "services": services,
    }
    if args.output:
        atomic_write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.strict and errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
