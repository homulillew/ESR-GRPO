"""实验一前置检查：数据、切分、qrels、语料和服务端点。"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from esr_grpo.browsecomp import load_examples, read_qrels, validate_split_manifest


def endpoint_check(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "esr-grpo-preflight/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            body = response.read(4096).decode("utf-8", errors="replace")
            return {"ok": 200 <= response.status < 300, "status": response.status, "body": body[:500]}
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return {"ok": False, "error": str(exc)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--corpus-jsonl")
    parser.add_argument("--retrieval-health-url")
    parser.add_argument("--policy-models-url")
    parser.add_argument("--verifier-models-url")
    parser.add_argument("--judge-models-url")
    parser.add_argument("--output")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    errors: list[str] = []
    warnings: list[str] = []
    examples = load_examples(args.dataset)
    if not examples:
        errors.append("dataset is empty")
    empty_questions = [item.query_id for item in examples if not item.question.strip()]
    empty_answers = [item.query_id for item in examples if not item.answer.strip()]
    if empty_questions:
        errors.append(f"empty questions: {empty_questions[:10]}")
    if empty_answers:
        errors.append(f"empty answers: {empty_answers[:10]}")

    manifest = json.loads(Path(args.split_manifest).read_text(encoding="utf-8"))
    try:
        validate_split_manifest(manifest, [item.query_id for item in examples])
    except ValueError as exc:
        errors.append(str(exc))

    qrels = read_qrels(args.qrels)
    missing_qrels = sorted({item.query_id for item in examples} - set(qrels))
    if missing_qrels:
        errors.append(f"queries without evidence qrels: {missing_qrels[:10]}")

    corpus = None
    if args.corpus_jsonl:
        count = 0
        empty_documents = 0
        docids: set[str] = set()
        with Path(args.corpus_jsonl).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                docid = str(row.get("docid", ""))
                content = str(row.get("content", row.get("text", row.get("contents", ""))))
                if not docid:
                    errors.append(f"corpus line {line_number} has no docid")
                    break
                if docid in docids:
                    errors.append(f"duplicate corpus docid: {docid}")
                    break
                docids.add(docid)
                empty_documents += int(not content.strip())
                count += 1
        corpus = {"documents": count, "unique_docids": len(docids), "empty_documents": empty_documents}
        if empty_documents:
            warnings.append(f"corpus contains {empty_documents} empty documents")

    services = {}
    for name, url in (
        ("retrieval", args.retrieval_health_url),
        ("policy", args.policy_models_url),
        ("verifier", args.verifier_models_url),
        ("judge", args.judge_models_url),
    ):
        if url:
            services[name] = endpoint_check(url)
            if not services[name]["ok"]:
                errors.append(f"{name} endpoint unavailable: {services[name].get('error', services[name])}")
        else:
            services[name] = {"ok": None, "skipped": True}
            warnings.append(f"{name} endpoint check skipped")

    report = {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "dataset": {"examples": len(examples), "empty_questions": len(empty_questions), "empty_answers": len(empty_answers)},
        "splits": {name: len(manifest.get(name, [])) for name in ("train", "validation", "test")},
        "qrels": {"queries": len(qrels), "missing_queries": len(missing_qrels)},
        "corpus": corpus,
        "services": services,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered)
    if args.strict and errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
