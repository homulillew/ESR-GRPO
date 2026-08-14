#!/usr/bin/env python3
"""使用 BC-Plus 官方风格提示评测 AREX run。"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any

from clients import _request_json, _request_json_with_headers
from io_utils import atomic_write_json, load_jsonl


GRADER_TEMPLATE = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

[correct_answer]: {correct_answer}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response].

[correct_answer]: Repeat the [correct_answer] given above.

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], in the context of this [question]. Judge semantic equivalence and allow harmless string variations or additional details only when those details are correct.

correct: Answer 'yes' if the extracted answer matches the correct answer, or is within a small margin of error for numerical problems. Answer 'no' for inconsistency, ambiguity, non-equivalency, or an incorrect answer.

confidence: Extract the confidence score between 0% and 100% from [response]. Put 100 if no confidence score is available.
"""


def _field(text: str, name: str) -> str | None:
    patterns = (
        rf"\*\*{re.escape(name)}:\*\*\s*(.*?)(?=\n|$)",
        rf"\*\*{re.escape(name)}\*\*:\s*(.*?)(?=\n|$)",
        rf"{re.escape(name)}:\s*(.*?)(?=\n|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
    return None


def parse_judgement(text: str) -> dict[str, Any]:
    extracted = _field(text, "extracted_final_answer")
    reasoning = _field(text, "reasoning")
    correct_text = _field(text, "correct")
    confidence_text = _field(text, "confidence")
    confidence_match = re.search(r"(\d+(?:\.\d+)?)", confidence_text or "")
    confidence = min(float(confidence_match.group(1)), 100.0) if confidence_match else None
    correct = None
    if correct_text:
        normalized = correct_text.lower().strip(" .")
        if normalized.startswith("yes"):
            correct = True
        elif normalized.startswith("no"):
            correct = False
    return {
        "extracted_final_answer": extracted,
        "reasoning": reasoning,
        "correct": correct,
        "confidence": confidence,
        "parse_error": extracted is None or correct is None or confidence is None,
        "raw": text,
    }


def read_qrels(path: Path) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) >= 4 and float(fields[3]) > 0:
            result.setdefault(fields[0], set()).add(fields[2])
    return result


def calibration_error(confidences: list[float], correctness: list[bool], bin_size: int = 100) -> float:
    ordered = sorted(zip(confidences, correctness), key=lambda item: item[0])
    total = len(ordered)
    error = 0.0
    for start in range(0, total, bin_size):
        rows = ordered[start : start + bin_size]
        mean_conf = sum(item[0] for item in rows) / len(rows)
        mean_correct = sum(float(item[1]) for item in rows) / len(rows)
        error += len(rows) / total * (mean_conf - mean_correct) ** 2
    return math.sqrt(error)


def final_response(run: dict[str, Any]) -> str:
    rows = [item for item in run.get("result", []) if item.get("type") == "output_text"]
    return str(rows[-1].get("output", "")) if rows else ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--qrels", required=True, type=Path)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--judge-base-url", default=os.getenv("AREX_JUDGE_BASE_URL"))
    parser.add_argument("--judge-model", default=os.getenv("AREX_JUDGE_MODEL", "Qwen3-32B"))
    parser.add_argument("--judge-api-key", default=os.getenv("AREX_JUDGE_API_KEY", "EMPTY"))
    parser.add_argument(
        "--judge-provider",
        default=os.getenv("AREX_JUDGE_PROVIDER", "openai"),
        choices=["openai", "anthropic"],
        help="Judge API 格式：openai（/chat/completions）或 anthropic（/v1/messages）",
    )
    parser.add_argument("--judge-max-tokens", type=int, default=2048)
    parser.add_argument("--exact-match-smoke", action="store_true")
    parser.add_argument("--overwrite-cache", action="store_true")
    args = parser.parse_args()
    if not args.exact_match_smoke and not args.judge_base_url:
        raise SystemExit("set --judge-base-url or use --exact-match-smoke")

    examples = {str(row["query_id"]): row for row in load_jsonl(args.dataset)}
    qrels = read_qrels(args.qrels)
    cache_namespace = "exact_match" if args.exact_match_smoke else re.sub(
        r"[^A-Za-z0-9_.-]+", "_", str(args.judge_model)
    )
    cache_dir = args.output.parent / "judge_cache" / cache_namespace
    cache_dir.mkdir(parents=True, exist_ok=True)
    details: list[dict[str, Any]] = []

    for path in sorted(args.runs.glob("*.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        query_id = str(run["query_id"])
        if query_id not in examples:
            continue
        row = examples[query_id]
        question = str(row.get("question", row.get("query", "")))
        answer = str(row.get("answer", ""))
        response = final_response(run)
        cache_path = cache_dir / f"{query_id}.json"
        if cache_path.is_file() and not args.overwrite_cache:
            judged = json.loads(cache_path.read_text(encoding="utf-8"))
        elif args.exact_match_smoke:
            judged = {
                "extracted_final_answer": response,
                "reasoning": "deterministic exact-match smoke",
                "correct": answer.lower().strip() in response.lower(),
                "confidence": 100.0,
                "parse_error": False,
                "raw": "",
            }
            atomic_write_json(cache_path, judged)
        else:
            prompt = GRADER_TEMPLATE.format(
                question=question, response=response, correct_answer=answer
            )
            if args.judge_provider == "anthropic":
                # Anthropic 原生 API：/v1/messages，x-api-key header，响应在 content[0].text
                raw = _request_json_with_headers(
                    f"{args.judge_base_url.rstrip('/')}/v1/messages",
                    payload={
                        "model": args.judge_model,
                        "max_tokens": args.judge_max_tokens,
                        "temperature": 0.0,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                    headers={
                        "Content-Type": "application/json",
                        "x-api-key": args.judge_api_key,
                        "anthropic-version": "2023-06-01",
                    },
                    timeout=300.0,
                )
                content_blocks = raw.get("content", [])
                text = str(content_blocks[0].get("text", "")) if content_blocks else ""
            else:
                # OpenAI 兼容：/chat/completions，Authorization Bearer，响应在 choices[0].message.content
                raw = _request_json(
                    f"{args.judge_base_url.rstrip('/')}/chat/completions",
                    payload={
                        "model": args.judge_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "max_tokens": args.judge_max_tokens,
                    },
                    api_key=args.judge_api_key,
                    timeout=300.0,
                )
                text = str(raw["choices"][0]["message"].get("content") or "")
            judged = parse_judgement(text)
            judged["raw_response"] = raw
            atomic_write_json(cache_path, judged)

        relevant = qrels.get(query_id, set())
        retrieved = set(str(item) for item in run.get("retrieved_docids", []))
        recall = len(retrieved & relevant) / len(relevant) if relevant else None
        details.append(
            {
                "query_id": query_id,
                "run_status": run.get("status"),
                "correct": judged.get("correct"),
                "confidence": judged.get("confidence"),
                "parse_error": judged.get("parse_error", True),
                "evidence_recall": recall,
                "extracted_final_answer": judged.get("extracted_final_answer"),
                "reasoning": judged.get("reasoning"),
                "all_citations_visited": bool(
                    (run.get("evidence_validation") or {}).get("all_citations_visited", False)
                ),
                "tool_calls": sum(int(value) for value in run.get("tool_call_counts", {}).values()),
                "turns": int(run.get("metadata", {}).get("turns", 0)),
            }
        )

    valid = [item for item in details if item["correct"] is not None]
    confidence_rows = [
        item for item in valid if isinstance(item.get("confidence"), (int, float))
    ]
    recalls = [item["evidence_recall"] for item in details if item["evidence_recall"] is not None]
    summary = {
        "queries": len(details),
        "completed_runs": sum(item["run_status"] == "completed" for item in details),
        "Completion Rate (%)": 100 * sum(item["run_status"] == "completed" for item in details) / len(details) if details else None,
        "Accuracy (%)": 100 * sum(bool(item["correct"]) for item in valid) / len(valid) if valid else None,
        "Recall (%)": 100 * sum(recalls) / len(recalls) if recalls else None,
        "Calibration Error (%)": (
            100
            * calibration_error(
                [float(item["confidence"]) / 100.0 for item in confidence_rows],
                [bool(item["correct"]) for item in confidence_rows],
            )
            if len(confidence_rows) >= 100
            else None
        ),
        "judge_parse_errors": sum(bool(item["parse_error"]) for item in details),
        "All Citations Visited (%)": 100 * sum(item["all_citations_visited"] for item in details) / len(details) if details else None,
        "Average Tool Calls": sum(item["tool_calls"] for item in details) / len(details) if details else None,
        "Average Turns": sum(item["turns"] for item in details) / len(details) if details else None,
        "per_query_metrics": details,
    }
    atomic_write_json(args.output, summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_query_metrics"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
