"""评测官方 run JSON：Accuracy、Evidence Recall 和修正后的 Calibration Error。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from esr_grpo.browsecomp import calibration_error, evidence_recall, load_examples, read_qrels
from esr_grpo.judge import ExactMatchJudge, OpenAICompatibleJudge


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--judge-base-url")
    parser.add_argument("--judge-model")
    parser.add_argument("--exact-match-smoke", action="store_true")
    args = parser.parse_args()

    examples = {item.query_id: item for item in load_examples(args.dataset)}
    qrels = read_qrels(args.qrels)
    if args.exact_match_smoke:
        judge = ExactMatchJudge()
    else:
        if not args.judge_base_url or not args.judge_model:
            raise SystemExit("Judge endpoint and model are required unless --exact-match-smoke is used")
        judge = OpenAICompatibleJudge(args.judge_base_url, args.judge_model)

    details: list[dict] = []
    for path in sorted(Path(args.runs).glob("*.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        query_id = str(run["query_id"])
        if query_id not in examples:
            continue
        result_rows = run.get("result", [])
        response = str(result_rows[-1].get("output", "")) if result_rows else ""
        judged = judge.judge(examples[query_id].question, response, examples[query_id].answer)
        recall = evidence_recall(run.get("retrieved_docids", []), qrels.get(query_id, set()))
        details.append(
            {
                "query_id": query_id,
                "correct": judged.correct,
                "confidence": judged.confidence,
                "evidence_recall": recall,
                "parse_error": judged.parse_error,
                "rationale": judged.rationale,
            }
        )
    confidences = [item["confidence"] for item in details]
    correctness = [item["correct"] for item in details]
    recalls = [item["evidence_recall"] for item in details if item["evidence_recall"] == item["evidence_recall"]]
    summary = {
        "queries": len(details),
        "Accuracy (%)": 100 * sum(correctness) / len(details) if details else 0.0,
        "Recall (%)": 100 * sum(recalls) / len(recalls) if recalls else None,
        "Calibration Error (%)": 100 * calibration_error(confidences, correctness) if details else None,
        "judge_parse_errors": sum(item["parse_error"] for item in details),
        "per_query_metrics": details,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "per_query_metrics"}, indent=2))


if __name__ == "__main__":
    main()
