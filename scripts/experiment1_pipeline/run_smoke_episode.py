"""无需网络和 GPU 的完整 ESR episode、回溯和掩码冒烟实验。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from esr_grpo.analysis import write_json
from esr_grpo.credit import CreditRouter
from esr_grpo.environment import ESREnvironment
from esr_grpo.models import RetrievedDocument, TokenSpan
from esr_grpo.retrieval import InMemoryRetriever
from esr_grpo.verification import KeywordVerifier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/smoke")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    database = output / "episode.sqlite"
    if database.exists():
        if not args.overwrite:
            raise SystemExit(f"{database} already exists; pass --overwrite")
        database.unlink()

    documents = [
        RetrievedDocument(
            "d1",
            "Northstar Systems sold Orion Analytics in 2019. The buyer later renamed it Atlas Search.",
            "2019 acquisition",
        ),
        RetrievedDocument(
            "d2",
            "Northstar acquired Vector Labs and its Orion Analytics product in 2016. "
            "Vector Labs introduced and developed Orion Analytics.",
            "Product history",
        ),
        RetrievedDocument(
            "d3",
            "Northstar redesigned the Orion Analytics interface after acquiring the product.",
            "Engineering update",
        ),
    ]
    environment = ESREnvironment(
        "Atlas Search 的前身 Orion Analytics 最初由哪家公司开发？",
        InMemoryRetriever.from_documents(documents),
        KeywordVerifier(("Vector Labs",)),
        store_path=database,
    )
    cursor = 0

    def span(length: int = 3) -> list[TokenSpan]:
        nonlocal cursor
        result = [TokenSpan(0, cursor, cursor + length)]
        cursor += length
        return result

    first_search = environment.search("Northstar Orion Atlas", token_spans=span(), turn_id="t1")
    first_page = environment.open_page(
        "d1", search_action_id=first_search["action_id"], token_spans=span(), turn_id="t2"
    )
    environment.update_state(
        "Northstar Systems",
        [{"evidence_id": first_page["evidence_id"], "finding": "只说明 Northstar 在2019年出售产品。"}],
        [first_page["evidence_id"]],
        token_spans=span(5),
        turn_id="t3",
    )
    environment.verify_answer(token_spans=span(), turn_id="t4")

    second_search = environment.search("Vector Labs Orion developed", token_spans=span(), turn_id="t5")
    # 两个 open_page 共享 turn_id，但各自有独立动作和 token 范围。
    page_results = environment.execute_parallel(
        [
            {
                "name": "open_page",
                "arguments": {"docid": "d2", "search_action_id": second_search["action_id"]},
                "token_spans": [vars(item) for item in span()],
            },
            {
                "name": "open_page",
                "arguments": {"docid": "d3", "search_action_id": second_search["action_id"]},
                "token_spans": [vars(item) for item in span()],
            },
        ],
        turn_id="t6",
    )
    useful, irrelevant = page_results
    environment.update_state(
        "Vector Labs",
        [
            {
                "evidence_id": useful["evidence_id"],
                "finding": "明确说明 Vector Labs 最初开发 Orion Analytics。",
            },
            {
                "evidence_id": irrelevant["evidence_id"],
                "finding": "只说明收购后的界面改造，与最初开发者无关。",
            },
        ],
        [first_page["evidence_id"], useful["evidence_id"]],
        token_spans=span(6),
        turn_id="t7",
    )
    environment.verify_answer(token_spans=span(), turn_id="t8")
    environment.submit_answer(token_spans=span(), turn_id="t9")

    router = CreditRouter()
    trace = router.route(environment.store, benchmark_success=True, group_advantage=1.25)
    masks = router.build_token_masks(environment.store, trace.selected_action_ids, {0: cursor})
    write_json(output / "episode.json", environment.snapshot())
    write_json(
        output / "credit.json",
        {
            "eligible": trace.eligible,
            "positive_advantage": trace.positive_advantage,
            "selected_action_ids": trace.selected_action_ids,
            "categories": trace.categories,
            "reasons": trace.reasons,
            "token_mask": masks,
        },
    )
    irrelevant_action = environment.store.get_action(irrelevant["action_id"])
    result = {
        "answer": environment.submitted_answer,
        "evidence_count": len(environment.store.list_evidence()),
        "selected_action_ids": trace.selected_action_ids,
        "credited_tokens": sum(masks[0]),
        "total_policy_tokens": cursor,
        "irrelevant_parallel_open_page_credited": irrelevant_action.action_id in trace.selected_action_ids,
        "output_dir": str(output.resolve()),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
