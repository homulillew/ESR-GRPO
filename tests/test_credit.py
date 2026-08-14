from __future__ import annotations

from esr_grpo.credit import CreditRouter, compute_group_advantages
from esr_grpo.environment import ESREnvironment
from esr_grpo.models import RetrievedDocument, TokenSpan
from esr_grpo.retrieval import InMemoryRetriever
from esr_grpo.verification import KeywordVerifier


def build_trace() -> tuple[ESREnvironment, str]:
    retriever = InMemoryRetriever.from_documents(
        [
            RetrievedDocument("useful", "Vector Labs developed Orion."),
            RetrievedDocument("irrelevant", "Orion received a new interface after acquisition."),
        ]
    )
    environment = ESREnvironment("Who developed Orion?", retriever, KeywordVerifier(("Vector Labs",)))
    search = environment.search("Vector Labs Orion interface", token_spans=[TokenSpan(0, 0, 2)])
    useful = environment.open_page(
        "useful", search_action_id=search["action_id"], token_spans=[TokenSpan(0, 2, 4)]
    )
    irrelevant = environment.open_page(
        "irrelevant", search_action_id=search["action_id"], token_spans=[TokenSpan(0, 4, 6)]
    )
    environment.update_state(
        "Vector Labs",
        [
            {"evidence_id": useful["evidence_id"], "finding": "Names the original developer."},
            {"evidence_id": irrelevant["evidence_id"], "finding": "Only covers a later redesign."},
        ],
        [useful["evidence_id"]],
        token_spans=[TokenSpan(0, 6, 10)],
    )
    environment.verify_answer(token_spans=[TokenSpan(0, 10, 12)])
    environment.submit_answer(token_spans=[TokenSpan(0, 12, 14)])
    return environment, irrelevant["action_id"]


def test_credit_excludes_irrelevant_parallel_page() -> None:
    environment, irrelevant_action = build_trace()
    result = CreditRouter().route(environment.store, benchmark_success=True, group_advantage=1.0)
    assert result.eligible
    assert irrelevant_action not in result.selected_action_ids
    mask = CreditRouter().build_token_masks(environment.store, result.selected_action_ids, {0: 14})
    assert mask[0][4:6] == [0, 0]
    assert mask[0][2:4] == [1, 1]


def test_failure_and_nonpositive_advantage_have_zero_credit() -> None:
    environment, _ = build_trace()
    router = CreditRouter()
    assert router.route(environment.store, benchmark_success=False, group_advantage=1).selected_action_ids == ()
    assert router.route(environment.store, benchmark_success=True, group_advantage=-1).selected_action_ids == ()


def test_group_advantages() -> None:
    values = compute_group_advantages([1, 0, 1, 0], ["q1", "q1", "q2", "q2"])
    assert values[0] > 0 and values[1] < 0
    assert values[2] > 0 and values[3] < 0
    assert compute_group_advantages([1], ["q"])[0] == 0
