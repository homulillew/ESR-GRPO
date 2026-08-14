from __future__ import annotations

from esr_grpo.diagnostics import audit_episode_store, audit_rollout_metadata, build_credit_debug_report
from esr_grpo.environment import ESREnvironment
from esr_grpo.models import RetrievedDocument, TokenSpan
from esr_grpo.retrieval import InMemoryRetriever
from esr_grpo.verification import KeywordVerifier


def completed_environment() -> ESREnvironment:
    environment = ESREnvironment(
        "Who developed Orion?",
        InMemoryRetriever.from_documents(
            [RetrievedDocument("d1", "Vector Labs developed Orion Analytics.")]
        ),
        KeywordVerifier(("Vector Labs",)),
        episode_id="diagnostic-episode",
    )
    search = environment.search("Vector Labs Orion", token_spans=[TokenSpan(0, 0, 2)])
    environment.open_page(
        "d1", search_action_id=search["action_id"], token_spans=[TokenSpan(0, 2, 4)]
    )
    environment.update_state(
        "Vector Labs",
        [{"evidence_id": "e1", "finding": "Vector Labs developed Orion Analytics."}],
        ["e1"],
        token_spans=[TokenSpan(0, 4, 6)],
    )
    environment.verify_answer(token_spans=[TokenSpan(0, 6, 8)])
    environment.submit_answer(token_spans=[TokenSpan(0, 8, 10)])
    return environment


def test_valid_episode_passes_protocol_audit() -> None:
    environment = completed_environment()
    report = audit_episode_store(environment.store)
    assert report["valid"] is True
    assert report["errors"] == 0


def test_credit_debug_report_contains_token_masks() -> None:
    environment = completed_environment()
    report = build_credit_debug_report(environment.store)
    assert report["eligible"] is True
    assert report["selected_action_ids"]
    assert report["credited_tokens"] > 0
    assert 0 < report["mask_density"] <= 1


def test_rollout_audit_detects_non_final_reward_and_bad_mask() -> None:
    report = audit_rollout_metadata(
        [
            {
                "rollout_id": "r1",
                "is_final": False,
                "reward": 1,
                "response_length": 2,
                "response_mask": [1, 0],
                "esr_response_credit_mask": [1],
            },
            {
                "rollout_id": "r1",
                "is_final": True,
                "reward": 1,
                "response_length": 2,
                "response_mask": [1, 1],
                "esr_response_credit_mask": [1, 0],
                "esr_legal_submit": True,
            },
        ]
    )
    assert report["valid"] is False
    assert {item["code"] for item in report["issues"]} >= {"CREDIT_MASK_LENGTH", "NON_FINAL_REWARD"}
