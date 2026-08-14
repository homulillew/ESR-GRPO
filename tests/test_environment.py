from __future__ import annotations

import sqlite3

import pytest

from esr_grpo.environment import ESREnvironment, IllegalActionError
from esr_grpo.models import RetrievedDocument, TokenSpan
from esr_grpo.retrieval import InMemoryRetriever
from esr_grpo.verification import KeywordVerifier


def make_environment() -> ESREnvironment:
    retriever = InMemoryRetriever.from_documents(
        [
            RetrievedDocument("d1", "Vector Labs developed Orion Analytics."),
            RetrievedDocument("d2", "Northstar later redesigned Orion Analytics."),
        ]
    )
    return ESREnvironment("Who developed Orion?", retriever, KeywordVerifier(("Vector Labs",)))


def acquire(environment: ESREnvironment, docid: str = "d1") -> dict:
    search = environment.search("Vector Labs Orion Northstar", token_spans=[TokenSpan(0, 0, 2)])
    return environment.open_page(
        docid, search_action_id=search["action_id"], token_spans=[TokenSpan(0, 2, 4)]
    )


def test_evidence_is_immutable_at_database_layer() -> None:
    environment = make_environment()
    page = acquire(environment)
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        environment.store._conn.execute(  # noqa: SLF001 - invariant test
            "UPDATE evidence SET payload_json = '{}' WHERE evidence_id = ?", (page["evidence_id"],)
        )


def test_update_requires_directory_coverage() -> None:
    environment = make_environment()
    acquire(environment)
    with pytest.raises(IllegalActionError, match="coverage"):
        environment.update_state("Vector Labs", [], ["e1"])
    assert environment.store.list_actions()[-1].legal is False


def test_changed_finding_requires_original_evidence_to_be_visible() -> None:
    environment = make_environment()
    page = acquire(environment)
    environment.update_state(
        "Vector Labs",
        [{"evidence_id": page["evidence_id"], "finding": "Vector Labs developed it."}],
        [page["evidence_id"]],
    )
    with pytest.raises(IllegalActionError, match="visible original"):
        environment.update_state(
            "Vector Labs",
            [{"evidence_id": page["evidence_id"], "finding": "Changed finding."}],
            [page["evidence_id"]],
        )
    environment.read_evidence(page["evidence_id"])
    state = environment.update_state(
        "Vector Labs",
        [{"evidence_id": page["evidence_id"], "finding": "Changed finding."}],
        [page["evidence_id"]],
    )
    assert state["task_state"]["evidence_directory"][0]["finding"] == "Changed finding."


def test_gaps_are_only_changed_by_verification() -> None:
    environment = make_environment()
    page = acquire(environment, "d2")
    environment.update_state(
        "Northstar", [{"evidence_id": page["evidence_id"], "finding": "Northstar redesigned it."}], ["e1"]
    )
    verified = environment.verify_answer()
    assert verified["verification_status"] == "needs_revision"
    gap_ids = [item["gap_id"] for item in verified["gaps"]]
    environment.read_evidence("e1")
    updated = environment.update_state(
        "Northstar", [{"evidence_id": "e1", "finding": "Still only a redesign claim."}], ["e1"]
    )
    assert [item["gap_id"] for item in updated["task_state"]["gaps"]] == gap_ids


def test_submit_is_gated_by_supported_latest_state() -> None:
    environment = make_environment()
    page = acquire(environment)
    environment.update_state(
        "Vector Labs", [{"evidence_id": "e1", "finding": "Names Vector Labs."}], ["e1"]
    )
    with pytest.raises(IllegalActionError, match="verify_answer"):
        environment.submit_answer()
    environment.verify_answer()
    assert environment.submit_answer()["answer"] == "Vector Labs"


def test_new_evidence_invalidates_previous_verification() -> None:
    environment = make_environment()
    acquire(environment)
    environment.update_state(
        "Vector Labs", [{"evidence_id": "e1", "finding": "Names Vector Labs."}], ["e1"]
    )
    environment.verify_answer()
    acquire(environment, "d2")
    with pytest.raises(IllegalActionError, match="latest Evidence"):
        environment.submit_answer()


def test_parallel_calls_have_independent_action_ids_and_spans() -> None:
    environment = make_environment()
    search = environment.search("Vector Labs Northstar Orion")
    results = environment.execute_parallel(
        [
            {
                "name": "open_page",
                "arguments": {"docid": "d1", "search_action_id": search["action_id"]},
                "token_spans": [{"segment_index": 0, "start": 0, "end": 2}],
            },
            {
                "name": "open_page",
                "arguments": {"docid": "d2", "search_action_id": search["action_id"]},
                "token_spans": [{"segment_index": 0, "start": 2, "end": 4}],
            },
        ],
        turn_id="t1",
    )
    assert results[0]["action_id"] != results[1]["action_id"]
    actions = environment.store.list_actions()[-2:]
    assert {item.turn_id for item in actions} == {"t1"}
    assert actions[0].token_spans != actions[1].token_spans
