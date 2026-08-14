"""面向三个实验阶段的确定性账本和 rollout 诊断。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

from .credit import CreditRouter
from .models import ActionKind, VerificationStatus
from .store import EpisodeStore


@dataclass(frozen=True)
class DiagnosticIssue:
    severity: str
    code: str
    message: str
    action_id: str | None = None
    state_version: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PLACEHOLDERS = {"todo", "tbd", "n/a", "unknown", "待补充", "占位", "不知道", "无意义"}


def audit_episode_store(store: EpisodeStore) -> dict[str, Any]:
    evidence = store.list_evidence()
    states = store.list_states()
    actions = store.list_actions()
    by_action = {item.action_id: item for item in actions}
    issues: list[DiagnosticIssue] = []

    def issue(
        severity: str,
        code: str,
        message: str,
        *,
        action_id: str | None = None,
        state_version: int | None = None,
    ) -> None:
        issues.append(DiagnosticIssue(severity, code, message, action_id, state_version))

    for expected, action in enumerate(actions):
        if action.sequence_index != expected:
            issue("error", "ACTION_SEQUENCE", f"expected sequence {expected}, got {action.sequence_index}", action_id=action.action_id)
        for span in action.token_spans:
            if span.end <= span.start:
                issue("error", "TOKEN_SPAN", "empty or reversed token span", action_id=action.action_id)

    for item in evidence:
        creator = by_action.get(item.created_by_action_id)
        if creator is None or creator.kind is not ActionKind.OPEN_PAGE:
            issue("error", "EVIDENCE_CREATOR", f"{item.evidence_id} has invalid creator", action_id=item.created_by_action_id)
        elif item.evidence_id not in creator.created_evidence_ids:
            issue("error", "EVIDENCE_CREATOR_LINK", f"creator does not list {item.evidence_id}", action_id=creator.action_id)
        if not item.content.strip():
            issue("error", "EMPTY_EVIDENCE", f"{item.evidence_id} has empty content")

    previous = None
    for expected_version, state in enumerate(states, start=1):
        if state.version != expected_version:
            issue("error", "STATE_VERSION", f"expected version {expected_version}, got {state.version}", state_version=state.version)
        expected_parent = previous.version if previous else None
        if state.parent_version != expected_parent:
            issue("error", "STATE_PARENT", f"expected parent {expected_parent}, got {state.parent_version}", state_version=state.version)
        creator = by_action.get(state.created_by_action_id)
        if creator is None:
            issue("error", "STATE_CREATOR", "state creator action is missing", action_id=state.created_by_action_id, state_version=state.version)
            previous = state
            continue
        if creator.state_version_after != state.version:
            issue("error", "STATE_ACTION_LINK", "creator does not point to state version", action_id=creator.action_id, state_version=state.version)

        available_ids = {
            item.evidence_id
            for item in evidence
            if item.created_by_action_id in by_action
            and by_action[item.created_by_action_id].sequence_index < creator.sequence_index
        }
        directory_ids = set(state.directory_map())
        if directory_ids != available_ids:
            issue(
                "error",
                "DIRECTORY_COVERAGE",
                f"missing={sorted(available_ids-directory_ids)}, extra={sorted(directory_ids-available_ids)}",
                action_id=creator.action_id,
                state_version=state.version,
            )
        if not set(state.supporting_evidence).issubset(directory_ids):
            issue("error", "SUPPORT_DIRECTORY", "supporting Evidence is outside directory", state_version=state.version)
        for finding in state.evidence_directory:
            normalized = finding.finding.strip().lower()
            if not normalized or normalized in PLACEHOLDERS or any(token in normalized for token in ("todo", "tbd")):
                issue("warning", "PLACEHOLDER_FINDING", f"placeholder finding for {finding.evidence_id}", state_version=state.version)

        if creator.kind is ActionKind.UPDATE_STATE:
            if state.verification_status is not VerificationStatus.UNVERIFIED:
                issue("error", "UPDATE_VERIFICATION", "update_state must reset verification", action_id=creator.action_id, state_version=state.version)
            if previous is not None and state.gaps != previous.gaps:
                issue("error", "UPDATE_CHANGED_GAPS", "update_state changed gaps", action_id=creator.action_id, state_version=state.version)
        elif creator.kind is ActionKind.VERIFY_ANSWER:
            if previous is None:
                issue("error", "VERIFY_WITHOUT_STATE", "verify_answer created the first state", action_id=creator.action_id)
            elif (
                state.answer != previous.answer
                or state.evidence_directory != previous.evidence_directory
                or state.supporting_evidence != previous.supporting_evidence
            ):
                issue("error", "VERIFY_CHANGED_CONTENT", "verify_answer changed answer or Evidence fields", action_id=creator.action_id, state_version=state.version)
        else:
            issue("error", "INVALID_STATE_ACTION", f"{creator.kind.value} created TaskState", action_id=creator.action_id)
        previous = state

    submit_id = store.get_metadata("submit_action_id")
    if submit_id is not None:
        submit = by_action.get(str(submit_id))
        final = states[-1] if states else None
        if submit is None or submit.kind is not ActionKind.SUBMIT_ANSWER or not submit.legal:
            issue("error", "SUBMIT_ACTION", "submit metadata does not point to a legal submit")
        if final is None or final.verification_status is not VerificationStatus.SUPPORTED or final.gaps:
            issue("error", "SUBMIT_STATE", "submitted final state is not supported with empty gaps")
        if final is not None and store.get_metadata("submitted_answer") != final.answer:
            issue("error", "SUBMIT_ANSWER_MISMATCH", "submitted answer differs from TaskState.answer")

    errors = sum(item.severity == "error" for item in issues)
    warnings = sum(item.severity == "warning" for item in issues)
    return {
        "episode_id": store.get_metadata("episode_id"),
        "valid": errors == 0,
        "errors": errors,
        "warnings": warnings,
        "counts": {"evidence": len(evidence), "states": len(states), "actions": len(actions)},
        "issues": [item.to_dict() for item in issues],
    }


def build_credit_debug_report(store: EpisodeStore) -> dict[str, Any]:
    router = CreditRouter()
    trace = router.structural_trace(store)
    actions = store.list_actions()
    selected = set(trace.selected_action_ids)
    segment_lengths: dict[int, int] = {}
    for action in actions:
        for span in action.token_spans:
            segment_lengths[span.segment_index] = max(segment_lengths.get(span.segment_index, 0), span.end)
    masks = router.build_token_masks(store, trace.selected_action_ids, segment_lengths) if segment_lengths else {}
    total_tokens = sum(segment_lengths.values())
    credited_tokens = sum(sum(mask) for mask in masks.values())
    return {
        "episode_id": store.get_metadata("episode_id"),
        "eligible": trace.eligible,
        "message": trace.message,
        "selected_action_ids": list(trace.selected_action_ids),
        "categories": {key: list(value) for key, value in trace.categories.items()},
        "reasons": {key: list(value) for key, value in trace.reasons.items()},
        "segment_lengths": segment_lengths,
        "token_masks": masks,
        "credited_tokens": credited_tokens,
        "observed_policy_tokens": total_tokens,
        "mask_density": credited_tokens / total_tokens if total_tokens else 0.0,
        "selected_actions_without_spans": [
            item.action_id for item in actions if item.action_id in selected and not item.token_spans
        ],
        "timeline": [
            {
                "action_id": item.action_id,
                "sequence_index": item.sequence_index,
                "turn_id": item.turn_id,
                "kind": item.kind.value,
                "legal": item.legal,
                "selected": item.action_id in selected,
                "state_before": item.state_version_before,
                "state_after": item.state_version_after,
                "evidence_created": list(item.created_evidence_ids),
                "evidence_referenced": list(item.referenced_evidence_ids),
                "active_gaps": list(item.active_gap_ids),
                "token_spans": [
                    {"segment_index": span.segment_index, "start": span.start, "end": span.end}
                    for span in item.token_spans
                ],
                "reasons": list(trace.reasons.get(item.action_id, ())),
            }
            for item in actions
        ],
    }


def audit_rollout_metadata(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    densities: list[float] = []
    final_counts: dict[str, int] = {}
    rewards_by_rollout: dict[str, float] = {}
    for row_index, row in enumerate(records):
        rollout_id = str(row.get("rollout_id", f"row-{row_index}"))
        is_final = bool(row.get("is_final", False))
        final_counts[rollout_id] = final_counts.get(rollout_id, 0) + int(is_final)
        response_mask = list(row.get("response_mask", []))
        credit_mask = list(row.get("esr_response_credit_mask", []))
        response_length = int(row.get("response_length", len(response_mask) or len(credit_mask)))
        if len(credit_mask) != response_length:
            issues.append({"row": row_index, "code": "CREDIT_MASK_LENGTH", "rollout_id": rollout_id})
        if response_mask and len(response_mask) != response_length:
            issues.append({"row": row_index, "code": "RESPONSE_MASK_LENGTH", "rollout_id": rollout_id})
        if response_mask and len(credit_mask) == len(response_mask):
            if any(bool(credit) and not bool(policy) for credit, policy in zip(credit_mask, response_mask)):
                issues.append({"row": row_index, "code": "CREDIT_ON_NON_POLICY_TOKEN", "rollout_id": rollout_id})
        denominator = sum(bool(item) for item in response_mask) if response_mask else response_length
        densities.append(sum(bool(item) for item in credit_mask) / denominator if denominator else 0.0)
        reward = float(row.get("reward", row.get("score", 0.0)) or 0.0)
        if not is_final and reward != 0.0:
            issues.append({"row": row_index, "code": "NON_FINAL_REWARD", "rollout_id": rollout_id})
        if is_final:
            rewards_by_rollout[rollout_id] = reward
        if bool(row.get("esr_legal_submit")) and not any(credit_mask):
            issues.append({"row": row_index, "code": "LEGAL_SUBMIT_EMPTY_MASK", "rollout_id": rollout_id})
        if bool(row.get("overlong")) and reward != 0.0:
            issues.append({"row": row_index, "code": "OVERLONG_REWARD", "rollout_id": rollout_id})
    for rollout_id, count in final_counts.items():
        if count != 1:
            issues.append({"code": "FINAL_SEGMENT_COUNT", "rollout_id": rollout_id, "count": count})
    return {
        "rows": len(records),
        "rollouts": len(final_counts),
        "valid": not issues,
        "issues": issues,
        "avg_mask_density": mean(densities) if densities else 0.0,
        "min_mask_density": min(densities) if densities else 0.0,
        "max_mask_density": max(densities) if densities else 0.0,
        "final_rewards": rewards_by_rollout,
    }


def load_json_or_jsonl(path: str) -> list[Mapping[str, Any]]:
    import json
    from pathlib import Path

    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        value = json.loads(text)
        if not isinstance(value, list):
            raise ValueError("JSON root must be a list")
        return value
    return [json.loads(line) for line in text.splitlines() if line.strip()]
