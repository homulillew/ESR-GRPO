"""根据最终 TaskState 回溯完整动作，并生成策略 token 掩码。"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Mapping, Sequence

from .models import ActionKind, ActionRecord, TaskState
from .store import EpisodeStore


@dataclass(frozen=True)
class CreditResult:
    eligible: bool
    positive_advantage: float
    selected_action_ids: tuple[str, ...]
    categories: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    reasons: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    message: str = ""


class CreditRouter:
    """实现答案、最终证据和已解决 gaps 三条回溯链。"""

    def structural_trace(self, store: EpisodeStore) -> CreditResult:
        actions = tuple(item for item in store.list_actions() if item.legal)
        states = store.list_states()
        submit_action_id = store.get_metadata("submit_action_id")
        if not actions or not states or not submit_action_id:
            return CreditResult(False, 0.0, (), message="episode has no legal submit")

        by_id = {item.action_id: item for item in actions}
        state_by_version = {item.version: item for item in states}
        final_state = states[-1]
        if final_state.verification_status.value != "supported" or final_state.gaps:
            return CreditResult(False, 0.0, (), message="submitted TaskState is not supported")
        if submit_action_id not in by_id:
            return CreditResult(False, 0.0, (), message="submit action is missing from the ledger")

        categories: dict[str, set[str]] = {
            "answer": set(),
            "evidence": set(),
            "gap_resolution": set(),
            "verification": set(),
            "submit": {submit_action_id},
        }
        reasons: dict[str, set[str]] = {}

        def select(category: str, action_id: str | None, reason: str) -> None:
            if action_id and action_id in by_id:
                categories[category].add(action_id)
                reasons.setdefault(action_id, set()).add(reason)

        final_verify = by_id.get(final_state.created_by_action_id)
        if final_verify and final_verify.kind is ActionKind.VERIFY_ANSWER:
            select("verification", final_verify.action_id, "最终状态通过验证")

        final_update = self._find_final_answer_update(actions, state_by_version, final_state)
        if final_update is not None:
            select("answer", final_update.action_id, "形成最终答案和主要证据集合")

        final_directory = final_state.directory_map()
        final_support = set(final_state.supporting_evidence)
        for evidence_id in final_state.supporting_evidence:
            evidence = store.get_evidence(evidence_id)
            select("evidence", evidence.created_by_action_id, f"首次保存最终证据 {evidence_id}")
            select("evidence", evidence.search_action_id, f"检索到最终证据 {evidence_id}")

            finding_writer = self._find_surviving_finding_writer(
                evidence_id, final_directory[evidence_id], actions, state_by_version
            )
            if finding_writer is not None:
                select("evidence", finding_writer.action_id, f"写入最终 finding：{evidence_id}")
                source_id = dict(finding_writer.metadata.get("finding_inputs", {})).get(evidence_id)
                if source_id in by_id:
                    source_action = by_id[source_id]
                    if source_action.kind is ActionKind.READ_EVIDENCE:
                        select("evidence", source_id, f"重新读取并实际用于 finding：{evidence_id}")
                    elif source_action.kind is ActionKind.OPEN_PAGE:
                        select("evidence", source_id, f"首次页面内容实际用于 finding：{evidence_id}")

            support_writer = self._find_surviving_support_writer(
                evidence_id, actions, state_by_version, final_state.version
            )
            if support_writer is not None:
                select("evidence", support_writer.action_id, f"将 {evidence_id} 纳入最终主要证据")

        # 对每次 needs_revision，寻找后来明确解决 gap 的 verify。只有区间内存在
        # 保留到最终状态的修改时，整段关系才进入信用集合。
        verify_actions = [item for item in actions if item.kind is ActionKind.VERIFY_ANSWER]
        for start in verify_actions:
            created_or_active = set(start.metadata.get("created_gap_ids", ()))
            after = state_by_version.get(start.state_version_after or -1)
            if after is not None:
                created_or_active.update(item.gap_id for item in after.gaps)
            if not created_or_active:
                continue
            resolving = next(
                (
                    item
                    for item in verify_actions
                    if item.sequence_index > start.sequence_index
                    and created_or_active.intersection(item.metadata.get("resolved_gap_ids", ()))
                ),
                None,
            )
            if resolving is None:
                continue
            interval = [
                item
                for item in actions
                if start.sequence_index < item.sequence_index < resolving.sequence_index
            ]
            surviving = [
                item
                for item in interval
                if self._action_survives_final(
                    item, state_by_version, final_state, final_support, store
                )
            ]
            if not surviving:
                continue
            select("gap_resolution", start.action_id, "发现了后来被解决的待查问题")
            select("gap_resolution", resolving.action_id, "确认待查问题减少或清空")
            for item in surviving:
                select("gap_resolution", item.action_id, "动作解决 gap，且修改保留到最终状态")

        ordered_ids = tuple(
            item.action_id
            for item in actions
            if any(item.action_id in values for values in categories.values())
        )
        return CreditResult(
            True,
            0.0,
            ordered_ids,
            {key: tuple(sorted(value, key=lambda aid: by_id[aid].sequence_index)) for key, value in categories.items()},
            {key: tuple(sorted(value)) for key, value in reasons.items()},
            "structural trace built",
        )

    def route(
        self,
        store: EpisodeStore,
        *,
        benchmark_success: bool,
        group_advantage: float,
    ) -> CreditResult:
        trace = self.structural_trace(store)
        positive = max(float(group_advantage), 0.0)
        if not benchmark_success:
            return CreditResult(False, positive, (), message="Benchmark final reward indicates failure")
        if positive <= 0.0:
            return CreditResult(False, 0.0, (), message="rollout has no positive group-relative advantage")
        if not trace.eligible:
            return CreditResult(False, positive, (), message=trace.message)
        return CreditResult(
            True,
            positive,
            trace.selected_action_ids,
            trace.categories,
            trace.reasons,
            "legal successful rollout with positive group-relative advantage",
        )

    def build_token_masks(
        self,
        store: EpisodeStore,
        selected_action_ids: Sequence[str],
        segment_lengths: Mapping[int, int],
        policy_masks: Mapping[int, Sequence[int | bool]] | None = None,
    ) -> dict[int, list[int]]:
        """完整动作整体进入或退出掩码，并与策略 token 掩码取交集。"""

        masks = {int(segment): [0] * int(length) for segment, length in segment_lengths.items()}
        selected = set(selected_action_ids)
        for action in store.list_actions():
            if action.action_id not in selected:
                continue
            for span in action.token_spans:
                if span.segment_index not in masks:
                    raise ValueError(f"missing segment length for segment {span.segment_index}")
                if span.end > len(masks[span.segment_index]):
                    raise ValueError(f"token span exceeds segment boundary: {span}")
                masks[span.segment_index][span.start : span.end] = [1] * (span.end - span.start)
        if policy_masks is not None:
            for segment, mask in masks.items():
                policy = list(policy_masks.get(segment, ()))
                if len(policy) != len(mask):
                    raise ValueError(f"policy mask length mismatch in segment {segment}")
                masks[segment] = [int(bool(left) and bool(right)) for left, right in zip(mask, policy)]
        return masks

    @staticmethod
    def masked_advantages(result: CreditResult, masks: Mapping[int, Sequence[int]]) -> dict[int, list[float]]:
        value = result.positive_advantage if result.eligible else 0.0
        return {segment: [value * int(bit) for bit in mask] for segment, mask in masks.items()}

    @staticmethod
    def _find_final_answer_update(
        actions: Sequence[ActionRecord],
        states: Mapping[int, TaskState],
        final_state: TaskState,
    ) -> ActionRecord | None:
        candidates: list[ActionRecord] = []
        for action in actions:
            if action.kind is not ActionKind.UPDATE_STATE or action.state_version_after is None:
                continue
            state = states[action.state_version_after]
            if (
                state.answer == final_state.answer
                and state.supporting_evidence == final_state.supporting_evidence
                and state.directory_map() == final_state.directory_map()
            ):
                candidates.append(action)
        return candidates[-1] if candidates else None

    @staticmethod
    def _find_surviving_finding_writer(
        evidence_id: str,
        final_finding: str,
        actions: Sequence[ActionRecord],
        states: Mapping[int, TaskState],
    ) -> ActionRecord | None:
        candidates: list[ActionRecord] = []
        for action in actions:
            if action.kind is not ActionKind.UPDATE_STATE or action.state_version_after is None:
                continue
            after = states[action.state_version_after]
            before = states.get(action.state_version_before or -1)
            before_value = before.directory_map().get(evidence_id) if before else None
            after_value = after.directory_map().get(evidence_id)
            if after_value == final_finding and before_value != final_finding:
                candidates.append(action)
        return candidates[-1] if candidates else None

    @staticmethod
    def _find_surviving_support_writer(
        evidence_id: str,
        actions: Sequence[ActionRecord],
        states: Mapping[int, TaskState],
        final_version: int,
    ) -> ActionRecord | None:
        candidates: list[ActionRecord] = []
        for action in actions:
            if action.kind is not ActionKind.UPDATE_STATE or action.state_version_after is None:
                continue
            after = states[action.state_version_after]
            before = states.get(action.state_version_before or -1)
            was_present = before is not None and evidence_id in before.supporting_evidence
            if evidence_id in after.supporting_evidence and not was_present and after.version <= final_version:
                candidates.append(action)
        return candidates[-1] if candidates else None

    @staticmethod
    def _action_survives_final(
        action: ActionRecord,
        states: Mapping[int, TaskState],
        final_state: TaskState,
        final_support: set[str],
        store: EpisodeStore,
    ) -> bool:
        if action.kind is ActionKind.UPDATE_STATE and action.state_version_after is not None:
            state = states[action.state_version_after]
            answer_survives = bool(action.metadata.get("answer_changed")) and state.answer == final_state.answer
            support_survives = bool(action.metadata.get("support_changed")) and bool(
                set(state.supporting_evidence).intersection(final_support)
            )
            finding_survives = any(
                final_state.directory_map().get(evidence_id) == state.directory_map().get(evidence_id)
                for evidence_id in action.metadata.get("changed_finding_ids", ())
            )
            return answer_survives or support_survives or finding_survives
        if action.kind in {ActionKind.OPEN_PAGE, ActionKind.READ_EVIDENCE}:
            ids = set(action.created_evidence_ids).union(action.referenced_evidence_ids)
            return bool(ids.intersection(final_support))
        if action.kind is ActionKind.SEARCH:
            return any(
                evidence.search_action_id == action.action_id and evidence.evidence_id in final_support
                for evidence in store.list_evidence()
            )
        return False


def compute_group_advantages(
    rewards: Sequence[float],
    group_ids: Sequence[str | int],
    *,
    normalize_by_std: bool = True,
    epsilon: float = 1e-6,
) -> list[float]:
    """按任务分组计算 GRPO 优势；组内只有一个样本时优势为零。"""

    if len(rewards) != len(group_ids):
        raise ValueError("rewards and group_ids must have the same length")
    groups: dict[str | int, list[int]] = {}
    for index, group_id in enumerate(group_ids):
        groups.setdefault(group_id, []).append(index)
    advantages = [0.0] * len(rewards)
    for indices in groups.values():
        if len(indices) <= 1:
            continue
        values = [float(rewards[index]) for index in indices]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        std = sqrt(variance)
        for index in indices:
            centered = float(rewards[index]) - mean
            advantages[index] = centered / (std + epsilon) if normalize_by_std and std > epsilon else centered
    return advantages
