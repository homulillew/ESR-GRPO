"""ECHO/verl 的 ESR-GRPO 优势函数和 rollout 元数据协议。"""

from __future__ import annotations

from typing import Any, Mapping, Sequence
import re

from ..credit import CreditRouter
from ..store import EpisodeStore


def build_segment_metadata(
    store: EpisodeStore,
    *,
    segment_lengths: Mapping[int, int],
    policy_masks: Mapping[int, Sequence[int | bool]] | None = None,
) -> dict[str, Any]:
    router = CreditRouter()
    trace = router.structural_trace(store)
    masks = router.build_token_masks(store, trace.selected_action_ids, segment_lengths, policy_masks)
    return {
        "esr_legal_submit": bool(trace.eligible),
        "esr_selected_action_ids": list(trace.selected_action_ids),
        "esr_credit_categories": {key: list(value) for key, value in trace.categories.items()},
        "esr_credit_reasons": {key: list(value) for key, value in trace.reasons.items()},
        "esr_response_credit_masks": masks,
    }


def extract_hermes_action_spans(
    tokenizer: Any,
    response_ids: Sequence[int],
    *,
    segment_index: int,
    segment_offset: int,
) -> list[dict[str, int]]:
    """从 Hermes `<tool_call>` 块取得每个并行调用的独立 token 范围。"""

    text = tokenizer.decode(list(response_ids), skip_special_tokens=False)
    matches = list(re.finditer(r"<tool_call>.*?</tool_call>", text, flags=re.DOTALL))
    if not matches:
        return []
    spans: list[dict[str, int]] = []
    try:
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = encoded["offset_mapping"]
        for match in matches:
            indices = [
                index
                for index, (start, end) in enumerate(offsets)
                if end > match.start() and start < match.end()
            ]
            if not indices:
                continue
            spans.append(
                {
                    "segment_index": segment_index,
                    "start": segment_offset + indices[0],
                    "end": segment_offset + indices[-1] + 1,
                }
            )
    except Exception:
        for match in matches:
            start = len(tokenizer.encode(text[: match.start()], add_special_tokens=False))
            end = len(tokenizer.encode(text[: match.end()], add_special_tokens=False))
            spans.append(
                {
                    "segment_index": segment_index,
                    "start": segment_offset + start,
                    "end": segment_offset + end,
                }
            )
    return spans


def finalize_echo_agent_data(agent_data: Any) -> dict[str, Any]:
    environment = getattr(agent_data, "_esr_environment", None)
    if environment is None:
        return {}
    segment_lengths = {
        index: len(trajectory.response_ids) for index, trajectory in enumerate(agent_data.trajectory_outputs)
    }
    policy_masks = {
        index: trajectory.response_mask for index, trajectory in enumerate(agent_data.trajectory_outputs)
    }
    metadata = build_segment_metadata(
        environment.store,
        segment_lengths=segment_lengths,
        policy_masks=policy_masks,
    )
    metadata["esr_submitted_answer"] = environment.submitted_answer
    metadata["esr_store_path"] = environment.store.path
    return metadata


def compute_esr_segmented_advantage(
    token_level_rewards,
    response_mask,
    index,
    rollout_id,
    is_final,
    overlong,
    esr_response_credit_mask,
    esr_legal_submit,
    *,
    success_threshold: float = 0.5,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    **_: Any,
):
    """在逻辑 rollout 层计算 GRPO，再将正优势写入 ESR 完整动作掩码。"""

    import numpy as np
    import torch

    with torch.no_grad():
        scores = token_level_rewards.float().sum(dim=-1)
        rollout_values = np.asarray(rollout_id, dtype=object).reshape(-1)
        group_values = np.asarray(index, dtype=object).reshape(-1)
        final_values = np.asarray(is_final).reshape(-1).astype(bool)
        overlong_values = np.asarray(overlong).reshape(-1).astype(bool)
        legal_values = np.asarray(esr_legal_submit).reshape(-1).astype(bool)
        credit_values = np.asarray(esr_response_credit_mask, dtype=object).reshape(-1)

        rollout_reward: dict[Any, float] = {}
        rollout_legal: dict[Any, bool] = {}
        rollout_overlong: set[Any] = set()
        groups: dict[Any, set[Any]] = {}
        for row, rollout in enumerate(rollout_values):
            group = group_values[row]
            groups.setdefault(group, set()).add(rollout)
            rollout_legal[rollout] = rollout_legal.get(rollout, False) or bool(legal_values[row])
            if overlong_values[row]:
                rollout_overlong.add(rollout)
            if final_values[row] and not overlong_values[row]:
                rollout_reward[rollout] = float(scores[row].item())

        rollout_advantage: dict[Any, float] = {}
        for rollouts in groups.values():
            valid = [item for item in rollouts if item in rollout_reward and item not in rollout_overlong]
            if len(valid) <= 1:
                rollout_advantage.update({item: 0.0 for item in rollouts})
                continue
            rewards = torch.tensor([rollout_reward[item] for item in valid], dtype=torch.float32)
            mean = float(rewards.mean().item())
            std = float(rewards.std(unbiased=False).item())
            for item in rollouts:
                if item not in valid:
                    rollout_advantage[item] = 0.0
                    continue
                centered = rollout_reward[item] - mean
                rollout_advantage[item] = (
                    centered / (std + epsilon)
                    if norm_adv_by_std_in_grpo and std > epsilon
                    else centered
                )

        advantages = torch.zeros_like(token_level_rewards, dtype=torch.float32)
        response_length = response_mask.shape[1]
        for row, rollout in enumerate(rollout_values):
            successful = (
                rollout not in rollout_overlong
                and rollout_legal.get(rollout, False)
                and rollout_reward.get(rollout, float("-inf")) >= success_threshold
            )
            advantage = max(rollout_advantage.get(rollout, 0.0), 0.0)
            if not successful or advantage <= 0.0:
                continue
            raw = credit_values[row]
            mask_values = raw.tolist() if isinstance(raw, np.ndarray) else list(raw or [])
            if len(mask_values) < response_length:
                mask_values.extend([0] * (response_length - len(mask_values)))
            mask = torch.tensor(mask_values[:response_length], device=response_mask.device, dtype=torch.bool)
            mask &= response_mask[row].bool()
            advantages[row] = advantage * mask.float()
        advantages = advantages.to(token_level_rewards.dtype)
        return advantages, advantages


def register_with_echo(name: str = "esr_grpo") -> Any:
    from verl.trainer.ppo.core_algos import register_adv_est

    return register_adv_est(name)(compute_esr_segmented_advantage)
