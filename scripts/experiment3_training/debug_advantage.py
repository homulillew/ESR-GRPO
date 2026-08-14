"""用一个最小两 rollout 例子检查 ESR 分段优势和动作 mask。"""

from __future__ import annotations


def main() -> None:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("需要训练依赖：pip install -r requirements-train-cu128.txt") from exc

    from esr_grpo.integrations.echo import compute_esr_segmented_advantage

    rewards = torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]])
    response_mask = torch.ones_like(rewards, dtype=torch.long)
    masks = [[1, 1, 0, 1], [1, 1, 1, 1]]
    advantages, returns = compute_esr_segmented_advantage(
        rewards,
        response_mask,
        index=["question-1", "question-1"],
        rollout_id=["rollout-good", "rollout-bad"],
        is_final=[True, True],
        overlong=[False, False],
        esr_response_credit_mask=masks,
        esr_legal_submit=[True, True],
    )
    expected = torch.tensor([[1.0, 1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]])
    if not torch.allclose(advantages, expected, atol=1e-5) or not torch.equal(advantages, returns):
        raise SystemExit(f"advantage smoke check failed: {advantages.tolist()}")
    print({"ok": True, "advantages": advantages.tolist(), "returns": returns.tolist()})


if __name__ == "__main__":
    main()
