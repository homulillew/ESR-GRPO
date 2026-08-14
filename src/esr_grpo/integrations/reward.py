"""使用合法 submit_answer 中的答案调用 ECHO BrowseComp Judge。"""

from __future__ import annotations

import json
from typing import Any


async def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    **kwargs: Any,
):
    answer = (extra_info or {}).get("esr_submitted_answer")
    if not answer:
        return {"score": 0.0, "acc": 0.0, "judge_correct": 0.0, "illegal_submit": 1.0}
    synthetic = "<tool_call>" + json.dumps(
        {"name": "finish", "arguments": {"answer": str(answer)}}, ensure_ascii=False
    ) + "</tool_call>"
    from verl.utils.reward_score.bc_p_llm_judge import compute_score as echo_compute_score

    result = await echo_compute_score(
        data_source=data_source,
        solution_str=synthetic,
        ground_truth=ground_truth,
        extra_info=extra_info,
        **kwargs,
    )
    result["illegal_submit"] = 0.0
    return result
