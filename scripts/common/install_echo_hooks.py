"""为本地 ECHO 快照安装最小训练器钩子。

脚本默认只检查。必须显式传入 --apply 才会修改两个 ECHO 文件，并在同目录创建
.esr-grpo.bak 备份。它不修改 ToolAgentLoop；精确动作元数据必须由 ESR rollout
适配器写入 AgentLoopOutput.extra_fields。
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


CORE_HOOK = """

# ESR-GRPO optional external advantage estimator.
try:
    from esr_grpo.integrations.echo import compute_esr_segmented_advantage
except ImportError:
    compute_esr_segmented_advantage = None
if compute_esr_segmented_advantage is not None:
    register_adv_est("esr_grpo")(compute_esr_segmented_advantage)
"""


def patch_ray_trainer(text: str) -> str:
    old = """        if adv_estimator == \"supo\" or (hasattr(adv_estimator, 'value') and getattr(adv_estimator, 'value', None) == \"supo\"):
"""
    new = """        _adv_name = getattr(adv_estimator, \"value\", adv_estimator)
        if _adv_name in {\"supo\", \"esr_grpo\"}:
"""
    if old not in text and new not in text:
        raise RuntimeError("ray_trainer advantage marker changed")
    text = text.replace(old, new, 1)
    marker = '            adv_kwargs["norm_adv_by_std_in_grpo"] = norm_adv_by_std_in_grpo\n'
    addition = """            if _adv_name == "esr_grpo":
                for _key in ("esr_response_credit_mask", "esr_legal_submit"):
                    if _key in data.non_tensor_batch:
                        adv_kwargs[_key] = data.non_tensor_batch[_key]
"""
    if addition not in text:
        if marker not in text:
            raise RuntimeError("ray_trainer metadata marker changed")
        text = text.replace(marker, marker + addition, 1)

    old_flag = """        self._is_supo = (adv_estimator == "supo" or
                         (hasattr(adv_estimator, 'value') and adv_estimator.value == "supo"))
"""
    new_flag = """        _adv_name = getattr(adv_estimator, "value", adv_estimator)
        self._is_supo = _adv_name in {"supo", "esr_grpo"}
"""
    if old_flag not in text and new_flag not in text:
        raise RuntimeError("ray_trainer split-rollout marker changed")
    return text.replace(old_flag, new_flag, 1)


def patch_tool_agent_loop(text: str) -> str:
    extract_marker = "        _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids)\n"
    extract_hook = """        _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids)
        # ESR-GRPO: preserve one exact policy-token range per Hermes tool call.
        try:
            from esr_grpo.integrations.echo import extract_hermes_action_spans
            _esr_segment_index = len(agent_data.trajectory_outputs)
            _esr_segment_offset = (
                len(agent_data.accumulated_response_ids) - len(agent_data.response_ids)
                if self.enable_summarization
                else len(agent_data.response_mask) - len(agent_data.response_ids)
            )
            agent_data._esr_pending_call_spans = extract_hermes_action_spans(
                self.tokenizer,
                agent_data.response_ids,
                segment_index=_esr_segment_index,
                segment_offset=_esr_segment_offset,
            )
        except ImportError:
            agent_data._esr_pending_call_spans = []
"""
    if extract_hook not in text:
        if extract_marker not in text:
            raise RuntimeError("tool_agent_loop extraction marker changed")
        text = text.replace(extract_marker, extract_hook, 1)

    task_old = """        for tool_call in agent_data.tool_calls[: self.max_parallel_calls]:
            tasks.append(self._call_tool(tool_call, agent_data.tools_kwargs, agent_data))
            tool_call_names.append(tool_call.name)
"""
    task_new = """        _esr_spans = getattr(agent_data, "_esr_pending_call_spans", [])
        for _esr_call_index, tool_call in enumerate(agent_data.tool_calls[: self.max_parallel_calls]):
            _esr_span = _esr_spans[_esr_call_index] if _esr_call_index < len(_esr_spans) else None
            tasks.append(self._call_tool(
                tool_call, agent_data.tools_kwargs, agent_data, esr_token_span=_esr_span
            ))
            tool_call_names.append(tool_call.name)
"""
    if task_old in text:
        text = text.replace(task_old, task_new, 1)
    elif task_new not in text:
        raise RuntimeError("tool_agent_loop parallel-call marker changed")

    signature_old = """    async def _call_tool(
        self, tool_call: FunctionCall, tools_kwargs: dict[str, Any], agent_data: AgentData
    ) -> tuple[ToolResponse, float, dict]:
"""
    signature_new = """    async def _call_tool(
        self, tool_call: FunctionCall, tools_kwargs: dict[str, Any], agent_data: AgentData,
        esr_token_span: dict[str, int] | None = None,
    ) -> tuple[ToolResponse, float, dict]:
"""
    if signature_old in text:
        text = text.replace(signature_old, signature_new, 1)
    elif signature_new not in text:
        raise RuntimeError("tool_agent_loop _call_tool signature changed")

    execute_old = """            tool_execution_response, tool_reward, res = await tool.execute(
                instance_id, tool_args, agent_data=agent_data
            )
"""
    execute_new = """            tool_execution_response, tool_reward, res = await tool.execute(
                instance_id, tool_args, agent_data=agent_data, esr_token_span=esr_token_span
            )
"""
    if execute_old in text:
        text = text.replace(execute_old, execute_new, 1)
    elif execute_new not in text:
        raise RuntimeError("tool_agent_loop tool execute marker changed")

    finish_old = '        is_finish_tool = any(name.lower() in ("finish", "stop", "submit") for name in tool_call_names)\n'
    finish_new = '        is_finish_tool = any(name.lower() in ("finish", "stop", "submit", "submit_answer") for name in tool_call_names)\n'
    if finish_old in text:
        text = text.replace(finish_old, finish_new, 1)
    elif finish_new not in text:
        raise RuntimeError("tool_agent_loop finish marker changed")

    loop_marker = "            for i, traj in enumerate(agent_data.trajectory_outputs):\n"
    loop_hook = """            try:
                from esr_grpo.integrations.echo import finalize_echo_agent_data
                _esr_metadata = finalize_echo_agent_data(agent_data)
            except ImportError:
                _esr_metadata = {}

            for i, traj in enumerate(agent_data.trajectory_outputs):
"""
    if loop_hook not in text:
        if loop_marker not in text:
            raise RuntimeError("tool_agent_loop final output marker changed")
        text = text.replace(loop_marker, loop_hook, 1)

    field_marker = '                        "echo_response_selection_mask": truncated_response_selection_mask,\n'
    field_hook = """                        "echo_response_selection_mask": truncated_response_selection_mask,
                        "esr_legal_submit": _esr_metadata.get("esr_legal_submit", False),
                        "esr_response_credit_mask": _esr_metadata.get(
                            "esr_response_credit_masks", {}
                        ).get(i, []),
                        "esr_selected_action_ids": _esr_metadata.get("esr_selected_action_ids", []),
                        "esr_credit_categories": _esr_metadata.get("esr_credit_categories", {}),
                        "esr_credit_reasons": _esr_metadata.get("esr_credit_reasons", {}),
                        "esr_submitted_answer": _esr_metadata.get("esr_submitted_answer"),
                        "esr_store_path": _esr_metadata.get("esr_store_path"),
"""
    if field_hook not in text:
        if field_marker not in text:
            raise RuntimeError("tool_agent_loop extra fields marker changed")
        text = text.replace(field_marker, field_hook, 1)
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--echo-root", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    root = Path(args.echo_root).resolve()
    core = root / "verl/trainer/ppo/core_algos.py"
    trainer = root / "verl/trainer/ppo/ray_trainer.py"
    agent_loop = root / "verl/experimental/agent_loop/tool_agent_loop.py"
    if not core.exists() or not trainer.exists() or not agent_loop.exists():
        raise SystemExit("not an ECHO/verl source tree")
    core_text = core.read_text(encoding="utf-8")
    trainer_text = patch_ray_trainer(trainer.read_text(encoding="utf-8"))
    agent_loop_text = patch_tool_agent_loop(agent_loop.read_text(encoding="utf-8"))
    if CORE_HOOK.strip() not in core_text:
        core_text += CORE_HOOK
    if not args.apply:
        print("hooks can be applied cleanly; rerun with --apply to modify ECHO")
        return
    for path in (core, trainer, agent_loop):
        backup = path.with_suffix(path.suffix + ".esr-grpo.bak")
        if not backup.exists():
            shutil.copy2(path, backup)
    core.write_text(core_text, encoding="utf-8")
    trainer.write_text(trainer_text, encoding="utf-8")
    agent_loop.write_text(agent_loop_text, encoding="utf-8")
    print("installed ESR-GRPO hooks into ECHO; backups use .esr-grpo.bak")


if __name__ == "__main__":
    main()
