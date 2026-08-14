"""与模型服务解耦的多轮 Agent Loop。

该模块用于 Benchmark 推理和协议调试。分布式 RL rollout 的精确 token 范围由
verl/SGLang 集成层提供，不能用这里的近似范围代替训练元数据。
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from .environment import ESREnvironment, IllegalActionError
from .models import TokenSpan, to_primitive


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: Mapping[str, Any]
    call_id: str = ""
    token_spans: tuple[TokenSpan, ...] = ()


@dataclass(frozen=True)
class PolicyTurn:
    content: str
    tool_calls: tuple[ToolCall, ...]
    raw: Mapping[str, Any] | None = None


class Policy(Protocol):
    def next_turn(self, messages: Sequence[Mapping[str, Any]], context: Mapping[str, Any]) -> PolicyTurn: ...


SYSTEM_PROMPT = """你是长程检索 Agent。系统通过 TaskState 保存当前答案、证据目录和待解决问题。
必须通过工具完成任务：search 查找候选文档；open_page 阅读并保存原文；update_state 整理全部新 Evidence；
verify_answer 根据主要证据检查答案；仅在 verification_status=supported 且 gaps 为空时调用 submit_answer。
不要在普通文本中绕过 submit_answer 给出最终答案。重新核对旧材料时使用 read_evidence。"""


OPENAI_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "在固定语料中检索文档",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_page",
            "description": "打开某次 search 返回的文档并保存为不可变 Evidence",
            "parameters": {
                "type": "object",
                "properties": {
                    "docid": {"type": "string"},
                    "search_action_id": {"type": "string"},
                },
                "required": ["docid", "search_action_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_evidence",
            "description": "按 Evidence ID 重新读取已保存的原文",
            "parameters": {
                "type": "object",
                "properties": {"evidence_id": {"type": "string"}},
                "required": ["evidence_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_state",
            "description": "写入或修正 finding，更新答案和主要证据；gaps 保持不变",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                    "evidence_findings": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "evidence_id": {"type": "string"},
                                "finding": {"type": "string"},
                            },
                            "required": ["evidence_id", "finding"],
                        },
                    },
                    "supporting_evidence": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["answer", "evidence_findings", "supporting_evidence"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verify_answer",
            "description": "让独立验证器根据主要 Evidence 检查当前答案",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_answer",
            "description": "提交已通过验证的 TaskState.answer",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


@dataclass
class OpenAIChatPolicy:
    base_url: str
    model: str
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout_seconds: float = 180.0

    def next_turn(self, messages: Sequence[Mapping[str, Any]], context: Mapping[str, Any]) -> PolicyTurn:
        contextual_messages = list(messages) + [
            {
                "role": "user",
                "content": "当前系统状态：\n" + json.dumps(context, ensure_ascii=False),
            }
        ]
        body = {
            "model": self.model,
            "messages": contextual_messages,
            "tools": OPENAI_TOOLS,
            "tool_choice": "auto",
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        api_key = os.getenv(self.api_key_env)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            raw = json.loads(response.read().decode("utf-8"))
        message = raw["choices"][0]["message"]
        calls: list[ToolCall] = []
        offset = 0
        for item in message.get("tool_calls", []):
            function = item["function"]
            arguments = json.loads(function.get("arguments") or "{}")
            # Chat Completions API 不返回 tool call 的 token 位置。这里的范围只用于
            # 推理日志可视化，训练时会由 SGLang tokenizer 覆盖。
            approximate_length = max(1, len(json.dumps(item, ensure_ascii=False)) // 4)
            calls.append(
                ToolCall(
                    name=function["name"],
                    arguments=arguments,
                    call_id=str(item.get("id", "")),
                    token_spans=(TokenSpan(0, offset, offset + approximate_length),),
                )
            )
            offset += approximate_length
        return PolicyTurn(str(message.get("content") or ""), tuple(calls), raw)


@dataclass
class AgentRunner:
    environment: ESREnvironment
    policy: Policy
    max_turns: int = 50
    max_parallel_calls: int = 5

    def run(self) -> dict[str, Any]:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self.environment.question},
        ]
        turns: list[dict[str, Any]] = []
        for turn_index in range(self.max_turns):
            turn = self.policy.next_turn(messages, self.environment.render_context())
            assistant = {
                "role": "assistant",
                "content": turn.content,
                "tool_calls": [to_primitive(item) for item in turn.tool_calls],
            }
            messages.append(assistant)
            if not turn.tool_calls:
                turns.append({"turn": turn_index, "error": "policy returned no tool call"})
                break
            calls = list(turn.tool_calls[: self.max_parallel_calls])
            parallel_names = {"search", "open_page", "read_evidence"}
            if len(calls) > 1 and all(item.name in parallel_names for item in calls):
                results = self.environment.execute_parallel(
                    [
                        {
                            "name": item.name,
                            "arguments": dict(item.arguments),
                            "token_spans": [to_primitive(span) for span in item.token_spans],
                        }
                        for item in calls
                    ],
                    turn_id=f"t{turn_index + 1}",
                )
            elif len(calls) == 1:
                item = calls[0]
                try:
                    result = self.environment.execute_tool(
                            item.name,
                            item.arguments,
                            token_spans=item.token_spans,
                            turn_id=f"t{turn_index + 1}",
                        )
                except IllegalActionError as exc:
                    result = {"error": str(exc), "action_id": exc.action_id, "legal": False}
                results = [result]
            else:
                turns.append({"turn": turn_index, "error": "state-changing actions must be serial"})
                break
            for call, result in zip(calls, results):
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.call_id,
                        "name": call.name,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )
            turns.append(
                {
                    "turn": turn_index,
                    "assistant": assistant,
                    "tool_results": results,
                }
            )
            if self.environment.is_submitted:
                break
        return {
            "submitted": self.environment.is_submitted,
            "answer": self.environment.submitted_answer,
            "turns": turns,
            "messages": messages,
            "episode": self.environment.snapshot(),
        }
