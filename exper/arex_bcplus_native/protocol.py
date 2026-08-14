"""解析 AREX 使用的 XML 工具调用协议。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


class ProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]
    raw_xml: str


_CALL_RE = re.compile(
    r"<tool_call>\s*<function=([^>\s]+)>\s*(.*?)\s*</function>\s*</tool_call>",
    re.DOTALL | re.IGNORECASE,
)
_PARAM_RE = re.compile(
    r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>", re.DOTALL | re.IGNORECASE
)


def parse_tool_call(text: str) -> ToolCall:
    matches = list(_CALL_RE.finditer(text or ""))
    if not matches:
        raise ProtocolError("model output contains no AREX <tool_call>")
    if len(matches) != 1:
        raise ProtocolError(f"AREX requires one tool per turn, found {len(matches)}")
    match = matches[0]
    name = match.group(1).strip()
    params: dict[str, Any] = {}
    for param in _PARAM_RE.finditer(match.group(2)):
        key = param.group(1).strip()
        if key in params:
            raise ProtocolError(f"duplicate parameter: {key}")
        value = param.group(2).strip()
        if key in {"query", "evidences"}:
            try:
                value = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ProtocolError(f"parameter {key} must be valid JSON: {exc}") from exc
        elif key == "url" and value.startswith("["):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ProtocolError(f"url array must be valid JSON: {exc}") from exc
        elif key == "url" and value.startswith('"'):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ProtocolError(f"quoted url must be valid JSON: {exc}") from exc
        params[key] = value
    return ToolCall(name=name, arguments=params, raw_xml=match.group(0))


def normalize_call(call: ToolCall) -> ToolCall:
    name = call.name
    args = dict(call.arguments)
    required = {
        "search": {"query"},
        "visit": {"url", "goal"},
        "update_context": {"context"},
        "finish": {"answer", "evidences", "confidence"},
    }
    if name not in required:
        raise ProtocolError(f"unsupported AREX tool in BC-Plus: {name}")
    missing = required[name] - set(args)
    if missing:
        raise ProtocolError(f"{name} missing required parameters: {sorted(missing)}")
    if name == "search":
        if isinstance(args["query"], str):
            args["query"] = [args["query"]]
        if not isinstance(args["query"], list) or not all(
            isinstance(item, str) and item.strip() for item in args["query"]
        ):
            raise ProtocolError("search.query must be a non-empty JSON string array")
    if name == "visit":
        if isinstance(args["url"], str):
            args["url"] = [args["url"]]
        if not isinstance(args["url"], list) or not args["url"]:
            raise ProtocolError("visit.url must be a string or non-empty string array")
    if name == "finish":
        if not str(args["answer"]).strip():
            raise ProtocolError("finish.answer is empty")
        if not isinstance(args["evidences"], list):
            raise ProtocolError("finish.evidences must be a JSON array")
        for index, item in enumerate(args["evidences"]):
            if not isinstance(item, dict) or set(item) != {"evidence", "url"}:
                raise ProtocolError(
                    f"finish.evidences[{index}] must contain exactly evidence and url"
                )
            if not str(item["evidence"]).strip() or not str(item["url"]).strip():
                raise ProtocolError(f"finish.evidences[{index}] contains an empty field")
        confidence_match = re.search(r"\d+(?:\.\d+)?", str(args["confidence"]))
        if not confidence_match or not 0 <= float(confidence_match.group(0)) <= 100:
            raise ProtocolError("finish.confidence must contain a value between 0 and 100")
    return ToolCall(name=name, arguments=args, raw_xml=call.raw_xml)


def extract_reasoning(text: str, call: ToolCall | None = None) -> str:
    prefix = text.split(call.raw_xml, 1)[0] if call and call.raw_xml in text else text
    match = re.search(r"<think>\s*(.*?)\s*</think>", prefix, re.DOTALL | re.IGNORECASE)
    return (match.group(1) if match else prefix).strip()
