"""最终奖励和 Benchmark 评测 Judge。"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class JudgeResult:
    correct: bool
    confidence: float
    extracted_answer: str
    rationale: str
    parse_error: bool = False


class Judge(Protocol):
    def judge(self, question: str, response: str, correct_answer: str) -> JudgeResult: ...


@dataclass
class ExactMatchJudge:
    """只用于评测管线测试。"""

    def judge(self, question: str, response: str, correct_answer: str) -> JudgeResult:
        normalized_response = " ".join(response.lower().split())
        normalized_answer = " ".join(correct_answer.lower().split())
        correct = normalized_answer in normalized_response
        return JudgeResult(correct, extract_confidence(response), response, "deterministic smoke judge")


@dataclass
class OpenAICompatibleJudge:
    base_url: str
    model: str
    api_key_env: str = "ESR_JUDGE_API_KEY"
    timeout_seconds: float = 180.0
    max_tokens: int = 2048

    def judge(self, question: str, response: str, correct_answer: str) -> JudgeResult:
        prompt = (
            "判断候选回答是否与标准答案语义一致。额外解释只有在内容正确时才允许。"
            "只输出 JSON："
            '{"correct":true|false,"extracted_answer":"...","rationale":"..."}。\n\n'
            f"问题：{question}\n标准答案：{correct_answer}\n候选回答：{response}"
        )
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "top_p": 1.0,
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
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response_obj:
            raw = json.loads(response_obj.read().decode("utf-8"))
        content = str(raw["choices"][0]["message"]["content"]).strip()
        try:
            parsed = _parse_json_object(content)
        except (ValueError, json.JSONDecodeError):
            return JudgeResult(False, extract_confidence(response), "", content, True)
        return JudgeResult(
            bool(parsed.get("correct", False)),
            extract_confidence(response),
            str(parsed.get("extracted_answer", "")),
            str(parsed.get("rationale", "")),
            False,
        )


def extract_confidence(response: str, default: float = 1.0) -> float:
    match = re.search(r"confidence\s*[:：]\s*(\d+(?:\.\d+)?)\s*%?", response, re.IGNORECASE)
    if not match:
        return default
    value = float(match.group(1))
    if value > 1.0:
        value /= 100.0
    return min(max(value, 0.0), 1.0)


def _parse_json_object(text: str) -> dict:
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].lstrip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("judge did not return JSON")
    return json.loads(text[start : end + 1])
