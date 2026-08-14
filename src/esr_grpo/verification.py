"""答案验证接口。

在线 verify_answer 与 Benchmark Judge 分离。前者只决定 TaskState 是否需要继续
搜索；后者在 episode 结束后提供最终奖励。
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Protocol, Sequence

from .models import Evidence, VerificationResult, VerificationStatus


class Verifier(Protocol):
    def verify(self, question: str, answer: str, evidence: Sequence[Evidence]) -> VerificationResult: ...


@dataclass
class KeywordVerifier:
    """测试用确定性验证器；正式实验不得把它当作 Benchmark Judge。"""

    required_terms: tuple[str, ...] = ()

    def verify(self, question: str, answer: str, evidence: Sequence[Evidence]) -> VerificationResult:
        combined = "\n".join(item.content for item in evidence).lower()
        missing = [term for term in self.required_terms if term.lower() not in combined]
        if not answer.strip():
            missing.insert(0, "当前答案为空")
        if missing:
            return VerificationResult(
                VerificationStatus.NEEDS_REVISION,
                tuple(f"缺少可核对信息：{item}" for item in missing),
                "规则验证未通过",
            )
        return VerificationResult(VerificationStatus.SUPPORTED, (), "规则验证通过")


@dataclass
class OpenAICompatibleVerifier:
    base_url: str
    model: str
    api_key_env: str = "ESR_VERIFIER_API_KEY"
    timeout_seconds: float = 120.0
    max_tokens: int = 1024

    def verify(self, question: str, answer: str, evidence: Sequence[Evidence]) -> VerificationResult:
        evidence_text = "\n\n".join(
            f"[{item.evidence_id}] {item.source.title}\n{item.content}" for item in evidence
        )
        prompt = (
            "根据给定原始证据检查当前答案。只输出 JSON："
            '{"verification_status":"supported|needs_revision","gaps":["具体待解决问题"],'
            '"rationale":"简短理由"}。supported 时 gaps 必须为空。\n\n'
            f"问题：{question}\n当前答案：{answer}\n\n原始证据：\n{evidence_text}"
        )
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
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
            result = json.loads(response.read().decode("utf-8"))
        content = result["choices"][0]["message"]["content"].strip()
        if content.startswith("```"):
            content = content.strip("`")
            if content.startswith("json"):
                content = content[4:].lstrip()
        parsed = json.loads(content)
        status = VerificationStatus(parsed["verification_status"])
        gaps = tuple(str(item).strip() for item in parsed.get("gaps", []) if str(item).strip())
        if status is VerificationStatus.SUPPORTED and gaps:
            raise ValueError("verifier returned supported with non-empty gaps")
        if status is VerificationStatus.NEEDS_REVISION and not gaps:
            raise ValueError("verifier returned needs_revision without gaps")
        return VerificationResult(status, gaps, str(parsed.get("rationale", "")))
