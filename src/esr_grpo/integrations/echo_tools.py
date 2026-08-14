"""ECHO 原生工具：共享 AgentData 上的 ESR episode 环境。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

from ..environment import ESREnvironment, IllegalActionError
from ..retrieval import EchoRetrievalClient
from ..verification import OpenAICompatibleVerifier


SCHEMAS: dict[str, dict[str, Any]] = {
    "search": {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the fixed BrowseComp-Plus corpus.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search query"}},
                "required": ["query"],
            },
        },
    },
    "open_page": {
        "type": "function",
        "function": {
            "name": "open_page",
            "description": "Open a document returned by search and archive the complete page as Evidence.",
            "parameters": {
                "type": "object",
                "properties": {
                    "docid": {"type": "string", "description": "Document ID"},
                    "search_action_id": {"type": "string", "description": "Parent search action ID"},
                },
                "required": ["docid", "search_action_id"],
            },
        },
    },
    "read_evidence": {
        "type": "function",
        "function": {
            "name": "read_evidence",
            "description": "Read the original full text saved under an Evidence ID.",
            "parameters": {
                "type": "object",
                "properties": {"evidence_id": {"type": "string", "description": "Evidence ID"}},
                "required": ["evidence_id"],
            },
        },
    },
    "update_state": {
        "type": "function",
        "function": {
            "name": "update_state",
            "description": "Update answer, findings and supporting Evidence. This action cannot edit gaps.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string", "description": "Current answer"},
                    "evidence_findings": {"type": "array", "description": "Evidence findings"},
                    "supporting_evidence": {"type": "array", "description": "Evidence IDs used by answer"},
                },
                "required": ["answer", "evidence_findings", "supporting_evidence"],
            },
        },
    },
    "verify_answer": {
        "type": "function",
        "function": {
            "name": "verify_answer",
            "description": "Verify the current answer against complete supporting Evidence.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    "submit_answer": {
        "type": "function",
        "function": {
            "name": "submit_answer",
            "description": "Submit TaskState.answer after supported verification and empty gaps.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
}


class ESRBaseTool(BaseTool):
    tool_name = ""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema | None = None):
        schema = tool_schema or OpenAIFunctionToolSchema.model_validate(SCHEMAS[self.tool_name])
        super().__init__(config, schema)

    def _environment(self, agent_data: Any) -> ESREnvironment:
        environment = getattr(agent_data, "_esr_environment", None)
        if environment is not None:
            return environment
        question = self._extract_question(agent_data.messages)
        store_dir = Path(self.config.get("store_dir", "outputs/echo_stores"))
        store_dir.mkdir(parents=True, exist_ok=True)
        environment = ESREnvironment(
            question,
            EchoRetrievalClient(self.config.get("retrieval_url", "http://127.0.0.1:8000")),
            OpenAICompatibleVerifier(
                self.config.get("verifier_base_url", "http://127.0.0.1:8001/v1"),
                self.config.get("verifier_model", "Qwen3-32B"),
                api_key_env=self.config.get("verifier_api_key_env", "ESR_VERIFIER_API_KEY"),
            ),
            store_path=store_dir / f"{agent_data.request_id}.sqlite",
            episode_id=agent_data.request_id,
            observation_char_limit=int(self.config.get("observation_char_limit", 16000)),
        )
        setattr(agent_data, "_esr_environment", environment)
        return environment

    @staticmethod
    def _extract_question(messages: list[dict[str, Any]]) -> str:
        for message in messages:
            if message.get("role") == "user":
                content = message.get("content", "")
                return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
        raise ValueError("raw prompt has no user question")

    @staticmethod
    def _execution_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
        agent_data = kwargs["agent_data"]
        raw_span = kwargs.get("esr_token_span")
        return {
            "token_spans": [raw_span] if raw_span else None,
            "turn_id": f"t{agent_data.assistant_turns}",
        }

    async def _run(self, parameters: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        environment = self._environment(kwargs["agent_data"])
        return environment.execute_tool(
            self.tool_name,
            parameters,
            **self._execution_kwargs(kwargs),
        )

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs: Any):
        try:
            result = await self._run(parameters, **kwargs)
        except IllegalActionError as exc:
            result = {"legal": False, "action_id": exc.action_id, "error": str(exc)}
        return ToolResponse(text=json.dumps(result, ensure_ascii=False)), 0.0, {"esr_tool": self.tool_name}


class ESRSearchTool(ESRBaseTool):
    tool_name = "search"


class ESROpenPageTool(ESRBaseTool):
    tool_name = "open_page"


class ESRReadEvidenceTool(ESRBaseTool):
    tool_name = "read_evidence"


class ESRUpdateStateTool(ESRBaseTool):
    tool_name = "update_state"


class ESRVerifyAnswerTool(ESRBaseTool):
    tool_name = "verify_answer"


class ESRSubmitAnswerTool(ESRBaseTool):
    tool_name = "submit_answer"

