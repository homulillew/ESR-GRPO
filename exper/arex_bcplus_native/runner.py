"""单道 BC-Plus 问题的原生 AREX Agent Loop。"""

from __future__ import annotations

import copy
import json
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable

from clients import AREXChatClient, BCPlusRetrievalClient
from io_utils import response_record
from prompts import build_messages
from protocol import ProtocolError, extract_reasoning, normalize_call, parse_tool_call


@dataclass
class AREXRunner:
    model: AREXChatClient
    retrieval: BCPlusRetrievalClient
    max_turns: int = 100
    on_checkpoint: Callable[[dict[str, Any]], None] | None = None

    def run(self, query_id: str, question: str) -> tuple[dict[str, Any], dict[str, Any]]:
        initial_messages = build_messages(question)
        messages = copy.deepcopy(initial_messages)
        trajectory: dict[str, Any] = {
            "schema_version": 1,
            "protocol": "AREX native XML adapted to BC-Plus fixed corpus",
            "query_id": query_id,
            "question": question,
            "model": self.model.model,
            "status": "running",
            "started_at_unix": time.time(),
            "initial_messages": initial_messages,
            "turns": [],
            "context_updates": [],
        }
        final: dict[str, Any] | None = None
        failure: dict[str, Any] | None = None

        for turn_index in range(self.max_turns):
            started = time.time()
            try:
                raw = self.model.complete(messages)
                choice = raw["choices"][0]
                message = choice.get("message", {})
                content = str(message.get("content") or "")
                call = normalize_call(parse_tool_call(content))
                turn: dict[str, Any] = {
                    "turn": turn_index + 1,
                    "elapsed_seconds": time.time() - started,
                    "assistant_content": content,
                    "reasoning": str(message.get("reasoning_content") or extract_reasoning(content, call)),
                    "tool_call": {"name": call.name, "arguments": call.arguments},
                    "raw_response": raw,
                }
                messages.append({"role": "assistant", "content": content})

                if call.name == "search":
                    result = self.retrieval.search(call.arguments["query"])
                elif call.name == "visit":
                    result = self.retrieval.visit(call.arguments["url"], str(call.arguments["goal"]))
                elif call.name == "update_context":
                    context = str(call.arguments["context"])
                    trajectory["context_updates"].append(
                        {"turn": turn_index + 1, "context": context}
                    )
                    result = {"status": "context_updated", "context": context}
                elif call.name == "finish":
                    final = {
                        "answer": str(call.arguments["answer"]).strip(),
                        "evidences": call.arguments["evidences"],
                        "confidence": str(call.arguments["confidence"]).strip(),
                    }
                    result = {"status": "finished"}
                else:  # pragma: no cover - normalize_call 已拦截
                    raise ProtocolError(f"unsupported tool: {call.name}")

                turn["tool_result"] = result
                trajectory["turns"].append(turn)
                if final is not None:
                    break

                tool_response = "<tool_response>\n" + json.dumps(
                    result, ensure_ascii=False
                ) + "\n</tool_response>"
                if call.name == "update_context":
                    # AREX 的 update_context 会用压缩状态替换旧历史，而不是只追加摘要。
                    messages = copy.deepcopy(initial_messages)
                    messages.append({"role": "user", "content": tool_response})
                    turn["active_context_reset"] = True
                else:
                    messages.append({"role": "user", "content": tool_response})
                self._checkpoint(trajectory)
            except Exception as exc:
                failure = {
                    "turn": turn_index + 1,
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
                trajectory["turns"].append(
                    {
                        "turn": turn_index + 1,
                        "elapsed_seconds": time.time() - started,
                        "error": failure,
                    }
                )
                break

        trajectory["finished_at_unix"] = time.time()
        trajectory["retrieved_docids"] = sorted(self.retrieval.retrieved_docids)
        trajectory["visited_docids"] = sorted(self.retrieval.visited_docids)
        visited_urls = {
            str(item.get("url"))
            for turn in trajectory["turns"]
            if turn.get("tool_call", {}).get("name") == "visit"
            for item in (turn.get("tool_result") or [])
            if isinstance(item, dict) and item.get("url")
        }
        if final is not None:
            cited_urls = {
                str(item.get("url"))
                for item in final.get("evidences", [])
                if isinstance(item, dict) and item.get("url")
            }
            final["evidence_validation"] = {
                "visited_urls": sorted(visited_urls),
                "cited_urls": sorted(cited_urls),
                "unvisited_cited_urls": sorted(cited_urls - visited_urls),
                "all_citations_visited": cited_urls.issubset(visited_urls) and bool(cited_urls),
            }
        trajectory["final"] = final
        if final is not None:
            trajectory["status"] = "completed"
        elif failure is not None:
            trajectory["status"] = "failed"
            trajectory["error"] = failure
        else:
            trajectory["status"] = "max_turns"
        self._checkpoint(trajectory)

        run = self._build_run(query_id, final, trajectory)
        return trajectory, run

    def _checkpoint(self, trajectory: dict[str, Any]) -> None:
        if self.on_checkpoint:
            self.on_checkpoint(trajectory)

    def _build_run(
        self, query_id: str, final: dict[str, Any] | None, trajectory: dict[str, Any]
    ) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        counts: dict[str, int] = {}
        for turn in trajectory["turns"]:
            call = turn.get("tool_call")
            if not call:
                continue
            name = str(call["name"])
            counts[name] = counts.get(name, 0) + 1
            results.append(
                {
                    "type": "tool_call",
                    "tool_name": name,
                    "arguments": call["arguments"],
                    "output": turn.get("tool_result"),
                }
            )
        if final:
            results.append(
                {
                    "type": "output_text",
                    "output": response_record(final["answer"], final["confidence"]),
                }
            )
        return {
            "metadata": {
                "model": self.model.model,
                "protocol": trajectory["protocol"],
                "temperature": self.model.temperature,
                "top_p": self.model.top_p,
                "top_k": self.model.top_k,
                "max_tokens_per_turn": self.model.max_tokens,
                "turns": len(trajectory["turns"]),
            },
            "query_id": query_id,
            "tool_call_counts": counts,
            "status": trajectory["status"],
            "retrieved_docids": sorted(self.retrieval.retrieved_docids),
            "visited_docids": sorted(self.retrieval.visited_docids),
            "evidences": final["evidences"] if final else [],
            "evidence_validation": final.get("evidence_validation") if final else None,
            "result": results,
        }
