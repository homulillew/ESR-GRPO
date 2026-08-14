"""ESR-GRPO Harness：执行工具、校验协议并记录动作关系。"""

from __future__ import annotations

import hashlib
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .models import (
    ActionKind,
    ActionRecord,
    Evidence,
    EvidenceFinding,
    EvidenceSource,
    Gap,
    TaskState,
    TokenSpan,
    VerificationStatus,
    normalize_spans,
    to_primitive,
)
from .retrieval import Retriever
from .store import EpisodeStore
from .verification import Verifier


class IllegalActionError(RuntimeError):
    def __init__(self, message: str, action_id: str | None = None) -> None:
        super().__init__(message)
        self.action_id = action_id


class ESREnvironment:
    """一个问题对应一个环境实例和一个追加式事件账本。"""

    def __init__(
        self,
        question: str,
        retriever: Retriever,
        verifier: Verifier,
        *,
        store: EpisodeStore | None = None,
        store_path: str | Path = ":memory:",
        episode_id: str | None = None,
        search_top_k: int = 5,
        observation_char_limit: int = 16_000,
        finding_char_limit: int = 600,
        max_gaps: int = 8,
    ) -> None:
        if not question.strip():
            raise ValueError("question must not be empty")
        self.question = question.strip()
        self.retriever = retriever
        self.verifier = verifier
        self.store = store or EpisodeStore(store_path)
        self.search_top_k = search_top_k
        self.observation_char_limit = observation_char_limit
        self.finding_char_limit = finding_char_limit
        self.max_gaps = max_gaps
        self._lock = threading.RLock()
        self._visible_evidence_inputs: dict[str, str] = {}

        stored_question = self.store.get_metadata("question")
        if stored_question is None:
            self.store.set_metadata_once("episode_id", episode_id or uuid.uuid4().hex)
            self.store.set_metadata_once("question", self.question)
        elif stored_question != self.question:
            raise ValueError("store belongs to a different question")

    @property
    def current_state(self) -> TaskState | None:
        return self.store.latest_state()

    @property
    def is_submitted(self) -> bool:
        return self.store.get_metadata("submit_action_id") is not None

    @property
    def submitted_answer(self) -> str | None:
        return self.store.get_metadata("submitted_answer")

    def search(
        self,
        query: str,
        *,
        token_spans: Sequence[TokenSpan | Mapping[str, int]] | None = None,
        turn_id: str | None = None,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            if not query.strip():
                return self._reject(ActionKind.SEARCH, "search query must not be empty", token_spans, turn_id)
            hits = self.retriever.search(query.strip(), top_k or self.search_top_k)
            action = self._make_action(
                ActionKind.SEARCH,
                token_spans,
                turn_id=turn_id,
                active_gap_ids=self._active_gap_ids(),
                metadata={"query": query.strip(), "hits": [to_primitive(item) for item in hits]},
            )
            self.store.add_action(action)
            return {"action_id": action.action_id, "results": [to_primitive(item) for item in hits]}

    def open_page(
        self,
        docid: str,
        *,
        search_action_id: str,
        token_spans: Sequence[TokenSpan | Mapping[str, int]] | None = None,
        turn_id: str | None = None,
    ) -> dict[str, Any]:
        """读取整篇文档，并在截断模型观察前持久化完整正文。"""

        with self._lock:
            try:
                search_action = self.store.get_action(search_action_id)
            except KeyError:
                return self._reject(
                    ActionKind.OPEN_PAGE, "search_action_id does not exist", token_spans, turn_id
                )
            if search_action.kind is not ActionKind.SEARCH or not search_action.legal:
                return self._reject(
                    ActionKind.OPEN_PAGE, "open_page parent must be a legal search action", token_spans, turn_id
                )
            hit_ids = {str(item.get("docid")) for item in search_action.metadata.get("hits", [])}
            if docid not in hit_ids:
                return self._reject(
                    ActionKind.OPEN_PAGE,
                    "document was not returned by the referenced search action",
                    token_spans,
                    turn_id,
                )

            document = self.retriever.get_document(docid)
            if not document.content:
                return self._reject(ActionKind.OPEN_PAGE, "retrieved document is empty", token_spans, turn_id)
            content_hash = hashlib.sha256(document.content.encode("utf-8")).hexdigest()
            source = EvidenceSource(document.docid, document.url, document.title)
            existing = self.store.find_evidence(source.key, content_hash)
            action_id, sequence_index = self._next_action_identity()
            evidence: Evidence | None = None
            if existing is None:
                evidence_id = f"e{len(self.store.list_evidence()) + 1}"
                evidence = Evidence(
                    evidence_id=evidence_id,
                    source=source,
                    content=document.content,
                    content_sha256=content_hash,
                    created_by_action_id=action_id,
                    search_action_id=search_action_id,
                    created_at=datetime.now(timezone.utc).isoformat(),
                )
            else:
                evidence_id = existing.evidence_id
            action = ActionRecord(
                action_id=action_id,
                sequence_index=sequence_index,
                turn_id=turn_id,
                kind=ActionKind.OPEN_PAGE,
                token_spans=normalize_spans(token_spans),
                legal=True,
                state_version_before=self._state_version(),
                parent_action_ids=(search_action_id,),
                referenced_evidence_ids=() if evidence is not None else (evidence_id,),
                created_evidence_ids=(evidence_id,) if evidence is not None else (),
                active_gap_ids=self._active_gap_ids(),
                metadata={"docid": docid, "duplicate": evidence is None},
            )
            self.store.commit_transition(action, evidence=evidence)
            self._visible_evidence_inputs[evidence_id] = action.action_id
            visible = document.content[: self.observation_char_limit]
            return {
                "action_id": action.action_id,
                "evidence_id": evidence_id,
                "source": to_primitive(source),
                "content": visible,
                "truncated": len(visible) < len(document.content),
                "stored_content_chars": len(document.content),
                "duplicate": evidence is None,
            }

    def read_evidence(
        self,
        evidence_id: str,
        *,
        token_spans: Sequence[TokenSpan | Mapping[str, int]] | None = None,
        turn_id: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            state = self.current_state
            if state is None or evidence_id not in state.directory_map():
                return self._reject(
                    ActionKind.READ_EVIDENCE,
                    "read_evidence requires an Evidence ID in the current evidence_directory",
                    token_spans,
                    turn_id,
                )
            evidence = self.store.get_evidence(evidence_id)
            action = self._make_action(
                ActionKind.READ_EVIDENCE,
                token_spans,
                turn_id=turn_id,
                referenced_evidence_ids=(evidence_id,),
                active_gap_ids=self._active_gap_ids(),
                metadata={"evidence_id": evidence_id},
            )
            self.store.add_action(action)
            self._visible_evidence_inputs[evidence_id] = action.action_id
            visible = evidence.content[: self.observation_char_limit]
            return {
                "action_id": action.action_id,
                "evidence_id": evidence_id,
                "source": to_primitive(evidence.source),
                "content": visible,
                "truncated": len(visible) < len(evidence.content),
                "stored_content_chars": len(evidence.content),
            }

    def update_state(
        self,
        answer: str,
        evidence_findings: Sequence[EvidenceFinding | Mapping[str, str]],
        supporting_evidence: Sequence[str],
        *,
        token_spans: Sequence[TokenSpan | Mapping[str, int]] | None = None,
        turn_id: str | None = None,
    ) -> dict[str, Any]:
        """追加 TaskState；保留 gaps，并将验证状态重置为 unverified。"""

        with self._lock:
            before = self.current_state
            old_directory = before.directory_map() if before else {}
            patch: dict[str, str] = {}
            finding_inputs: dict[str, str] = {}
            for raw in evidence_findings:
                item = raw if isinstance(raw, EvidenceFinding) else EvidenceFinding(**dict(raw))
                finding = item.finding.strip()
                if not finding:
                    return self._reject(
                        ActionKind.UPDATE_STATE, f"finding for {item.evidence_id} is empty", token_spans, turn_id
                    )
                if len(finding) > self.finding_char_limit:
                    return self._reject(
                        ActionKind.UPDATE_STATE,
                        f"finding for {item.evidence_id} exceeds character limit",
                        token_spans,
                        turn_id,
                    )
                if item.evidence_id in patch:
                    return self._reject(
                        ActionKind.UPDATE_STATE, f"duplicate finding: {item.evidence_id}", token_spans, turn_id
                    )
                try:
                    self.store.get_evidence(item.evidence_id)
                except KeyError:
                    return self._reject(
                        ActionKind.UPDATE_STATE, f"unknown Evidence ID: {item.evidence_id}", token_spans, turn_id
                    )
                changed = old_directory.get(item.evidence_id) != finding
                if changed and item.evidence_id not in self._visible_evidence_inputs:
                    return self._reject(
                        ActionKind.UPDATE_STATE,
                        f"changed finding requires visible original Evidence: {item.evidence_id}",
                        token_spans,
                        turn_id,
                    )
                if changed:
                    finding_inputs[item.evidence_id] = self._visible_evidence_inputs[item.evidence_id]
                patch[item.evidence_id] = finding

            new_directory = {**old_directory, **patch}
            archive_ids = [item.evidence_id for item in self.store.list_evidence()]
            missing = [item for item in archive_ids if item not in new_directory]
            extra = [item for item in new_directory if item not in set(archive_ids)]
            if missing or extra:
                return self._reject(
                    ActionKind.UPDATE_STATE,
                    f"evidence_directory coverage failed; missing={missing}, extra={extra}",
                    token_spans,
                    turn_id,
                )
            support = tuple(dict.fromkeys(str(item) for item in supporting_evidence))
            if len(support) != len(supporting_evidence):
                return self._reject(ActionKind.UPDATE_STATE, "supporting_evidence contains duplicates", token_spans, turn_id)
            unknown_support = [item for item in support if item not in new_directory]
            if unknown_support:
                return self._reject(
                    ActionKind.UPDATE_STATE,
                    f"supporting_evidence is not in directory: {unknown_support}",
                    token_spans,
                    turn_id,
                )
            normalized_answer = answer.strip()
            if normalized_answer and not support:
                return self._reject(
                    ActionKind.UPDATE_STATE, "a non-empty answer requires supporting_evidence", token_spans, turn_id
                )

            action_id, sequence_index = self._next_action_identity()
            state = TaskState(
                version=(before.version + 1) if before else 1,
                parent_version=before.version if before else None,
                evidence_directory=tuple(
                    EvidenceFinding(evidence_id, new_directory[evidence_id]) for evidence_id in archive_ids
                ),
                answer=normalized_answer,
                supporting_evidence=support,
                gaps=before.gaps if before else (),
                verification_status=VerificationStatus.UNVERIFIED,
                created_by_action_id=action_id,
            )
            action = ActionRecord(
                action_id=action_id,
                sequence_index=sequence_index,
                turn_id=turn_id,
                kind=ActionKind.UPDATE_STATE,
                token_spans=normalize_spans(token_spans),
                legal=True,
                state_version_before=before.version if before else None,
                state_version_after=state.version,
                parent_action_ids=tuple(dict.fromkeys(finding_inputs.values())),
                referenced_evidence_ids=tuple(new_directory),
                active_gap_ids=self._active_gap_ids(),
                metadata={
                    "finding_inputs": finding_inputs,
                    "changed_finding_ids": [
                        key for key, value in new_directory.items() if old_directory.get(key) != value
                    ],
                    "answer_changed": before is None or before.answer != normalized_answer,
                    "support_changed": before is None or before.supporting_evidence != support,
                },
            )
            self.store.commit_transition(action, state=state)
            self._visible_evidence_inputs.clear()
            return {"action_id": action.action_id, "task_state": to_primitive(state)}

    def verify_answer(
        self,
        *,
        token_spans: Sequence[TokenSpan | Mapping[str, int]] | None = None,
        turn_id: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            before = self.current_state
            error = self._verification_error(before)
            if error:
                return self._reject(ActionKind.VERIFY_ANSWER, error, token_spans, turn_id)
            assert before is not None
            evidence = [self.store.get_evidence(item) for item in before.supporting_evidence]
            result = self.verifier.verify(self.question, before.answer, evidence)
            if result.status is VerificationStatus.SUPPORTED and result.gaps:
                return self._reject(
                    ActionKind.VERIFY_ANSWER, "supported verifier result contains gaps", token_spans, turn_id
                )
            if result.status is VerificationStatus.NEEDS_REVISION and not result.gaps:
                return self._reject(
                    ActionKind.VERIFY_ANSWER, "needs_revision verifier result has no gap", token_spans, turn_id
                )
            if len(result.gaps) > self.max_gaps:
                return self._reject(
                    ActionKind.VERIFY_ANSWER, "verifier returned too many gaps", token_spans, turn_id
                )

            action_id, sequence_index = self._next_action_identity()
            old_by_text = {item.description.strip().lower(): item for item in before.gaps}
            gap_count = len({gap.gap_id for state in self.store.list_states() for gap in state.gaps})
            gaps: list[Gap] = []
            for description in result.gaps:
                text = description.strip()
                existing = old_by_text.get(text.lower())
                if existing is not None:
                    gaps.append(existing)
                else:
                    gap_count += 1
                    gaps.append(Gap(f"g{gap_count}", text, action_id))
            old_ids = {item.gap_id for item in before.gaps}
            new_ids = {item.gap_id for item in gaps}
            state = TaskState(
                version=before.version + 1,
                parent_version=before.version,
                evidence_directory=before.evidence_directory,
                answer=before.answer,
                supporting_evidence=before.supporting_evidence,
                gaps=tuple(gaps),
                verification_status=result.status,
                created_by_action_id=action_id,
            )
            action = ActionRecord(
                action_id=action_id,
                sequence_index=sequence_index,
                turn_id=turn_id,
                kind=ActionKind.VERIFY_ANSWER,
                token_spans=normalize_spans(token_spans),
                legal=True,
                state_version_before=before.version,
                state_version_after=state.version,
                referenced_evidence_ids=before.supporting_evidence,
                active_gap_ids=tuple(old_ids),
                metadata={
                    "verification_status": result.status.value,
                    "created_gap_ids": sorted(new_ids - old_ids),
                    "resolved_gap_ids": sorted(old_ids - new_ids),
                    "rationale": result.rationale,
                },
            )
            self.store.commit_transition(action, state=state)
            self._visible_evidence_inputs.clear()
            return {
                "action_id": action.action_id,
                "verification_status": result.status.value,
                "gaps": [to_primitive(item) for item in gaps],
                "rationale": result.rationale,
            }

    def submit_answer(
        self,
        *,
        token_spans: Sequence[TokenSpan | Mapping[str, int]] | None = None,
        turn_id: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            state = self.current_state
            error = self._submission_error(state)
            if error:
                return self._reject(ActionKind.SUBMIT_ANSWER, error, token_spans, turn_id)
            assert state is not None
            action = self._make_action(
                ActionKind.SUBMIT_ANSWER,
                token_spans,
                turn_id=turn_id,
                referenced_evidence_ids=state.supporting_evidence,
                metadata={"answer": state.answer},
            )
            self.store.commit_transition(
                action,
                metadata_once={"submitted_answer": state.answer, "submit_action_id": action.action_id},
            )
            return {"action_id": action.action_id, "answer": state.answer}

    def execute_parallel(self, calls: Sequence[Mapping[str, Any]], *, turn_id: str) -> list[dict[str, Any]]:
        """并行调用只允许 search/open_page/read_evidence，并为每项分配独立 action_id。"""

        allowed = {"search", "open_page", "read_evidence"}
        if any(str(call.get("name")) not in allowed for call in calls):
            raise ValueError("parallel execution only supports search/open_page/read_evidence")

        def run(call: Mapping[str, Any]) -> dict[str, Any]:
            name = str(call["name"])
            arguments = dict(call.get("arguments", {}))
            arguments["turn_id"] = turn_id
            arguments["token_spans"] = call.get("token_spans")
            try:
                return getattr(self, name)(**arguments)
            except IllegalActionError as exc:
                return {"error": str(exc), "action_id": exc.action_id, "legal": False}

        with ThreadPoolExecutor(max_workers=max(1, len(calls))) as pool:
            return list(pool.map(run, calls))

    def execute_tool(
        self,
        name: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        token_spans: Sequence[TokenSpan | Mapping[str, int]] | None = None,
        turn_id: str | None = None,
    ) -> dict[str, Any]:
        arguments = dict(arguments or {})
        arguments["token_spans"] = token_spans
        arguments["turn_id"] = turn_id
        try:
            method = getattr(self, name)
        except AttributeError as exc:
            raise ValueError(f"unknown ESR tool: {name}") from exc
        return method(**arguments)

    def render_context(self) -> dict[str, Any]:
        state = self.current_state
        evidence = self.store.list_evidence()
        return {
            "question": self.question,
            "task_state": to_primitive(state) if state else None,
            "evidence_archive_count": len(evidence),
            "unindexed_evidence_ids": self._missing_directory_ids(state),
            "evidence_sources": [
                {"evidence_id": item.evidence_id, "source": to_primitive(item.source)} for item in evidence
            ],
            "allowed_actions": [item.value for item in ActionKind],
        }

    def rebuild_context(self) -> dict[str, Any]:
        self._visible_evidence_inputs.clear()
        return self.render_context()

    def snapshot(self) -> dict[str, Any]:
        return self.store.snapshot()

    def _make_action(
        self,
        kind: ActionKind,
        spans: Sequence[TokenSpan | Mapping[str, int]] | None,
        *,
        turn_id: str | None,
        legal: bool = True,
        parent_action_ids: Sequence[str] = (),
        referenced_evidence_ids: Sequence[str] = (),
        created_evidence_ids: Sequence[str] = (),
        active_gap_ids: Sequence[str] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> ActionRecord:
        action_id, sequence_index = self._next_action_identity()
        return ActionRecord(
            action_id=action_id,
            sequence_index=sequence_index,
            turn_id=turn_id,
            kind=kind,
            token_spans=normalize_spans(spans),
            legal=legal,
            state_version_before=self._state_version(),
            parent_action_ids=tuple(parent_action_ids),
            referenced_evidence_ids=tuple(referenced_evidence_ids),
            created_evidence_ids=tuple(created_evidence_ids),
            active_gap_ids=tuple(active_gap_ids),
            metadata=dict(metadata or {}),
        )

    def _reject(
        self,
        kind: ActionKind,
        message: str,
        spans: Sequence[TokenSpan | Mapping[str, int]] | None,
        turn_id: str | None,
    ) -> Any:
        action = self._make_action(kind, spans, turn_id=turn_id, legal=False, metadata={"error": message})
        self.store.add_action(action)
        raise IllegalActionError(message, action.action_id)

    def _next_action_identity(self) -> tuple[str, int]:
        index = len(self.store.list_actions())
        return f"a{index + 1}", index

    def _state_version(self) -> int | None:
        state = self.current_state
        return state.version if state else None

    def _active_gap_ids(self) -> tuple[str, ...]:
        state = self.current_state
        return tuple(item.gap_id for item in state.gaps) if state else ()

    def _missing_directory_ids(self, state: TaskState | None) -> list[str]:
        directory = set(state.directory_map()) if state else set()
        return [item.evidence_id for item in self.store.list_evidence() if item.evidence_id not in directory]

    def _verification_error(self, state: TaskState | None) -> str | None:
        if state is None:
            return "TaskState does not exist"
        if self._missing_directory_ids(state):
            return "latest Evidence has not been written to evidence_directory"
        if not state.answer:
            return "current answer is empty"
        if not state.supporting_evidence:
            return "current answer has no supporting Evidence"
        if state.verification_status is not VerificationStatus.UNVERIFIED:
            return "current TaskState has already been verified; call update_state before verifying again"
        return None

    def _submission_error(self, state: TaskState | None) -> str | None:
        if self.is_submitted:
            return "episode has already been submitted"
        if state is None:
            return "TaskState does not exist"
        if self._missing_directory_ids(state):
            return "latest Evidence has not been written to evidence_directory"
        if not state.answer:
            return "current answer is empty"
        if not state.supporting_evidence:
            return "current answer has no supporting Evidence"
        if state.gaps:
            return "current answer has unresolved gaps"
        if state.verification_status is not VerificationStatus.SUPPORTED:
            return "current answer has not passed verify_answer"
        return None


ESR_TOOL_SCHEMAS: tuple[dict[str, Any], ...] = (
    {"name": "read_evidence", "parameters": {"evidence_id": "string"}},
    {
        "name": "update_state",
        "parameters": {
            "answer": "string",
            "evidence_findings": "array[{evidence_id,finding}]",
            "supporting_evidence": "array[string]",
        },
    },
    {"name": "verify_answer", "parameters": {}},
    {"name": "submit_answer", "parameters": {}},
)
