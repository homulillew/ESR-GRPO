"""单 episode 的 SQLite 追加式存储。"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Mapping

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
    to_primitive,
)


class EpisodeStore:
    """保存 Evidence、TaskState 版本和动作事件。

    所有研究对象只允许 INSERT。触发器会拒绝 UPDATE/DELETE，从数据库层保证
    Evidence 和历史状态不会被训练或调试代码意外覆盖。
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        # 并行只发生在只读检索/页面动作层。环境使用同一把 RLock 串行提交，
        # check_same_thread=False 允许这些逻辑动作由不同 worker thread 发起。
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_schema()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "EpisodeStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _create_schema(self) -> None:
        self._conn.executescript(
            """
            PRAGMA foreign_keys = ON;
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS evidence (
                evidence_id TEXT PRIMARY KEY,
                source_key TEXT NOT NULL,
                content_sha256 TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                UNIQUE(source_key, content_sha256)
            );
            CREATE TABLE IF NOT EXISTS task_states (
                version INTEGER PRIMARY KEY,
                payload_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS actions (
                action_id TEXT PRIMARY KEY,
                sequence_index INTEGER NOT NULL UNIQUE,
                payload_json TEXT NOT NULL
            );
            CREATE TRIGGER IF NOT EXISTS evidence_no_update
            BEFORE UPDATE ON evidence BEGIN SELECT RAISE(ABORT, 'Evidence is immutable'); END;
            CREATE TRIGGER IF NOT EXISTS evidence_no_delete
            BEFORE DELETE ON evidence BEGIN SELECT RAISE(ABORT, 'Evidence is immutable'); END;
            CREATE TRIGGER IF NOT EXISTS states_no_update
            BEFORE UPDATE ON task_states BEGIN SELECT RAISE(ABORT, 'TaskState is append-only'); END;
            CREATE TRIGGER IF NOT EXISTS states_no_delete
            BEFORE DELETE ON task_states BEGIN SELECT RAISE(ABORT, 'TaskState is append-only'); END;
            CREATE TRIGGER IF NOT EXISTS actions_no_update
            BEFORE UPDATE ON actions BEGIN SELECT RAISE(ABORT, 'ActionRecord is append-only'); END;
            CREATE TRIGGER IF NOT EXISTS actions_no_delete
            BEFORE DELETE ON actions BEGIN SELECT RAISE(ABORT, 'ActionRecord is append-only'); END;
            """
        )
        self._conn.commit()

    def set_metadata_once(self, key: str, value: Any) -> None:
        self._conn.execute(
            "INSERT INTO metadata(key, value_json) VALUES (?, ?)",
            (key, json.dumps(to_primitive(value), ensure_ascii=False, sort_keys=True)),
        )
        self._conn.commit()

    def get_metadata(self, key: str, default: Any = None) -> Any:
        row = self._conn.execute("SELECT value_json FROM metadata WHERE key = ?", (key,)).fetchone()
        return default if row is None else json.loads(row["value_json"])

    def add_evidence(self, evidence: Evidence) -> None:
        self._conn.execute(
            "INSERT INTO evidence(evidence_id, source_key, content_sha256, payload_json) VALUES (?, ?, ?, ?)",
            (
                evidence.evidence_id,
                evidence.source.key,
                evidence.content_sha256,
                json.dumps(to_primitive(evidence), ensure_ascii=False, sort_keys=True),
            ),
        )
        self._conn.commit()

    def find_evidence(self, source_key: str, content_sha256: str) -> Evidence | None:
        row = self._conn.execute(
            "SELECT payload_json FROM evidence WHERE source_key = ? AND content_sha256 = ?",
            (source_key, content_sha256),
        ).fetchone()
        return None if row is None else self._decode_evidence(json.loads(row["payload_json"]))

    def get_evidence(self, evidence_id: str) -> Evidence:
        row = self._conn.execute(
            "SELECT payload_json FROM evidence WHERE evidence_id = ?", (evidence_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown Evidence ID: {evidence_id}")
        return self._decode_evidence(json.loads(row["payload_json"]))

    def list_evidence(self) -> tuple[Evidence, ...]:
        rows = self._conn.execute("SELECT payload_json FROM evidence ORDER BY rowid").fetchall()
        return tuple(self._decode_evidence(json.loads(row["payload_json"])) for row in rows)

    def add_state(self, state: TaskState) -> None:
        self._conn.execute(
            "INSERT INTO task_states(version, payload_json) VALUES (?, ?)",
            (state.version, json.dumps(to_primitive(state), ensure_ascii=False, sort_keys=True)),
        )
        self._conn.commit()

    def get_state(self, version: int) -> TaskState:
        row = self._conn.execute(
            "SELECT payload_json FROM task_states WHERE version = ?", (version,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown TaskState version: {version}")
        return self._decode_state(json.loads(row["payload_json"]))

    def latest_state(self) -> TaskState | None:
        row = self._conn.execute(
            "SELECT payload_json FROM task_states ORDER BY version DESC LIMIT 1"
        ).fetchone()
        return None if row is None else self._decode_state(json.loads(row["payload_json"]))

    def list_states(self) -> tuple[TaskState, ...]:
        rows = self._conn.execute("SELECT payload_json FROM task_states ORDER BY version").fetchall()
        return tuple(self._decode_state(json.loads(row["payload_json"])) for row in rows)

    def add_action(self, action: ActionRecord) -> None:
        self._conn.execute(
            "INSERT INTO actions(action_id, sequence_index, payload_json) VALUES (?, ?, ?)",
            (
                action.action_id,
                action.sequence_index,
                json.dumps(to_primitive(action), ensure_ascii=False, sort_keys=True),
            ),
        )
        self._conn.commit()

    def commit_transition(
        self,
        action: ActionRecord,
        *,
        evidence: Evidence | None = None,
        state: TaskState | None = None,
        metadata_once: Mapping[str, Any] | None = None,
    ) -> None:
        """原子追加一个动作及其产生的 Evidence、TaskState 或提交元数据。"""

        with self._conn:
            self._conn.execute(
                "INSERT INTO actions(action_id, sequence_index, payload_json) VALUES (?, ?, ?)",
                (
                    action.action_id,
                    action.sequence_index,
                    json.dumps(to_primitive(action), ensure_ascii=False, sort_keys=True),
                ),
            )
            if evidence is not None:
                self._conn.execute(
                    "INSERT INTO evidence(evidence_id, source_key, content_sha256, payload_json) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        evidence.evidence_id,
                        evidence.source.key,
                        evidence.content_sha256,
                        json.dumps(to_primitive(evidence), ensure_ascii=False, sort_keys=True),
                    ),
                )
            if state is not None:
                self._conn.execute(
                    "INSERT INTO task_states(version, payload_json) VALUES (?, ?)",
                    (state.version, json.dumps(to_primitive(state), ensure_ascii=False, sort_keys=True)),
                )
            for key, value in (metadata_once or {}).items():
                self._conn.execute(
                    "INSERT INTO metadata(key, value_json) VALUES (?, ?)",
                    (key, json.dumps(to_primitive(value), ensure_ascii=False, sort_keys=True)),
                )

    def get_action(self, action_id: str) -> ActionRecord:
        row = self._conn.execute(
            "SELECT payload_json FROM actions WHERE action_id = ?", (action_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown action ID: {action_id}")
        return self._decode_action(json.loads(row["payload_json"]))

    def list_actions(self) -> tuple[ActionRecord, ...]:
        rows = self._conn.execute("SELECT payload_json FROM actions ORDER BY sequence_index").fetchall()
        return tuple(self._decode_action(json.loads(row["payload_json"])) for row in rows)

    def snapshot(self) -> dict[str, Any]:
        return {
            "episode_id": self.get_metadata("episode_id"),
            "question": self.get_metadata("question"),
            "submitted_answer": self.get_metadata("submitted_answer"),
            "submit_action_id": self.get_metadata("submit_action_id"),
            "evidence": [to_primitive(item) for item in self.list_evidence()],
            "states": [to_primitive(item) for item in self.list_states()],
            "actions": [to_primitive(item) for item in self.list_actions()],
        }

    @staticmethod
    def _decode_evidence(data: dict[str, Any]) -> Evidence:
        return Evidence(
            evidence_id=data["evidence_id"],
            source=EvidenceSource(**data["source"]),
            content=data["content"],
            content_sha256=data["content_sha256"],
            created_by_action_id=data["created_by_action_id"],
            search_action_id=data.get("search_action_id"),
            created_at=data["created_at"],
        )

    @staticmethod
    def _decode_state(data: dict[str, Any]) -> TaskState:
        return TaskState(
            version=int(data["version"]),
            parent_version=data.get("parent_version"),
            evidence_directory=tuple(EvidenceFinding(**item) for item in data["evidence_directory"]),
            answer=data["answer"],
            supporting_evidence=tuple(data["supporting_evidence"]),
            gaps=tuple(Gap(**item) for item in data["gaps"]),
            verification_status=VerificationStatus(data["verification_status"]),
            created_by_action_id=data["created_by_action_id"],
        )

    @staticmethod
    def _decode_action(data: dict[str, Any]) -> ActionRecord:
        return ActionRecord(
            action_id=data["action_id"],
            sequence_index=int(data["sequence_index"]),
            turn_id=data.get("turn_id"),
            kind=ActionKind(data["kind"]),
            token_spans=tuple(TokenSpan(**item) for item in data["token_spans"]),
            legal=bool(data["legal"]),
            state_version_before=data.get("state_version_before"),
            state_version_after=data.get("state_version_after"),
            parent_action_ids=tuple(data.get("parent_action_ids", [])),
            referenced_evidence_ids=tuple(data.get("referenced_evidence_ids", [])),
            created_evidence_ids=tuple(data.get("created_evidence_ids", [])),
            active_gap_ids=tuple(data.get("active_gap_ids", [])),
            metadata=data.get("metadata", {}),
        )
