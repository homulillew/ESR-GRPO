"""ESR-GRPO 的稳定数据协议。

数据对象使用不可变 dataclass。SQLite 中的 Evidence、TaskState 和 ActionRecord
同样只允许追加，防止训练回溯时看到被覆盖的历史。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any, Mapping, Sequence


class ActionKind(str, Enum):
    SEARCH = "search"
    OPEN_PAGE = "open_page"
    READ_EVIDENCE = "read_evidence"
    UPDATE_STATE = "update_state"
    VERIFY_ANSWER = "verify_answer"
    SUBMIT_ANSWER = "submit_answer"


class VerificationStatus(str, Enum):
    UNVERIFIED = "unverified"
    NEEDS_REVISION = "needs_revision"
    SUPPORTED = "supported"


@dataclass(frozen=True)
class TokenSpan:
    """一次完整动作在某个轨迹分段中的左闭右开 token 范围。"""

    segment_index: int
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.segment_index < 0 or self.start < 0 or self.end <= self.start:
            raise ValueError(f"invalid token span: {self}")


@dataclass(frozen=True)
class EvidenceSource:
    docid: str
    url: str = ""
    title: str = ""

    @property
    def key(self) -> str:
        return self.docid or self.url


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    source: EvidenceSource
    content: str
    content_sha256: str
    created_by_action_id: str
    search_action_id: str | None
    created_at: str


@dataclass(frozen=True)
class EvidenceFinding:
    evidence_id: str
    finding: str


@dataclass(frozen=True)
class Gap:
    gap_id: str
    description: str
    created_by_action_id: str


@dataclass(frozen=True)
class TaskState:
    version: int
    parent_version: int | None
    evidence_directory: tuple[EvidenceFinding, ...]
    answer: str
    supporting_evidence: tuple[str, ...]
    gaps: tuple[Gap, ...]
    verification_status: VerificationStatus
    created_by_action_id: str

    def directory_map(self) -> dict[str, str]:
        return {item.evidence_id: item.finding for item in self.evidence_directory}

    def gap_map(self) -> dict[str, Gap]:
        return {item.gap_id: item for item in self.gaps}


@dataclass(frozen=True)
class ActionRecord:
    action_id: str
    sequence_index: int
    turn_id: str | None
    kind: ActionKind
    token_spans: tuple[TokenSpan, ...]
    legal: bool
    state_version_before: int | None = None
    state_version_after: int | None = None
    parent_action_ids: tuple[str, ...] = ()
    referenced_evidence_ids: tuple[str, ...] = ()
    created_evidence_ids: tuple[str, ...] = ()
    active_gap_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchHit:
    docid: str
    snippet: str
    score: float = 0.0
    title: str = ""
    url: str = ""


@dataclass(frozen=True)
class RetrievedDocument:
    docid: str
    content: str
    title: str = ""
    url: str = ""


@dataclass(frozen=True)
class VerificationResult:
    status: VerificationStatus
    gaps: tuple[str, ...]
    rationale: str = ""


def normalize_spans(values: Sequence[TokenSpan | Mapping[str, int]] | None) -> tuple[TokenSpan, ...]:
    if not values:
        return ()
    spans = tuple(value if isinstance(value, TokenSpan) else TokenSpan(**dict(value)) for value in values)
    for left, right in zip(spans, spans[1:]):
        if left.segment_index == right.segment_index and left.end > right.start:
            raise ValueError("token spans of one action must not overlap")
    return spans


def to_primitive(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {key: to_primitive(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): to_primitive(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_primitive(item) for item in value]
    return value

