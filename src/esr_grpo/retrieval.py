"""固定语料检索接口及不依赖第三方库的调试实现。"""

from __future__ import annotations

import json
import math
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

from .models import RetrievedDocument, SearchHit


class Retriever(Protocol):
    def search(self, query: str, top_k: int = 5) -> list[SearchHit]: ...

    def get_document(self, docid: str) -> RetrievedDocument: ...


@dataclass
class InMemoryRetriever:
    """用于测试和 CPU 冒烟实验的确定性 BM25 风格检索器。"""

    documents: dict[str, RetrievedDocument]

    @classmethod
    def from_documents(cls, documents: Iterable[RetrievedDocument]) -> "InMemoryRetriever":
        return cls({item.docid: item for item in documents})

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "InMemoryRetriever":
        documents: list[RetrievedDocument] = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                documents.append(
                    RetrievedDocument(
                        docid=str(row["docid"]),
                        content=str(row.get("content", row.get("text", row.get("contents", "")))),
                        title=str(row.get("title", "")),
                        url=str(row.get("url", "")),
                    )
                )
        return cls.from_documents(documents)

    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        query_tokens = _tokens(query)
        ranked: list[tuple[float, RetrievedDocument]] = []
        for document in self.documents.values():
            document_tokens = _tokens(f"{document.title} {document.content}")
            counts = {token: document_tokens.count(token) for token in query_tokens}
            score = sum((1.0 + math.log(count)) if count else 0.0 for count in counts.values())
            if score > 0:
                ranked.append((score, document))
        ranked.sort(key=lambda item: (-item[0], item[1].docid))
        return [
            SearchHit(
                docid=document.docid,
                snippet=document.content[:1600],
                score=score,
                title=document.title,
                url=document.url,
            )
            for score, document in ranked[:top_k]
        ]

    def get_document(self, docid: str) -> RetrievedDocument:
        try:
            return self.documents[docid]
        except KeyError as exc:
            raise KeyError(f"unknown document: {docid}") from exc


@dataclass
class EchoRetrievalClient:
    """调用 ECHO BrowseComp 检索服务的轻量客户端。"""

    base_url: str = "http://127.0.0.1:8000"
    timeout_seconds: float = 60.0

    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        payload = self._post("/retrieve", {"queries": [query], "topk": top_k})
        rows = payload.get("result", payload.get("results", payload))
        if rows and isinstance(rows[0], list):
            rows = rows[0]
        return [
            SearchHit(
                docid=str(row.get("docid", row.get("id", ""))),
                snippet=str(row.get("content", row.get("text", "")))[:1600],
                score=float(row.get("score", 0.0)),
                title=str(row.get("title", "")),
                url=str(row.get("url", "")),
            )
            for row in rows
        ]

    def get_document(self, docid: str) -> RetrievedDocument:
        row = self._post("/get_doc", {"docid": docid})
        if isinstance(row.get("result"), dict):
            row = row["result"]
        return RetrievedDocument(
            docid=str(row.get("docid", docid)),
            content=str(row.get("content", row.get("text", ""))),
            title=str(row.get("title", "")),
            url=str(row.get("url", "")),
        )

    def _post(self, endpoint: str, payload: dict[str, object]) -> dict:
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}{endpoint}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))


def _tokens(text: str) -> list[str]:
    return re.findall(r"[\w\u4e00-\u9fff]+", text.lower())

