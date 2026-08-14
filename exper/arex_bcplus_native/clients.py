"""AREX 模型服务和 ECHO BC-Plus 检索服务客户端。"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _request_json(
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    api_key: str | None = None,
    timeout: float = 120.0,
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json", "User-Agent": "arex-bcplus-native/1.0"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return _request_json_with_headers(url, payload=payload, headers=headers, timeout=timeout)


def _request_json_with_headers(
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str],
    timeout: float = 120.0,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="GET" if payload is None else "POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body[:2000]}") from exc


@dataclass
class AREXChatClient:
    base_url: str
    model: str = "AREX-Turbo"
    api_key: str = "EMPTY"
    timeout_seconds: float = 600.0
    max_tokens: int = 8192
    temperature: float = 1.0
    top_p: float = 0.95
    presence_penalty: float = 1.5
    top_k: int = 20

    def models(self) -> dict[str, Any]:
        return _request_json(
            f"{self.base_url.rstrip('/')}/models",
            api_key=self.api_key,
            timeout=30.0,
        )

    def complete(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "presence_penalty": self.presence_penalty,
            "extra_body": {"top_k": self.top_k},
        }
        # vLLM 接受 top_k 作为顶层扩展字段，而不是 OpenAI SDK 的 extra_body 包装。
        payload["top_k"] = payload.pop("extra_body")["top_k"]
        return _request_json(
            f"{self.base_url.rstrip('/')}/chat/completions",
            payload=payload,
            api_key=self.api_key,
            timeout=self.timeout_seconds,
        )


@dataclass
class CorpusURLIndex:
    """只加载 corpus parquet 的 docid 和 url 两列，不读取大正文。"""

    values: dict[str, str]

    @classmethod
    def from_parquet(cls, path: str | Path | None) -> "CorpusURLIndex":
        if not path:
            return cls({})
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("--corpus-file requires pyarrow; install requirements.txt") from exc
        table = pq.read_table(path, columns=["docid", "url"])
        docids = table.column("docid").to_pylist()
        urls = table.column("url").to_pylist()
        return cls({str(docid): str(url or "") for docid, url in zip(docids, urls)})

    def url_for(self, docid: str) -> str:
        return self.values.get(docid) or f"bcplus://document/{urllib.parse.quote(docid, safe='')}"


@dataclass
class BCPlusRetrievalClient:
    base_url: str
    corpus_urls: CorpusURLIndex = field(default_factory=lambda: CorpusURLIndex({}))
    top_k: int = 10
    snippet_chars: int = 4000
    visit_chars: int = 24000
    timeout_seconds: float = 120.0
    url_to_docid: dict[str, str] = field(default_factory=dict)
    retrieved_docids: set[str] = field(default_factory=set)
    visited_docids: set[str] = field(default_factory=set)

    def health(self) -> dict[str, Any]:
        return _request_json(f"{self.base_url.rstrip('/')}/health", timeout=30.0)

    def search(self, queries: list[str]) -> list[dict[str, Any]]:
        payload = _request_json(
            f"{self.base_url.rstrip('/')}/retrieve",
            payload={"queries": queries, "topk": self.top_k},
            timeout=self.timeout_seconds,
        )
        groups = payload.get("result", payload.get("results", []))
        if groups and isinstance(groups, list) and groups and isinstance(groups[0], dict):
            groups = [groups]
        output: list[dict[str, Any]] = []
        for query, rows in zip(queries, groups):
            hits: list[dict[str, Any]] = []
            for row in rows or []:
                docid = str(row.get("docid", row.get("id", "")))
                document = row.get("document") if isinstance(row.get("document"), dict) else {}
                contents = str(
                    row.get("content")
                    or row.get("text")
                    or document.get("contents")
                    or document.get("text")
                    or ""
                )
                title = contents.splitlines()[0].strip() if contents else ""
                url = str(row.get("url") or self.corpus_urls.url_for(docid))
                self.url_to_docid[url] = docid
                self.retrieved_docids.add(docid)
                hits.append(
                    {
                        "docid": docid,
                        "url": url,
                        "title": title,
                        "score": float(row.get("score", 0.0)),
                        "snippet": contents[: self.snippet_chars],
                    }
                )
            output.append({"query": query, "results": hits})
        return output

    def visit(self, urls: list[str], goal: str) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for url in urls:
            docid = self.resolve_docid(str(url))
            row = _request_json(
                f"{self.base_url.rstrip('/')}/get_doc",
                payload={"docid": docid},
                timeout=self.timeout_seconds,
            )
            if row.get("error"):
                results.append({"url": url, "docid": docid, "error": row["error"]})
                continue
            document = row.get("document") if isinstance(row.get("document"), dict) else {}
            contents = str(
                row.get("content")
                or row.get("text")
                or document.get("contents")
                or document.get("text")
                or ""
            )
            self.visited_docids.add(docid)
            results.append(
                {
                    "url": self.corpus_urls.url_for(docid),
                    "docid": docid,
                    "goal": goal,
                    "content": contents[: self.visit_chars],
                    "truncated": len(contents) > self.visit_chars,
                    "original_chars": len(contents),
                }
            )
        return results

    def resolve_docid(self, url: str) -> str:
        if url in self.url_to_docid:
            return self.url_to_docid[url]
        prefix = "bcplus://document/"
        if url.startswith(prefix):
            return urllib.parse.unquote(url[len(prefix) :])
        # 模型偶尔直接传回 docid；仅允许已经由 search 返回过的 ID。
        if url in self.retrieved_docids:
            return url
        for docid, real_url in self.corpus_urls.values.items():
            if real_url == url and docid in self.retrieved_docids:
                return docid
        raise ValueError(f"visit URL was not returned by search: {url}")

