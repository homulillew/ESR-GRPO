"""检查 search 返回的 docid 是否能由 open_page 服务读取完整正文。"""

from __future__ import annotations

import argparse
import json

from esr_grpo.retrieval import EchoRetrievalClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-url", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--open-count", type=int, default=3)
    args = parser.parse_args()
    client = EchoRetrievalClient(args.retrieval_url)
    hits = client.search(args.query, args.top_k)
    rows = []
    for hit in hits[: args.open_count]:
        document = client.get_document(hit.docid)
        rows.append(
            {
                "docid": hit.docid,
                "score": hit.score,
                "snippet_chars": len(hit.snippet),
                "document_chars": len(document.content),
                "roundtrip_docid_matches": document.docid == hit.docid,
                "document_nonempty": bool(document.content.strip()),
            }
        )
    report = {
        "query": args.query,
        "hits": len(hits),
        "opened": len(rows),
        "valid": bool(hits) and all(row["roundtrip_docid_matches"] and row["document_nonempty"] for row in rows),
        "documents": rows,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
