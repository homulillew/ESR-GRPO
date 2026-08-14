from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clients import BCPlusRetrievalClient, CorpusURLIndex


class RetrievalClientTests(unittest.TestCase):
    def test_echo_document_contents_are_preserved(self):
        responses = [
            {
                "result": [
                    [
                        {
                            "docid": "d1",
                            "score": 2.5,
                            "document": {"contents": "Title\nFull snippet"},
                        }
                    ]
                ]
            },
            {"docid": "d1", "document": {"contents": "Title\nComplete page"}},
        ]
        with patch("clients._request_json", side_effect=responses):
            client = BCPlusRetrievalClient(
                "http://retrieval", CorpusURLIndex({"d1": "https://example.test/d1"})
            )
            searched = client.search(["query"])
            self.assertEqual(searched[0]["results"][0]["snippet"], "Title\nFull snippet")
            visited = client.visit(["https://example.test/d1"], "verify")
            self.assertEqual(visited[0]["content"], "Title\nComplete page")
            self.assertEqual(client.retrieved_docids, {"d1"})
            self.assertEqual(client.visited_docids, {"d1"})

    def test_unsearched_url_is_rejected(self):
        client = BCPlusRetrievalClient("http://retrieval")
        with self.assertRaises(ValueError):
            client.resolve_docid("https://outside.test")


if __name__ == "__main__":
    unittest.main()

