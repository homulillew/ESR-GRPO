from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runner import AREXRunner


class FakeModel:
    model = "AREX-Turbo"
    temperature = 1.0
    top_p = 0.95
    top_k = 20
    max_tokens = 8192

    def __init__(self, outputs):
        self.outputs = iter(outputs)

    def complete(self, messages):
        content = next(self.outputs)
        return {
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }


class FakeRetrieval:
    def __init__(self):
        self.retrieved_docids = set()
        self.visited_docids = set()

    def search(self, queries):
        self.retrieved_docids.add("d1")
        return [{"query": queries[0], "results": [{"docid": "d1", "url": "bcplus://document/d1"}]}]

    def visit(self, urls, goal):
        self.visited_docids.add("d1")
        return [{"docid": "d1", "url": urls[0], "content": "Vector Labs developed it."}]


class RunnerTests(unittest.TestCase):
    def test_complete_native_episode(self):
        outputs = [
            '<tool_call><function=search><parameter=query>["orion developer"]</parameter></function></tool_call>',
            '<tool_call><function=visit><parameter=url>bcplus://document/d1</parameter><parameter=goal>identify developer</parameter></function></tool_call>',
            '<tool_call><function=finish><parameter=answer>Vector Labs</parameter>'
            '<parameter=evidences>[{"evidence":"developed it","url":"bcplus://document/d1"}]</parameter>'
            '<parameter=confidence>95%</parameter></function></tool_call>',
        ]
        trajectory, run = AREXRunner(FakeModel(outputs), FakeRetrieval(), max_turns=5).run(
            "q1", "Who developed it?"
        )
        self.assertEqual(trajectory["status"], "completed")
        self.assertEqual(trajectory["final"]["answer"], "Vector Labs")
        self.assertEqual(run["status"], "completed")
        self.assertEqual(run["retrieved_docids"], ["d1"])
        self.assertIn("Confidence: 95%", run["result"][-1]["output"])

    def test_protocol_failure_is_persistable(self):
        trajectory, run = AREXRunner(FakeModel(["plain answer"]), FakeRetrieval()).run(
            "q2", "question"
        )
        self.assertEqual(trajectory["status"], "failed")
        self.assertEqual(run["status"], "failed")
        self.assertEqual(trajectory["error"]["type"], "ProtocolError")


if __name__ == "__main__":
    unittest.main()

