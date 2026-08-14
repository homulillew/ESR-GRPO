from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from protocol import ProtocolError, normalize_call, parse_tool_call


class ProtocolTests(unittest.TestCase):
    def test_search_array(self):
        call = normalize_call(
            parse_tool_call(
                "<think>plan</think><tool_call><function=search>"
                '<parameter=query>["alpha", "beta"]</parameter>'
                "</function></tool_call>"
            )
        )
        self.assertEqual(call.name, "search")
        self.assertEqual(call.arguments["query"], ["alpha", "beta"])

    def test_visit_string_is_normalized(self):
        call = normalize_call(
            parse_tool_call(
                "<tool_call><function=visit><parameter=url>bcplus://document/d1</parameter>"
                "<parameter=goal>verify author</parameter></function></tool_call>"
            )
        )
        self.assertEqual(call.arguments["url"], ["bcplus://document/d1"])

    def test_finish_schema(self):
        call = normalize_call(
            parse_tool_call(
                "<tool_call><function=finish><parameter=answer>Vector Labs</parameter>"
                '<parameter=evidences>[{"evidence":"fact","url":"bcplus://document/d1"}]</parameter>'
                "<parameter=confidence>92%</parameter></function></tool_call>"
            )
        )
        self.assertEqual(call.arguments["confidence"], "92%")

    def test_multiple_calls_are_rejected(self):
        text = (
            "<tool_call><function=search><parameter=query>[\"a\"]</parameter></function></tool_call>"
            "<tool_call><function=search><parameter=query>[\"b\"]</parameter></function></tool_call>"
        )
        with self.assertRaises(ProtocolError):
            parse_tool_call(text)

    def test_google_scholar_is_rejected_for_fixed_corpus(self):
        call = parse_tool_call(
            "<tool_call><function=google_scholar><parameter=query>[\"a\"]</parameter>"
            "</function></tool_call>"
        )
        with self.assertRaises(ProtocolError):
            normalize_call(call)


if __name__ == "__main__":
    unittest.main()

