from __future__ import annotations

from esr_grpo.integrations.echo import extract_hermes_action_spans


class CharacterTokenizer:
    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(item) for item in ids)

    def encode(self, text, add_special_tokens=False):
        return [ord(item) for item in text]

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        return {"offset_mapping": [(index, index + 1) for index in range(len(text))]}


def test_parallel_hermes_calls_get_independent_spans() -> None:
    text = (
        'think<tool_call>{"name":"open_page","arguments":{"docid":"d1"}}</tool_call>'
        '<tool_call>{"name":"open_page","arguments":{"docid":"d2"}}</tool_call>'
    )
    spans = extract_hermes_action_spans(
        CharacterTokenizer(),
        [ord(item) for item in text],
        segment_index=2,
        segment_offset=10,
    )
    assert len(spans) == 2
    assert spans[0]["segment_index"] == spans[1]["segment_index"] == 2
    assert spans[0]["end"] <= spans[1]["start"]
    assert spans[0]["start"] >= 10
