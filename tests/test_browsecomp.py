from __future__ import annotations

import pytest

from esr_grpo.browsecomp import calibration_error, make_split_manifest, validate_split_manifest


def test_calibration_uses_the_last_bin() -> None:
    confidence = [1.0] * 100
    correct = [False] * 100
    assert calibration_error(confidence, correct) == pytest.approx(1.0)


def test_calibration_handles_partial_last_bin() -> None:
    confidence = [0.0] * 100 + [1.0] * 30
    correct = [False] * 130
    assert calibration_error(confidence, correct) > 0


def test_custom_split_is_deterministic_disjoint_and_complete() -> None:
    query_ids = [str(index) for index in range(20)]
    first = make_split_manifest(query_ids, seed=7)
    second = make_split_manifest(query_ids, seed=7)
    assert first == second
    validate_split_manifest(first, query_ids)
