from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluate import calibration_error, parse_judgement


class EvaluateTests(unittest.TestCase):
    def test_official_style_judge_output(self):
        result = parse_judgement(
            "extracted_final_answer: Vector Labs\n"
            "[correct_answer]: Vector Labs\n"
            "reasoning: semantically equivalent\n"
            "correct: yes\n"
            "confidence: 95%\n"
        )
        self.assertFalse(result["parse_error"])
        self.assertTrue(result["correct"])
        self.assertEqual(result["confidence"], 95.0)

    def test_calibration_uses_final_partial_bin(self):
        value = calibration_error([1.0, 0.0, 1.0], [True, False, False], bin_size=2)
        self.assertGreater(value, 0.0)


if __name__ == "__main__":
    unittest.main()

