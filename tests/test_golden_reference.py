from __future__ import annotations

from pathlib import Path

from ubs_forecasting.reference import (
    BASELINE_PROBABILITIES_SHA256,
    BASELINE_SUBMISSION_SHA256,
    verify_reference_script,
)


def test_appendix_b_script_is_immutable_golden_fixture() -> None:
    verify_reference_script(Path(__file__).parents[1] / "ubs_baseline.py")
    assert BASELINE_SUBMISSION_SHA256 == ("c977ba7b3c52d3f0273081f4d5869adc81e66ee84191b1d07b43e7354e37bdfb")
    assert BASELINE_PROBABILITIES_SHA256 == ("a0acd1079e8cc07d4eda76652ac4d67e0fad8b018a025c7cccf2ae7ac1513baa")
