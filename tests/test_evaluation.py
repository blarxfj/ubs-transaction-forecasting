from __future__ import annotations

import pytest

from ubs_forecasting.evaluation import validate_leak_regression


def test_expected_masking_leak_contrast_passes() -> None:
    validate_leak_regression(0.36, 0.63)


def test_leak_safe_model_regression_fails() -> None:
    with pytest.raises(RuntimeError, match="leak-safe"):
        validate_leak_regression(0.36, 0.50)


def test_missing_leak_contrast_fails() -> None:
    with pytest.raises(RuntimeError, match="contrast"):
        validate_leak_regression(0.55, 0.63)
