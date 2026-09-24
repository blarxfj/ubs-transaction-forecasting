"""Immutable hashes for the verified Appendix B reference implementation."""

from __future__ import annotations

from pathlib import Path

from .data import sha256_file

BASELINE_SCRIPT_SHA256 = "d7e28439a4a4996921df1add8a57c18f5eafc85d18d840fe9f58d250bd8c278d"
BASELINE_SUBMISSION_SHA256 = "c977ba7b3c52d3f0273081f4d5869adc81e66ee84191b1d07b43e7354e37bdfb"
BASELINE_PROBABILITIES_SHA256 = "a0acd1079e8cc07d4eda76652ac4d67e0fad8b018a025c7cccf2ae7ac1513baa"


def verify_reference_script(path: str | Path) -> None:
    """Verify that the retained monolith is byte-identical to Appendix B."""

    actual = sha256_file(path)
    if actual != BASELINE_SCRIPT_SHA256:
        raise ValueError(f"reference script hash mismatch: {actual}")


def verify_reference_submission(path: str | Path) -> None:
    """Verify a submission produced by the retained Appendix B baseline."""

    actual = sha256_file(path)
    if actual != BASELINE_SUBMISSION_SHA256:
        raise ValueError(f"reference submission hash mismatch: {actual}")
