from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from ubs_forecasting.vocab import description_evidence, parse_description


@pytest.mark.parametrize(
    ("description", "bases", "stripped", "exact"),
    [
        ("cloud access", {"cloud access"}, 0, True),
        ("pay cloud access online", {"cloud access"}, 2, True),
        ("prem plan", {"premium plan"}, 0, True),
        ("access", {"cloud access", "software access", "video access"}, 0, False),
        # Stable prefix/suffix loop order resolves this grammar tie as in the verified baseline.
        ("billing digital", {"saas billing"}, 1, False),
        ("totally alien vendor", set(), 99, False),
    ],
)
def test_closed_grammar_parser(description: str, bases: set[str], stripped: int, exact: bool) -> None:
    parsed_bases, parsed_stripped, parsed_exact = parse_description(description)
    assert set(parsed_bases) == bases
    assert parsed_stripped == stripped
    assert parsed_exact is exact


def test_ambiguous_family_weights() -> None:
    kind, weights = description_evidence("digital plus")
    assert kind == "fam"
    assert weights == {"mobile": 1 / 3, "music": 1 / 3, "streaming": 1 / 3}

    kind, weights = description_evidence("premium plan")
    assert kind == "fam"
    assert weights == {"music": 1 / 3, "software": 1 / 3, "streaming": 1 / 3}


def test_parser_is_hash_seed_independent() -> None:
    script = """
import json
from ubs_forecasting.vocab import description_evidence
values = ['digital', 'billing digital', 'service', 'premium plan', 'access']
print(json.dumps([description_evidence(value) for value in values], sort_keys=True))
"""
    outputs = []
    for seed in ("1", "777"):
        environment = {**os.environ, "PYTHONHASHSEED": seed}
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        outputs.append(json.loads(result.stdout))
    assert outputs[0] == outputs[1]
