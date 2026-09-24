from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

from ubs_forecasting.data import TRANSACTION_SPLITS
from ubs_forecasting.vocab import description_evidence


@pytest.mark.full_data
def test_parser_covers_all_but_24_transaction_rows() -> None:
    data_value = os.environ.get("UBS_DATA_DIR")
    if not data_value:
        pytest.skip("set UBS_DATA_DIR to run full-data grammar coverage")
    data_dir = Path(data_value)
    counts: Counter[str] = Counter()
    for split in TRANSACTION_SPLITS:
        descriptions = pd.read_json(data_dir / f"{split}_transactions.jsonl", lines=True).description
        counts.update(descriptions)
    unknown_rows = sum(
        count for description, count in counts.items() if description_evidence(description)[0] == "unknown"
    )
    assert len(counts) == 2899
    assert unknown_rows == 24
