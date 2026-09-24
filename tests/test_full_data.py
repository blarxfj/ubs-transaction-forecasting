from __future__ import annotations

import os
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

from ubs_forecasting.data import TRANSACTION_SPLITS
from ubs_forecasting.vocab import description_evidence


@pytest.mark.full_data
def test_parser_covers_all_but_24_transaction_rows() -> None:
    data_value = os.environ.get("UBS_DATA_DIR")
    zip_value = os.environ.get("UBS_DATA_ZIP")
    if not data_value and not zip_value:
        pytest.skip("set UBS_DATA_DIR or UBS_DATA_ZIP to run full-data grammar coverage")
    counts: Counter[str] = Counter()
    for split in TRANSACTION_SPLITS:
        name = f"{split}_transactions.jsonl"
        if zip_value:
            with zipfile.ZipFile(zip_value) as archive:
                members = [member for member in archive.namelist() if Path(member).name == name]
                assert len(members) == 1
                with archive.open(members[0]) as handle:
                    descriptions = pd.read_json(handle, lines=True).description
        else:
            descriptions = pd.read_json(Path(data_value) / name, lines=True).description
        counts.update(descriptions)
    unknown_rows = sum(
        count for description, count in counts.items() if description_evidence(description)[0] == "unknown"
    )
    assert len(counts) == 2899
    assert unknown_rows == 24
