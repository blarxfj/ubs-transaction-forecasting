# Deliverables of the final protocol run

Written by `uv run ubs-forecast ensemble --data "$UBS_DATA_DIR" --output <dir> --jobs 4` at the
commit recorded in `metrics.json` (`git_revision`), with the frozen recipe in
`src/ubs_forecasting/ensemble.py`. Regenerate rather than edit.

| File | Rows | Content |
| --- | ---: | --- |
| `oof.csv` | 2,791 | seed-0 out-of-fold class probabilities for every development client, with the hashed fold |
| `oof_seed0.csv`, `oof_seed1.csv`, `oof_seed2.csv` | 2,791 each | the same for each repetition seed |
| `lockbox.csv` | 209 | final-model probabilities for the valid-only lockbox clients |
| `test_proba.csv` | 1,000 | final-model probabilities for the test clients, in sample-submission order |
| `submission.csv` | 1,000 | contract-valid submission (`client_id,predicted_next_recurring_merchant`) |
| `metrics.json` | | protocol scores with bootstrap intervals, per-class F1, recipe, data hashes |

Two independent runs (the second under a different `PYTHONHASHSEED`) produced byte-identical
`submission.csv` with SHA-256:

```text
092cdd3f1e56c439ca5a0e70d8d212b1feb1a36f3a3ff53a75c136b79fe0d345
```

Predicted label counts: none 253, software 126, gym 124, cloud 110, mobile 109, insurance 100,
streaming 92, music 86.
