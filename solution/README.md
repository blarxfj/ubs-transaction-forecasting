# Next recurring merchant family - solution code

Predicts, for each client, the merchant family (cloud, gym, insurance, mobile, music,
software, streaming) whose recurring payment comes next after the 2026-01-01 cutoff, or
`none`. See `../report.md` for the approach, findings and scores.

## Layout

| file | role |
| --- | --- |
| `streams.py` | data loading, merchant-name vocabulary, family keyword mapping |
| `features.py` | family assignment of recurring events and per-family stream features |
| `pairwise.py` | wide-to-long conversion (one row per client x family) and the CV harness for the stage-1 model |
| `stage2.py` | logistic calibration of the 7 family scores into 8-class probabilities |
| `protocol.py` | lockbox / fold hashing, macro-F1, bootstrap |
| `train.py` | end-to-end run: CV under the protocol, final bagged model, deliverables |
| `results/` | `oof.csv`, `lockbox.csv`, `test_proba.csv`, `submission.csv`, `results.json`, `feature_importance.csv` |

## Reproduce

Python 3.12. From a directory that contains the unpacked `data/` folder
(`train_transactions.jsonl`, `valid_transactions.jsonl`, `test_transactions.jsonl`,
`train_labels.csv`, `valid_labels.csv`, `sample_submission.csv`):

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r solution/requirements.txt
# or: uv venv -p 3.12 .venv && uv pip install -p .venv/bin/python -r solution/requirements.txt
.venv/bin/python solution/train.py --data data --out results --rounds 450 --bag 5
```

Runtime is about 5 minutes on an 8-core laptop (feature building under a minute, the
rest is the 3-seed x 5-fold protocol CV plus the bagged final model). `--skip_cv` produces
only the final model and deliverables in about 30 seconds. The run is deterministic: the same command yields a
byte-identical `submission.csv` (its SHA-256 is recorded in `results.json`).

Optional experiments that are not part of the final model (kept for reference):
`cv.py` (direct 8-class model), `pseudo.py` + `pseudo_model.py` (pseudo-cutoff
self-supervision on the unlabeled histories).
