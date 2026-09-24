# UBS Transaction Activity Forecasting

A deterministic Python solution for predicting each client's next recurring merchant family after the `2026-01-01` cutoff. It detects recurring amount/currency streams, infers merchant-family evidence, and combines a shared family ranker with a client-level `none` gate.

The repository generates and validates a local `submission.csv`; it contains no upload or organiser-submission integration.

## Results

Repeated 5-fold client-held-out CV uses fold seeds 0 and 17. The test-noise stress result averages two independently corrupted views of the same valid clients.

| Evaluation view | Macro-F1 (95% client-bootstrap CI) | none-gate AUC |
| --- | ---: | ---: |
| Valid noise | **0.672** (0.639-0.700) | 0.903 |
| Test-level noise | **0.668** (0.637-0.697) | 0.902 |

The paired test-noise minus valid difference is -0.004 (95% CI -0.013 to +0.004). These are conditional estimates on a validation set used during development, not independent evidence or a hidden-test promise.

- [Full validation results and per-class F1](results/VALIDATION.md)
- [Parser-conditioned noise calibration](results/CALIBRATION.md)
- [Measured model explanations](results/INTERPRETABILITY.md)
- [Verified Appendix B reference](results/REFERENCE.md)

## Method

1. **Closed-grammar parser.** Normalize prefixes, suffixes, abbreviations, and one-token truncations into 47 supplied-corpus base descriptions. Ambiguous names retain fractional evidence across families.
2. **Recurrence detection.** Cluster card-payment candidates by currency and single-linkage log amount at 3.5%. Summarize cadence, jitter, amount stability, recency, and matched refunds.
3. **Stream posterior.** Combine description, MCC, clean amount-prior, and refund evidence into a seven-family posterior. Amount priors come from the unlabeled histories.
4. **Features.** Build one row per `(client, family)` plus client-level activity, recurrence, timing, currency, and refund aggregates.
5. **Shift repair.** Re-draw card-payment corruption for **every** train subscription-candidate row without using labels. Confident recurring streams use their inferred family; sparse or ambiguous candidates sample from their inferred posterior. Affix, mask, and MCC rates are calibrated against parser-derived test statistics.
6. **Two-part model.** A LightGBM binary ranker chooses among seven families for non-`none` clients. A separate wide LightGBM gate predicts `none`. Three deterministic model seeds are averaged; the final decision is untuned argmax.

The retained [`ubs_baseline.py`](ubs_baseline.py) is the byte-identical Appendix B reference. The modular package first reproduced its scores and deterministic submission hash before the augmentation hardening was added.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12. All direct and transitive dependencies are locked in `uv.lock`.

```bash
uv sync --locked
```

Keep challenge data outside the repository. Either unpack it explicitly:

```bash
export UBS_DATA_DIR=/absolute/path/outside/this/repository/ubs-data
DATA_DIR="$UBS_DATA_DIR" DATASET_ZIP=/path/to/dataset.zip make data
```

or unpack it yourself and point `--data` / `UBS_DATA_DIR` at the directory containing all seven challenge files. The dataset and generated artifacts are ignored by Git.

## Run

The `ubs-forecast` CLI is the single entry point:

```bash
# Two fold seeds, two corruption seeds, bootstrap CIs, and bounded ablations
uv run ubs-forecast evaluate --data "$UBS_DATA_DIR" --output artifacts/evaluation --jobs 7

# Train and persist the final ensemble
uv run ubs-forecast train --data "$UBS_DATA_DIR" --model artifacts/model.pkl --jobs 7

# Produce submission.csv, probabilities, and explanations
uv run ubs-forecast predict --data "$UBS_DATA_DIR" \
  --model artifacts/model.pkl --output artifacts/prediction --jobs 7

# Evaluate, train, and predict in one run
uv run ubs-forecast all --data "$UBS_DATA_DIR" --output artifacts/run --jobs 7

# Recompute corruption calibration statistics
uv run ubs-forecast calibrate --data "$UBS_DATA_DIR" \
  --output artifacts/calibration --jobs 7
```

Equivalent shortcuts are available through `make cv`, `make model`, `make submission`, and `make all`.

Evaluation retains every out-of-fold probability table under `OUTPUT/oof/`, plus `evaluation.md` and a manifest containing the Git revision and SHA-256 of every input file. Prediction writes:

- `submission.csv`
- `test_probabilities.csv`
- `feature_importance.csv` and grouped importance
- `local_explanations.csv`

The submission writer uses explicit exceptions rather than `assert`, checks literal column names, exact sample ID membership, uniqueness, non-null values, and allowed labels, then reads the CSV back and validates it again.

## Key data insights

### Train-only masking shortcut

Description quality is label-associated in raw train but not comparably in valid. A train-only model using noise-quality features scores 0.366 macro-F1; dropping four such features reaches 0.630. Across all raw train subscription-candidate rows, unambiguous generic descriptions occur in 15.5% of `none` rows versus 1.2% of non-`none` rows. After label-independent redraw of every candidate row, the rates are 40.9% and 40.9%.

This is a split-dependent shortcut, not direct access to future labels. The final model reuses merchant-confidence features only after the redraw.

### Split noise differs materially

Generic everyday masks rise from 2.4% in train to 48.2% in valid and 67.8% in test. The detected-stream subscription-mask estimate rises from 4.3% to 36.1% and 52.1%. Final test-level train redraw measures 51.6%, and two valid stress views measure 53.1%. See [the calibration report](results/CALIBRATION.md) for MCC and affix rates and the detector limitations.

### Descriptions follow a compact grammar

The supplied corpus has 2,899 descriptions over 73 tokens. The deterministic parser emits evidence for all but 24 of 1,047,053 rows. Shared names such as `digital plus`, `premium plan`, and one-token truncations remain explicitly ambiguous rather than being forced to one family. This coverage supports a compact rule normalizer for this corpus; it is not a claim that unseen production text would follow the same grammar.

### What the label represents

The documented target is the first recurring merchant family within the 90-day future horizon, or `none`. Among reconstructed active streams, choosing the earliest expected next date scores 0.503 macro-F1, versus 0.420 for the most-paid stream and 0.359 for the most recent. This makes timing important, but observed history does not determine every future label: churn, new starts, stream-reconstruction error, and unobserved factors remain plausible explanations.

### Interpretability

Measured LightGBM gain assigns 71.6% of family-ranker gain to timing/recurrence features. For the `none` gate, refunds contribute 37.6%, timing 24.3%, background activity 19.1%, and merchant evidence 16.3%. Local contribution examples show an ended refunded stream pushing one client toward `none`, while an active gym stream due in two days pushes another toward `gym`.

## Reproducibility and tests

```bash
make lint
make test
UBS_DATA_DIR=/absolute/path/to/data uv run pytest tests/test_full_data.py
```

CI runs lint, unit/contract tests, parser determinism checks, and a two-run byte-identity check on a tiny synthetic model fixture without private data. On the full supplied data, independently repeated train/predict runs under different `PYTHONHASHSEED` values produced byte-identical CSVs.

## Limitations

- Valid labels informed model development. Repeated folds measure client-held-out behavior, but they do not create a prospectively untouched final holdout.
- Valid and test-noised-valid are paired views of the same clients. Their difference is a stress test, not an independent replication.
- Noise rates are parser/detector-conditioned estimates, not known generator corruption probabilities. Sparse, ambiguous, and heavily masked streams are underrepresented.
- Refund descriptions are more stable and informative across splits, but are not assumed literally noise-free.
- Amount-only single-linkage can merge nearby streams or split drifting ones. Ordered-gap and secondary-stream timing additions were tested and rejected because gains were inconsistent across fold seeds/noise views.
- Ranker scores are normalized for decision-making but are not claimed to be calibrated customer-facing probabilities.
- No hidden-test score, stochastic data-generating mechanism, or irreducible performance ceiling is asserted.
