# UBS Transaction Activity Forecasting

A deterministic Python solution for predicting each client's next recurring merchant family after the `2026-01-01` cutoff. It detects recurring amount/currency streams, infers merchant-family evidence, and scores eight candidates per client (seven families and `none`) with one shared listwise model.

The repository generates and validates a local `submission.csv`; it contains no upload or organiser-submission integration.

## Results

Scores are macro-F1 over the fixed eight labels under the shared protocol in
[results/PROTOCOL.md](results/PROTOCOL.md): development set = 2,000 train clients plus 791
non-lockbox valid clients, five hashed client folds repeated with seeds 0, 1, 2, and a 209-client
valid-only lockbox predicted once by the final model. Validation-only CV and the lockbox are the
headline numbers; pooled CV is flattered by the much cleaner train clients. Intervals are 95%
client bootstraps. Differences under about 0.02 are noise.

| Solution | Pooled CV | Validation-only CV | Lockbox |
| --- | ---: | ---: | ---: |
| Iteration 0: ranker + none gate (`ubs-forecast all`) | 0.655 (0.637-0.674) | 0.643 (0.605-0.673) | 0.703 (0.632-0.763) |
| Iteration 1: keyword streams (`--components keyword_streams`) | 0.662 (0.645-0.680) | 0.603 (0.567-0.634) | 0.641 (0.567-0.703) |
| Iteration 2: amount-kernel listwise (`--components listwise_candidates`) | 0.683 (0.666-0.700) | 0.616 (0.580-0.645) | 0.592 (0.516-0.661) |
| Iteration 3: parser features + eight-candidate softmax (`ubs-forecast ensemble`) | 0.688 (0.671-0.706) | 0.679 (0.643-0.711) | 0.668 (0.593-0.729) |
| Iteration 4: equal six-method blend, none multiplier 0.8 | 0.694 (0.676-0.710) | 0.666 (0.630-0.699) | 0.703 (0.632-0.764) |
| **This branch (iteration 5): iteration 3 recipe, retained after a further improvement round** | **0.688 (0.671-0.706)** | **0.679 (0.643-0.711)** | 0.668 (0.593-0.729) |

This branch keeps iteration 3's frozen recipe: a further round of ideas (self-supervised
pseudo-cutoff labels from the unlabeled histories, stronger churn summaries for the `none` row,
calibration, blends with iteration 4, training-set cleaning, bagging and regularization) produced
nothing outside seed noise, and the analysis in [CHANGES.md](CHANGES.md) shows why the label
process leaves little room above this score. CHANGES.md lists every tested variant with its paired
bootstrap against iteration 3; [results/PROTOCOL.md](results/PROTOCOL.md) has the side-by-side
protocol scores of all iterations. The earlier single-split results of iteration 0 remain in
[results/VALIDATION.md](results/VALIDATION.md).

## Method

1. **Closed-grammar parser.** Normalize prefixes, suffixes, abbreviations, and one-token truncations into 47 supplied-corpus base descriptions. Ambiguous names retain fractional evidence across families.
2. **Recurrence detection.** Cluster card-payment candidates by currency and single-linkage log amount at 3.5%. Summarize cadence, jitter, amount stability, recency, and matched refunds.
3. **Stream posterior.** Combine description, MCC, clean amount-prior, and refund evidence into a seven-family posterior. Amount priors come from the unlabeled histories.
4. **Features.** Build one row per `(client, family)` plus client-level activity, recurrence, timing, currency, and refund aggregates.
5. **Shift repair.** Re-draw card-payment corruption for **every** train subscription-candidate row without using labels. Confident recurring streams use their inferred family; sparse or ambiguous candidates sample from their inferred posterior. Affix, mask, and MCC rates are calibrated against parser-derived test statistics.
6. **Eight-candidate softmax scorer.** Every client becomes eight rows: the seven family rows plus one `none` row carrying only client-level features, each with label-free summaries of the client's strongest family evidence. One gradient-boosted scorer is trained with a per-client softmax cross-entropy so the `none` decision competes directly with the family evidence. Three deterministic model seeds are averaged; the final decision is untuned argmax. The earlier two-part head (binary family ranker plus a separate `none` gate) remains available through `ubs-forecast all`.

The retained [`ubs_baseline.py`](ubs_baseline.py) is the byte-identical Appendix B reference. The modular package first reproduced its scores and deterministic submission hash before the augmentation hardening was added.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12. All direct and transitive dependencies are locked in `uv.lock` (the ported iteration-2 component needs the symmetric-tree boosting library, which the lock pins; on macOS it may require `brew install libomp`).

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

The `ubs-forecast` CLI is the single entry point. The protocol run below builds the features,
runs the repeated hashed-fold CV, refits on the development set, scores the lockbox once, and
writes `oof.csv`, `oof_seed{0,1,2}.csv`, `lockbox.csv`, `test_proba.csv`, `submission.csv`, and
`metrics.json` (about eight minutes on eight cores):

```bash
uv run ubs-forecast ensemble --data "$UBS_DATA_DIR" --output artifacts/protocol --jobs 8

# Re-score the two ported earlier attempts under the same protocol (the second one takes about
# an hour because of its 2,093-column listwise ranker)
uv run ubs-forecast ensemble --data "$UBS_DATA_DIR" --output artifacts/rescore \
  --components parser_softmax,keyword_streams,listwise_candidates --jobs 8

# Re-score the iteration-0 head under the protocol and score any probability directory
uv run python scripts/rescore_iteration0.py --data "$UBS_DATA_DIR" --output artifacts/iteration0
uv run python scripts/score_protocol.py --data "$UBS_DATA_DIR" --directory artifacts/iteration0
```

The tracked copies of the final run's outputs live under [deliverables/](deliverables/).

The development experiments behind CHANGES.md refit only the scorer on cached feature tables
(about three minutes per variant; the pseudo-cutoff variants first build the unlabeled views,
about six minutes):

```bash
uv run python scripts/protocol_experiments.py --data "$UBS_DATA_DIR" \
  --cache artifacts/cache --output artifacts/experiments --variant base --variant pseudo90_w03
```

The original single-split workflows of iteration 0 are unchanged:

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
