# Recurring-transaction forecasting

## Result

**Primary score: 0.6816 mean macro-F1 across 15 held-out folds**, with a **95% client-bootstrap interval of [0.6627, 0.6960]**. The corresponding non-lockbox validation-only score is **0.6103**, interval **[0.5678, 0.6360]**.

| Repetition | Mean fold macro-F1 | Validation-only mean fold macro-F1 |
|---|---:|---:|
| Seed 0 | 0.6888 | 0.6274 |
| Seed 1 | 0.6727 | 0.5930 |
| Seed 2 | 0.6832 | 0.6105 |
| **Mean** | **0.6816** | **0.6103** |

For comparison, averaging the three **pooled** OOF scores gives **0.6834**; the analogous pooled validation-only average is **0.6156**. These are not the primary fold-averaged statistics. Exact values and all 15 fold scores are in `metrics.json`.

Per-class F1, averaged across the 15 folds:

| cloud | gym | insurance | mobile | music | software | streaming | none |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.6633 | 0.6618 | 0.6748 | 0.6990 | 0.6277 | 0.6567 | 0.6476 | 0.8215 |

**One-shot lockbox macro-F1: 0.5924 on 209 clients**, reported separately from development scores. Per-class results are in `lockbox_metrics.json`. No estimator, feature, ensemble weight, or decision multiplier was changed after this evaluation. The lower result reinforces using the noisy-domain validation estimate rather than the combined development score as a conservative guide.

There is no claimed test score: test labels are unavailable.

The deliverable is a CPU-only, deterministic Python workflow, its fitted estimators, development out-of-fold probabilities, a separately evaluated lockbox, and a contract-valid 1,000-client submission. No challenge submission was made.

## Approach

The central idea is to share statistical strength across merchant families rather than rely only on one wide eight-class client profile. Represent every client as eight candidate rows: one per merchant family, plus `none`. A shared decision-tree scoring function learns across merchant families. Client-level folds keep all eight rows together.

1. **Reconstruct merchant families.** Use MCC and a small, auditable description vocabulary. Exclude recognizable everyday purchases and non-card transaction types. Resolve corrupted or generic descriptions using similar-amount transactions in the *same client's* history. Music and streaming share an MCC, so their distinctive description tokens and amount neighborhoods matter.
2. **Describe recurring streams.** Compute recency, counts in several recent windows, gaps, amount dispersion and trend, refund activity, and approximate periodic phase. Repeat those summaries for a dense stable-amount neighborhood and a recently active neighborhood. These help separate genuine recurring streams from one-off lookalikes and multiple subscriptions in one family.
3. **Forecast cadence.** Compare conventional 7/14/30/60/90/180/365-day cadences, allowing missing occurrences; fit short-window time-versus-cycle trends; calculate projected due dates and overdue ratios. These are features, not brittle hard prediction rules.
4. **Compare competing streams.** Add relative ranks, feature extrema, and attributes of the four earliest plausible competing streams. The `none` candidate sees the same client context, even though it has no own payment stream.
5. **Handle degraded descriptions.** Treat `monthly plan` as mostly generic rather than as decisive mobile evidence. Add client- and family-level generic-description fractions. This adjustment was motivated by a measurable train/validation text-quality shift, not by any lockbox observations.

The selected recipe is **one shared listwise symmetric-tree ranker**, trained with client-group softmax loss: 1,000 trees, depth 5, learning rate 0.04, L2 regularization 5, and four CPU threads. The eight scores become a softmax distribution; the `none` probability is multiplied by **0.8**, followed by renormalization and argmax. The final fit uses seed 0. No ensemble was retained: the single scorer achieved the best seed-0 selection score, **0.6888**. The exact frozen configuration is `recipe.json`.

The final feature representation contains 289 per-family descriptors, reused in candidate/context summaries to form 2,093 numeric inputs. No client identifier enters the predictors. There are no external services, embeddings, merchant enrichment, or pretrained language components. The optional unlabeled transaction file was not used.

The full-fit feature-importance artifact shows strong use of relative refund counts, observed stream support, recent active-core support, and recency/due-date features. Those are associations, not proof that refunds cause cancellations. Several nominal-period due features are simply different offsets of recency; their importance should not be mistaken for evidence of annual payment cycles.

Implementation: `features.py:33` reconstructs families; `features.py:66` computes temporal statistics; `features.py:125` extracts client profiles; `features.py:199` builds candidate/context matrices; `experiment.py:15` defines the estimators and objectives.

## Data findings

Counts and description statistics are reproducible with `audit.py`; its output is `results/data_audit.json`. Inferred-family coverage, measured from development feature tensors, is recorded in `results/target_family_coverage.json`.

- Development comprises **2,000 train clients and 791 non-lockbox validation clients**. The validation lockbox contains **209 clients**; test contains **1,000**.
- Development histories contain **147,459 train transactions** and **57,896 non-lockbox validation transactions**. Mean histories are 73.7 and 73.2 transactions per client. Train history lengths range from 8 to 162.
- Transactions run from November 2024 through December 2025; feature extraction asserts every timestamp precedes `2026-01-01`.
- `none` is 829/2,791 development labels (29.7%); each other class has 255–300 examples. Fixed-label macro-F1 therefore matters more than raw accuracy.
- Vague everyday-purchase descriptions (`card purchase`, `digital order`, `merchant charge`, `service payment`) account for **0.87% of train transactions versus 16.79% of validation transactions**.
- Among transactions described exactly as `monthly plan`, mobile MCC 4814 accounts for **65.5% in train but only 21.5% in validation**. Treating that phrase as a strong mobile alias damages robustness.
- MCC alone cannot separate music from streaming and confuses subscriptions with some ordinary purchases. Amount continuity and within-client semantic evidence reduce that ambiguity.
- The inferred target family has no assigned historical card payment for about **5.2% of non-`none` train targets and 4.1% of non-`none` validation targets**. This is not proof of label error: new future commitments and imperfect family reconstruction are both plausible.

## Experiments and unsuccessful approaches

All selection used development data. The lockbox was excluded from training, feature diagnostics, and selection. Unless explicitly marked as a one-fold screen, values below are the mean of the five seed-0 fold macro-F1 scores; the validation column pools only non-lockbox validation clients' held-out predictions.

| Method | Seed-0 mean fold F1 | Validation-only pooled F1 |
|---|---:|---:|
| Always predict `none` | 0.0573 | 0.0567 |
| Stable-amount earliest-due heuristic | 0.5030 | 0.4792 |
| Initial shared binary scorer | 0.6517 | 0.5885 |
| Binary scorer + cadence/competitors | 0.6701 | 0.6029 |
| Listwise scorer + cadence/competitors | 0.6776 | 0.6056 |
| Alternative leaf-wise binary scorer | 0.6647 | 0.6043 |
| Alternative leaf-wise listwise scorer | 0.6697 | 0.6096 |
| Description-robust binary scorer | 0.6790 | 0.6249 |
| Description-robust leaf-wise listwise scorer | 0.6721 | 0.6154 |
| Description-robust listwise scorer, unadjusted | 0.6850 | 0.6167 |
| **Selected listwise scorer, `none` multiplier 0.8** | **0.6888** | **0.6294** |

The best tested mixed binary/listwise ensemble scored **0.6855**, below the selected single scorer. A 75% listwise / 25% leaf-wise mixture scored **0.6854**. The full final-feature blend/multiplier grid is `results/selection_grid.csv`.

Selection prioritizes the combined development metric. The description-robust binary scorer with the same 0.8 multiplier scored **0.6824** overall but **0.6393** on validation alone, slightly above the selected method's validation score. That tradeoff is worth retaining when judging likely test-domain shift.

The earliest-due heuristic's best tested age threshold was 35 days. Increasing it to 90 days reduced its mean fold score from **0.5030 to 0.4847**: accepting old streams creates false future activity. Aggregating every assigned transaction rather than stable-amount cores scored only **0.4527** at the same 35-day threshold.

The wide eight-class classifier was stopped after its first fold: **0.5526**, versus **0.6538** for the corresponding shared-candidate baseline on that exact fold. A compact candidate representation also lost in a first-fold screen: **0.6574** versus **0.6744** for the full listwise representation. Neither screening number is presented as a completed cross-validation result.

Cadence reconstruction and joint competitor context improved the independent candidate classifier from **0.6517 to 0.6701**. Accounting for description degradation then improved it to **0.6790**, with validation-only pooled F1 rising from **0.6029 to 0.6249** before any decision adjustment. This was a more useful gain than simply changing the boosting implementation.

A coarse search over a small number of ensemble weights and `none` multipliers was used. Development cross-validation is consequently a model-selection estimate, not an unbiased nested-CV estimate. The separately reserved lockbox is the cleaner final check. Archived feature implementations and experiment outputs are under `ablations/` and `results/experiments/`.

## Evaluation protocol and safeguards

- A validation client is locked if `int(sha256(client_id).hexdigest(), 16) % 5 == 0`; all training clients remain eligible. The implementation is `features.py:14–25`.
- For each seed 0, 1, 2, a client's fold is `int(sha256(f"{seed}:{client_id}").hexdigest(), 16) % 5`. All observations/candidate rows of that client remain in that fold. Fold preprocessing uses only that client's pre-cutoff transactions; no learned global feature transform is fitted on held-out clients.
- Every scoring call fixes labels to `cloud, gym, insurance, mobile, music, software, streaming, none`. The primary statistic averages macro-F1 across **all 15 folds**. Pooled OOF scores are additionally reported to eliminate aggregation ambiguity.
- Validation-only evaluation also scores the non-lockbox validation subset **inside each fold**, then averages the 15 scores. Pooled validation-only scores are additional diagnostics, not a replacement.
- The 95% interval uses **2,000 client bootstrap resamples**, seed 1729. The same client resample is used across all repeated seeds; the three predictions for one client are not treated as independent clients. Each replicate recomputes the full mean-of-15-folds statistic. This interval is conditional on the fitted models and does not account for model-selection or refitting uncertainty. See `evaluate.py:15`.
- The final recipe is frozen before lockbox evaluation. Final fitting uses the 2,791 development clients, **not** lockbox labels. Locked targets are evaluated once, after final predictions exist. `run.py:60` separates that stage and refuses a second evaluation in the same output directory.
- Test prediction runs deterministically with fixed seeds, fixed package versions, fixed feature order, four estimator threads, and stable CSV row order. `submission.csv` uses exactly the sample submission's identifiers and schema.

## Limitations and recommendation

There is a substantial input-quality shift between train and validation, and validation-only performance is more conservative than the combined development score. Future cancellations, new subscriptions, exact timing jitter, and heavily masked descriptions are not fully identifiable from historical records. Shared or overlapping amounts can join unrelated streams; drifting amounts can split one stream. Currency conversion is not modeled. The output probabilities are normalized prediction scores adjusted for macro-F1, not demonstrated calibrated event probabilities.

The lockbox is small, especially by class. Treat a modest difference between methods as uncertain; do not retune against the lockbox. A larger untouched validation cohort would be a better basis for production decisions. For this dataset, the reusable engineering result is the within-client recurrence reconstruction plus shared candidate scoring, rather than a large end-to-end sequence architecture.

Recommendation: retain this reproducible candidate-ranking baseline and its strict client-level evaluation. The package is runnable as delivered; no repository source change or external submission is needed to reproduce it.

## Exact reproduction

Prerequisites: `uv`, `unzip`, and the original `dataset.zip`. On macOS, the leaf-wise boosting dependency may require `brew install libomp`. Tested environment details are in `results/environment.json`. No data is included in this package.

Expected archive SHA-256:

```text
1afc95470f4e8641601503172be3e698ef9eaf91528d911a6a01a120911634c6
```

Run from this package directory:

```bash
export UV_CACHE_DIR="$PWD/work/uv-cache"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
uv venv --python 3.12.12 .venv
uv pip sync --python .venv/bin/python requirements.txt
mkdir -p work/data
unzip /absolute/path/to/dataset.zip -d work/data
.venv/bin/python -m unittest -v test_solution.py
.venv/bin/python audit.py --data work/data --output work/data_audit.json

# All fifteen held-out folds, the bootstrap interval, and seed-0 OOF file.
.venv/bin/python run.py cv --data work/data --work work --output reproduced

# Fit on development; predict test and evaluate the lockbox once.
.venv/bin/python run.py final --data work/data --work work \
  --output reproduced --evaluate-lockbox

# Independent refit to verify identical submission without reevaluating
# or regenerating lockbox predictions.
.venv/bin/python run.py final --data work/data --work work \
  --output repeated --test-only
cmp reproduced/submission.csv repeated/submission.csv
.venv/bin/python validate.py --data work/data --output reproduced \
  --repeat-submission repeated/submission.csv
sha256sum reproduced/submission.csv   # macOS: shasum -a 256
```

To rerun earlier feature ablations, execute the corresponding `ablations/v1/features.py` or `ablations/v2/features.py` with `--data work/data --cache work/v1` (or `work/v2`), then its neighboring `experiment.py` with that cache. Exact tested mode/depth/iteration/learning-rate values are stored in `results/experiments/*/metrics.json`. For the transparent heuristic sweep:

```bash
.venv/bin/python baselines.py --data work/data --cache work/v1 \
  --output work/heuristic_baselines.json
```

## Verification and artifact inventory

Checks completed:

- `python -m unittest -v test_solution.py`: **8 tests passed** (`results/tests.log`), covering family reconstruction, timestamp rejection, transaction-order invariance, cadence projection, fixed-label scoring, deterministic hashing, probability adjustment, and lockbox exclusion from the determinism check.
- `python -m compileall -q .`: passed.
- The development feature tensors were **exactly identical** when generated alone versus alongside all 4,000 clients (`results/feature_isolation.json`); adding held-out clients does not alter any training feature.
- `validate.py` confirmed unique/correct identifiers, exact CSV columns, all probability row sums, seed-0 fold hashes, and submission/argmax agreement. It recomputed all 15 development fold scores directly with `sklearn.metrics.f1_score(..., average="macro", labels=LABELS)`; the difference from the reported primary score was **0.0**.
- Two independent full-development fits produced **byte-identical `submission.csv`**. Their frozen recipe/source manifests were also byte-identical. The comparison refit used test-only prediction, so this check did not reevaluate or regenerate lockbox predictions.

Verification output is `results/validation.json`. The submission SHA-256 is:

```text
48c853ed135fb5daee38760661352213ad19b0b7e6ff8a48727f9d585d38cba4
```

Core artifacts:

- `oof.csv`: **2,791 rows**, `client_id`, seed-0 `fold`, and the eight class probabilities.
- `oof_seed0.csv`, `oof_seed1.csv`, `oof_seed2.csv`: all three adjusted OOF sets used for the final metrics and bootstrap.
- `lockbox.csv`: **209 rows**, final class probabilities; locked labels are not bundled.
- `test_proba.csv`: **1,000 rows**, final test class probabilities.
- `submission.csv`: **1,000 rows**, exactly `client_id,predicted_next_recurring_merchant`, in sample-submission order.
- `component0.cbm`: fitted full-development scorer; `component0_importance.csv`: feature importance.
- `recipe.json`, `frozen_manifest.json`, `requirements.txt`: configuration, source/dependency fingerprints, and complete version pins.
- `metrics.json`, `lockbox_metrics.json`: full numerical results, including per-seed and per-class scores.
- `results/experiments/`: completed seed-0 ablations and explicitly labeled single-fold screens; `results/selection_grid.csv`: final-feature ensemble/decision-adjustment comparisons.
- `results/components/0/`: unadjusted component OOF probabilities, retained so `assemble.py` can reproduce the final score aggregation without retraining.

The source repository was left unchanged. Dataset files and credentials are deliberately excluded from the deliverable.
