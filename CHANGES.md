# Changes versus iteration 3

Iteration 3 (parser features, eight-candidate listwise softmax; its own change notes are in
[results/ITERATION3.md](results/ITERATION3.md)) is the starting point and the previous best under
the shared protocol. This iteration tested the remaining ideas that looked
capable of a gain of 0.02 or more on validation-only macro-F1, found none that beats iteration 3
outside seed noise, and therefore keeps iteration 3's recipe unchanged: the frozen model,
`deliverables/`, and `submission.csv` are byte-identical to iteration 3. What is new is the
evidence, the tooling that makes every tested variant reproducible, and an explanation of why the
score is close to a ceiling for this label process.

All scores are macro-F1 over the fixed eight labels under [results/PROTOCOL.md](results/PROTOCOL.md):
development set = 2,000 train clients + 791 non-lockbox valid clients, five hashed client folds
repeated with seeds 0, 1, 2, and a 209-client valid-only lockbox predicted once by the final model.
Validation-only CV and the lockbox are the headline numbers. Differences under about 0.02 are noise.

## Side by side

| Solution | Pooled CV | Validation-only CV | Lockbox (209 clients) |
| --- | ---: | ---: | ---: |
| Iteration 3: parser features, eight-candidate softmax | 0.688 (0.671-0.706) | 0.679 (0.643-0.711) | 0.668 (0.593-0.729) |
| Iteration 4: equal six-method blend, none multiplier 0.8 | 0.694 (0.676-0.710) | 0.666 (0.630-0.699) | 0.703 (0.632-0.764) |
| **Iteration 5 (this branch): iteration 3 recipe, retained** | **0.688 (0.671-0.706)** | **0.679 (0.643-0.711)** | 0.668 (0.593-0.729) |

Paired client bootstrap of iteration 5 minus iteration 3 on the validation-only CV: +0.000 by
construction (identical predictions). Iteration 4 minus iteration 3 on the same clients: -0.013
(validation-only, per-seed paired differences -0.008, -0.011, -0.018); on the lockbox +0.035
(95% interval -0.019 to +0.086), not distinguishable from zero on 209 clients.

## What was tested and did not help

Validation-only CV, mean over fold seeds 0/1/2, with the paired client-bootstrap difference
versus iteration 3 (mean of the per-seed intervals). Every variant is a `--variant` of
[scripts/protocol_experiments.py](scripts/protocol_experiments.py) and was selected against
development out-of-fold data only.

| Variant | Validation-only CV | Paired vs iteration 3 | Note |
| --- | ---: | ---: | --- |
| Iteration 3 recipe (`base`) | 0.679 | +0.000 | reference; reproduces `deliverables/oof_seed*.csv` exactly |
| Pseudo-cutoff views, unlabeled clients re-noised at test and valid level, family-only, weight 0.3 (`pseudo90_w03`) | 0.674 | -0.005 (-0.025 to +0.015) | |
| Same, weight 1.0 (`pseudo90_w1`) | 0.649 | -0.029 (-0.054 to -0.006) | worse with more weight |
| Same, weight 0.3, `none` row supervised too (`pseudo90_full_w03`) | 0.663 | -0.016 (-0.037 to +0.003) | observed futures are 7% `none`, labels 29% |
| Plus 180-day pseudo cutoffs (`pseudo90_180_w1`) | 0.629 | -0.049 (-0.076 to -0.023) | worse with more pseudo data |
| Plus pseudo cutoffs of the development train clients (`pseudo90_train_w1`) | 0.649 | -0.030 (-0.054 to -0.005) | |
| Clean unlabeled pseudo views, no re-noising (`pseudo90_clean_w1`) | 0.668 | -0.011 (-0.034 to +0.011) | |
| Five model seeds instead of three (`seeds5`) | 0.675 | -0.004 (-0.013 to +0.004) | |
| `min_child_samples` 40, `reg_lambda` 3 (`regularized`) | 0.677 | -0.002 (-0.015 to +0.010) | |
| Earliest-due stream summaries on every candidate row (`earliest_summaries`) | 0.675 | -0.004 (-0.018 to +0.010) | 20 churn features of the earliest-due stream, visible to the `none` row |
| Drop training clients whose label family has no stream and no single event (`drop_unsupported`) | 0.681 | +0.002 (-0.013 to +0.017) | |
| Both of the previous two | 0.673 | -0.006 (-0.021 to +0.008) | |

Post-hoc analyses on the saved out-of-fold tables (no refit):

| Analysis | Validation-only CV | Note |
| --- | ---: | --- |
| `none` probability multiplier, selected on the other four folds of each seed | 0.677 | -0.002; the unnested curve peaks at +0.005 |
| Per-class multipliers by nested coordinate ascent | 0.673 | -0.006 |
| Blend with iteration 4's frozen blend, weight 0.3, unnested | 0.689 | +0.011 |
| Blend with any iteration 4 component, weight and `none` multiplier selected nested | 0.680 | +0.001; the unnested gains are selection optimism |

## Why the pseudo-cutoff idea fails here

The idea: cut every history 90 or 180 days before the real cutoff, re-date it, and label the client
from its own observed future (the family of the earliest recurring event in the 90-day window).
The 10,000 unlabeled histories are nearly noise-free, so their observed futures give accurate
targets, and their pasts can be re-noised to valid and test level. Implemented in
`src/ubs_forecasting/pseudo.py`, with soft targets (the stream posterior) and an objective option
that supervises only the seven-family ranking (`family_only`), leaving the `none` decision to the
real labels.

Two measurements on the clean train clients show why it cannot transfer:

1. **The label process is more random than the observed transaction process.** Among clients
   with two or more active streams, the stream with the earliest expected next date is the label
   60% of the time and the second-earliest 34% of the time, with the label's stream due within
   5 days of the earliest in three quarters of the cases. In the same clients' own observed
   histories, the earliest-expected stream is the one that actually fires first 72% of the time
   at a 90-day pseudo cutoff and 81% at a 180-day one. Observed futures therefore teach a
   sharper timing rule than the labels follow, and the more weight they get, the worse the score.
2. **The `none` label is not observed churn.** 29% of clients are labeled `none`, and the rate is
   flat at about 25% whether a client has one, two, three, or four active streams. In observed
   90-day windows only 7% of clients have no recurring event. No alternative next-due rule
   (mean, last, minimum, or maximum gap; day-of-month anchoring; schedule anchored at the first
   event; fixed 30 days) matches the labels better than the median gap.

## Where the remaining errors are

Per seed on the 791 validation-only clients, iteration 3 makes about 147 family-to-family errors,
39 family-to-`none` errors, and 44 `none`-to-family errors.

- 81 of the family-to-family errors have the true family present as an active, confidently
  identified stream; half of those are due within 7 days of the predicted family's stream. This is
  the timing randomness above.
- 62 have no detected recurring stream of the true family at all: 38 clients (4.8%) have no
  event of the label family in their history (accuracy 0 on them), and about 70 have a single
  event only, on which the model is right about 40% of the time (the single-event prior is 24%).
- Description and MCC noise cost almost nothing in identification: the label family has a detected
  stream for 84.4% of family-labeled train clients on clean histories and 84.5% on test-level
  re-noised ones; the corresponding valid rate is 80.7% with and without extra noise.
- Family amounts are continuous ranges, not price points (about 900 distinct stream amounts per
  family in the unlabeled histories), so a sharper amount prior cannot identify masked streams.
- Refund features are the strongest `none` signal (client-level AUC 0.72-0.74) and are already in
  the feature set.

## Code changes

- `src/ubs_forecasting/pseudo.py`: shifted histories and soft pseudo labels from observed futures.
- `src/ubs_forecasting/pipeline.py`: `FeatureSpec.shift`, the pseudo-cutoff table specs, and
  `pseudo_label_tables`.
- `src/ubs_forecasting/model.py`: the listwise objective accepts soft targets, per-table sample
  weights (applied explicitly, since the boosting library does not weight custom objectives), and
  family-only blocks; `candidate_rows` can add earliest-due stream summaries. With the default
  arguments the fitted scorers and their outputs are unchanged to the last bit.
- `src/ubs_forecasting/ensemble.py`: `Recipe.pseudo_views` (empty in the frozen recipe).
- `scripts/protocol_experiments.py`: cached-feature protocol CV for every variant above, with the
  paired bootstrap against a reference solution.
- `tests/test_pseudo.py`: shift, pseudo-label, soft-target, and objective tests.

## Process notes

- Everything above was selected on development out-of-fold data only; the pseudo-label rule was
  checked against the real label distribution before any protocol run.
- The lockbox was scored by the final run of the retained recipe. Because the recipe is
  iteration 3's, the lockbox probabilities equal iteration 3's; an earlier end-to-end equality
  check of the refactored code also produced these identical files. No differently configured
  model was ever scored on the lockbox in this iteration.
- Determinism: the final run reproduces iteration 3's `submission.csv` byte for byte; its SHA-256
  is recorded in [deliverables/README.md](deliverables/README.md).
