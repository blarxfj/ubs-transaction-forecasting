# Transaction Activity Forecasting - next recurring merchant family

Independent attempt built only from `README.md` and `data/dataset.zip`.
Final solution code, deliverables and exact run commands are in `solution/` next to this report.

## Result summary

| measurement | macro-F1 |
| --- | --- |
| Protocol CV, mean over seeds 0/1/2 (5-fold, development set = train + non-lockbox valid, 2365 clients) | **0.6493** |
| per seed | 0.6513 / 0.6533 / 0.6434 |
| non-lockbox valid clients only, inside those folds (mean over seeds) | 0.5931 |
| 95% bootstrap over clients, seed 0 | [0.629, 0.672] |
| Lockbox (635 clients, evaluated once with the final model) | **0.6605**, bootstrap [0.621, 0.696] |

Per-class F1 (seed 0 CV / lockbox):

| class | cloud | gym | insurance | mobile | music | software | streaming | none |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CV seed 0 | 0.634 | 0.630 | 0.667 | 0.672 | 0.588 | 0.599 | 0.652 | 0.767 |
| lockbox | 0.607 | 0.686 | 0.696 | 0.640 | 0.574 | 0.657 | 0.644 | 0.780 |

Reference points from the same protocol: a hand-written "earliest due active stream" rule scores 0.38; a direct 8-class LightGBM on the same features scores 0.58. The final model is a shared per-(client, family) binary LightGBM with a logistic second stage.

Test submission label distribution: none 301, insurance 112, software 110, gym 103, mobile 101, streaming 99, cloud 97, music 77. The run is deterministic: two independent runs produced byte-identical `submission.csv` (SHA-256 `b738094d7cd58516e335a98de0a58a61032e02167133a6e99216ba67a9d835da`) and identical `test_proba.csv`.

## Data findings

Sizes: 2000 train, 1000 valid, 1000 test clients with about 74 transactions each between 2024-11-07 and 2025-12-31; 10000 unlabeled clients over the same window (no post-cutoff data anywhere). Label mix is about 30% `none` and 9-11% per family (valid has slightly more gym and fewer cloud than train).

How recurring streams look in the data (all verified on train+valid+test):

- A stream is a sequence of `card_payment`/`out` events with a nearly constant amount (log-amount spread about 1-2%, occasional +10-20% outliers) and a period that is mostly ~30 days (median gap 30, interquartile 28-33) with a minority of ~14-day streams and a few 45-90 day ones. Gap jitter is several days, and some streams drift in price by a few percent per month.
- Each family has four named merchants plus four generic names shared by all families, and the description of each event is drawn from that pool (verified by clustering events by amount within client and voting the family of named events):
  - cloud (MCC 5732): cloud access, cloud backup, service plan, storage plan
  - gym (7997): urban gym, gym membership, fit club, fitness monthly
  - insurance (6300): cover plan, safe cover, policy premium, insurance monthly
  - mobile (4814): phone contract, service bill, monthly plan, digital plus
  - music (5812): audio streaming, member pass, premium plan, digital plus
  - software (5734): saas billing, productivity suite, software access, premium plan
  - streaming (5812): media streaming, video access, premium plan, digital plus
  - generic for all: monthly plan, member plan, digital service, subscription charge
  Descriptions are further corrupted with prefixes (pay, billing, member), suffixes (core, digital, online, plus, service), abbreviations (dgtl, prem, mth, prod, stream) and truncation to one token. MCCs are swapped to another family's MCC on roughly 5-10% of events. Music and streaming share MCC 5812, so `premium plan` / `digital plus` events can only be attributed through their amount.
- Non-recurring noise: salary, ATM, p2p, fees, and card payments named `merchant charge`, `service payment`, `card purchase`, `digital order` at random MCCs and amounts. Refunds mirror recent payments.
- Stream starts are uniform from the start of the window to about 5 months before the cutoff and then stop, so every stream has at least a few observed events; new streams after the cutoff cannot be anticipated.
- Visible churn is rare far from the cutoff and increases sharply toward it: of streams active at day -180, 99% produced another event in the next 90 days; at day -120, 95%; at day -90, 92%.

What determines the label (measured on the 2365 development clients):

- Even a client with exactly one clearly active stream (>=4 events, last event within 45 days) gets that family as label only 53% of the time, `none` 29% and another family 18%. So a large part of the label is hidden churn at or after the cutoff plus streams that start after it: 8% of family labels have no event of that family in the history at all, and a further 5% have fewer than three.
- Among clients with several active families whose label is one of them (896 clients, 2.6 candidates on average), picking the earliest predicted due date is right 53% of the time versus 41% for a random pick; every other criterion (most events, largest amount, oldest, most regular) is worse. The label therefore behaves like "earliest next event of the streams that survive", with heavy timing jitter.
- Churn is partly predictable from noise on the stream itself. For single-active-stream clients: MCC-swap fraction above 0.35 gives 77% `none` (vs 21% with no swaps); a refund matching the stream amount in the last 60 days gives 43% `none` (vs 25%); noisy or generic descriptions in the last three events, a last gap 1.3x longer than usual (80% `none`), a duplicate charge, an amount trend, or an unusually young stream all raise the `none` rate. These signals are correlated (0.3-0.45) and look like a per-stream "flakiness" that the generator ties to termination.

## Approach

1. Family assignment of recurring candidates (`features.py:assign_family`). Candidates are out-going card payments that are neither known non-recurring merchants nor generic noise names. Named descriptions map to a family through a keyword table; generic or truncated descriptions are linked to the nearest named amount anchor of the same client (within 8% in log amount, weighted by anchor support), then fall back to the MCC, and 5812 events that cannot be resolved stay in an `amb5812` bucket.
2. Per-family stream features (`features.py:timeline_feats`, about 70 per family): counts in 30/45/60/90/180-day windows, first/last event, median and mean gap, gap spread, last gap ratio, predicted due date (last + median gap), overdue phase, coverage, regularity, day-of-month spread, amount level and spread, last-amount deviation, outliers, slope, MCC-swap fraction (overall, last event, last three), generic and noisy-description fractions, a combined noise score and clean-event count, fees, currency changes, matched refunds (count, recency, after-last-event flag). Plus client-level counts (transaction types, refunds by window, generic and noise counts, salary and spend) and cross-family context (due rank, event-count rank, due date minus the earliest other due date, number of active families).
3. Stage 1 (`pairwise.py:to_long`, `train.py`): the wide client table is unpivoted into one row per (client, family) with family-agnostic column names, a family id and "other families" context, giving 7x the training rows with a single binary target "this family is the label". A LightGBM binary model (15 leaves, min 40 rows per leaf, feature fraction 0.5, learning rate 0.03, 450 rounds, bagged over 5 seeds for the final model) shares all churn and timing signal across families. This is the single biggest gain: 0.58 to 0.645 over a direct 8-class model on the same features.
4. Stage 2 (`stage2.py`): a multinomial logistic regression on the 7 family log-odds (plus top-2 and margin) fitted on inner out-of-fold stage-1 scores turns them into calibrated 8-class probabilities (`oof.csv`, `lockbox.csv`, `test_proba.csv`); the prediction is the argmax. A tuned threshold on the raw max score or per-class weights tuned for macro-F1 gave the same score within noise, so the simplest calibrated argmax is used.
5. Evaluation strictly follows the shared protocol: lockbox = sha256(client_id) % 5 == 0 (635 clients, touched once at the end), folds = sha256(f"{seed}:{client_id}") % 5, seeds 0/1/2, macro-F1 over the fixed 8 labels, valid-only score inside the folds, bootstrap over clients.

## What was tried and did not help (protocol CV, mean macro-F1 unless noted)

| variant | score | note |
| --- | --- | --- |
| rule: earliest due active stream | 0.382 | also "most events" 0.382, "latest event" 0.323 |
| direct 8-class LightGBM, first feature set (330 features) | 0.5773 | per-family columns used by only ~15% of rows each |
| direct 8-class LightGBM, family-first features (510) | 0.5805 | valid-only 0.521 |
| per-family binary model, family-first features | 0.6451 | valid-only 0.610 |
| + smaller trees (7 leaves, min 80) | 0.6403 | |
| + larger trees (31 leaves) / feature fraction 0.3, L2 5 / lr 0.02, ff 0.7 | 0.6453 / 0.6444 / 0.6436 | all within noise |
| + pseudo-cutoff self-supervision: per-family model trained on 42000 pseudo-labelled histories (all 14000 clients at cutoffs -90/-150/-210 days, label = first family event in the following 90 days) and its score added as a feature | 0.6462 | pseudo labels have only 12-34% `none` and no hidden churn, so they teach little beyond the timing features; dropped |
| + description-noise features (final) | 0.6513 (early stopping) / 0.6493 (fixed 450 rounds, stage 2) | |
| stage-2 decision: raw-score threshold vs calibrated argmax vs per-class weights | 0.6451 / 0.6454 / 0.6443 | equivalent |

Alternative due-date estimators for ordering competing streams (least-squares phase fit, robust median phase, day-of-month) were all slightly worse than last event + median gap (0.51-0.52 vs 0.53 pick accuracy).

## Error structure (seed 0 out-of-fold, 2365 clients, 799 errors)

| category | count |
| --- | --- |
| wrong family, both present in history | 267 |
| true `none`, predicted a family | 234 |
| label family has fewer than 3 history events | 118 |
| label family absent from history | 110 |
| true family, predicted `none` | 70 |

Roughly a third of the errors (label family absent or nearly absent) are not predictable from the history; the rest are hidden churn and stream ordering under timing jitter, where the churn signals above already carry most of the available information. Music is the weakest class because its generic-named events are indistinguishable from streaming without an amount match.

## Limitations

- The evaluation ceiling is set by the generator's hidden churn and post-cutoff stream starts, so gains beyond about 0.65-0.67 are unlikely without new information; the valid-only score (0.59) is lower than the pooled score, suggesting a mild train/valid shift or simply the smaller sample.
- The family vocabulary and non-recurring merchant list are hard-coded from this synthetic dataset; real data would need a learned merchant normaliser.
- Amount linking uses an 8% tolerance, which can misattribute generic-named events when two streams of one client have similar amounts.
- Feature building is a pure pandas loop per client (about 45 seconds for 4000 clients); it would need vectorising for millions of clients.

## Reproduction

```bash
unzip data/dataset.zip -d data            # from the challenge repository
python3.12 -m venv .venv && .venv/bin/pip install -r solution/requirements.txt   # or: uv venv -p 3.12 .venv && uv pip install -p .venv/bin/python -r solution/requirements.txt
.venv/bin/python solution/train.py --data data --out results --rounds 450 --bag 5
```

Outputs in `results/`: `oof.csv` (seed-0 out-of-fold probabilities with fold ids), `lockbox.csv`, `test_proba.csv`, `submission.csv`, `results.json` (all scores above, per class, bootstrap, submission hash), `feature_importance.csv`. `--skip_cv` skips the 3-seed CV. The full run takes about 5 minutes on an 8-core laptop (feature building under a minute). A second run from a fresh directory and fresh virtual environment reproduced the same scores and the same `submission.csv` hash.

Evidence files: `solution/results/results.json` (scores), `solution/results/feature_importance.csv` (top features: last event day, event counts in the last 45/60/30 days, MCC-swap fraction, clean-event count, due date versus the earliest other due date).
