# Repeated client-hash comparison

## Headline results

The frozen blend improves development-valid macro-F1 but **does not establish an overall win**: its lockbox score is effectively identical to the strongest previous method. Differences below about 0.02 should be treated as noise, not substantive improvements.

Each entry is macro-F1 with a 95% client-bootstrap interval.

| Fixed method | Pooled CV, 2,791 clients | Validation-only CV, 791 clients | One-shot lockbox, 209 clients |
| --- | ---: | ---: | ---: |
| Main / iteration 0 | 0.6551 (0.6379–0.6723) | 0.6426 (0.6060–0.6755) | 0.7031 (0.6319–0.7623) |
| Iteration 1, corrected split | 0.6623 (0.6443–0.6791) | 0.6032 (0.5659–0.6358) | 0.6408 (0.5652–0.7029) |
| Iteration 2 | 0.6834 (0.6666–0.6994) | 0.6156 (0.5809–0.6484) | 0.5924 (0.5169–0.6585) |
| Iteration 4, frozen blend | **0.6937 (0.6764–0.7099)** | **0.6662 (0.6295–0.6993)** | **0.7026 (0.6321–0.7640)** |

Pooled CV misleadingly ranks iteration 2 above main. Validation-only and lockbox reverse that ranking. The validation-only blend-minus-main difference is +0.0236, with a paired client-bootstrap interval of +0.0094 to +0.0374. This interval conditions on the selected recipe and does **not** adjust for model-selection optimism. Lockbox difference is −0.0005: no meaningful improvement.

## Exact protocol

- Lock only valid clients satisfying `int(sha256(client_id).hexdigest(), 16) % 5 == 0`.
- Development contains all 2,000 train clients and the remaining 791 valid clients. Correcting iteration 1 restores 426 train clients previously excluded from development.
- Five folds use `int(sha256(f"{seed}:{client_id}").hexdigest(), 16) % 5`, for seeds 0, 1, 2. Every augmented view follows its client's fold; explicit disjointness guards reject leakage.
- Primary CV scores average the three seeds' pooled OOF macro-F1 scores, always using all eight fixed labels. They do not average fold-specific F1 or treat repeated client predictions as independent clients.
- Headline intervals use 2,000 percentile bootstrap samples with seed 2026. Each client is sampled jointly across the three repetitions. Paired development differences use seed 2028. Provisional grid diagnostics use 200 samples; their intervals do not select the recipe or replace the headline intervals.
- The final models use development labels only. Selection was completed before a single final lockbox scoring stage. The selected recipe was not changed afterward.

Main and iteration 1 were refitted under these assignments. Iteration 1 retains its nested training-only score calibration and final five-seed bag. Iteration 2's saved probabilities already use the correct development IDs and all three exact fold assignments; every row was checked before reuse. Its final lockbox and test probabilities come from the same pinned revision, not a newly fitted substitute.

Pinned earlier revisions:

- Main: `196f09b8c6676202d8f14d7fbe98c2d0902f9bd7`
- Iteration 1: `70b07f0fe971d310c351953fb0b800086a6d7718`
- Iteration 2: `347ccad233b0297be1d21d392a4e25f12bdd859b`

## Improvements and controls

The existing grammar, uncertain family evidence, unlabeled amount prior, label-independent train re-noising, recurrence/refund features, and separate none gate are retained.

| Development experiment | Validation-only macro-F1 | Interpretation |
| --- | ---: | --- |
| Previous strongest method, main | 0.6426 | Baseline |
| Seven-family listwise softmax + none gate | 0.6457 | +0.0031; noise-sized |
| Listwise + recent activity/refund chronology | 0.6466 | +0.0009 over listwise; noise-sized |
| Listwise + double valid-view training weight | 0.6467 | +0.0011 over listwise; noise-sized |
| Equal blend of the three previous methods, none weight 0.8 | 0.6643 | Most of the development gain comes from diversity |
| Equal blend of all six methods, no further none adjustment | 0.6660 | Only +0.0017 over the old-only blend |
| Frozen six-method blend, none weight 0.8 | 0.6662 | Best measured development score; not a decisive gain over simpler blending |

The selection grid contains 162 recipes: six individual methods, 45 pairwise mixtures, and three equal-weight group mixtures, each with none multipliers 0.8, 1.0, or 1.2. Only development-valid scores select the recipe. Selection and all candidate predictions are retained for audit.

The final recipe averages main, iteration 1, iteration 2, listwise, activity, and valid-weighted probabilities equally, multiplies the blended none score by 0.8, and renormalizes. No class-specific weight search is used. The new objective/features alone do not demonstrate a substantive gain; their +0.0020 contribution over the old-only blend is noise-sized. The 0.8 none adjustment itself adds only 0.0002 to the six-way blend.

Per-seed validation-only scores are 0.6692, 0.6652, and 0.6643. Averaging the three OOF probability tables before argmax instead gives 0.6678 validation-only and 0.6967 pooled. That alternative estimand is what a direct score of the exported averaged `oof.csv` produces; it is not substituted for the headline repeated-CV scores.

The extra candidates' one-shot lockbox scores are 0.7121 listwise, 0.7006 activity, and 0.6981 valid-weighted. These were reported after freezing the recipe, not used to choose a replacement. Their small differences are not a basis for a new selection round.

## Reproduction and artifacts

Requires Python 3.12, the unchanged locked dependency set, and the pinned previous commits in the local Git object database. A normal full clone includes the earlier branches. Run from the repository root with the private ZIP outside the repository:

```bash
uv sync --locked
make compare DATASET_ZIP=/external/dataset.zip OUT=artifacts/comparison
make lint
UBS_DATA_ZIP=/external/dataset.zip make test
```

Use a fresh output directory for changed code. Cache identity checks reject changed input ZIPs or earlier revision selections. The archive is read directly; no private transaction files are extracted or committed.

Outputs under `artifacts/comparison/selected/`:

- `oof.csv`: 2,791 rows, mean of three OOF probability tables.
- `oof_seed0.csv`, `oof_seed1.csv`, `oof_seed2.csv`: probabilities and exact fold IDs for headline scoring.
- `lockbox.csv`: 209 rows.
- `test_proba.csv`: 1,000 rows.
- `submission.csv`: exactly the 1,000 sample IDs and required two-column schema.

Other ignored outputs contain each baseline's OOF tables, all development and final metrics, the selection grid, frozen recipe, input/source hashes, and a lockbox-opened marker. Source code and summarized Markdown are committed; generated probability tables remain ignored.

A test-only refit checks byte identity without reopening lockbox scoring:

```bash
PYTHONHASHSEED=123 uv run python scripts/compare.py verify \
  --data /external/dataset.zip --output artifacts/comparison
```

Submission SHA-256: `208de9cf7ac4ecd08b1f7cef9b4b7e86cb04d66b84035fcffe6fa7db92a30f89`.

## Verification

`make lint` passed. Standard tests passed 25 with one expected data-dependent skip; the external-ZIP invocation passed all 26 tests. A refit under `PYTHONHASHSEED=123` produced byte-identical component probability tables and submission. An additional fresh-feature rebuild and refit under `PYTHONHASHSEED=321` reproduced the same submission hash; all six base feature tables matched exactly. Neither check rescored the lockbox. The versioned iteration 2 component was reused rather than refitted. Submission schema, exact sample membership, uniqueness, allowed labels, and read-back validation all passed.

## Evidence limits

The corrected lockbox was excluded from current fitting and selection, but its clients were part of earlier development/evaluation histories. It is not a prospectively pristine sample. Repeated-CV uncertainty is conditional on these clients and the developed recipes; the 162-way selection can be optimistic. Test descriptions are noisier than valid descriptions. Normalized ranker/blend scores are not claimed to be calibrated customer probabilities. No hidden-test performance or improvement is asserted.

The defensible conclusion is a development gain from probability blending, not a proven improvement over main on the held-out view. A future improvement claim needs new untouched evaluation data rather than repeated optimization against this lockbox.
