# Changes

- Standardized all three earlier methods on 2,791 development clients, 209 valid-only locked clients, and three SHA-256 five-fold repetitions. Previous best by validation-only score is main, not the method with the highest pooled CV score.
- Added seven-family listwise training, description-independent activity/refund chronology, a valid-view weighting control, and development-selected probability blending. Kept the grammar, label-independent re-noising, shared family model, and none gate.
- Frozen equal six-method blend with none multiplier 0.8: validation-only macro-F1 **0.6662**, versus main **0.6426**; paired difference **+0.0236** (95% CI +0.0094 to +0.0374, conditional on selection).
- One-shot lockbox: **0.7026 versus 0.7031**. This is a tie, **not a demonstrated overall win**. The recipe was not changed after this result.
- What worked: blending previous methods already reached **0.6643**. What did not demonstrate a meaningful gain: listwise alone **0.6457**, activity features **0.6466**, valid-view weighting **0.6467**. The final blend's +0.0020 over old-only blending is noise-sized.
- Added direct external-ZIP input, fixed-eight-label client-clustered bootstrap intervals, fold/leakage checks, one-shot scoring guards, source/input hashes, and deterministic test-only refitting. Dependencies remain pinned in `uv.lock`; the golden reference is unchanged.

Run `make compare DATASET_ZIP=/external/dataset.zip OUT=artifacts/comparison` from a full clone. Generated CSVs are retained under ignored `artifacts/comparison/selected/`. See [the full comparison](results/ITERATION4.md) for side-by-side intervals, limitations, and reproduction details.
