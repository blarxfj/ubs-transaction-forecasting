# Model explanations

These measurements come from the final three-seed ensemble fit on all labeled data. Gain importance is an internal tree-model diagnostic, not a causal effect.

## Grouped LightGBM gain

| Component | Signal group | Gain share |
| --- | --- | ---: |
| Family ranker | timing and recurrence | 71.6% |
| Family ranker | merchant evidence | 11.5% |
| Family ranker | refunds | 7.6% |
| Family ranker | background activity | 7.3% |
| Family ranker | other | 2.0% |
| `none` gate | refunds | 37.6% |
| `none` gate | timing and recurrence | 24.3% |
| `none` gate | background activity | 19.1% |
| `none` gate | merchant evidence | 16.3% |
| `none` gate | other | 2.7% |

The split supports the intended model shape: expected timing dominates which-family ranking, while refund structure and broad activity carry much of the `none` decision.

## Contrasting local examples

The prediction command writes the ten largest absolute LightGBM logit contributions for one high-`none` and one high-family client to `local_explanations.csv`.

- **High `none`:** an ended stream carrying a refund contributes +2.24 to the `none` logit. Three subscription refunds, low active-stream length, and a distant earliest next date also push toward `none`.
- **High family (`gym`):** the selected gym stream is due in two days. Earliest timing rank, recent payment, active-stream evidence, and its refund pattern push the family ranker toward gym, while zero ended refunded streams and a long active stream push the gate away from `none`.

These examples explain model behavior on synthetic challenge clients; they do not establish causal customer behavior.

## Bounded feature experiments

Two cheap sequence/stream additions were evaluated with the same LightGBM learner, both fold seeds, and both corruption views:

- last-three gaps, largest gap, skipped-cycle fraction, and latest amount change;
- earliest next date and 30/60/90-day mass across secondary streams of the same family.

Neither met the predeclared retention rule of at least +0.01 macro-F1 on both views and both fold seeds. They remain implemented for reproducible ablation but are excluded from the final models. Detailed paired scores are in [VALIDATION.md](VALIDATION.md).
