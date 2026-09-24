# Changes versus the previous best solution

The previous best solution under the shared protocol is the main branch (iteration 0): a
per-(client, family) binary ranker plus a separate client-level `none` gate, trained on
label-independently re-noised train histories. This branch keeps its parser, its re-noising, and
its stream features, and replaces the model head.

All scores are macro-F1 over the fixed eight labels under the shared protocol described in
[results/PROTOCOL.md](results/PROTOCOL.md): development set = 2,000 train clients + 791
non-lockbox valid clients, five hashed client folds repeated with seeds 0, 1, 2, and a 209-client
valid-only lockbox predicted once at the end. Validation-only CV and the lockbox are the headline
numbers. Differences under about 0.02 are noise.

## Side by side

| Solution | Pooled CV | Validation-only CV | Lockbox (209 clients) |
| --- | ---: | ---: | ---: |
| Iteration 0 (main): ranker + none gate | 0.655 (0.637-0.674) | 0.643 (0.605-0.673) | 0.703 (0.632-0.763) |
| Iteration 1: keyword streams, binary + logistic stage | 0.662 (0.645-0.680) | 0.603 (0.567-0.634) | 0.641 (0.567-0.703) |
| Iteration 2: amount-kernel candidates, listwise ranker | 0.683 (0.666-0.700) | 0.616 (0.580-0.645) | 0.592 (0.516-0.661) |
| **Iteration 3 (this branch): parser features, eight-candidate softmax** | **0.688 (0.671-0.706)** | **0.679 (0.643-0.711)** | 0.668 (0.593-0.729) |

Paired client bootstrap of iteration 3 minus iteration 0 on the validation-only CV: +0.032
(95% interval +0.012 to +0.052). On the lockbox the paired difference is -0.036 (95% interval
-0.090 to +0.018): not distinguishable from zero on 209 clients. Against iteration 2 the lockbox
difference is +0.075 (+0.009 to +0.141).

Under the shared protocol iteration 0 was already the best of the three earlier attempts on both
headline views. Its pooled score is the lowest because it never trains on clean train-level
descriptions: held-out train clients are scored on their clean histories by a model that only
saw re-noised ones.

## What changed

1. **Eight-candidate listwise softmax head** (`model.py:candidate_rows`, `fit_listwise`). Each
   client becomes eight rows: the seven family rows of the existing feature table plus one `none`
   row that carries only the client-level features. One gradient-boosted scorer is trained with a
   per-client softmax cross-entropy (custom objective) instead of a binary ranker plus a separate
   `none` gate, so the `none` decision competes directly with the family evidence. This is the
   change that matters: validation-only CV moves from 0.643 to 0.672 on the unchanged feature set
   and the `none` F1 from 0.74 to 0.81.
2. **Client summaries on every candidate row** (`x_max_prob`, `x_max_n`, `x_min_next`, `x_n_fam`,
   `x_max_active`). Removing them costs about 0.01 (0.679 to 0.669).
3. **Churn and window features on the primary stream** (`streams.py`, `features.py`): counts in
   30/60/90/180-day windows, last-gap ratio, duplicates, missed cycles, gap regularity, last-amount
   deviation, outliers, MCC swaps and generic names among the last events, fee rate, day-of-month
   spread, refunds after the last payment, plus family event counts, client window counts, and
   cross-family count and recency ranks. Worth about +0.007 for the softmax head (0.672 to 0.679)
   and +0.004 for the old head: within noise on their own, kept because they are consistent across
   seeds.
4. **Shared protocol module and re-scoring tools** (`protocol.py`, `scripts/`), and the two
   earlier attempts ported as components (`components/keyword_streams.py`,
   `components/listwise_candidates.py`) so that all three re-scores and any blend can be
   reproduced from this branch. The ports reproduce the original out-of-fold probabilities to
   machine precision when fed the same clients in the same order.

## What was tried and did not help

Validation-only CV, mean over seeds 0/1/2, old head unless stated; baseline 0.643.

| Variant | Validation-only CV | Note |
| --- | ---: | --- |
| Include the sequence feature group previously left out | 0.643 | no change |
| Include all experimental features | 0.645 / softmax 0.674 | no change / slightly worse |
| Two extra re-noised train views and one extra noised valid view | 0.640 | no change |
| Same, with view weights halved | 0.645 | no change |
| Valid rows weighted 2x | 0.640 | no change |
| Only test-level re-noised train views | 0.647 | no change |
| Only valid-level re-noised train views | 0.640 | no change |
| 800 trees at learning rate 0.015 | 0.643 / softmax 0.678 | no change |
| 31 leaves | 0.635 | slightly worse |
| 7 leaves, 600 trees | softmax 0.671 | slightly worse |
| Column subsampling 0.5 | 0.647 | no change |
| Lambdarank objective instead of softmax | 0.668 | worse than softmax |
| Blend of the three earlier solutions (0.6/0.2/0.2 of iterations 0/1/2) | 0.664 | +0.02 over iteration 0, but below the softmax head alone |
| Blend of the softmax head with iterations 1 and 2 | 0.686 at best | +0.005 to +0.008: noise |
| Nested logistic stacking of the three earlier solutions | 0.632-0.636 | worse than a fixed blend |
| `none` probability multiplier 0.7-1.2 on the blend | 0.657-0.667 | flat; left at 1.0 |

**Blending under test-level noise.** Valid histories were additionally corrupted to the measured
test noise level and scored inside the seed-0 folds. The softmax head holds at 0.682 (raw 0.677),
the keyword-stream component drops from 0.592 to 0.564, and the amount-kernel listwise component
from 0.629 to 0.578. Blends that include them gain at most +0.003 on the noised view. The final
recipe therefore gives all weight to the softmax head; the ported components remain runnable
through `ubs-forecast ensemble --components ...` for re-scoring.

## Process notes

- Everything above was selected on development out-of-fold data only.
- The lockbox was scored more than once. The final model's lockbox number was produced twice by
  identical fits: during an end-to-end smoke test of the pipeline with the primary component
  alone, and by the final run. A three-component run launched before the recipe was frozen also
  scored the lockbox for its components and for the provisional 0.6/0.2/0.2 blend (0.707,
  0.636-0.766). The recipe was frozen on development evidence before that number existed and was
  not changed afterwards; it is reported here so the lockbox is not mistaken for an untouched
  holdout. The earlier attempts' lockbox scores come from their own final models.
- Running the ported components inside this branch reproduces the earlier attempts' out-of-fold,
  lockbox, and test probabilities to the 1e-10 precision of the written CSV files.
- Determinism: two independent full runs produce byte-identical `submission.csv`; the SHA-256 is
  recorded in [deliverables/README.md](deliverables/README.md).
