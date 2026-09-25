# Iteration history

This is a presentation-ready record of the six solution iterations. Scores are macro-F1. The comparison table uses the shared protocol and rounds scores and intervals to three decimals; validation-only CV and the locked check are the headline views.

## Shared comparison

Pooled CV includes the much cleaner training clients and can overstate performance on the noisier validation and test populations. Differences below about 0.02 are treated as noise at this sample size.

| Iteration | Pooled CV (95% CI) | Validation-only CV (95% CI) | Locked check (95% CI) |
| --- | ---: | ---: | ---: |
| 0 — first solution | 0.655 (0.637–0.674) | 0.643 (0.605–0.673) | 0.703 (0.632–0.763) |
| 1 — independent attempt A | 0.662 (0.645–0.680) | 0.603 (0.567–0.634) | 0.641 (0.567–0.703) |
| 2 — independent attempt B | 0.683 (0.666–0.700) | 0.616 (0.580–0.645) | 0.592 (0.516–0.661) |
| 3 — improvement round 1, attempt A | 0.688 (0.671–0.706) | 0.679 (0.643–0.711) | 0.668 (0.593–0.729) |
| 4 — improvement round 1, attempt B | 0.694 (0.676–0.710) | 0.666 (0.630–0.699) | 0.703 (0.632–0.764) |
| 5 — improvement round 2 | 0.688 (0.671–0.706) | 0.679 (0.643–0.711) | 0.668 (0.593–0.729) |

The shared protocol uses five client-held-out folds repeated with seeds 0, 1, and 2. The development set contains 2,000 train clients and 791 non-lockbox validation clients. The locked check contains 209 validation clients selected by a fixed rule and is scored once at the end. Scores use all eight labels and client-level bootstrap intervals. The validation-only and locked-check values in the table are therefore directly comparable; the original runs described below are retained as historical context where their split or scoring setup differed.

## Iteration 0 — first solution

**Idea.** Detect recurring payment streams from noisy descriptions, then rank which merchant family recurs first, with a separate `none` gate.

**What changed and what we learned.** The description vocabulary is a closed grammar: a small rule parser maps all but 24 of 1,047,053 rows. Training also contains a shortcut: generic descriptions occur disproportionately on `none` clients, but valid and test do not share that pattern. A train-only model using the shortcut falls from about 0.63 to 0.37 (the recorded values are 0.630 and 0.366). The test split is the noisiest, so training histories were re-noised to test level. Refunds were a particularly stable signal.

**Original score.** The initial evaluation scored 0.672 on validation (range about 0.64–0.70) and 0.668 at test-level noise. It was deterministic and took about three minutes. The shared-protocol re-score is the iteration-0 row in the table above.

**Lesson.** The main win came from understanding the data generator—not from using a bigger model.

## Iteration 1 — independent attempt A

**Idea.** Rebuild the generator rules independently. Each family has four named merchants plus four generic names shared by all families; descriptions are corrupted by prefixes, suffixes, abbreviations, truncation, and about 5–10% merchant-code swaps. Streams have near-constant amounts on roughly 30-day cycles, with some 14-day and 45–90-day cycles. The final approach used one shared per-client/per-family binary tree scorer followed by a small logistic second stage.

**Original score.** The branch’s own results recorded 0.649 pooled CV, 0.593 on validation clients alone, and 0.661 on its locked check. That run locked away 635 clients, including train and validation clients, rather than the intended approximately 209 validation-only clients, so its locked value is not directly comparable. After the corrected shared-protocol re-score, the comparable values are 0.662 pooled, 0.603 validation-only, and 0.641 on the locked check.

**Lesson.** An independent route rediscovered the same generator structure as iteration 0. One shared scorer per family candidate beat a single eight-class scorer by about 0.06 in both independent attempts, supporting this framing. The fair re-score also showed that the higher pooled score was driven by the easier training clients.

## Iteration 2 — independent attempt B

**Idea.** Represent each client as eight candidate rows—one per merchant family plus `none`—and learn one shared tree-based scoring function across families, trained listwise so rare families can borrow strength from common ones. Add client- and family-level generic-description fractions, while treating “monthly plan” as mostly generic.

**Original score.** The branch’s own metrics recorded 0.682 pooled CV (95% interval 0.663–0.696), a mean-fold validation-only score of 0.610, and 0.592 on the 209-client locked check. Its pooled validation-only score was 0.616; that is the value used in the shared comparison table. The pooled score was again flattered by cleaner training clients.

**Lesson.** Validation-only and locked-check numbers are the honest comparison views, and they show the cost of the train/test noise gap. Every attempt should be compared on those views rather than on pooled CV alone.

## Iteration 3 — improvement round 1, attempt A

**Idea.** Keep iteration 0’s parser, re-noising, and stream features; add churn and window features; and replace the binary family ranker plus separate `none` gate with one shared eight-candidate listwise softmax scorer. The seven family rows and one `none` row compete directly.

**Score.** Validation-only CV reached 0.679 versus 0.643 for iteration 0. The paired validation-only gain was +0.032 (95% interval +0.012 to +0.052), a real gain on that measure. The locked check was 0.668 versus 0.703; the difference was within noise. Across three seeds, validation-only scores were 0.677, 0.677, and 0.682.

**Lesson.** Letting `none` compete head-to-head with the families in one scorer beat both the separate gate and blending. Blends added at most about +0.005–0.008 on top, so the single shared scorer was retained.

## Iteration 4 — improvement round 1, attempt B

**Idea.** Re-score the earlier solutions fairly, then blend the existing solutions and their components: six components at equal weights, with a tuned 0.8 multiplier on `none`. Single-scorer changes—listwise ranking plus a `none` gate, activity and refund chronology, and extra validation weighting—each contributed less than +0.005.

**Score.** Validation-only CV was 0.6662 versus 0.6426 for iteration 0, a +0.0236 difference. The locked check was 0.7026 versus 0.7031, a −0.0005 difference. Rounded in the comparison table, this is 0.666 versus 0.643 and 0.703 versus 0.703. A fresh rebuild reproduced the same deterministic submission.

**Lesson.** Blending diverse solutions beat every single-scorer tweak, but the gain was at the edge of the noise and the locked check did not move. The honest conclusion was “slightly better, not proven.”

## Iteration 5 — improvement round 2

**Idea.** Start from iteration 3 and test 12 variants, including pseudo-labelled views of the 10,000 unlabeled clients, blends with iteration 4, stronger stream-end signals, calibration of the `none` row, training-set cleaning, bagging, and regularization.

**Score.** No variant beat iteration 3 outside noise, so the retained recipe remained identical: 0.688 pooled, 0.679 validation-only, and 0.668 locked check. The best changes were within about +0.002/−0.005; giving pseudo-labelled views full weight reduced validation-only CV by 0.029, and adding longer pseudo-cutoff views reduced it by 0.049. The retained outputs were reproducible and byte-identical to iteration 3’s outputs.

**Lesson.** This was a diminishing-returns round, consistent with substantial randomness in which stream recurs first. The stopping rule was applied: a further round that did not produce a gain of at least 0.02 did not justify continuing.

## Summary for slides

- **Best retained solution:** iteration 3’s eight-candidate softmax scorer on parser and stream features: validation-only macro-F1 0.679 (0.643–0.711), locked check 0.668.
- **Biggest wins, in order:** understand the data generator (closed grammar, train-only shortcut, and noise gap); re-noise training data to test level; use one shared scorer where `none` competes with the families.
- **What did not help:** bigger single models, heavy pseudo-labelling of unlabeled clients, and blending beyond small gains.
- **Process:** two independent blind attempts, fair identical re-scoring, then improvement rounds with a stopping rule and a locked final check. This made the comparisons explicit and kept pooled CV from being mistaken for the honest headline.
