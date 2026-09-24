# Held-out validation results

Five-fold stratified CV was repeated with fold seeds 0, 17. Each held-out client was excluded from both the original-valid and additionally noised training copies. Test-level noise is the mean of corruption seeds 0, 2. Intervals are 95% client-bootstrap percentile intervals (2,000 resamples).

**Evidence scope:** Repeated client-held-out CV on a validation set used during development; intervals are conditional paired resampling, not an independent hidden-test guarantee.

| Condition | Macro-F1 (95% CI) | Accuracy | none AUC |
| --- | ---: | ---: | ---: |
| Valid noise | 0.672 (0.639-0.700) | 0.692 | 0.903 |
| Test-level noise | 0.668 (0.637-0.697) | 0.690 | 0.902 |

The paired test-noise minus valid macro-F1 difference is -0.004 (95% CI -0.013 to +0.004).

## Fold-partition sensitivity

| Fold seed | Valid macro-F1 | Test-level-noise macro-F1 |
| ---: | ---: | ---: |
| 0 | 0.665 | 0.664 |
| 17 | 0.668 | 0.667 |

## Per-class F1

| Class | Valid noise (95% CI) | Test-level noise (95% CI) |
| --- | ---: | ---: |
| cloud | 0.663 (0.579-0.734) | 0.660 (0.579-0.729) |
| gym | 0.703 (0.636-0.760) | 0.703 (0.634-0.764) |
| insurance | 0.653 (0.573-0.724) | 0.647 (0.570-0.719) |
| mobile | 0.700 (0.624-0.772) | 0.710 (0.634-0.780) |
| music | 0.622 (0.535-0.702) | 0.615 (0.532-0.694) |
| software | 0.632 (0.549-0.699) | 0.625 (0.543-0.696) |
| streaming | 0.624 (0.543-0.696) | 0.604 (0.520-0.679) |
| none | 0.778 (0.739-0.815) | 0.782 (0.741-0.818) |

## Recurrence heuristics

| Active-stream selection | Valid macro-F1 |
| --- | ---: |
| earliest next | 0.503 |
| most payments | 0.420 |
| most recent | 0.359 |

## Candidate feature ablations

Candidate groups are retained only for a gain of at least 0.01 on both views and both fold seeds. Unretained groups remain reproducibly computable but are dropped by the final model.

| Candidate | Fold seed | Valid gain | Test-noise gain | Retained |
| --- | ---: | ---: | ---: | --- |
| ordered sequence features | 0 | +0.004 | +0.005 | no |
| ordered sequence features | 17 | -0.007 | -0.008 | no |
| secondary stream timing | 0 | -0.002 | +0.004 | no |
| secondary stream timing | 17 | -0.000 | -0.003 | no |

## Leakage regression

A train-only model using noise-quality features scores 0.366; dropping those features raises it to 0.630. The final model uses those useful features only after every subscription-candidate row is re-noised without labels.
