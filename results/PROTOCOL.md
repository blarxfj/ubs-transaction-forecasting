# Shared-protocol comparison

Development set: all train clients plus non-lockbox valid clients (2,791). Folds: five hashed client folds repeated for seeds 0, 1, 2. Lockbox: the 209 valid clients whose hashed identifier is divisible by five, predicted once by the final model. Macro-F1 is over the fixed eight labels. Intervals are 95% client bootstraps (2,000 resamples; the same resample is applied across seeds).

Validation-only and lockbox are the headline numbers: pooled scores mix in the much cleaner train clients. Differences under about 0.02 are within noise.

| Solution | Pooled CV (95% CI) | Validation-only CV (95% CI) | Lockbox (95% CI) |
| --- | ---: | ---: | ---: |
| Iteration 0 (main): ranker + none gate | 0.655 (0.637-0.674) | 0.643 (0.605-0.673) | 0.703 (0.632-0.763) |
| Iteration 1: keyword streams | 0.662 (0.645-0.680) | 0.603 (0.567-0.634) | 0.641 (0.567-0.703) |
| Iteration 2: amount-kernel listwise | 0.683 (0.666-0.700) | 0.616 (0.580-0.645) | 0.592 (0.516-0.661) |
| Iteration 3: parser softmax | 0.688 (0.671-0.706) | 0.679 (0.643-0.711) | 0.668 (0.593-0.729) |

## Per-seed validation-only macro-F1

| Solution | Seed 0 | Seed 1 | Seed 2 |
| --- | ---: | ---: | ---: |
| Iteration 0 (main): ranker + none gate | 0.650 | 0.638 | 0.640 |
| Iteration 1: keyword streams | 0.600 | 0.615 | 0.594 |
| Iteration 2: amount-kernel listwise | 0.629 | 0.602 | 0.616 |
| Iteration 3: parser softmax | 0.677 | 0.677 | 0.682 |

## Per-class F1 (validation-only, mean over seeds)

| Solution | cloud | gym | insurance | mobile | music | software | streaming | none |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Iteration 0 (main): ranker + none gate | 0.637 | 0.697 | 0.585 | 0.689 | 0.575 | 0.599 | 0.610 | 0.749 |
| Iteration 1: keyword streams | 0.641 | 0.662 | 0.570 | 0.628 | 0.482 | 0.586 | 0.558 | 0.698 |
| Iteration 2: amount-kernel listwise | 0.654 | 0.659 | 0.541 | 0.658 | 0.486 | 0.580 | 0.595 | 0.751 |
| Iteration 3: parser softmax | 0.664 | 0.695 | 0.645 | 0.723 | 0.613 | 0.624 | 0.652 | 0.814 |

## Per-class F1 (lockbox)

| Solution | cloud | gym | insurance | mobile | music | software | streaming | none |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Iteration 0 (main): ranker + none gate | 0.756 | 0.622 | 0.731 | 0.757 | 0.700 | 0.667 | 0.625 | 0.768 |
| Iteration 1: keyword streams | 0.667 | 0.667 | 0.800 | 0.556 | 0.585 | 0.471 | 0.625 | 0.756 |
| Iteration 2: amount-kernel listwise | 0.698 | 0.600 | 0.640 | 0.452 | 0.595 | 0.462 | 0.558 | 0.736 |
| Iteration 3: parser softmax | 0.682 | 0.667 | 0.745 | 0.615 | 0.703 | 0.579 | 0.560 | 0.789 |
