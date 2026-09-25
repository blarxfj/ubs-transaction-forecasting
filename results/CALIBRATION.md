# Noise calibration

Statistics are conditioned on the final parser and regular >=4-payment detected streams; they are not exact noise rates in the data. The mask estimate uses a 4/3 adjustment because monthly plan is both a mobile name and a generic mask.

| Split/view | Everyday mask | Everyday MCC mismatch | Subscription mask estimate | Family-MCC match | Affixed/truncated | Streams |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| unlabeled_pretrain | 0.000 | 0.000 | 0.030 | 0.946 | 0.032 | 15812 |
| train | 0.024 | 0.013 | 0.043 | 0.935 | 0.083 | 3087 |
| valid | 0.482 | 0.091 | 0.361 | 0.833 | 0.430 | 1286 |
| test | 0.678 | 0.127 | 0.521 | 0.735 | 0.431 | 1272 |
| train_redrawn_test | 0.683 | 0.119 | 0.516 | 0.722 | 0.415 | 2746 |
| train_redrawn_valid | 0.484 | 0.091 | 0.365 | 0.836 | 0.418 | 2967 |
| valid_test_seed0 | 0.680 | 0.124 | 0.531 | 0.724 | 0.436 | 1186 |
| valid_test_seed2 | 0.676 | 0.125 | 0.531 | 0.716 | 0.434 | 1210 |

## All train subscription candidates

| State | none generic rate | non-none generic rate |
| --- | ---: | ---: |
| Before redraw | 0.155 | 0.012 |
| After redraw | 0.409 | 0.409 |
