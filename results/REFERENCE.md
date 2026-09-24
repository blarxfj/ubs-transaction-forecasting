# Verified Appendix B reference

`ubs_baseline.py` is retained without edits as the immutable reference implementation.

| Artifact | SHA-256 |
| --- | --- |
| Dataset zip used locally | `1afc95470f4e8641601503172be3e698ef9eaf91528d911a6a01a120911634c6` |
| `ubs_baseline.py` | `d7e28439a4a4996921df1add8a57c18f5eafc85d18d840fe9f58d250bd8c278d` |
| Reference `submission.csv` | `c977ba7b3c52d3f0273081f4d5869adc81e66ee84191b1d07b43e7354e37bdfb` |
| Reference `test_probabilities.csv` | `a0acd1079e8cc07d4eda76652ac4d67e0fad8b018a025c7cccf2ae7ac1513baa` |

The local reference run reproduced:

- valid macro-F1: 0.6758
- additionally test-noised valid macro-F1: 0.6650
- none-gate AUC: 0.9017 / 0.9020
- 1,000 contract-valid test rows
- class counts: none 242, software 132, gym 122, cloud 112, mobile 110, insurance 103, streaming 96, music 83

The modular package reproduced the reference submission byte-for-byte before augmentation hardening. The hashes are encoded in `src/ubs_forecasting/reference.py` and protected by `tests/test_golden_reference.py`. Generated CSVs and challenge data are deliberately not committed.

The final hardened pipeline produces a different local CSV because it re-noises all sparse/ambiguous candidates and uses recalibrated corruption rates. Its two independently trained runs were byte-identical:

| Final local artifact | SHA-256 |
| --- | --- |
| `submission.csv` | `2644e38847013adbe53cda1675ccd453d9bf790671633f1c8dff620463a4c88a` |
| `test_probabilities.csv` | `2e2a07e9ed3c27bb04e11bd45f852a9734ebd5beefa092bd55211e94b3f5c2b7` |

No CSV was uploaded by this project.
