# Recurring-transaction forecasting

A reproducible CPU workflow that reconstructs recurring payment streams and scores eight candidate merchant families per client.

- Read [report.md](report.md) for the approach, measured results, limitations, and exact reproduction commands.
- Install the fully pinned dependencies in `requirements.txt` with Python 3.12.12.
- `run.py cv` runs repeated client-level cross-validation and its bootstrap interval.
- `run.py final` fits the frozen `recipe.json` and creates probabilities plus a submission.
- `test_solution.py` and `validate.py` check feature invariants, split integrity, probability schemas, and submission reproducibility.

The original dataset is required but is not included. No external API or challenge submission is performed by this code.
