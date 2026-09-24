# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

- Use Python 3.12 through `uv sync --locked`; run `make lint` and `make test` before committing.
- Keep the challenge dataset outside the repository and pass it with `--data` or `UBS_DATA_DIR`; see `README.md` for workflows.
- `ubs_baseline.py` is an immutable golden reference. Its script/output hashes live in `src/ubs_forecasting/reference.py` and are protected by `tests/test_golden_reference.py`.
- Generated models, OOF tables, probabilities, and submissions belong under ignored artifact/result subdirectories. Commit only summarized Markdown under `results/`.
- The canonical architecture and evidence boundaries are documented in `README.md`; shared-protocol results are in `results/PROTOCOL.md` and `CHANGES.md`, iteration-0 single-split results in `results/VALIDATION.md` and `results/CALIBRATION.md`.
- `deliverables/` is the one tracked exception for generated tables: the final protocol run's out-of-fold, lockbox, test probability, and submission files. Regenerate them with `ubs-forecast ensemble`; never edit them by hand.
- Model selection uses development data only (`src/ubs_forecasting/protocol.py`); the valid-only lockbox is scored once by the final run and must not steer any choice.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
