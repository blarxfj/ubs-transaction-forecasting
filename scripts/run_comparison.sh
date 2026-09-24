#!/usr/bin/env bash
# Run all development comparisons before opening the final holdout.
set -euo pipefail
DATA=${1:?Usage: bash scripts/run_comparison.sh /external/dataset.zip [output]}
OUT=${2:-artifacts/comparison}
uv sync --locked
run() { uv run python scripts/compare.py "$@" --data "$DATA" --output "$OUT"; }
run prepare
run previous2
run previous1
for model in main listwise activity weighted; do
  run cv --model "$model"
done
run summarize
run select
for model in main listwise activity weighted; do
  run predict --model "$model"
done
run previous1-final
run final
PYTHONHASHSEED=123 run verify
