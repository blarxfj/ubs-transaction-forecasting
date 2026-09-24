DATA_DIR ?= $(UBS_DATA_DIR)
DATASET_ZIP ?=
OUT ?= artifacts/run
MODEL ?= artifacts/model.pkl
JOBS ?= 8

.PHONY: sync data lint test cv model submission all compare

sync:
	uv sync --locked

data:
	@test -n "$(DATA_DIR)" || (echo "Set DATA_DIR to a directory outside the repository"; exit 2)
	@test -n "$(DATASET_ZIP)" || (echo "Set DATASET_ZIP to dataset.zip"; exit 2)
	mkdir -p "$(DATA_DIR)"
	unzip -q -o "$(DATASET_ZIP)" -d "$(DATA_DIR)"

lint:
	uv run ruff check .
	uv run ruff format --check src tests scripts

test:
	uv run pytest

cv:
	@test -n "$(DATA_DIR)" || (echo "Set DATA_DIR or UBS_DATA_DIR"; exit 2)
	uv run ubs-forecast evaluate --data "$(DATA_DIR)" --output "$(OUT)" --jobs "$(JOBS)"

model:
	@test -n "$(DATA_DIR)" || (echo "Set DATA_DIR or UBS_DATA_DIR"; exit 2)
	uv run ubs-forecast train --data "$(DATA_DIR)" --model "$(MODEL)" --jobs "$(JOBS)"

submission: model
	uv run ubs-forecast predict --data "$(DATA_DIR)" --model "$(MODEL)" --output "$(OUT)" --jobs "$(JOBS)"

all:
	@test -n "$(DATA_DIR)" || (echo "Set DATA_DIR or UBS_DATA_DIR"; exit 2)
	uv run ubs-forecast all --data "$(DATA_DIR)" --output "$(OUT)" --jobs "$(JOBS)"

compare:
	@test -n "$(DATASET_ZIP)" || (echo "Set DATASET_ZIP to the external dataset.zip"; exit 2)
	bash scripts/run_comparison.sh "$(DATASET_ZIP)" "$(OUT)"
