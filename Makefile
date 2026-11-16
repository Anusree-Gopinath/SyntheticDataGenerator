.PHONY: install test lint app reference experiments

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check synthgen tests

app:
	streamlit run synthgen/app/main.py

reference:
	python -m synthgen.reference_world.simulator --config config/reference_world.yaml
	python -m synthgen.reference_world.stores --config config/reference_world.yaml

experiments:
	python -m synthgen.orchestration.run_experiments --plan config/experiment_plan.yaml
