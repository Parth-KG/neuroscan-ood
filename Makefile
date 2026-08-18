.PHONY: install test lint format prepare audit r1 r2 r3 r4 r5

install:
	pip install -r requirements.txt && pip install -e .

test:
	pytest -q

lint:
	ruff check src scripts tests && ruff format --check src scripts tests

format:
	ruff format src scripts tests

# NEUROSCAN_ROOT must point at your working folder (holds data/ and runs/)
prepare:
	python scripts/prepare_data.py --raw-root $(NEUROSCAN_ROOT)/data/raw --out-root $(NEUROSCAN_ROOT)/data/prepared

audit:
	python scripts/audit_sources.py --config configs/audit.yaml

r1:
	python scripts/run_r1.py --config configs/r1.yaml --seeds 0 1 2

r2:
	python scripts/run_r2.py --config configs/r1.yaml

r3:
	python scripts/run_r3.py --config configs/r1.yaml

r4:
	python scripts/run_r4.py --config configs/r1.yaml --severity 3

r5:
	python scripts/run_r5.py --config configs/r1.yaml --severity 3

