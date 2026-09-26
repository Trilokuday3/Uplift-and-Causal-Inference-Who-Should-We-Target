.PHONY: data ab-test train policy test

DATA_DIR ?= data
RAW_CSV ?= $(DATA_DIR)/raw/criteo-uplift-v2.1.csv
PROCESSED_DIR ?= $(DATA_DIR)/processed
MODELS_DIR ?= models

data:
	python -m uplift.etl --input $(RAW_CSV) --output-dir $(PROCESSED_DIR)

ab-test:
	python -m uplift.ab_test --input-dir $(PROCESSED_DIR) --output docs/ab_test_results.json

train:
	python -m uplift.train --input-dir $(PROCESSED_DIR) --output docs/uplift_results.json --artifacts-dir $(MODELS_DIR)

policy:
	python -m uplift.policy --artifacts-dir $(MODELS_DIR) --output docs/policy_results.json

test:
	pytest tests/ -v
