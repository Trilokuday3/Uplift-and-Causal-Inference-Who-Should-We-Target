.PHONY: data ab-test train policy replay replay-files stream-files stream score benchmark serve dashboard up down test

DATA_DIR ?= data
RAW_CSV ?= $(DATA_DIR)/raw/criteo-uplift-v2.1.csv
PROCESSED_DIR ?= $(DATA_DIR)/processed
MODELS_DIR ?= models
KAFKA ?= localhost:29092
EPS ?= 500
LIMIT ?= 200000
SAMPLE_SIZE ?= 500000
TEST_SAMPLE_SIZE ?= 1000000
N_TRIALS ?= 5
N_BOOT ?= 100

# One-time setup: pip install -r requirements.txt && pip install -e .

data:
	python -m uplift.etl --input $(RAW_CSV) --output-dir $(PROCESSED_DIR)

ab-test:
	python -m uplift.ab_test --input-dir $(PROCESSED_DIR) --output docs/ab_test_results.json

## Defaults reproduce the run reported in README.md; override e.g. `make train SAMPLE_SIZE=200000` for a quick local run.
train:
	python -m uplift.train --input-dir $(PROCESSED_DIR) --output docs/uplift_results.json --artifacts-dir $(MODELS_DIR) \
		--sample-size $(SAMPLE_SIZE) --test-sample-size $(TEST_SAMPLE_SIZE) --n-trials $(N_TRIALS) --n-boot $(N_BOOT)

policy:
	python -m uplift.policy --artifacts-dir $(MODELS_DIR) --output docs/policy_results.json

# Simulated real-time replay of the test split through Kafka (needs `make up`).
replay:
	python -m streaming.producer --input-dir $(PROCESSED_DIR) --bootstrap-servers $(KAFKA) --eps $(EPS) --limit $(LIMIT)

score:
	python -m streaming.scorer --artifacts-dir $(MODELS_DIR) --bootstrap-servers $(KAFKA)

stream:
	python -m streaming.ab_stream --source kafka --bootstrap-servers $(KAFKA) --output docs/streaming_results.json

# Same pipeline without Kafka: replay to parquet files, then stream them with Spark.
replay-files:
	python -m streaming.producer --sink files --out-dir $(DATA_DIR)/stream --input-dir $(PROCESSED_DIR) --eps $(EPS) --limit $(LIMIT)

stream-files:
	python -m streaming.ab_stream --source files --exposures-path $(DATA_DIR)/stream/exposures --outcomes-path $(DATA_DIR)/stream/outcomes --output docs/streaming_results.json

benchmark:
	python -m streaming.benchmark --artifacts-dir $(MODELS_DIR) --input-dir $(PROCESSED_DIR)

serve:
	uvicorn api.main:app --port 8000

dashboard:
	streamlit run app/dashboard.py

up:
	docker compose up -d --build

down:
	docker compose down

test:
	pytest tests/ -v
