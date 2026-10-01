.PHONY: data ab-test train policy train-full drift-demo load-test replay replay-files stream-files stream score benchmark serve dashboard up down test

DATA_DIR ?= data
RAW_CSV ?= $(DATA_DIR)/raw/criteo-uplift-v2.1.csv
PROCESSED_DIR ?= $(DATA_DIR)/processed
MODELS_DIR ?= models
KAFKA ?= localhost:29092
EPS ?= 500
LIMIT ?= 200000

# One-time setup: pip install -r requirements.txt && pip install -e .

data:
	python -m uplift.etl --input $(RAW_CSV) --output-dir $(PROCESSED_DIR)

ab-test:
	python -m uplift.ab_test --input-dir $(PROCESSED_DIR) --output docs/ab_test_results.json

train:
	python -m uplift.train --input-dir $(PROCESSED_DIR) --output docs/uplift_results.json --artifacts-dir $(MODELS_DIR)

# Full-split training (hours on a laptop). Override N_TRIALS / N_BOOT to taste.
train-full:
	python -m uplift.train --input-dir $(PROCESSED_DIR) --output docs/uplift_results_full.json --artifacts-dir $(MODELS_DIR)_full --sample-size 999999999 --n-trials 30 --n-boot 200

drift-demo:
	python -m streaming.drift_demo --input-dir $(PROCESSED_DIR)

# Terminal 1: make load-test; terminal 2: make score; terminal 3: make replay EPS=20000 LIMIT=200000
load-test:
	python -m streaming.load_test --bootstrap-servers $(KAFKA) --expected $(LIMIT)

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
