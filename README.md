# Uplift & Causal Inference: Who Should We Target?

An end-to-end uplift-modeling project on the Criteo Uplift Prediction Dataset (v2.1, ~14M rows from a
randomized ad campaign). It answers two questions: **did the campaign work**, and **who should we
target next time** to get the most incremental visits per rupee spent.

## Result in one page

| Question | Answer |
| --- | --- |
| Did it work? | Yes. Treated users visited +1.03pp more (95% CI +1.01 to +1.06pp), a 27% relative lift. Randomization checks (SRM, covariate balance) passed. |
| Can a model tell who to target? | Every model beats random ranking. **The best uplift model (causal forest) beats a plain "who visits" classifier by a small margin that is *not* statistically clear** (paired Qini AUC difference +0.0056, 95% CI -0.0024 to +0.0117). |
| Does targeting matter for money? | Yes. With illustrative prices (INR 0.5 per impression, INR 20 per incremental visit), targeting everyone loses about INR 293k on the 1M-user test sample; targeting the top 10% by uplift earns about INR 75.6k, against INR 61.2k for the response-model ranking at its best budget. |
| Is the streaming path correct? | On a simulated replay of 200,000 events, the streaming ATE equals the batch ATE exactly (0.0082125), and single-process scoring runs at about 17k events/s with a 29 ms p95 batch latency. |

All lifts are from Criteo's non-uniformly sub-sampled data: they are method comparisons, not claims
about Criteo's real ad economics. Rupee figures are labelled assumptions, not Criteo data.

## What is in the repo

| Path | What it does |
| --- | --- |
| `src/uplift/etl.py` | PySpark CSV to Parquet; hard-gated SRM and covariate-balance checks; deterministic treatment x visit stratified split |
| `src/uplift/ab_test.py` | ATE, normal and Spark bootstrap CIs, CUPED-style adjustment, power/MDE, CACE via IV |
| `src/uplift/models.py` | Seven models behind one interface: random, response baseline, S/T/X-learner, class transformation, causal forest |
| `src/uplift/evaluate.py` | Qini AUC, uplift@k, decile table, segments, paired bootstrap CIs |
| `src/uplift/train.py` | Subsampled training, MLflow logging (local SQLite), Optuna tuning on validation only, final test evaluation |
| `src/uplift/policy.py` | Budget simulation, rupee impact, operating point, savings from not targeting low-uplift users |
| `streaming/` | Replay producer (delayed outcomes), micro-batch scorer, Spark stream-stream join with running ATE, mSPRT vs peeking, KS drift monitor |
| `api/main.py` | FastAPI `/score`, `/policy`, `/health` |
| `app/dashboard.py` | Functional Streamlit view (no styling work) |
| `docker-compose.yml`, `Dockerfile`, `.github/workflows/ci.yml` | Kafka (KRaft), Postgres, MLflow, API, scorer, dashboard; CI runs the test suite |
| `docs/` | Written results: `ab-test-verdict.md`, `uplift-results.md`, and the JSON they come from |

## Method notes worth knowing

- **Leakage guard.** `treatment` (randomized assignment) is the only valid effect variable.
  `exposure` happens after randomization and is never a feature: models see only `f0`-`f11`, the API
  rejects requests containing anything else, and replay events never carry outcomes or `exposure`.
- **Splits.** Stratified on treatment x visit with a content-hash ordering, so the split is
  identical regardless of Spark partitioning. Tuning and model selection use validation only; the
  test split is scored once.
- **Class transformation** is reweighted for the 85/15 treatment split; unweighted, it behaves like
  a response model on this data (near-identical Qini AUC in an ad-hoc comparison, not reproduced in
  the committed results). Reweighted, it scores 0.063 Qini AUC (`docs/uplift_results.json`).
- **Peeking demo.** On the replayed stream the naive p-value first drops below 0.05 at n = 8,000,
  while the always-valid mSPRT confidence sequence first excludes zero at n = 46,500. The wider
  sequence is the price of being valid however often you look.

## Honest limits

- Model training used a 500k-row subsample and 1M test rows, 5 Optuna trials per tuned model and
  100 bootstrap resamples, on a Windows laptop. Full-split training (`make train-full`) has not been run, and the near-tie
  with the response model is the fragile part of the conclusion.
- **Kafka path, run once on a small sample.** Against a live single-broker Kafka (`docker compose`),
  the replay producer sent 3,000 exposures plus 3,000 delayed outcomes; the scorer consumed the
  exposures, scored them with the real causal-forest model and wrote 3,000 decisions to both the
  `decisions` topic and Postgres; and the Spark job joined all 3,000 and reproduced the batch ATE
  exactly (`docs/streaming_kafka_results.json`). The 329-treated-at-10%-target count was read off
  Postgres during that run and wasn't captured to a committed artifact, so treat it as observed, not
  reproducible from this repo. The API and dashboard images were built and their containers answered
  `/health`, `/score`, `/policy` and the Streamlit health check. This was a single functional run,
  not a load test: the 17k events/s figure is in-process scoring only, and the `scorer`/`mlflow`
  compose services were not started.
- Model loading: the scorer and API read `models/best_model.joblib` by default. Setting `MODEL_URI`
  (e.g. `models:/uplift@production`, alongside `MLFLOW_TRACKING_URI`) loads it from the MLflow Model
  Registry instead (`make train --register-model uplift` registers it); that path is covered by
  `tests/test_registry.py` but not exercised by CI.
- Drift detection uses a Bonferroni-corrected KS test (SciPy) rather than Evidently, to avoid a
  heavy dependency; it is unit-tested and wired into the scorer (publishes to the `drift` topic), the
  producer can inject a mid-stream shift (`--drift-feature f0`), and `make drift-demo` replays one
  offline to `docs/drift_demo.json`, but no live drift demo run is recorded yet.
- "Streaming ATE equals batch exactly" compares the Spark job's running ATE against a batch ATE
  computed from the *same replayed rows* (the producer's `meta.json`), not against the full
  classic A/B test on the whole test split (`docs/ab_test_results.json`). It proves the streaming
  join doesn't lose or double-count events, not that streaming and full-sample batch analysis agree.

## Run it

```bash
pip install -r requirements.txt && pip install -e .
# Criteo v2.1 CSV -> data/raw/criteo-uplift-v2.1.csv (Hugging Face: criteo/criteo-uplift)
make data ab-test train policy      # batch path: ETL, A/B analysis, models, policy
make replay-files stream-files      # streaming path without Kafka
make serve                          # API on :8000
make dashboard                      # Streamlit on :8501
make up                             # full docker stack (Kafka, Postgres, MLflow, API, scorer, app)
make test
```

Requires Java 17 for Spark. On Windows, `src/uplift/spark_env.py` sets JAVA_HOME/HADOOP_HOME
defaults (edit the paths there if yours differ).
