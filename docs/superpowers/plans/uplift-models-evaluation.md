# Uplift Models & Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans (inline). TDD every task.

**Goal:** Fit and honestly evaluate seven uplift-ranking methods on the Criteo splits, log to MLflow, tune the top two, and write up results.

**Architecture:** `models.py` (adapters + registry) and `evaluate.py` (pure metric functions) are independent; `train.py` composes them with pandas-loaded splits, MLflow and Optuna. Notebook renders plots from a results JSON.

**Tech Stack:** EconML, scikit-uplift, LightGBM, MLflow (local file store), Optuna, pandas/pyarrow, pytest.

**Spec:** `docs/superpowers/specs/uplift-models-evaluation-design.md`

## Global Constraints

- `exposure` is never a feature; models see only `FEATURE_COLS` (from `uplift.constants`).
- Tuning uses the validation split only; test split is evaluated once at the end.
- No dates in file/folder names. Branch is `feat/uplift-models` (no "claude"). Commits only on this branch, never `main`.
- Tests must not touch the network or real data; use synthetic data. Fast: model tests use <= 2000 rows and tiny estimators.
- Every model: `fit(X: DataFrame, treatment: ndarray, y: ndarray)` and `predict_uplift(X: DataFrame) -> ndarray` of length len(X), finite.

## Review Focus

1. All-one-arm / constant-outcome input to metrics (no positives): must not crash silently with NaN; raise a clear error or return 0.
2. Bootstrap resamples that lack a treated or control row: must be skipped/resampled, not crash.
3. Subsample must preserve treatment × outcome strata proportions and be deterministic per seed.
4. Optuna tuning must never read the test split.
5. Ties in predicted uplift (e.g. `random` degenerate, constant scores) in decile/segment code: no empty-bin crash.

---

### Task 1: Dependencies, registry skeleton, `random` and `response_model`

**Files:** Modify `requirements.txt`, `.gitignore` (add `mlruns/`); Create `src/uplift/models.py`, `tests/test_models.py`; Modify `conftest.py` (add `uplift_frame` fixture).

**Interfaces:** Produces `MODEL_REGISTRY: dict[str, Callable[..., UpliftModel]]`, `build_model(name, seed=42, **params)`, `feature_matrix(df) -> DataFrame` (FEATURE_COLS only).

- [ ] Add `econml`, `lightgbm`, `mlflow`, `optuna`, `matplotlib`, `scikit-learn` to requirements.txt.
- [ ] `uplift_frame` fixture (conftest): factory `(n=1500, seed=0)` returning DataFrame with f0..f11, treatment (p=0.5 for balanced test power), visit where uplift is strongly driven by f0: `p = 0.10 + treatment * (0.25 if f0 > 0 else -0.05)`; plus exposure column.
- [ ] Tests: `test_registry_has_all_seven_models`; `test_feature_matrix_excludes_leaky_columns` (columns == FEATURE_COLS); `test_random_model_shape_finite_and_seeded` (same seed same output, different seed differs); `test_response_model_ignores_treatment` (predict_uplift is a probability in [0,1]).
- [ ] Watch fail (ImportError), implement, watch pass, commit.

### Task 2: S-, T-, X-learner adapters (EconML)

**Files:** Modify `src/uplift/models.py`, `tests/test_models.py`.

- [ ] Parametrized test over `["s_learner","t_learner","x_learner"]`: fit on `uplift_frame(1500)`, predict on held-out frame; assert length, finite, and **directional correctness**: mean predicted uplift for rows with f0 > 0.5 exceeds mean for rows with f0 < -0.5.
- [ ] Implement with `LGBMRegressor(n_estimators=60, learning_rate=0.1, num_leaves=15, verbose=-1, random_state=seed)` as base learner (regressor on 0/1 outcome estimates probability); XLearner also needs propensity model (`LogisticRegression`) . Params override via `**params`.
- [ ] Commit.

### Task 3: `class_transformation` and `causal_forest` adapters

- [ ] Same parametrized directional test for `["class_transformation","causal_forest"]` (causal forest `n_estimators=100`, `min_samples_leaf=20`, discrete treatment, LGBM nuisance models).
- [ ] Implement `ClassTransformation(estimator=LGBMClassifier(...))` from `sklift.models` (its `fit(X, y, treat)`; adapter reorders args) and `CausalForestDML(model_y=LGBMRegressor, model_t=LGBMClassifier, discrete_treatment=True, ...)` with `.effect(X)`.
- [ ] Commit.

### Task 4: evaluate — Qini, uplift@k, decile table

**Files:** Create `src/uplift/evaluate.py`, `tests/test_evaluate.py`.

**Produces:** `qini_curve(y, treatment, scores) -> (x, y)`, `qini_auc(y, treatment, scores) -> float`, `uplift_at_k(y, treatment, scores, k) -> float`, `decile_table(y, treatment, scores, n_bins=10) -> DataFrame[decile, n, treated_rate, control_rate, uplift]`.

- [ ] Tests on `uplift_frame`: oracle score (f0) has higher `qini_auc` than reversed score (-f0), which is lower than random-noise score; `uplift_at_k(..., 0.3)` for oracle > for reversed; `decile_table` returns 10 rows, `n` sums to len(y), decile 1 (highest score) uplift > decile 10 uplift for oracle; constant scores (all ties) does not crash and returns non-empty table; input with no treated rows raises `ValueError`.
- [ ] Implement over `sklift.metrics` (`qini_curve`, `qini_auc_score`, `uplift_at_k` with `strategy="overall"`). Decile via `pd.qcut(rank(method="first"))` so ties cannot create empty bins.
- [ ] Commit.

### Task 5: evaluate — segments and bootstrap CI

**Produces:** `segment_table(y, treatment, uplift_scores, response_scores, eps=0.0) -> DataFrame[segment, n, share, observed_uplift]`, `bootstrap_qini_ci(y, treatment, scores_by_model: dict[str, ndarray], n_boot=200, seed=42, alpha=0.05) -> dict[str, dict[str, float]]` with keys `auc, ci_low, ci_high`.

- [ ] Tests: four segment names present and `n` sums to len(y); oracle persuadables (f0>0 region) observed_uplift > 0 and sleeping dogs observed_uplift < persuadables; `bootstrap_qini_ci`: oracle CI low > random-score CI high on `uplift_frame(3000)`; deterministic for same seed; resamples lacking an arm are skipped (test with tiny 6-row imbalanced frame, must not raise).
- [ ] Implement (shared resample indices across models so comparisons are paired).
- [ ] Commit.

### Task 6: train.py — data loading, stratified subsample, evaluation harness

**Files:** Create `src/uplift/train.py`, `tests/test_train.py`.

**Produces:** `load_split(input_dir, name) -> DataFrame`, `stratified_subsample(df, n, seed, outcome_col) -> DataFrame`, `evaluate_models(models: dict, X, treatment, y) -> dict[name -> {qini_auc, uplift_at_10, uplift_at_20, uplift_at_30}]`, `fit_all(df_train, outcome_col, seed, names=None) -> dict[str, fitted]`.

- [ ] Tests: subsample of 1000 from a 5000-row frame keeps each treatment×outcome cell share within 2pp and is deterministic per seed, `n >= len(df)` returns the frame unchanged; `fit_all` on 1200-row frame with names=["random","response_model","s_learner"] returns three fitted models; `fit_all` never passes columns other than FEATURE_COLS (spy adapter records `X.columns`).
- [ ] Commit.

### Task 7: MLflow logging + Optuna tuning

**Produces:** `log_run(name, params, metrics, tracking_uri)`, `tune_model(name, train_df, val_df, outcome_col, n_trials, seed) -> dict params`.

- [ ] Tests: `log_run` with tmp_path file store creates a run whose metrics contain `qini_auc` (read back via `MlflowClient`); `tune_model("s_learner", ..., n_trials=3)` returns dict with keys of `SEARCH_SPACES["s_learner"]`; tuning function signature takes no test frame (guard against test leakage); `tune_model` on non-tunable name (`random`) raises `ValueError`.
- [ ] Implement `SEARCH_SPACES` in models.py: LightGBM-based learners tune `num_leaves`, `learning_rate`, `n_estimators`, `min_child_samples`; `causal_forest` tunes `n_estimators`, `min_samples_leaf`.
- [ ] Commit.

### Task 8: End-to-end CLI

**Produces:** `python -m uplift.train --input-dir DIR --output JSON --sample-size N --outcome visit --n-trials T --n-boot B --mlflow-uri URI [--models a,b,c]`.

Pipeline: load train/val/test; subsample train (and val) to `--sample-size`; fit all; evaluate on val; log to MLflow; pick top 2 non-baseline by val Qini AUC; tune each with Optuna (val only); refit tuned on train; evaluate all final models on **test** (full test, or capped by `--test-sample-size`); bootstrap CIs; write JSON containing per-model test metrics + CI, val metrics, tuned params, best-model decile table, segment table, qini curve points (downsampled to <= 200 points) for every model incl. random line.

- [ ] Test (subprocess, like `tests/test_ab_test_cli.py`): write synthetic train/val/test parquet from `uplift_frame` into tmp_path, run CLI with `--models random,response_model,s_learner,t_learner --sample-size 800 --n-trials 2 --n-boot 20`; assert JSON keys, best model's test `qini_auc` > random's, MLflow dir contains runs, `exposure` absent from any logged param.
- [ ] Commit.

### Task 9: Notebook + Makefile

- [ ] `notebooks/03_uplift.ipynb` (nbformat, built via script like existing notebooks): loads `docs/uplift_results.json`, plots all Qini curves + random, decile bars, prints segment and CI tables. Extend `tests/test_notebooks.py` pattern: notebook is valid nbformat and contains no stored outputs referencing absolute paths.
- [ ] Makefile `train` target; `.gitignore` `mlruns/`.
- [ ] Commit.

### Task 10: Real-data run and write-up

- [ ] Re-download Criteo v2.1 from HF mirror (`criteo/criteo-uplift`, `criteo-research-uplift-v2.1.csv.gz`) into `data/raw/`, run `python -m uplift.etl` to produce splits, run `python -m uplift.train` (dev: sample 300k for tuning; final: larger train sample, documented), write `docs/uplift_results.json` and `docs/uplift-results.md` with honest numbers and caveats.
- [ ] Commit.

### Task 11: Final review and finish

- [ ] Whole-branch review, one TDD fix pass, ledger, then finishing-a-development-branch (user approval for merge/push).
