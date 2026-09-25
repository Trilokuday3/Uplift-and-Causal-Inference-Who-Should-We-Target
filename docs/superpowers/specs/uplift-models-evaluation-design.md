# Sub-project 2 — Uplift Models & Evaluation — Design

**Date:** 2026-09-25
**Status:** Approved by user ("complete this") after in-chat design review
**Parent:** `uplift-targeting-roadmap.md`
**Depends on:** sub-project 1 (train/val/test Parquet splits, `FEATURE_COLS`, `TREATMENT_COL`, `VISIT_COL`, `CONVERSION_COL`)
**Blocks:** sub-project 3 (policy simulation consumes the fitted models and `evaluate.py` curves)

## Purpose

Answer the project's second question — **who should we target?** — by predicting individual
uplift τ(x) = P(Y=1 | treated, x) − P(Y=1 | control, x) with several methods, then evaluating
them honestly on held-out data. Matches source-doc Phases 3–4.

**In scope:** seven ranking methods behind one interface, Qini/uplift curves, uplift@k, decile
analysis, four-segment identification, bootstrap CIs on Qini AUC, Optuna tuning of the top two
models (validation split only), MLflow (local file store) run logging, a notebook that renders
the plots, a written results summary.
**Out of scope:** rupee-denominated policy simulation (sub-project 3), streaming/serving (4, 5),
any `app/` code.

## Decisions (made with the user)

- **Library stack:** EconML (S/T/X-learner, `CausalForestDML`) + scikit-uplift (`ClassTransformation`,
  all Qini/uplift metrics) + LightGBM (base learner and response-model baseline). `causalml` is
  deliberately not used (Windows install risk; no algorithmic gain over EconML's versions).
- **Scale:** develop and tune on a stratified subsample (treatment × visit); one full-train-split
  run for the final reported numbers. Sample size is a CLI flag, default documented in the CLI help.
- **MLflow:** local file store (`mlruns/`, gitignored). Sub-project 5 only repoints
  `MLFLOW_TRACKING_URI`.
- **Primary outcome:** `visit` (the source doc's development target); `conversion` selectable via flag.

## Components

| File | Responsibility |
| --- | --- |
| `src/uplift/models.py` | `MODEL_REGISTRY` name → factory `(seed, **params) -> UpliftModel`. Every model exposes `fit(X, treatment, y)` and `predict_uplift(X) -> np.ndarray`. Keys: `random`, `response_model`, `s_learner`, `t_learner`, `x_learner`, `class_transformation`, `causal_forest`. `SEARCH_SPACES` defines Optuna spaces per tunable model. |
| `src/uplift/evaluate.py` | `qini_curve`, `qini_auc`, `uplift_at_k`, `decile_table`, `segment_table`, `bootstrap_qini_ci`. Returns arrays/DataFrames only — no plotting. |
| `src/uplift/train.py` | CLI + functions: load splits (pandas), stratified subsample, fit-all, evaluate, MLflow logging, Optuna tuning, final test evaluation, JSON results. |
| `notebooks/03_uplift.ipynb` | Renders Qini/uplift curves (all models + random), decile bars, segment table from the results JSON. |
| `docs/uplift-results.md` | Written summary of real-data results, with caveats. |

## Leakage guards

`exposure` is never a feature: models are fit only on `FEATURE_COLS`, and a test asserts the
feature matrix handed to every model excludes `exposure`, `treatment`, `visit`, `conversion`.
Hyperparameter tuning uses the validation split only; the test split is touched once, for final
numbers. Splits come from sub-project 1 unchanged.

## Segments (definition, stated because the four groups are unobservable per person)

Using predicted uplift τ̂ and a baseline response score p̂ (the response model): persuadables
τ̂ > ε; sleeping dogs τ̂ < −ε; the rest are sure things if p̂ ≥ median(p̂) else lost causes.
Each segment reports size and the *observed* treated−control rate, which is the sanity check
(persuadables should be clearly positive, sleeping dogs ≤ 0).

## Success criteria

1. All seven models fit and produce a finite score per row on synthetic and real data.
2. Every run logged to MLflow with Qini AUC and uplift@10/20/30%.
3. Best uplift model beats `random` and `response_model` on **held-out test** Qini AUC; bootstrap
   CIs reported. If CIs overlap, the write-up says so plainly (honest result, not a failure).
4. Decile table for the best model: observed uplift generally decreasing across deciles.
5. Test suite green; no test touches the network or the real dataset.

## Risks

- Causal forest / X-learner are slow at scale → subsample-for-dev decision above.
- Rare `conversion` makes uplift noisy → `visit` is primary.
- Criteo sub-sampling: results are method comparisons, not claims about real ad economics.
