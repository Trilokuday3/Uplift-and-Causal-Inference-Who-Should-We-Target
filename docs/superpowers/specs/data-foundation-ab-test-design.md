# Sub-project 1 — Data Foundation & Classic A/B Test — Design

**Date:** 2026-09-25
**Status:** Draft, pending user review
**Parent:** `uplift-targeting-roadmap.md`
**Depends on:** nothing (first sub-project)
**Blocks:** sub-project 2 (Uplift Models & Evaluation) and everything downstream — the split
produced here is reused, unchanged, by every later sub-project.

## Purpose

Turn the raw Criteo v2.1 CSV into a validated, reproducibly-split dataset, and answer the first
of the project's two core questions — **did the campaign work?** — with a rigorous classical A/B
analysis, before any causal ML modeling starts. Matches source-doc Phases 1–2.

**In scope:** download + PySpark ETL to Parquet, SRM check, covariate balance check, stratified
train/val/test split, EDA notebook, ATE + CIs (normal + bootstrap), CUPED-style variance
reduction, power analysis, optional CACE/IV, one-page written verdict.
**Out of scope:** uplift/causal ML modeling (sub-project 2), Qini/uplift evaluation (sub-project
2), targeting policy (sub-project 3), streaming (sub-project 4), serving (sub-project 5).

## Success criteria

Directly from the source doc's evaluation table, scoped to what this sub-project produces:

1. **SRM p-value > 0.01** — chi-square test confirms the observed treatment/control split matches
   the designed 85/15 ratio (randomization wasn't broken by a bug in the download/split pipeline).
2. **Covariate SMD < 0.1** for every feature `f0`–`f11` between treatment and control (groups are
   balanced pre-treatment).
3. **ATE + 95% CI on `visit`** excludes 0 (the primary success signal), computed two independent
   ways (normal approximation, bootstrap) that agree.
4. **CUPED-style variance reduction reported** as a percentage, with the narrower resulting CI
   shown — any reduction counts as success, the number just has to be honest.
5. **Power analysis** states the minimum detectable effect at this sample size and how large a
   smaller test would need to be for the same MDE.
6. A written one-page verdict answers: did it work, by how much, how sure are we — committed to
   the repo, not just left in a notebook.

## Data

Criteo Uplift Prediction Dataset v2.1 (not v1 — v1 had a known treatment-leakage issue).

| column | type | role |
| --- | --- | --- |
| `f0`–`f11` | dense float | anonymized, randomly-projected covariates |
| `treatment` | int 0/1 | randomized assignment (intent-to-treat) — **the only valid effect-measurement variable** |
| `exposure` | int 0/1 | post-randomization; user actually saw the ad — **forbidden as a model feature or as the ATE grouping variable**, usable only as the IV-analysis outcome-of-interest for optional CACE |
| `visit` | int 0/1 | primary outcome for Phases 1–2 (avg. rate ~4.13%) |
| `conversion` | int 0/1 | secondary/harder outcome (avg. rate ~0.23%) |

Loaded via `sklift.datasets.fetch_criteo(version="2.1")` or the Hugging Face mirror if the
`sklift` download is unavailable; either loader must be pinned to v2.1 explicitly and the choice
recorded in `etl.py`'s docstring, not left implicit.

## Pipeline

1. **Download** (`Makefile` target `make data`) — fetches the raw CSV into `data/raw/` (gitignored).
2. **`src/uplift/etl.py`** (PySpark):
   - Read CSV, cast dtypes, write Parquet to `data/processed/`.
   - Log row count in and out (a mismatch is a load failure, not a silent drop).
   - **SRM check**: chi-square goodness-of-fit test, observed `treatment` counts vs. expected
     85/15. Fails loudly (non-zero exit / raised error) if p ≤ 0.01 — this is a hard gate, not a
     warning, because every downstream number is meaningless if randomization is broken.
   - **Covariate balance**: standardized mean difference for each of `f0`–`f11` between
     `treatment=1` and `treatment=0`; asserts all < 0.1, reports the table either way.
   - **Stratified split** 60/20/20 (train/val/test) stratified on `treatment × visit`, written as
     three separate Parquet outputs with a fixed seed. This exact split is reused unchanged by
     every later sub-project — recorded in `generation_meta.json`-equivalent (`split_meta.json`)
     alongside seed, row counts per split, and per-split treatment/visit rates for a leakage
     sanity check.
3. **`notebooks/01_eda.ipynb`**: outcome rates by group, `f0`–`f11` distributions, correlation —
   read-only exploration against the Parquet output, no logic that later code depends on.
4. **`src/uplift/ab_test.py`**:
   - ATE on `visit` and `conversion`: difference in means + relative lift.
   - 95% CI two ways: normal approximation (closed-form) and bootstrap (resampled in Spark —
     `B` configurable, default 2,000 resamples).
   - CUPED-style adjustment: `Y_cuped = Y - θ(X - X̄)`, `θ = Cov(Y,X)/Var(X)`, fit on the
     available pre-treatment covariates `f0`–`f11` (there's no true pre-experiment period in this
     dataset, so this is regression adjustment in the CUPED style, not classic CUPED — the
     `ab_test.py` docstring and the written verdict both say so explicitly, per the source doc's
     honesty note). Reports % variance reduction.
   - Power analysis: minimum detectable effect at the actual `n`, plus the `n` a smaller test
     would need for the same MDE.
   - Optional CACE: 2SLS/Wald estimator using `treatment` as an instrument for `exposure`, effect
     on the compliers (actually-exposed) — behind a flag, not required for sub-project success.
5. **`notebooks/02_ab_test.ipynb`**: runs `ab_test.py`'s functions, renders the CI plots and the
   CUPED before/after comparison.
6. **Written verdict** (`docs/ab-test-verdict.md`): one page — did the campaign work, effect size,
   CI, confidence, caveats (Criteo's non-uniform sub-sampling means this isn't a claim about
   Criteo's real ads).

## Leakage guards (defined here, enforced downstream)

- `exposure` is never read by any function in `ab_test.py` except the optional, explicitly-flagged
  CACE path — asserted by a test that scans `ab_test.py`'s ATE/CI functions' inputs.
- The train/val/test split is treatment-and-outcome-stratified but otherwise untouched by any
  model-fitting step in this sub-project (no model is trained here) — sub-project 2 is the first
  to fit anything, and it must load this split rather than re-deriving it, so every later
  sub-project's numbers are comparable to this one's.

## Component boundaries

| unit | does | consumed via | depends on |
| --- | --- | --- | --- |
| `src/uplift/etl.py` | CSV → Parquet, SRM check, balance check, stratified split | `make data` / CLI | PySpark |
| `src/uplift/ab_test.py` | ATE, CIs, CUPED-style adjustment, power, optional CACE | import (notebooks, tests) | statsmodels, SciPy |
| `notebooks/01_eda.ipynb`, `02_ab_test.ipynb` | exploration + reporting | Jupyter | `etl.py` / `ab_test.py` outputs |
| `tests/` | asserts SRM/balance/split/CI correctness | `pytest` | `src/uplift` |

## Tests

| test | asserts |
| --- | --- |
| `test_srm_detects_break` | injecting an artificial 90/10 split into the chi-square check yields p ≤ 0.01 (the gate actually fires) |
| `test_srm_passes_on_clean_split` | the real 85/15 data passes at p > 0.01 |
| `test_covariate_balance_all_features` | SMD computed and returned for all 12 features, real data < 0.1 |
| `test_split_is_stratified` | per-split treatment rate and visit rate are within a small tolerance of the full-dataset rates |
| `test_split_determinism` | same seed ⇒ identical split row counts / hashes |
| `test_ate_bootstrap_matches_normal_approx` | bootstrap CI and normal-approximation CI overlap substantially on the same data |
| `test_cuped_reduces_variance` | CUPED-adjusted outcome has strictly lower variance than raw outcome on data with correlated covariates |
| `test_exposure_never_used_in_ate` | static/functional check that `exposure` isn't read by the default ATE/CI code path |

Unit tests run against a small synthetic sample (a few thousand rows with known injected SRM
breaks / balance violations) so they run in seconds; the full 14M-row run is exercised by
`make data ab-test`, not by unit tests.

## Deliverables checklist

- [ ] `src/uplift/etl.py` — download wiring, PySpark CSV→Parquet, SRM check, balance check, stratified split, `split_meta.json`
- [ ] `src/uplift/ab_test.py` — ATE, dual CIs, CUPED-style adjustment, power analysis, optional CACE
- [ ] `notebooks/01_eda.ipynb`, `notebooks/02_ab_test.ipynb`
- [ ] `tests/` unit suite (table above)
- [ ] `docs/ab-test-verdict.md` — the one-page written verdict
- [ ] `.gitignore` covers `data/`
- [ ] `Makefile` targets: `make data`, `make ab-test` (at minimum; more targets added by later sub-projects)

## Risks / decisions

- **Spark is used even though this stage's split could run in pandas** — deliberate, matching the
  roadmap's stated reasoning: the Big Data skill is part of the portfolio story.
- **No true pre-experiment period** — CUPED here is regression adjustment on pre-treatment
  covariates, labeled as CUPED-*style* everywhere (code comments, notebook, verdict doc), never
  presented as classic CUPED.
- **SRM/balance checks are hard gates in `etl.py`**, not just notebook diagnostics — a broken
  split must fail the pipeline loudly, because every number downstream (this sub-project and all
  later ones) depends on randomization actually holding.
