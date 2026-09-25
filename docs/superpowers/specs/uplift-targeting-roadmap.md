# Uplift & Causal Inference: Who Should We Target? — Roadmap

**Date:** 2026-09-25
**Status:** Approved (design). Sub-project specs and plans produced one at a time.
**Source brief:** `../../../Uplift & Causal Inference Who Should We Target.md` (the original
project doc, kept at repo root as the canonical source of dataset facts, target metrics, and
interview prep — this roadmap scopes it into buildable sub-projects; it doesn't replace it).

## Goal

A portfolio-proof, end-to-end uplift-modeling system on the Criteo Uplift Prediction Dataset
(v2.1, ~14M rows) that answers two questions: did a randomized ad campaign work, and who should
be targeted next time to maximize incremental conversions per rupee. Learning goal stated
explicitly by the user: by the end, they should be able to independently do *everything* in this
project **except** frontend/UI craft — experiment validation, causal ML, uplift evaluation,
policy design, streaming systems, and serving/MLOps are all first-class learning targets; the
Streamlit dashboard is built only to the extent needed to prove the pipeline end-to-end.

## Scope decision: seven phases, five sub-projects

The source doc lays out 7 phases over ~4 weeks. Building all of it as one implementation plan
would be too large to review or execute safely, so it's split into sub-projects along the same
boundaries the source doc's own week-by-week timeline already implies:

| Sub-project | Source doc phases | Focus |
| --- | --- | --- |
| 1 — Data Foundation & A/B Test | Phases 1–2 | PySpark ETL, SRM/balance checks, stratified split, classic A/B analysis (ATE, CIs, CUPED, power, optional CACE), written verdict |
| 2 — Uplift Models & Evaluation | Phases 3–4 | S/T/X-learners, causal forest, class transformation; Qini/uplift curves, decile analysis, segment identification |
| 3 — Targeting Policy & Business Impact | Phase 5 | Budget simulation, ₹-denominated impact, sleeping-dogs/sure-things savings |
| 4 — Real-Time Replay | Phase 6 | Kafka (KRaft), replay producer with delayed outcomes, scoring service, Spark Structured Streaming A/B join, sequential testing (mSPRT), drift demo |
| 5 — Serving & Presentation | Phase 7 | FastAPI `/score` + `/policy`, minimal Streamlit view (functional only, no UX polish), Docker/Makefile/CI, README case study |

Each sub-project gets its own design doc under `docs/superpowers/specs/` and implementation plan
under `docs/superpowers/plans/`, written and approved one at a time — later sub-projects are not
designed until the ones before them are implemented, so design decisions can reflect what
actually got built rather than assumptions made weeks earlier.

## Approach (chosen, from the source doc)

- **PySpark + Parquet** for batch ETL — the dataset is ~14M rows, and processing it with Spark
  (rather than pandas, which could technically hold it) is a deliberate choice: the "Big Data"
  skill is part of the portfolio story, not incidental.
- **statsmodels / SciPy** for the classical A/B statistics (CIs, CUPED-style regression
  adjustment, power, mSPRT).
- **CausalML / EconML + scikit-uplift** for meta-learners, causal forests, and Qini/uplift
  evaluation.
- **LightGBM / XGBoost** as the base learners inside the meta-learners and as the "naive response
  model" baseline the uplift models must beat.
- **MLflow** as the model registry — the live scoring service (sub-project 4) loads its
  "Production"-stage model from here, not from a local file.
- **Kafka (KRaft mode) + Spark Structured Streaming** for the replay/streaming path — always
  labeled "simulated real-time replay," never presented as live ad traffic.
- **FastAPI + Streamlit + Docker Compose** for serving; Streamlit is scoped to functional only
  (budget slider, Qini curve, decile chart, A/B verdict card render correctly) per the stated
  learning goal — no frontend design pass.
- **Evidently** for drift monitoring on sliding feature windows.
- **GitHub Actions** running tests + a small-sample training smoke test.

## Dataset facts carried forward (see source doc for full detail)

- Use **Criteo v2.1** only (`sklift.datasets.fetch_criteo` or the Hugging Face mirror) — v1 had a
  known treatment-leakage issue.
- `treatment` (randomized, intent-to-treat) is the only valid ground truth for effect
  measurement. `exposure` is post-randomization and **must never be a model feature** — this rule
  is carried into every sub-project's design and tests.
- Treatment is imbalanced (~85/15); conversions are rare (~0.23%) — `visit` is the primary
  development target, `conversion` the harder secondary one. Every stratified split is on
  treatment × visit.
- Criteo's non-uniform sub-sampling means absolute lift numbers aren't a claim about Criteo's real
  ads — results are presented as method comparisons (uplift ranking vs. response ranking vs.
  random), per the source doc's honesty notes.

## Repo/tooling conventions

- This repo (`Uplift & Causal Inference Who Should We Target/`) is its own git repository, not
  nested inside the user's home-directory repo — set up 2026-09-25 specifically to avoid mixing
  project history with unrelated personal files.
- CLAUDE.md's Git Commit Rule and branch-naming rule (no "claude" in branch names) apply from the
  first commit onward.
- `data/` is gitignored — never commit the downloaded Criteo CSV/Parquet.

## Status of this roadmap

Sub-project 1 design is being written next (`data-foundation-ab-test-design.md`), followed by
its implementation plan. Sub-projects 2–5 are named and ordered here but not designed yet.
