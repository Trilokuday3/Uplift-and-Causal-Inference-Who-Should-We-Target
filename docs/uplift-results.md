# Uplift Models & Evaluation — Results

Real run on Criteo Uplift v2.1 (`visit` outcome). Reproduce with `make data` then
`python -m uplift.train --sample-size 500000 --test-sample-size 1000000 --n-trials 5 --n-boot 100`.
Numbers below come from `docs/uplift_results.json`; the plots are in `notebooks/03_uplift.ipynb`.

## Setup (and what was reduced)

- Train on a stratified (treatment x visit) **500k-row subsample** of the 8.39M-row train split;
  tune on a 250k subsample of validation; evaluate on a **1M-row stratified subsample of the 2.8M
  test split**. Full-split training was not run (time/memory on a Windows laptop).
- Optuna: **5 trials** per tuned model (top two uplift models by validation Qini AUC). With so few
  trials the sampler is still in its random start-up phase, so this is close to a 5-point random
  search, and both tuned models happened to receive identical parameters (same seed, same space).
- Bootstrap CIs: 100 resamples, paired across models. Best model chosen on **validation**, never test.
- Base learners are small LightGBM models (<= 200 trees); causal forest uses 100 trees.

## Held-out test results

| Model | Qini AUC | 95% CI | uplift@10% | uplift@30% |
| --- | --- | --- | --- | --- |
| random | 0.0011 | [-0.0045, 0.0079] | 0.0120 | 0.0107 |
| response_model | 0.0860 | [0.0763, 0.0961] | 0.0517 | 0.0295 |
| s_learner | 0.0862 | [0.0774, 0.0953] | 0.0666 | 0.0308 |
| t_learner | 0.0802 | [0.0709, 0.0904] | 0.0602 | 0.0289 |
| **x_learner** (best on validation) | **0.0875** | [0.0750, 0.0977] | 0.0614 | 0.0299 |
| class_transformation | 0.0856 | [0.0765, 0.0957] | 0.0512 | 0.0293 |
| causal_forest | 0.0807 | [0.0699, 0.0922] | 0.0607 | 0.0279 |

## What this does and does not show

- **Every real model clearly beats random** (Qini AUC ~0.08 vs ~0.001; CIs do not overlap).
- **The success criterion "best uplift model beats the response model with non-overlapping CIs" is
  NOT met.** X-learner's AUC is 0.0875 vs the response model's 0.0860 — a difference far inside the
  bootstrap noise (CIs overlap almost entirely). Honest reading: on this data, with this budget,
  uplift models are statistically indistinguishable from a good "who visits" classifier on overall
  Qini AUC.
- A plausible reason (not proven here): in Criteo the treatment effect scales with baseline visit
  propensity, so ranking by response is already a strong ranking by uplift.
- Where uplift models do look better is the **top of the ranking**: uplift@10% is 0.061–0.067 for
  S/T/X/causal-forest vs 0.052 for the response model. This is a point estimate; no CI was computed
  for uplift@k, so treat it as suggestive.
- **Decile check (x_learner):** observed uplift by predicted-uplift decile falls sharply from
  +6.1pp (decile 1) to +1.1pp (2), +0.3pp (3), then sits near zero for deciles 4–9. It is **not
  strictly monotone**: decile 10 ticks back up to +0.4pp.
- **Segments** (defined from model scores relative to the median; see `evaluate.segment_table`):
  persuadables (11% of users) show +5.8pp observed uplift; sure things (36%) +0.5pp; lost causes
  (50%) ~0; the lowest-uplift group (3%) is slightly negative (-0.06pp, i.e. essentially zero — not
  strong evidence of true "sleeping dogs"). Individual response types are not observable, so these
  are model-based groupings with observed uplift as the check.

## Caveats

- Criteo's data is non-uniformly sub-sampled; absolute lifts are not claims about Criteo's real ads.
- Single seed, single split, reduced tuning and bootstrap counts. Larger training samples and more
  Optuna trials could move the model ranking; the response-model tie is the most fragile conclusion.
- Qini AUC here is scikit-uplift's normalised score on the sampled test set, not comparable to
  numbers from other papers without the same normalisation.
