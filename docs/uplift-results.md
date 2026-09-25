# Uplift Models & Evaluation — Results

Real run on Criteo Uplift v2.1 (`visit` outcome). Reproduce with `make data` then
`python -m uplift.train --sample-size 500000 --test-sample-size 1000000 --n-trials 5 --n-boot 100`.
Numbers below come from `docs/uplift_results.json`; the plots are in `notebooks/03_uplift.ipynb`.

## Setup (and what was reduced)

- Train on a stratified (treatment x visit) **500k-row subsample** of the 8.39M-row train split;
  tune on a 250k subsample of validation; evaluate on a **1M-row stratified subsample of the 2.8M
  test split**. Full-split training was not run (time/memory on a Windows laptop).
- Optuna: **5 trials** per tuned model (top two uplift models by validation Qini AUC: causal forest
  and X-learner). With so few trials the sampler is still in its random start-up phase, so this is
  close to a 5-point random search.
- Bootstrap: 100 resamples, shared across models so differences are **paired**. The best model was
  chosen on **validation**, never test. Tuned models are scored by their best-of-5 validation
  result while the others use one untuned fit, which slightly favours the tuned ones
  (untuned validation metrics are kept in the JSON under `untuned_val_metrics`).
- Base learners are small LightGBM models (<= 200 trees).
- `class_transformation` is reweighted so treated and control carry equal mass. scikit-uplift's
  version assumes a 50/50 split; on Criteo's 85/15 split the unweighted version behaves like a
  response model. A first run without the fix scored 0.0856, essentially identical to the response
  model; after the fix it scores 0.0626.

## Held-out test results

Paired difference = model Qini AUC minus response-model Qini AUC, with a 95% bootstrap CI.

| Model | Qini AUC | 95% CI | vs response model (paired) | uplift@10% | uplift@30% |
| --- | --- | --- | --- | --- | --- |
| random | -0.0001 | [-0.0052, 0.0062] | -0.0861 [-0.0989, -0.0735] | 0.0080 | 0.0105 |
| response_model | 0.0860 | [0.0758, 0.0964] | — | 0.0516 | 0.0295 |
| s_learner | 0.0882 | [0.0788, 0.0986] | +0.0022 [-0.0028, +0.0062] | 0.0676 | 0.0299 |
| t_learner | 0.0796 | [0.0702, 0.0908] | -0.0064 [-0.0188, +0.0057] | 0.0607 | 0.0288 |
| x_learner | 0.0886 | [0.0808, 0.0973] | +0.0026 [-0.0088, +0.0141] | 0.0623 | 0.0307 |
| class_transformation | 0.0626 | [0.0537, 0.0734] | -0.0234 [-0.0325, -0.0132] | 0.0595 | 0.0257 |
| **causal_forest** (best on validation) | **0.0916** | [0.0819, 0.1035] | +0.0056 [-0.0024, +0.0117] | 0.0628 | 0.0302 |

## What this does and does not show

- **Every real model beats random** (Qini AUC 0.06–0.09 vs ~0; paired CIs exclude zero).
- **The success criterion "best uplift model beats the response model with a clear margin" is NOT
  met.** Causal forest is +0.0056 above the response model, but the paired 95% CI
  [-0.0024, +0.0117] includes zero. S-, X-learner and causal forest are all slightly ahead in point
  estimate and none is distinguishable from the response model. `class_transformation` is
  significantly *worse* than it (its CI is entirely below zero), and T-learner is not distinguishable.
- A plausible reason (not proven here): in Criteo the treatment effect scales with baseline visit
  propensity, so ranking by response is already a strong ranking by uplift.
- The **top of the ranking** is where uplift models look best: uplift@10% is 0.060–0.068 for
  S/T/X/causal-forest/class-transformation vs 0.052 for the response model. No CI was computed
  for uplift@k, so treat this as suggestive.
- **Decile check (causal forest):** observed uplift by predicted-uplift decile is +6.3pp for
  decile 1, +1.1pp for decile 2, +0.2pp for decile 3, and about zero for deciles 4–10 (it is
  flat and noisy rather than strictly decreasing there). Essentially all the incremental effect
  sits in the top ~20% of the ranking.
- **Segments** (defined from model scores relative to the median; see `evaluate.segment_table`):
  persuadables (17% of users) show +4.6pp observed uplift; sure things (31%) +0.2pp; lost causes
  (50%) ~0; the lowest-uplift group (2%) shows +0.5pp — i.e. **not negative**, so this data gives no
  evidence of a real "sleeping dogs" group. Individual response types are not observable, so these
  are model-based groupings with observed uplift as the check.

## Caveats

- Criteo's data is non-uniformly sub-sampled; absolute lifts are not claims about Criteo's real ads.
- Single seed, single split, reduced tuning and bootstrap counts. Larger training samples and more
  Optuna trials could move the ranking; the near-tie with the response model is the fragile part.
- Qini AUC here is scikit-uplift's normalised score on the sampled test set, not comparable to
  numbers from other papers without the same normalisation.
