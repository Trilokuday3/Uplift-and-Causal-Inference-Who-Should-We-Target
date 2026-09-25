# A/B Test Verdict: Did the Campaign Work?

**Data:** Criteo Uplift Prediction Dataset v2.1 (confirmed — see Caveats), 13,979,592 users, 85.00% treated / 15.00% control. SRM check passed, all 12 covariates balanced (max SMD 0.0488, threshold 0.1). Full pipeline output: `ab_test_results.json`.

## Yes — the campaign worked, on both outcomes, with high confidence.

### Visit

| Metric | Value |
| --- | --- |
| Control rate | 3.820% |
| Treated rate | 4.854% |
| ATE | **+1.034 percentage points** |
| Relative lift | **+27.07%** |
| 95% CI (normal approx.) | [1.006pp, 1.063pp] — excludes 0 |
| 95% CI (bootstrap, 100 resamples, seed 42) | [1.005pp, 1.059pp] — excludes 0, agrees closely with the normal-approx CI |

### Conversion (harder, rarer outcome)

| Metric | Value |
| --- | --- |
| Control rate | 0.194% |
| Treated rate | 0.309% |
| ATE | **+0.115 percentage points** |
| Relative lift | **+59.45%** |
| 95% CI (normal approx.) | [0.108pp, 0.122pp] — excludes 0 |
| 95% CI (bootstrap) | [0.107pp, 0.122pp] — excludes 0, agrees closely |

Both outcomes show a clear, statistically significant positive effect. The two independent CI methods (closed-form normal approximation and Spark-native bootstrap) agree closely on both outcomes, which is itself evidence the estimate is stable rather than an artifact of one method's assumptions.

## How much of that precision came from CUPED

Regression-adjustment on the pre-treatment covariates f0–f11 (CUPED-style — see Caveats) reduced outcome variance by:
- **28.01%** for visit
- **11.89%** for conversion

| | Raw ATE | Raw 95% CI | CUPED-adjusted ATE | CUPED-adjusted 95% CI |
| --- | --- | --- | --- | --- |
| Visit | +1.034pp | [1.006pp, 1.063pp] (width 0.057pp) | +0.698pp | [0.673pp, 0.723pp] (width 0.050pp) |
| Conversion | +0.115pp | [0.108pp, 0.122pp] (width 0.013pp) | +0.092pp | [0.085pp, 0.098pp] (width 0.013pp) |

The adjusted intervals are narrower, as expected from the variance reduction — but the **adjusted point estimate also moved**, from +1.034pp to +0.698pp for visit (about 33% smaller) and from +0.115pp to +0.092pp for conversion (about 20% smaller). This is not a bug: CUPED/regression adjustment is unbiased *in expectation over repeated randomizations*, not guaranteed to exactly reproduce the raw ATE on any one realized sample. With 12 covariates, even the small imbalances already reported in the balance check (all SMDs under 0.05) combine with their fitted regression coefficients into a real, non-trivial shift here — concretely, the shift equals `Σ θᵢ × (mean_treated,ᵢ − mean_control,ᵢ)` almost exactly, which is the textbook mechanism by which finite-sample covariate imbalance leaks into a regression-adjusted estimator. **We treat the raw ATE as the headline number for this verdict** — it directly compares the two randomized groups with no model in between — and report CUPED's adjusted estimate for completeness and as a demonstration of the variance-reduction technique, not as a replacement point estimate. Both intervals exclude 0 either way, so this doesn't change the overall conclusion.

The larger reduction on visit than conversion is expected: visit is more common and more strongly correlated with the available covariates (two features, f8 and f11, carry most of the adjustment weight — θ of −1.098 and −0.145 respectively for visit), while conversion is rarer and closer to noise even after adjustment.

## Power: this test is enormously overpowered

At the actual sample size (11,882,655 treated / 2,096,937 control), the minimum detectable effect at 80% power is a **0.040pp** lift on visit — twenty-five times smaller than the 1.034pp effect actually observed. Put the other way: detecting the observed visit effect at 80% power would have needed only ~6,068 users per group; detecting the observed conversion effect would have needed ~29,269 per group. A far smaller (and cheaper) experiment would have already been conclusive on both outcomes — a directly actionable finding for planning the next test.

## Optional: effect on the actually-exposed (CACE)

Using `treatment` as an instrument for `exposure` (Wald estimator): first-stage compliance (effect of treatment assignment on actually being exposed) is 3.60pp. The resulting CACE is **+28.70 percentage points** of visit probability among users who were actually exposed to the ad — an *absolute* effect, not a relative lift, so the correct comparison is against the ITT's +1.034 percentage points, not its 27.07% relative figure. CACE = ITT ÷ first-stage compliance = 1.034pp ÷ 0.0360 ≈ 28.7pp, roughly **28× larger** than the intent-to-treat effect. That's expected: CACE concentrates the effect onto only the subset of treated users who were actually exposed (about 3.6% of assignment translates to exposure), rather than diluting it across everyone assigned to treatment regardless of whether the ad was actually shown.

## Caveats

- **This is not a claim about Criteo's real ad performance.** Criteo sub-sampled this benchmark dataset non-uniformly for privacy before releasing it. The relative-lift comparisons (uplift ranking vs. random, etc. — later sub-projects) are valid; the absolute percentages above should not be read as "Criteo's ads lift visits by 27%."
- **CUPED here is regression adjustment, not classic pre-period CUPED.** This dataset has no true pre-experiment period; theta is estimated by regressing the outcome on pre-treatment covariates f0–f11, pooled across both arms. The direction and validity of the variance reduction is the same idea CUPED is built on, but it is not literally the pre-period covariate technique from the original CUPED paper.
- **Bootstrap `n_boot=100`, not the design doc's suggested 2,000.** At 14M rows, each bootstrap resample is a full-dataset Spark pass; 2,000 resamples was not tractable on this machine (16GB RAM, local-mode Spark, no cluster) in reasonable time. 100 resamples already agrees closely with the closed-form normal-approximation CI on both outcomes, which is itself good evidence 100 is enough here — but this is a known, documented reduction from the original plan, not an oversight.
- **Dataset version:** confirmed v2.1 via the official Hugging Face mirror (`huggingface.co/datasets/criteo/criteo-uplift`, file `criteo-research-uplift-v2.1.csv.gz`) — the `sklift` package's own downloader returned `403 Forbidden` from its S3 bucket during this run, so the HF mirror (the design doc's documented fallback) was used instead.
- **`EXPECTED_TREATMENT_RATE` was corrected from 0.846 to 0.85** after the SRM hard gate correctly caught the mismatch against real data on the first pipeline run — the true design ratio, confirmed against the dataset's own official documentation, is 85.00%, not the source doc's approximate 84.6%.
