from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def _running_estimates(y, treatment, checkpoints) -> pd.DataFrame:
    y, treatment = np.asarray(y, dtype=float), np.asarray(treatment)
    rows = []
    for n in checkpoints:
        yy, tt = y[:n], treatment[:n]
        n_t, n_c = int((tt == 1).sum()), int((tt == 0).sum())
        if n_t == 0 or n_c == 0:
            continue
        p_t, p_c = yy[tt == 1].mean(), yy[tt == 0].mean()
        variance = p_t * (1 - p_t) / n_t + p_c * (1 - p_c) / n_c
        rows.append({"n": n, "ate": p_t - p_c, "variance": max(variance, 1e-12)})
    return pd.DataFrame(rows)


def naive_pvalue_path(y, treatment, checkpoints) -> pd.DataFrame:
    """Two-sided z-test p-value recomputed at every look. Repeated peeking inflates false positives."""
    est = _running_estimates(y, treatment, checkpoints)
    est["p_value"] = 2 * (1 - norm.cdf(np.abs(est["ate"]) / np.sqrt(est["variance"])))
    return est[["n", "ate", "p_value"]]


def msprt_confidence_sequence(y, treatment, checkpoints, alpha: float = 0.05, tau: float = 0.05) -> pd.DataFrame:
    """Always-valid confidence sequence from a normal-mixture mSPRT (mixing sd `tau` on the effect).

    The test statistic is Lambda = sqrt(V/(V+tau^2)) * exp(tau^2 * ate^2 / (2V(V+tau^2))); the null is
    rejected once Lambda >= 1/alpha, which inverts to the radius below. Unlike the naive interval it
    stays valid however often you look."""
    est = _running_estimates(y, treatment, checkpoints)
    v, tau2 = est["variance"].to_numpy(), tau**2
    radius = np.sqrt(2 * v * (v + tau2) / tau2 * (np.log(1 / alpha) + 0.5 * np.log((v + tau2) / v)))
    est["ci_low"] = est["ate"] - radius
    est["ci_high"] = est["ate"] + radius
    return est[["n", "ate", "ci_low", "ci_high"]]
