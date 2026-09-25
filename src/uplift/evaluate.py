from __future__ import annotations

import numpy as np
import pandas as pd
from sklift import metrics as sk_metrics

SEGMENT_NAMES = ("persuadables", "sure_things", "lost_causes", "sleeping_dogs")


def _validate(y, treatment, scores, shuffle: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y, treatment, scores = np.asarray(y), np.asarray(treatment), np.asarray(scores, dtype=float)
    if not (len(y) == len(treatment) == len(scores)):
        raise ValueError("y, treatment and scores must have the same length")
    if treatment.sum() == 0 or (treatment == 0).sum() == 0:
        raise ValueError("Need at least one treated and one control row to measure uplift")
    if len(np.unique(y)) < 2:
        raise ValueError("Outcome has a single class; uplift is undefined without both 0s and 1s")
    # Callers' rows are often grouped by treatment/outcome stratum; a fixed shuffle stops
    # position-based tie-breaking from linking tied scores to the treatment arm.
    if not shuffle:
        return y, treatment, scores
    perm = np.random.default_rng(0).permutation(len(y))
    return y[perm], treatment[perm], scores[perm]


def qini_curve(y, treatment, scores) -> tuple[np.ndarray, np.ndarray]:
    y, treatment, scores = _validate(y, treatment, scores)
    x_curve, y_curve = sk_metrics.qini_curve(y, scores, treatment)
    return np.asarray(x_curve), np.asarray(y_curve)


def qini_auc(y, treatment, scores) -> float:
    y, treatment, scores = _validate(y, treatment, scores)
    return float(sk_metrics.qini_auc_score(y, scores, treatment))


def uplift_at_k(y, treatment, scores, k: float) -> float:
    y, treatment, scores = _validate(y, treatment, scores)
    return float(sk_metrics.uplift_at_k(y, scores, treatment, strategy="overall", k=k))


def _rate(values: np.ndarray) -> float:
    return float(values.mean()) if len(values) else float("nan")


def decile_table(y, treatment, scores, n_bins: int = 10) -> pd.DataFrame:
    """Decile 1 holds the highest predicted uplift. Ranking with method='first' breaks ties
    by position so constant scores still produce n_bins equal-sized, non-empty bins."""
    y, treatment, scores = _validate(y, treatment, scores)
    order = pd.Series(-scores).rank(method="first")
    decile = pd.qcut(order, n_bins, labels=False) + 1
    rows = []
    for d in range(1, n_bins + 1):
        mask = (decile == d).to_numpy()
        treated_rate = _rate(y[mask & (treatment == 1)])
        control_rate = _rate(y[mask & (treatment == 0)])
        rows.append(
            {
                "decile": d,
                "n": int(mask.sum()),
                "treated_rate": treated_rate,
                "control_rate": control_rate,
                "uplift": treated_rate - control_rate,
            }
        )
    return pd.DataFrame(rows)


def segment_table(
    y, treatment, uplift_scores, response_scores, eps: float | None = None
) -> pd.DataFrame:
    """Segments are not observable per person, so they are defined from model scores, relative to
    the population: persuadables have predicted uplift above median + eps, sleeping dogs below
    median - eps, and the middle band splits into sure things / lost causes by whether the
    baseline response score is above its median. eps defaults to 0.5 x the score std. Because
    Criteo's overall lift is positive, "sleeping dogs" here means lowest predicted uplift; only
    the `observed_uplift` column (treated rate - control rate inside each segment) says whether
    that group is truly negative."""
    y, treatment, uplift_scores = _validate(y, treatment, uplift_scores, shuffle=False)
    response_scores = np.asarray(response_scores, dtype=float)
    center = np.median(uplift_scores)
    if eps is None:
        eps = 0.5 * float(np.std(uplift_scores))
    high_response = response_scores >= np.median(response_scores)
    masks = {
        "persuadables": uplift_scores > center + eps,
        "sleeping_dogs": uplift_scores < center - eps,
    }
    neutral = ~(masks["persuadables"] | masks["sleeping_dogs"])
    masks["sure_things"] = neutral & high_response
    masks["lost_causes"] = neutral & ~high_response
    rows = []
    for name in SEGMENT_NAMES:
        mask = masks[name]
        rows.append(
            {
                "segment": name,
                "n": int(mask.sum()),
                "share": float(mask.mean()),
                "observed_uplift": _rate(y[mask & (treatment == 1)]) - _rate(y[mask & (treatment == 0)]),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_qini_ci(
    y,
    treatment,
    scores_by_model: dict[str, np.ndarray],
    n_boot: int = 200,
    seed: int = 42,
    alpha: float = 0.05,
    reference: str | None = None,
) -> dict[str, dict]:
    """Percentile bootstrap of Qini AUC. All models share each resample, so differences between
    models are paired. Resamples missing a treated or control row are skipped. With `reference`,
    each other model also gets `diff_vs_reference`: the paired AUC difference and its CI."""
    y = np.asarray(y)
    treatment = np.asarray(treatment)
    scores_by_model = {k: np.asarray(v, dtype=float) for k, v in scores_by_model.items()}
    for scores in scores_by_model.values():
        _validate(y, treatment, scores)

    rng = np.random.default_rng(seed)
    n = len(y)
    draws: dict[str, list[float]] = {name: [] for name in scores_by_model}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        t_b = treatment[idx]
        if t_b.sum() == 0 or (t_b == 0).sum() == 0 or len(np.unique(y[idx])) < 2:
            continue
        for name, scores in scores_by_model.items():
            draws[name].append(qini_auc(y[idx], t_b, scores[idx]))

    result = {}
    for name, scores in scores_by_model.items():
        if not draws[name]:
            raise ValueError("Every bootstrap resample lacked a treated row, a control row, or both outcome classes")
        low, high = np.percentile(draws[name], [100 * alpha / 2, 100 * (1 - alpha / 2)])
        result[name] = {"auc": qini_auc(y, treatment, scores), "ci_low": float(low), "ci_high": float(high)}
    if reference is not None:
        ref_draws = np.asarray(draws[reference])
        for name, scores in scores_by_model.items():
            if name == reference:
                continue
            diffs = np.asarray(draws[name]) - ref_draws
            low, high = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
            result[name]["diff_vs_reference"] = {
                "estimate": result[name]["auc"] - result[reference]["auc"],
                "ci_low": float(low),
                "ci_high": float(high),
            }
    return result
