from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from uplift.evaluate import segment_table

DEFAULT_FRACTIONS = [round(f, 2) for f in np.arange(0.05, 1.0001, 0.05)]


def budget_curve(y, treatment, scores, fractions=None) -> pd.DataFrame:
    """Estimated incremental outcomes when only the top `fraction` of users (by score) is targeted.

    Incremental = users targeted x (treated rate - control rate) among those users, so it is an
    estimate from the randomized data, not an observed count."""
    fractions = DEFAULT_FRACTIONS if fractions is None else fractions
    y, treatment, scores = np.asarray(y), np.asarray(treatment), np.asarray(scores, dtype=float)
    n = len(y)
    perm = np.random.default_rng(0).permutation(n)  # break ties independently of input order
    y, treatment, scores = y[perm], treatment[perm], scores[perm]
    order = np.argsort(-scores, kind="stable")
    y, treatment = y[order], treatment[order]
    cum_t = np.cumsum(treatment)
    cum_c = np.cumsum(1 - treatment)
    cum_ty = np.cumsum(y * treatment)
    cum_cy = np.cumsum(y * (1 - treatment))
    rows = []
    for fraction in fractions:
        k = max(int(round(fraction * n)), 1)
        n_t, n_c = cum_t[k - 1], cum_c[k - 1]
        uplift = cum_ty[k - 1] / n_t - cum_cy[k - 1] / n_c if n_t and n_c else float("nan")
        rows.append(
            {"fraction": float(fraction), "n_targeted": k, "uplift": float(uplift), "incremental_outcomes": float(k * uplift)}
        )
    return pd.DataFrame(rows)


def business_impact(curve: pd.DataFrame, cost_per_impression: float, value_per_outcome: float) -> pd.DataFrame:
    """Adds rupee columns. Both prices are assumptions supplied by the caller, not Criteo facts."""
    out = curve.copy()
    out["cost"] = out["n_targeted"] * cost_per_impression
    out["revenue"] = out["incremental_outcomes"] * value_per_outcome
    out["profit"] = out["revenue"] - out["cost"]
    return out


def operating_point(impact: pd.DataFrame) -> pd.Series:
    return impact.loc[impact["profit"].idxmax()]


def segment_savings(
    y, treatment, uplift_scores, response_scores, cost_per_impression: float, value_per_outcome: float
) -> pd.DataFrame:
    """Money saved by not targeting sure things and the lowest-uplift group ('sleeping dogs')."""
    segments = segment_table(y, treatment, uplift_scores, response_scores).set_index("segment")
    rows = []
    for name in ("sure_things", "sleeping_dogs"):
        n = int(segments.loc[name, "n"])
        observed = float(np.nan_to_num(segments.loc[name, "observed_uplift"]))
        saved = n * cost_per_impression
        forgone = n * observed * value_per_outcome
        rows.append(
            {
                "segment": name,
                "n": n,
                "observed_uplift": observed,
                "impression_cost_saved": saved,
                "value_forgone": forgone,
                "net_saving": saved - forgone,
            }
        )
    total = {"segment": "do_not_target_total"}
    for column in ("n", "impression_cost_saved", "value_forgone", "net_saving"):
        total[column] = sum(r[column] for r in rows)
    total["observed_uplift"] = float("nan")
    return pd.DataFrame(rows + [total])


def _records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records"))


def run(artifacts_dir: str, output: str, cost_per_impression: float, value_per_outcome: float) -> dict:
    artifacts = Path(artifacts_dir)
    meta = json.loads((artifacts / "model_meta.json").read_text())
    scores = pd.read_parquet(artifacts / "test_scores.parquet")
    y, t = scores["outcome"].to_numpy(), scores["treatment"].to_numpy()
    columns = {"uplift": meta["model_name"], "response_model": "response_model", "random": "random"}

    strategies, best_points = {}, {}
    for strategy, column in columns.items():
        impact = business_impact(budget_curve(y, t, scores[column].to_numpy()), cost_per_impression, value_per_outcome)
        best = operating_point(impact)
        strategies[strategy] = {"score_column": column, "curve": _records(impact)}
        best_points[strategy] = json.loads(best.to_json())

    savings = segment_savings(
        y, t, scores[meta["model_name"]].to_numpy(), scores["response_model"].to_numpy(),
        cost_per_impression, value_per_outcome,
    )
    results = {
        "assumptions": {
            "cost_per_impression": cost_per_impression,
            "value_per_outcome": value_per_outcome,
            "currency": "INR",
            "outcome": meta.get("outcome", "visit"),
            "note": "Illustrative prices, not Criteo data. Incremental outcomes are randomized-data estimates.",
        },
        "population": int(len(y)),
        "model": meta["model_name"],
        "strategies": strategies,
        "best_by_strategy": best_points,
        "operating_point": {"strategy": "uplift", **best_points["uplift"]},
        "segment_savings": _records(savings),
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(results, indent=2))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Budget-constrained targeting simulation.")
    parser.add_argument("--artifacts-dir", default="models")
    parser.add_argument("--output", default="docs/policy_results.json")
    parser.add_argument("--cost-per-impression", type=float, default=0.5, help="INR per targeted user (assumption)")
    parser.add_argument("--value-per-outcome", type=float, default=20.0, help="INR per incremental visit (assumption)")
    args = parser.parse_args()
    results = run(args.artifacts_dir, args.output, args.cost_per_impression, args.value_per_outcome)
    for strategy, point in results["best_by_strategy"].items():
        print(f"{strategy:15s} best at {point['fraction']:.0%} targeted: profit INR {point['profit']:,.0f}")


if __name__ == "__main__":
    main()
