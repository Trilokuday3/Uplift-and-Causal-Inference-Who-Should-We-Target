from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from uplift.policy import interpolate_curve


def load_json(docs_dir: str | Path, name: str) -> dict | None:
    path = Path(docs_dir) / name
    return json.loads(path.read_text()) if path.exists() else None


def qini_frame(uplift_results: dict) -> pd.DataFrame:
    """All models' Qini curves interpolated onto one shared x axis, ready for a line chart."""
    curves = uplift_results["qini_curves"]
    grid = np.unique(np.concatenate([np.asarray(c["x"], dtype=float) for c in curves.values()]))
    return pd.DataFrame({name: np.interp(grid, c["x"], c["y"]) for name, c in curves.items()}, index=grid)


def decile_frame(uplift_results: dict) -> pd.DataFrame:
    return pd.DataFrame(uplift_results["decile_table"])


def ab_verdict(ab_results: dict, outcome: str = "visit") -> dict:
    ate = ab_results[outcome]["ate"]
    ci = ab_results[outcome]["ci_normal"]
    worked = ci["ci_low"] > 0 or ci["ci_high"] < 0
    headline = (
        f"{'The campaign worked' if worked else 'Inconclusive - the CI includes 0'}: {outcome} rate "
        f"{ate['ate'] * 100:+.2f}pp (95% CI {ci['ci_low'] * 100:.2f} to {ci['ci_high'] * 100:.2f}pp), "
        f"relative lift {ate['relative_lift'] * 100:.1f}%"
    )
    return {"worked": worked, "headline": headline, "n_treated": ate["n_treated"], "n_control": ate["n_control"]}


def policy_at_budget(policy_results: dict, budget: float, strategy: str = "uplift") -> dict:
    return interpolate_curve(policy_results["strategies"][strategy]["curve"], budget)
