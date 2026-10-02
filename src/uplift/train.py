from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd

from uplift.constants import FEATURE_COLS, TREATMENT_COL, VISIT_COL
from uplift.evaluate import (
    bootstrap_qini_ci,
    decile_table,
    qini_auc,
    qini_curve,
    segment_table,
    uplift_at_k,
)
from uplift.models import MODEL_REGISTRY, SEARCH_SPACES, build_model, feature_matrix

MAX_CURVE_POINTS = 200


def load_split(input_dir: str | Path, name: str) -> pd.DataFrame:
    return pd.read_parquet(Path(input_dir) / name)


def stratified_subsample(df: pd.DataFrame, n: int, seed: int, outcome_col: str = VISIT_COL) -> pd.DataFrame:
    # groupby.sample returns rows grouped by stratum, so always shuffle before handing rows to
    # learners that fold or bag in row order.
    if n < len(df):
        df = df.groupby([TREATMENT_COL, outcome_col], group_keys=False).sample(
            frac=n / len(df), random_state=seed
        )
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def fit_all(df: pd.DataFrame, outcome_col: str, seed: int, names: list[str] | None = None) -> dict:
    X = feature_matrix(df)
    treatment, y = df[TREATMENT_COL].to_numpy(), df[outcome_col].to_numpy()
    return {name: build_model(name, seed=seed).fit(X, treatment, y) for name in (names or list(MODEL_REGISTRY))}


def score_all(models: dict, df: pd.DataFrame) -> dict[str, np.ndarray]:
    X = feature_matrix(df)
    return {name: np.asarray(model.predict_uplift(X), dtype=float) for name, model in models.items()}


def summarize(y, treatment, scores) -> dict[str, float]:
    return {
        "qini_auc": qini_auc(y, treatment, scores),
        "uplift_at_10": uplift_at_k(y, treatment, scores, 0.1),
        "uplift_at_20": uplift_at_k(y, treatment, scores, 0.2),
        "uplift_at_30": uplift_at_k(y, treatment, scores, 0.3),
    }


def log_run(name: str, params: dict, metrics: dict, tracking_uri: str, experiment: str = "uplift") -> None:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=name):
        mlflow.log_params({k: str(v) for k, v in params.items()})
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})


def _suggest(trial: optuna.Trial, name: str, spec: tuple):
    kind, low, high, *step = spec
    if kind == "int":
        return trial.suggest_int(name, low, high, step=step[0] if step else 1)
    if kind == "float_log":
        return trial.suggest_float(name, low, high, log=True)
    raise ValueError(f"Unknown search-space kind {kind!r}")


def tune_model(
    name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    outcome_col: str,
    n_trials: int,
    seed: int,
) -> dict:
    """Optuna search maximising validation Qini AUC. Takes no test frame by design."""
    if name not in SEARCH_SPACES:
        raise ValueError(f"{name!r} is not tunable; choose from {sorted(SEARCH_SPACES)}")
    X_train = feature_matrix(train_df)
    t_train, y_train = train_df[TREATMENT_COL].to_numpy(), train_df[outcome_col].to_numpy()
    X_val = feature_matrix(val_df)
    t_val, y_val = val_df[TREATMENT_COL].to_numpy(), val_df[outcome_col].to_numpy()

    def objective(trial: optuna.Trial) -> float:
        params = {p: _suggest(trial, p, spec) for p, spec in SEARCH_SPACES[name].items()}
        model = build_model(name, seed=seed, **params).fit(X_train, t_train, y_train)
        return qini_auc(y_val, t_val, np.asarray(model.predict_uplift(X_val), dtype=float))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return dict(study.best_params)


def _downsample_curve(x: np.ndarray, y: np.ndarray) -> dict[str, list[float]]:
    idx = np.unique(np.linspace(0, len(x) - 1, min(len(x), MAX_CURVE_POINTS)).astype(int))
    return {"x": [float(v) for v in x[idx]], "y": [float(v) for v in y[idx]]}


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def run(
    input_dir: str,
    output: str,
    outcome_col: str = VISIT_COL,
    sample_size: int = 200_000,
    test_sample_size: int | None = None,
    n_trials: int = 10,
    n_boot: int = 200,
    mlflow_uri: str = "sqlite:///mlflow.db",
    names: list[str] | None = None,
    seed: int = 42,
    artifacts_dir: str | None = None,
    register_model: str | None = None,
) -> dict:
    if register_model and not artifacts_dir:
        raise ValueError("--register-model needs --artifacts-dir (the saved model is what gets registered)")
    names = names or list(MODEL_REGISTRY)
    tracking_uri = mlflow_uri

    train = stratified_subsample(load_split(input_dir, "train"), sample_size, seed, outcome_col)
    val = stratified_subsample(load_split(input_dir, "val"), max(sample_size // 2, 1), seed, outcome_col)
    test = load_split(input_dir, "test")
    if test_sample_size:
        test = stratified_subsample(test, test_sample_size, seed, outcome_col)

    y_val, t_val = val[outcome_col].to_numpy(), val[TREATMENT_COL].to_numpy()
    base_params = {"outcome": outcome_col, "n_train": len(train), "n_val": len(val), "seed": seed}

    fitted = fit_all(train, outcome_col, seed, names)
    val_metrics = {n: summarize(y_val, t_val, s) for n, s in score_all(fitted, val).items()}
    for name, metrics in val_metrics.items():
        log_run(name, {**base_params, "stage": "untuned_val"}, metrics, tracking_uri)

    untuned_val_metrics = {n: dict(m) for n, m in val_metrics.items()}
    candidates = sorted((n for n in names if n in SEARCH_SPACES), key=lambda n: -val_metrics[n]["qini_auc"])
    tuned_params: dict[str, dict] = {}
    for name in candidates[:2]:
        params = tune_model(name, train, val, outcome_col, n_trials, seed)
        tuned_params[name] = params
        fitted[name] = build_model(name, seed=seed, **params).fit(
            feature_matrix(train), train[TREATMENT_COL].to_numpy(), train[outcome_col].to_numpy()
        )
        val_metrics[name] = summarize(y_val, t_val, score_all({name: fitted[name]}, val)[name])
        log_run(name, {**base_params, "stage": "tuned_val", **params}, val_metrics[name], tracking_uri)

    y_test, t_test = test[outcome_col].to_numpy(), test[TREATMENT_COL].to_numpy()
    test_scores = score_all(fitted, test)
    test_metrics = {n: summarize(y_test, t_test, s) for n, s in test_scores.items()}
    for name, metrics in test_metrics.items():
        log_run(name, {**base_params, "stage": "final_test", "n_test": len(test)}, metrics, tracking_uri)

    best = max(candidates, key=lambda n: val_metrics[n]["qini_auc"]) if candidates else None
    reference = "response_model" if "response_model" in test_scores else None
    ci = bootstrap_qini_ci(y_test, t_test, test_scores, n_boot=n_boot, seed=seed, reference=reference)

    results = {
        "outcome": outcome_col,
        "sizes": {"train": len(train), "val": len(val), "test": len(test)},
        "best_model": best,
        "reference_model": reference,
        "untuned_val_metrics": untuned_val_metrics,
        "tuned_params": tuned_params,
        "val_metrics": val_metrics,
        "test_metrics": {n: {**test_metrics[n], **ci[n]} for n in test_metrics},
        "qini_curves": {n: _downsample_curve(*qini_curve(y_test, t_test, s)) for n, s in test_scores.items()},
    }
    if best:
        results["decile_table"] = decile_table(y_test, t_test, test_scores[best]).to_dict(orient="records")
        if "response_model" in test_scores:
            results["segment_table"] = segment_table(
                y_test, t_test, test_scores[best], test_scores["response_model"]
            ).to_dict(orient="records")

    if artifacts_dir:
        out = Path(artifacts_dir)
        out.mkdir(parents=True, exist_ok=True)
        scores_frame = pd.DataFrame({"treatment": t_test, "outcome": y_test, **test_scores})
        scores_frame.to_parquet(out / "test_scores.parquet")
        val_scores = score_all(fitted, val)
        pd.DataFrame({"treatment": t_val, "outcome": y_val, **val_scores}).to_parquet(out / "val_scores.parquet")
        if best:
            joblib.dump(fitted[best], out / "best_model.joblib")
            meta = {"model_name": best, "outcome": outcome_col, "feature_cols": FEATURE_COLS, "n_train": len(train), "n_val": len(val)}
            (out / "model_meta.json").write_text(json.dumps(meta, indent=2))
            if register_model:
                from uplift.registry import register_model as register

                results["registered_model_version"] = register(out / "best_model.joblib", register_model, tracking_uri)

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(_clean(results), indent=2))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit, tune and evaluate uplift models.")
    parser.add_argument("--input-dir", default="data/processed")
    parser.add_argument("--output", default="docs/uplift_results.json")
    parser.add_argument("--outcome", default=VISIT_COL, choices=["visit", "conversion"])
    parser.add_argument("--sample-size", type=int, default=200_000, help="stratified train subsample size")
    parser.add_argument("--test-sample-size", type=int, default=None, help="cap test rows (default: all)")
    parser.add_argument("--n-trials", type=int, default=10, help="Optuna trials per tuned model")
    parser.add_argument("--n-boot", type=int, default=200)
    parser.add_argument("--mlflow-uri", default="sqlite:///mlflow.db", help="local SQLite store by default")
    parser.add_argument("--models", default=None, help="comma-separated subset of the registry")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--artifacts-dir", default=None, help="save best model + test scores here")
    parser.add_argument("--register-model", default=None, help="register the best model in MLflow under this name (alias: production)")
    args = parser.parse_args()
    results = run(
        args.input_dir,
        args.output,
        outcome_col=args.outcome,
        sample_size=args.sample_size,
        test_sample_size=args.test_sample_size,
        n_trials=args.n_trials,
        n_boot=args.n_boot,
        mlflow_uri=args.mlflow_uri,
        names=args.models.split(",") if args.models else None,
        seed=args.seed,
        artifacts_dir=args.artifacts_dir,
        register_model=args.register_model,
    )
    for name, m in results["test_metrics"].items():
        print(f"{name:22s} qini_auc={m['qini_auc']:.5f}  CI=[{m['ci_low']:.5f}, {m['ci_high']:.5f}]")
    print("best model (chosen on validation):", results["best_model"])


if __name__ == "__main__":
    main()
