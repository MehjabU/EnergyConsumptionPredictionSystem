"""
src/models/train.py

Same fixed train/test split (random_state=42, deterministically sorted by
ewrb_id in prepare_data.py) is used for both feature sets, so differences
in metrics are attributable to the model/features, not the data. As a
sanity check, the median baseline should be identical across v1 and v2.
If it isn't, something changed in preprocessing/splitting that shouldn't
have.

NOTE on Linear Regression: it's included as a baseline comparison
alongside the tree-based models, not excluded despite scoring far worse.
On this dataset its performance is substantially weaker than Random
Forest under the current feature representation, reported plainly in
the results table below like any other model, without assuming a
specific cause (e.g. multicollinearity) that hasn't actually been
demonstrated.

Model selection uses MAE as the primary metric (see PRIMARY_METRIC below),
not R² or RMSE. The target has extreme outliers, so RMSE is dominated by
a handful of observations and isn't a reliable single number to select
on. All three metrics are still reported for every model.

Run from the project root as a module:

    python -m src.models.train
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.models.prepare_data import prepare_data, FEATURE_SETS

# Primary metric for selecting the "best" model (see module docstring for
# why MAE rather than R²/RMSE). MAE_BETTER defines how to compare two
# values of this metric — min() since lower MAE is better.
PRIMARY_METRIC = "MAE"
PRIMARY_METRIC_BETTER = min

def evaluate(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    """Compute MAE, RMSE, and R² for a set of predictions, always on the
    original (raw) target scale — this makes every model's metrics
    directly comparable, whether or not it was trained on a transformed
    target."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def train_and_predict(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    preprocessor,
    log_transform: bool = False,
) -> np.ndarray:
    """
    Fit `model` inside a fresh Pipeline (preprocessor cloned so pipelines
    never share fitted state) and return predictions on X_test, always
    converted back to the original target scale.
    """
    pipeline = Pipeline(steps=[
        ("preprocessor", clone(preprocessor)),
        ("model", model),
    ])

    y_fit = np.log1p(y_train) if log_transform else y_train
    pipeline.fit(X_train, y_fit)

    y_pred = pipeline.predict(X_test)
    if log_transform:
        y_pred = np.expm1(y_pred)

    return y_pred

def print_results_table(experiments: list) -> None:
    print("\n" + "=" * 76)
    print(f"{'Model':<40}{'MAE':>12}{'RMSE':>12}{'R2':>12}")
    print("-" * 76)
    for exp in experiments:
        metrics = exp["metrics"]
        print(f"{exp['label']:<40}{metrics['MAE']:>12.3f}"
              f"{metrics['RMSE']:>12.3f}{metrics['R2']:>12.4f}")

def print_largest_errors(
    df: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
    top_n: int = 10,
) -> None:
    """
    Metrics alone say how well a model performs, not where it fails.
    This prints the largest absolute errors WITH the building's
    identifying/contextual features, looked up from df via the shared
    index, plus a SIGNED error so it's clear whether the model is
    under- or over-predicting each case.
    """
    context_cols = [
        "ewrb_id", "city", "primary_property_type", "self_property_type",
        "largest_property_type", "postal_code",
    ]
    context_cols = [c for c in context_cols if c in df.columns]
    context_df = df.loc[X_test.index, context_cols].reset_index(drop=True)

    errors_df = context_df.copy()
    errors_df["actual"] = y_test.values
    errors_df["predicted"] = y_pred
    # Signed error: positive = underprediction (actual > predicted),
    # negative = overprediction (actual < predicted).
    errors_df["error"] = errors_df["actual"] - errors_df["predicted"]
    errors_df["absolute_error"] = errors_df["error"].abs()
    errors_df = errors_df.sort_values("absolute_error", ascending=False).head(top_n)

    print(f"\nLargest prediction errors ({model_name}):")
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(errors_df.to_string(index=False))

def print_best_model(experiments: list) -> None:
    """
    Which experiment scored best on each metric, across ALL models
    (including Linear Regression, so poor performance is reported, not
    treated as grounds for exclusion). MAE is the primary selection
    metric for this project (see module docstring).
    """
    print("\n" + "=" * 76)
    print("BEST MODEL BY METRIC (all models included)")
    for metric_name, better in [("MAE", min), ("RMSE", min), ("R2", max)]:
        best_exp = better(experiments, key=lambda e: e["metrics"][metric_name])
        tag = " <- PRIMARY" if metric_name == PRIMARY_METRIC else ""
        print(f"  {metric_name}: {best_exp['label']} "
              f"({best_exp['metrics'][metric_name]:.4f}){tag}")

def print_feature_set_comparison(experiments: list) -> None:
    """
    Directly answers: did v2's engineered features actually improve each
    model over v1? Matches experiments by model name across feature sets
    and reports the change in MAE, RMSE, and R².
    """
    by_key = {(e["feature_set"], e["model_name"]): e["metrics"] for e in experiments}
    model_names = sorted({e["model_name"] for e in experiments})

    print("\n" + "=" * 76)
    print("FEATURE SET COMPARISON (v2 vs v1)")
    print(f"{'Model':<28}{'MAE (v1->v2)':>20}{'RMSE (v1->v2)':>20}{'R2 (v1->v2)':>24}")
    print("-" * 92)
    for model_name in model_names:
        v1 = by_key.get(("v1", model_name))
        v2 = by_key.get(("v2", model_name))
        if v1 is None or v2 is None:
            continue
        mae_str = f"{v1['MAE']:.3f} -> {v2['MAE']:.3f}"
        rmse_str = f"{v1['RMSE']:.3f} -> {v2['RMSE']:.3f}"
        r2_str = f"{v1['R2']:.4f} -> {v2['R2']:.4f}"
        print(f"{model_name:<28}{mae_str:>20}{rmse_str:>20}{r2_str:>24}")

def check_baseline_consistency(experiments: list) -> None:
    """
    Sanity check: the median baseline should be IDENTICAL across v1 and
    v2, since it only depends on y_train, and y_train should be the same
    set of rows in the same split either way. If it differs, something
    unexpected changed in preprocessing or splitting between the two
    feature-set runs.
    """
    baselines = {
        e["feature_set"]: e["metrics"]
        for e in experiments if e["model_name"] == "Median Baseline"
    }
    if "v1" not in baselines or "v2" not in baselines:
        return

    # Compare the specific metric values with np.isclose rather than
    # dict equality, since that expresses the intent more directly than
    # relying on exact dict equality.
    metrics_match = all(
        np.isclose(baselines["v1"][m], baselines["v2"][m])
        for m in ["MAE", "RMSE", "R2"]
    )

    if metrics_match:
        print("\nSanity check: Median Baseline (v1) == Median Baseline (v2). "
              "Same underlying split confirmed.")
    else:
        print("\nWARNING: Median Baseline differs between v1 and v2 — "
              "expected identical results since the baseline only depends "
              "on y_train. Check that the split is truly the same for "
              "both feature sets.")

def run_feature_set(feature_set: str, experiments: list):
    """
    Run baseline + all models for one feature set, appending each result
    (metrics AND predictions/context) to `experiments` so error analysis
    can later be run on whichever model actually scored best.
    """
    categorical = FEATURE_SETS[feature_set]["categorical"]
    numeric = FEATURE_SETS[feature_set]["numeric"]
    print("\n" + "#" * 76)
    print(f"FEATURE SET: {feature_set} — {categorical + numeric}")

    X_train, X_test, y_train, y_test, preprocessor, df = prepare_data(feature_set)

    def add_experiment(model_name: str, y_pred: np.ndarray):
        experiments.append({
            "feature_set": feature_set,
            "model_name": model_name,
            "label": f"{model_name} ({feature_set})",
            "metrics": evaluate(y_test, y_pred),
            "y_pred": y_pred,
            "X_test": X_test,
            "y_test": y_test,
            "df": df,
        })

    median_value = y_train.median()
    y_pred_baseline = np.full(shape=len(y_test), fill_value=median_value)
    add_experiment("Median Baseline", y_pred_baseline)

    y_pred_lr = train_and_predict(LinearRegression(), X_train, X_test, y_train, preprocessor)
    add_experiment("Linear Regression", y_pred_lr)

    y_pred_rf = train_and_predict(
        RandomForestRegressor(random_state=42), X_train, X_test, y_train, preprocessor
    )
    add_experiment("Random Forest", y_pred_rf)

    y_pred_lr_log = train_and_predict(
        LinearRegression(), X_train, X_test, y_train, preprocessor, log_transform=True
    )
    add_experiment("Linear Regression log1p", y_pred_lr_log)

    y_pred_rf_log = train_and_predict(
        RandomForestRegressor(random_state=42), X_train, X_test, y_train, preprocessor,
        log_transform=True,
    )
    add_experiment("Random Forest log1p", y_pred_rf_log)

def main():
    experiments = []

    for feature_set in FEATURE_SETS:
        run_feature_set(feature_set, experiments)

    print_results_table(experiments)
    print_best_model(experiments)
    print_feature_set_comparison(experiments)
    check_baseline_consistency(experiments)

    # Error analysis on whichever experiment actually scored best on the
    # chosen primary metric (MAE). Not based on R², since R² and RMSE are both
    # sensitive to the extreme outliers in this target.
    best_exp = PRIMARY_METRIC_BETTER(experiments, key=lambda e: e["metrics"][PRIMARY_METRIC])
    print(f"\nRunning error analysis on the best model by {PRIMARY_METRIC}: {best_exp['label']}")
    print_largest_errors(
        best_exp["df"], best_exp["X_test"], best_exp["y_test"], best_exp["y_pred"],
        model_name=best_exp["label"],
    )

if __name__ == "__main__":
    main()