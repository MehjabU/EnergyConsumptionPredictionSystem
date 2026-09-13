import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
 
from src.models.prepare_data import prepare_data
 
 
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
 
    If log_transform is True, the model is trained on log1p(y_train) and
    predictions are converted back with expm1 before being returned, so
    the caller can evaluate every model on the same raw-scale metrics.
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
 
 
def print_results_table(results: dict) -> None:
    print("\n" + "=" * 64)
    print(f"{'Model':<28}{'MAE':>12}{'RMSE':>12}{'R2':>12}")
    print("-" * 64)
    for model_name, metrics in results.items():
        print(f"{model_name:<28}{metrics['MAE']:>12.3f}"
              f"{metrics['RMSE']:>12.3f}{metrics['R2']:>12.3f}")
 
 
def print_largest_errors(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
    top_n: int = 10,
) -> None:
    """
    Metrics alone say how well a model performs, not where it fails.
    This prints the largest absolute errors so it's clear whether a
    handful of extreme observations (e.g. the 16117.3 outlier from EDA)
    are dominating RMSE, versus errors being spread evenly across
    buildings.
    """
    errors_df = pd.DataFrame({
        "row_index": X_test.index,
        "actual": y_test.values,
        "predicted": y_pred,
    })
    errors_df["absolute_error"] = (errors_df["actual"] - errors_df["predicted"]).abs()
    errors_df = errors_df.sort_values("absolute_error", ascending=False).head(top_n)
 
    print(f"\nLargest prediction errors ({model_name}):")
    print(f"{'Actual':>12}{'Predicted':>14}{'Abs Error':>14}")
    for _, row in errors_df.iterrows():
        print(f"{row['actual']:>12.2f}{row['predicted']:>14.2f}{row['absolute_error']:>14.2f}")
 
 
def print_best_model(results: dict) -> None:
    """
    Simple summary of which model scored best on each metric. This is a
    starting point for discussion, not a final verdict — MAE and RMSE can
    disagree about which model is "best" precisely because RMSE is far
    more sensitive to the extreme outliers in site_eui_gj_m2.
    """
    print("\n" + "=" * 64)
    print("BEST MODEL BY METRIC (lower is better for MAE/RMSE, higher for R2)")
    for metric_name, better in [("MAE", min), ("RMSE", min), ("R2", max)]:
        best_model = better(results, key=lambda name: results[name][metric_name])
        print(f"  {metric_name}: {best_model} ({results[best_model][metric_name]:.3f})")
 
 
def main():
    # 1. Call prepare_data()
    X_train, X_test, y_train, y_test, preprocessor = prepare_data()
 
    results = {}
 
    # 2. Median baseline
    # Uses the training median only, so no test-set information leaks into
    # the baseline.
    median_value = y_train.median()
    y_pred_baseline = np.full(shape=len(y_test), fill_value=median_value)
    print(f"\nMedian baseline prediction: {median_value:.3f}")
    results["Median Baseline"] = evaluate(y_test, y_pred_baseline)
 
    # 3. Linear Regression (raw target)
    y_pred_lr = train_and_predict(
        LinearRegression(), X_train, X_test, y_train, preprocessor
    )
    results["Linear Regression"] = evaluate(y_test, y_pred_lr)
 
    # 4. Random Forest (raw target)
    y_pred_rf = train_and_predict(
        RandomForestRegressor(random_state=42), X_train, X_test, y_train, preprocessor
    )
    results["Random Forest"] = evaluate(y_test, y_pred_rf)
 
    print_results_table(results)
 
    # 5. Analyze prediction errors 
    print_largest_errors(X_test, y_test, y_pred_rf, model_name="Random Forest")
 
    # 6. Log-transform target and retrain both models
    # site_eui_gj_m2 is extremely right-skewed (median ~0.737, max
    # ~16117). Training on log1p(y) compresses that skew
    y_pred_lr_log = train_and_predict(
        LinearRegression(), X_train, X_test, y_train, preprocessor,
        log_transform=True,
    )
    results["Linear Regression (log1p)"] = evaluate(y_test, y_pred_lr_log)
 
    y_pred_rf_log = train_and_predict(
        RandomForestRegressor(random_state=42), X_train, X_test, y_train, preprocessor,
        log_transform=True,
    )
    results["Random Forest (log1p)"] = evaluate(y_test, y_pred_rf_log)
 
    # 7. Compare all five results together
    print_results_table(results)
    print_largest_errors(X_test, y_test, y_pred_rf_log, model_name="Random Forest (log1p)")
 
    # 8. starting point for the next decision (further feature work, or committing to one approach),
    # not a final answer.
    print_best_model(results)
 
 
if __name__ == "__main__":
    main()