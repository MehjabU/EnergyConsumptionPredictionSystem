import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
 
from src.models.prepare_data import prepare_data
 
 
def evaluate(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    """Compute MAE, RMSE, and R² for a set of predictions."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}
 
 
def print_results_table(results: dict) -> None:
    print("\n" + "=" * 60)
    print(f"{'Model':<20}{'MAE':>12}{'RMSE':>12}{'R2':>12}")
    print("-" * 56)
    for model_name, metrics in results.items():
        print(f"{model_name:<20}{metrics['MAE']:>12.3f}"
              f"{metrics['RMSE']:>12.3f}{metrics['R2']:>12.3f}")
 
 
def main():
    # 1. Call prepare_data()
    X_train, X_test, y_train, y_test, preprocessor = prepare_data()
 
    results = {}
 
    # 2. Create a median baseline
    # Uses the training median only
    # test-set information into the baseline.
    median_value = y_train.median()
    y_pred_baseline = np.full(shape=len(y_test), fill_value=median_value)
    print(f"\nMedian baseline prediction: {median_value:.3f}")
    results["Median Baseline"] = evaluate(y_test, y_pred_baseline)
 
    # 3. Train LinearRegression in a Pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor", clone(preprocessor)),
        ("model", LinearRegression()),
    ])
    pipeline.fit(X_train, y_train)
    y_pred_lr = pipeline.predict(X_test)
 
    # 4. Evaluate baseline vs Linear Regression
    results["Linear Regression"] = evaluate(y_test, y_pred_lr)
 
    # 5. Train RandomForestRegressor in a Pipeline
    rf_pipeline = Pipeline(steps=[
        ("preprocessor", clone(preprocessor)),
        ("model", RandomForestRegressor(random_state=42)),
    ])
    rf_pipeline.fit(X_train, y_train)
    y_pred_rf = rf_pipeline.predict(X_test)
 
    # 6. Evaluate baseline vs Linear Regression vs Random Forest
    results["Random Forest"] = evaluate(y_test, y_pred_rf)
    print_results_table(results)
 
 
if __name__ == "__main__":
    main()