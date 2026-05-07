import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
MODELS_DIR = BASE_DIR / "models"

# Dataset configurations - map each dataset to its CSV file and column name
DATASETS = {
    "overall_hicp": {
        "csv_path": DATA_DIR / "Overall_Hicp_Monthly.csv",
        "value_col": "HICP Inflation rate - Total - Annual rate of change (HICP.M.U2.N.000000.4D0.ANR)",
        "label": "HICP Inflation Overall Rate",
    },
    "energy": {
        "csv_path": DATA_DIR / "Energy_monthly.csv",
        "value_col": "HICP Inflation rate - Energy - Annual rate of change (HICP.M.U2.N.NRGY00.4D0.ANR)",
        "label": "HICP Energy Rate",
    },
    "housing": {
        "csv_path": DATA_DIR / "Housing_monthly.csv",
        "value_col": "HICP Inflation rate - Housing, water, electricity, gas and other fuels - Annual rate of change (HICP.M.U2.N.040000.4D0.ANR)",
        "label": "HICP Housing Rate",
    },
    "food": {
        "csv_path": DATA_DIR / "Food_monthly.csv",
        "value_col": "HICP Inflation rate - Food including alcohol and tobacco - Annual rate of change (HICP.M.U2.N.FOOD00.4D0.ANR)",
        "label": "HICP Food Rate",
    },
}

# XGBoost model configuration
XGB_PARAMS = dict(
    objective="reg:squarederror",
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.7,
    colsample_bytree=0.8,
)

TEST_MONTHS = 12  # Hold out last 12 months for backtesting

# Global bounds - set per dataset during training
PREDICTION_LOWER_BOUND = None
PREDICTION_UPPER_BOUND = None


def calculate_bounds(df: pd.DataFrame) -> tuple:
    """Calculate realistic prediction bounds for monthly inflation-rate changes.
    
    Instead of fixed bounds that may be too tight or too loose, we use the 5th 
    and 95th percentile of the actual monthly changes. This lets ~90% of real
    data pass through naturally while still catching extreme outliers.
    """
    changes = df["Inflation_Change"].dropna()
    lower = np.percentile(changes, 5)
    upper = np.percentile(changes, 95)
    return float(lower), float(upper)

# Feature engineering - creates lagged and rolling features for the model
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create lagged inflation changes and seasonal features."""
    d = df.copy()
    
    # Lagged changes from the previous monthly movement in the annual inflation rate
    # These help the model learn short and long-term patterns
    d["lag_1"]  = d["Inflation_Change"].shift(1)
    d["lag_2"]  = d["Inflation_Change"].shift(2)
    d["lag_3"]  = d["Inflation_Change"].shift(3)
    d["lag_12"] = d["Inflation_Change"].shift(12)  # Year-over-year pattern
    
    # Rolling averages to smooth out noise
    d["rolling_mean_3"] = d["Inflation_Change"].rolling(3).mean()
    d["rolling_mean_6"] = d["Inflation_Change"].rolling(6).mean()
    
    # Seasonal features - month and quarter of the year
    d["month"]   = d.index.month
    d["quarter"] = d.index.quarter
    
    return d.dropna()


FEATURE_COLS = [
    "lag_1", "lag_2", "lag_3", "lag_12",
    "rolling_mean_3", "rolling_mean_6",
    "month", "quarter",
]


# Data loading and preprocessing
def load_dataset(csv_path: str, value_col: str) -> pd.DataFrame:
    """Load and prepare inflation data from CSV.
    
    Process:
    1. Read the CSV and parse dates
    2. Treat the ECB ANR column as an annual inflation rate in percent
    3. Calculate the month-to-month difference in that annual rate
    """
    df = pd.read_csv(csv_path)
    
    # Clean up any unused columns from the CSV
    if "TIME PERIOD" in df.columns:
        df = df.drop(columns=["TIME PERIOD"])
    
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.set_index("DATE").sort_index()
    df = df.rename(columns={value_col: "Index_Value"})
    
    # ECB ANR series or annual inflation rates in percent
    df["Inflation_Rate"] = pd.to_numeric(df["Index_Value"], errors="coerce")
    
    # Remove any problematic values
    df = df[~np.isinf(df["Inflation_Rate"])]
    df = df.dropna(subset=["Inflation_Rate"])
    
    # Calculate the month-to-month difference in the annual inflation rate
    # This is what we actually predict - it's more stable than predicting absolute rates
    df["Inflation_Change"] = df["Inflation_Rate"].diff()
    
    return df[["Index_Value", "Inflation_Rate", "Inflation_Change"]]


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error, handling zero values."""
    mask = actual != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def persistence_baseline(y: pd.Series, test_months: int) -> pd.Series:
    """Use the previous observed monthly change as a simple forecast baseline."""
    return y.shift(1).iloc[-test_months:]


def validate_features(df: pd.DataFrame, stage: str, lower: float, upper: float) -> None:
    """Log a warning if we find inflation changes outside the prediction bounds."""
    changes = df["Inflation_Change"].dropna()
    extreme_values = np.sum((changes < lower) | (changes > upper))
    if extreme_values > 0:
        print(f"  [INFO] {stage}: Found {extreme_values} extreme values outside [{lower:.2f}%, {upper:.2f}%]")


def train_and_save(key: str, config: dict, out_dir: str):
    """Train an XGBoost model on inflation changes and save artifacts."""
    print(f"\n{'='*50}")
    print(f"Training: {config['label']}")
    print(f"{'='*50}")
    
    # Load and clean the data
    df = load_dataset(config["csv_path"], config["value_col"])
    print(f"  Rows after cleaning: {len(df)}")
    
    # Calculate realistic bounds from the data itself
    global PREDICTION_LOWER_BOUND, PREDICTION_UPPER_BOUND
    PREDICTION_LOWER_BOUND, PREDICTION_UPPER_BOUND = calculate_bounds(df)
    print(f"  Bounds: [{PREDICTION_LOWER_BOUND:.2f}%, {PREDICTION_UPPER_BOUND:.2f}%]")
    
    # Check for extreme values and report them
    validate_features(df, "After loading", PREDICTION_LOWER_BOUND, PREDICTION_UPPER_BOUND)
    
    # Build features for the model
    df_feat = build_features(df)
    print(f"  Rows after feature engineering: {len(df_feat)}")
    
    # Prepare training and test data (last 12 months held out for backtest)
    X = df_feat[FEATURE_COLS]
    y = df_feat["Inflation_Change"]  # We predict the change, not the absolute rate
    
    X_train, X_test = X.iloc[:-TEST_MONTHS], X.iloc[-TEST_MONTHS:]
    y_train, y_test = y.iloc[:-TEST_MONTHS], y.iloc[-TEST_MONTHS:]
    
    # Train the model
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    
    # Evaluate on the test set
    predictions = model.predict(X_test)
    mae_val  = mean_absolute_error(y_test, predictions)
    mape_val = mape(y_test.values, predictions)

    # Baseline: predict each test month using the previous observed monthly change
    baseline_predictions = persistence_baseline(y, TEST_MONTHS)
    baseline_mae_val = mean_absolute_error(y_test, baseline_predictions)
    baseline_mape_val = mape(y_test.values, baseline_predictions.values)
    baseline_improvement = (
        (baseline_mae_val - mae_val) / baseline_mae_val * 100
        if baseline_mae_val != 0 else float("nan")
    )
    
    print(f"  MAE  = {mae_val:.4f}")
    print(f"  MAPE = {mape_val:.2f}%")
    print(f"  Baseline MAE  = {baseline_mae_val:.4f}")
    print(f"  Baseline MAPE = {baseline_mape_val:.2f}%")
    print(f"  Improvement vs baseline = {baseline_improvement:.1f}%")
    
    # Check if any predictions went out of bounds (would have been clipped)
    oob_count = np.sum((predictions < PREDICTION_LOWER_BOUND) | (predictions > PREDICTION_UPPER_BOUND))
    if oob_count > 0:
        print(f"  WARNING: {oob_count} predictions exceed bounds [{PREDICTION_LOWER_BOUND:.2f}%, {PREDICTION_UPPER_BOUND:.2f}%]")
    
    # Create backtest dataframe for later visualization
    backtest_df = pd.DataFrame({
        "Actual":     y_test.values,
        "Predicted":  predictions,
        "Baseline":   baseline_predictions.values,
    }, index=y_test.index)
    
    # Prepare all stats to save
    stats = {
        "mae":       mae_val,
        "mape":      mape_val,
        "baseline_mae": baseline_mae_val,
        "baseline_mape": baseline_mape_val,
        "baseline_improvement": baseline_improvement,
        "backtest":  backtest_df,
        "label":     config["label"],
        "lower_bound": PREDICTION_LOWER_BOUND,
        "upper_bound": PREDICTION_UPPER_BOUND,
    }
    
    # Save model and supporting files
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(model,        os.path.join(out_dir, f"{key}_model.pkl"))
    joblib.dump(FEATURE_COLS, os.path.join(out_dir, f"{key}_features.pkl"))
    joblib.dump(stats,        os.path.join(out_dir, f"{key}_stats.pkl"))
    df.to_pickle(os.path.join(out_dir, f"{key}_data.pkl"))  # Full dataset for forecasting
    
    print(f"  Artifacts saved to '{out_dir}/'")
    return stats


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
if __name__ == "__main__":
    OUT_DIR = MODELS_DIR
    summary = {}
    
    # Train all datasets
    for key, config in DATASETS.items():
        summary[key] = train_and_save(key, config, OUT_DIR)
    
    # Print summary
    print("\n\n=== TRAINING COMPLETE ===")
    for key, stats in summary.items():
        print(f"  {stats['label']}")
        print(f"    MAE: {stats['mae']:.4f}  |  MAPE: {stats['mape']:.2f}%")
    
    print(f"\nAll models saved to '{OUT_DIR}/'")
