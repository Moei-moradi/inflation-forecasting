import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

# Dataset configurations - map each dataset to its CSV file and column name
DATASETS = {
    "overall_hicp": {
        "csv_path": "hicp_monthly_euroarea_Dec2025.csv",
        "value_col": "HICP - Overall index (ICP.M.U2.N.000000.4.ANR)",
        "label": "Overall HICP (Euro Area)",
    },
    "alcohol": {
        "csv_path": "Alcohol_monthly_euroarea_Dec2025.csv",
        "value_col": "HICP - ALCOHOLIC BEVERAGES, TOBACCO (ICP.M.U2.N.020000.4.ANR)",
        "label": "Alcoholic Beverages & Tobacco (Euro Area)",
    },
    "coal": {
        "csv_path": "Coal_monthly_euroarea_Dec2025.csv",
        "value_col": "HICP - Coal (ICP.M.U2.N.045410.4.ANR)",
        "label": "Coal (Euro Area)",
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
    """Calculate realistic prediction bounds from actual data using percentiles.
    
    Instead of fixed bounds that may be too tight or too loose, we use the 5th 
    and 95th percentile of the actual inflation rates. This lets ~90% of real 
    data pass through naturally while still catching extreme outliers.
    """
    inf_rates = df["Inflation_Rate"].dropna()
    lower = np.percentile(inf_rates, 5)
    upper = np.percentile(inf_rates, 95)
    return float(lower), float(upper)

# Feature engineering - creates lagged and rolling features for the model
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create lagged inflation changes and seasonal features."""
    d = df.copy()
    
    # Lagged changes from the previous month-over-month inflation change
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
    2. Calculate month-over-month percentage changes
    3. Calculate first-order differences (change from previous change)
    """
    df = pd.read_csv(csv_path)
    
    # Clean up any unused columns from the CSV
    if "TIME PERIOD" in df.columns:
        df = df.drop(columns=["TIME PERIOD"])
    
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.set_index("DATE").sort_index()
    df = df.rename(columns={value_col: "Index_Value"})
    
    # Month-over-month percentage change of the index
    df["Inflation_Rate"] = df["Index_Value"].pct_change() * 100
    
    # Remove any problematic values (infinities from zero index values)
    df = df[~np.isinf(df["Inflation_Rate"])]
    df = df.dropna(subset=["Inflation_Rate"])
    
    # Calculate the change in inflation from one month to the next
    # This is what we actually predict - it's more stable than predicting absolute rates
    df["Inflation_Change"] = df["Inflation_Rate"].diff()
    
    return df[["Index_Value", "Inflation_Rate", "Inflation_Change"]]


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error, handling zero values."""
    mask = actual != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def validate_features(df: pd.DataFrame, stage: str, lower: float, upper: float) -> None:
    """Log a warning if we find extreme inflation values outside the bounds."""
    inf_rates = df["Inflation_Rate"].dropna()
    extreme_values = np.sum((inf_rates < lower) | (inf_rates > upper))
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
    
    print(f"  MAE  = {mae_val:.4f}")
    print(f"  MAPE = {mape_val:.2f}%")
    
    # Check if any predictions went out of bounds (would have been clipped)
    oob_count = np.sum((predictions < PREDICTION_LOWER_BOUND) | (predictions > PREDICTION_UPPER_BOUND))
    if oob_count > 0:
        print(f"  ⚠️  {oob_count} predictions exceed bounds [{PREDICTION_LOWER_BOUND:.2f}%, {PREDICTION_UPPER_BOUND:.2f}%]")
    
    # Create backtest dataframe for later visualization
    backtest_df = pd.DataFrame({
        "Actual":     y_test.values,
        "Predicted":  predictions,
    }, index=y_test.index)
    
    # Prepare all stats to save
    stats = {
        "mae":       mae_val,
        "mape":      mape_val,
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
    OUT_DIR = "models"
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