# Inflation Forecasting Dashboard

A machine learning powered web application for forecasting Euro Area inflation rates using XGBoost

## Overview

This project predicts month-over-month inflation changes across three Euro Area categories:
- Overall HICP (Harmonised Index of Consumer Prices)
- Alcoholic Beverages & Tobacco
- Coal

The model uses recursive forecasting with data-driven prediction bounds to provide realistic inflation forecasts up to 12 months ahead.

## Features

- **Model Performance Page**: View backtest results on real data with MAE and MAPE metrics
- **Future Forecast Page**: Generate recursive forecasts with 3 years of historical context
- **Dataset Selection**: Compare different inflation categories side-by-side
- **Prediction Bounds**: Automatically calculated from historical data to prevent unrealistic values
- **Interactive Dashboard**: Built with Streamlit for easy exploration

## Project Structure

```
├── train_models.py          # Model training and evaluation script
├── app.py                   # Streamlit application
├── requirements.txt         # Python dependencies
├── models/                  # Trained model artifacts (created after training)
├── *.csv                    # Historical inflation data from ECB
└── README.md                # This file
```

## Quick Start

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

1. Clone the repository:
```bash
git clone https://github.com/"username"/inflation-forecasting.git
cd inflation-forecasting
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Training Models

Train all models on the latest data:
```bash
python3 train_models.py
```

This will:
- Load and process inflation data from CSV files
- Create engineered features (lagged changes, rolling averages, seasonality)
- Train XGBoost models on each dataset
- Generate prediction bounds based on data percentiles
- Save trained models to the `models/` directory

**Output**: Performance metrics for each dataset
```
=== TRAINING COMPLETE ===
  Overall HICP (Euro Area)
    MAE: 2.4457  |  MAPE: 41.91%
  Alcoholic Beverages & Tobacco (Euro Area)
    MAE: 4.3953  |  MAPE: 55.84%
  Coal (Euro Area)
    MAE: 61.9138  |  MAPE: 56.10%
```

### Running the Dashboard

Start the Streamlit app:
```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`

## How It Works

### Training Process

1. **Data Loading**: Read inflation rates from CSV files
2. **Feature Engineering**: Create lagged inflation changes (lag 1, 2, 3, 12 months) and rolling averages
3. **Bounds Calculation**: Compute 5th and 95th percentiles of historical inflation to define realistic prediction ranges
4. **Model Training**: Train XGBoost on inflation changes (not absolute rates) using last 12 months for backtesting
5. **Evaluation**: Calculate MAE and MAPE on test data

### Forecasting Process

1. **Recursive Prediction**: For each future month, predict the change in inflation from the current month
2. **Bounds Clipping**: Keep predictions within data-driven bounds to avoid extreme values
3. **Accumulation**: Add predicted change to previous inflation rate to get new rate
4. **Feature Update**: Use new rate to generate features for the next month's prediction

### Key Design Decisions

- **Predict Changes, Not Rates**: We predict inflation changes (deltas) rather than absolute rates. This avoids recursive error convergence that would cause all predictions to converge to the model mean.
- **Data-Driven Bounds**: Instead of fixed bounds, we use percentiles (5th-95th) from actual historical data. This allows natural variation while catching outliers.
- **Delta Feature Engineering**: All features (lags and rolling means) are built from inflation changes, ensuring consistency with the target variable.

## Model Performance

### Latest Results (March 2026)

| Dataset | Rows | MAE | MAPE | Bounds |
|---------|------|-----|------|--------|
| Overall HICP | 344 | 2.45 | 41.91% | [-50.00%, 35.91%] |
| Alcohol & Tobacco | 347 | 4.40 | 55.84% | [-17.17%, 19.58%] |
| Coal | 96 | 61.91 | 56.10% | [-78.75%, 106.25%] |

**Notes**:
- Coal dataset has limited history (starts December 2017) but shows reasonable performance
- MAPE values reflect the natural volatility of each commodity
- Bounds automatically adjust per dataset based on historical data patterns

## Files Explanation

### `train_models.py`
- Loads CSV files and calculates month-over-month inflation changes
- Builds lagged and rolling average features
- Trains XGBoost model with 12-month holdout for backtesting
- Calculates realistic bounds using data percentiles
- Saves trained models and associated metadata

**Key Functions**:
- `load_dataset()`: Read and preprocess inflation data
- `build_features()`: Create lagged and seasonal features
- `calculate_bounds()`: Compute percentile-based bounds
- `train_and_save()`: Full training pipeline

### `app.py`
- Interactive Streamlit dashboard
- Two pages: Model Performance (backtest results) and Future Forecast (forward predictions)
- Loads cached models from disk
- Implements recursive forecasting with bounds
- Visualizes historical data + forecast with annotations

**Key Functions**:
- `recursive_forecast()`: Generate month-by-month predictions
- `validate_forecast()`: Check if predictions hit bounds
- `style_axis()`: Consistent chart formatting

## Data Format

CSV files should contain:
- `DATE`: Date in YYYY-MM-DD format
- `[HICP Column Name]`: Monthly inflation index values

The script automatically:
- Calculates month-over-month % changes
- Computes inflation changes (delta)
- Handles missing or invalid values

## Requirements

See `requirements.txt` for full list. Main dependencies:
- `xgboost`: Machine learning model
- `streamlit`: Web dashboard
- `pandas`, `numpy`: Data processing
- `matplotlib`: Charting
- `scikit-learn`: Metrics

## Extending the Project

### Add a New Dataset

1. Add CSV file to project directory
2. Update `DATASETS` in both `train_models.py` and `app.py`:
   ```python
   "new_dataset": {
       "csv_path": "new_data.csv",
       "value_col": "Column Name In CSV",
       "label": "Display Name",
   },
   ```
3. Run `python3 train_models.py` to train

### Adjust Forecast Horizon

In `app.py`, modify the slider range:
```python
horizon = st.slider("Months to forecast", min_value=1, max_value=24, value=6)
```

### Change Model Hyperparameters

Edit `XGB_PARAMS` in `train_models.py`:
```python
XGB_PARAMS = dict(
    n_estimators=1000,        # More trees
    learning_rate=0.01,       # Slower learning
    max_depth=5,              # Deeper trees
)
```

## Limitations

- **Recursive Error Compounding**: Longer forecasts (7+ months) are less reliable as errors accumulate
- **Data Quality**: Coal dataset has extreme volatility during 2022 energy crisis; predictions reflect this
- **Limited History**: Coal data starts December 2017; older datasets have more training data
- **Seasonal Patterns**: Model captures seasonality but may not adapt to structural breaks


**Last Updated**: March 26, 2026  
**Data Source**: ECB (European Central Bank) Official Website
