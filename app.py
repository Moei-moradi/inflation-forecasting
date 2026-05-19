import os
import subprocess
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error

st.set_page_config(
    page_title="Forecasting Dashboard",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
TRAINING_SCRIPT = BASE_DIR / "train_models.py"

# Available datasets for selection
DATASETS = {
    "overall_hicp": "HICP Inflation Overall Rate",
    "energy":       "HICP Energy Rate",
    "housing":      "HICP Housing Rate",
    "food":         "HICP Food Rate",
}

# Features used during training
FEATURE_COLS = [
    "lag_1", "lag_2", "lag_3", "lag_12",
    "rolling_mean_3", "rolling_mean_6",
    "month", "quarter",
]


@st.cache_resource
def load_all_assets() -> dict:
    """Load pre-trained models and their supporting data into memory.
    
    If models are missing, automatically train them first.
    """
    assets = {}
    
    # Check if any models are missing
    missing_models = []
    for key in DATASETS:
        model_path = os.path.join(MODELS_DIR, f"{key}_model.pkl")
        if not os.path.exists(model_path):
            missing_models.append(key)
    
    # Train missing models if any
    if missing_models:
        st.info(f"Training missing models for: {', '.join(missing_models)}...")
        try:
            # Run the training script
            result = subprocess.run(
                [sys.executable, str(TRAINING_SCRIPT)],
                capture_output=True,
                text=True,
                cwd=BASE_DIR,
                timeout=300,
            )
            if result.returncode == 0:
                st.success("Models trained successfully.")
            else:
                st.error(f"Training failed: {result.stderr}")
                return {}
        except subprocess.TimeoutExpired:
            st.error("Training timed out after 5 minutes")
            return {}
        except Exception as e:
            st.error(f"Training error: {str(e)}")
            return {}
    
    # Load all models
    for key in DATASETS:
        model_path = os.path.join(MODELS_DIR, f"{key}_model.pkl")
        if not os.path.exists(model_path):
            st.error(f"Model for {key} still missing after training")
            assets[key] = None
            continue
        
        # Load all saved artifacts for this model
        assets[key] = {
            "model":    joblib.load(os.path.join(MODELS_DIR, f"{key}_model.pkl")),
            "features": joblib.load(os.path.join(MODELS_DIR, f"{key}_features.pkl")),
            "stats":    joblib.load(os.path.join(MODELS_DIR, f"{key}_stats.pkl")),
            "data":     pd.read_pickle(os.path.join(MODELS_DIR, f"{key}_data.pkl")),
        }
    return assets

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create lagged and seasonal features from inflation changes."""
    d = df.copy()
    d["lag_1"]  = d["Inflation_Change"].shift(1)
    d["lag_2"]  = d["Inflation_Change"].shift(2)
    d["lag_3"]  = d["Inflation_Change"].shift(3)
    d["lag_12"] = d["Inflation_Change"].shift(12)
    d["rolling_mean_3"] = d["Inflation_Change"].rolling(3).mean()
    d["rolling_mean_6"] = d["Inflation_Change"].rolling(6).mean()
    d["month"]   = d.index.month
    d["quarter"] = d.index.quarter
    return d


def recursive_forecast(df: pd.DataFrame, model, horizon: int, lower_bound: float, upper_bound: float) -> pd.DataFrame:
    """Forecast inflation changes for the given number of months ahead.
    
    How it works:
    1. Each month, predict the change in inflation from current month
    2. Apply bounds to keep predictions realistic
    3. Add the change to the previous inflation rate to get new rate
    4. Use that new rate as input for the next month's prediction
    """
    current_df = df.copy()
    results = []
    
    for _ in range(horizon):
        temp = build_features(current_df)
        next_date = current_df.index[-1] + pd.DateOffset(months=1)
        
        # Predict the change in inflation
        X_next = temp.tail(1)[FEATURE_COLS]
        predicted_change = float(model.predict(X_next)[0])
        
        # Keep prediction within realistic bounds
        predicted_change = np.clip(predicted_change, lower_bound, upper_bound)
        
        # Add change to previous rate to get new inflation rate
        prev_inflation = current_df["Inflation_Rate"].iloc[-1]
        new_inflation = prev_inflation + predicted_change
        
        results.append({"Date": next_date, "Forecast": new_inflation})
        
        # Add this new value to the dataset for next iteration
        placeholder = pd.DataFrame(
            {"Index_Value": [np.nan], "Inflation_Rate": [new_inflation], "Inflation_Change": [predicted_change]},
            index=[next_date],
        )
        current_df = pd.concat([current_df, placeholder])
    
    return pd.DataFrame(results).set_index("Date")


def style_axis(ax):
    """Apply consistent styling to the chart axes."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.set_ylabel("Inflation Rate (%)", fontsize=11)


def add_zero_line(ax):
    """Add a reference line at zero inflation."""
    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")


def validate_forecast(forecast_df: pd.DataFrame, lower_bound: float, upper_bound: float) -> dict:
    """Check if forecasts hit the prediction bounds."""
    predictions = forecast_df["Forecast"].values
    clipped_count = np.sum(
        (predictions == lower_bound) | (predictions == upper_bound)
    )
    return {
        "clipped_count": clipped_count,
        "total_count": len(predictions),
        "has_clipped": clipped_count > 0,
        "lower": lower_bound,
        "upper": upper_bound
    }


# Load all models at startup
all_assets = load_all_assets()

missing = [k for k, v in all_assets.items() if v is None]
if missing:
    st.error(
        f"Model files not found for: {', '.join(missing)}.\n\n"
        "Please run `python3 train_models.py` first to generate the model files."
    )
    st.stop()


# Initialize session state for page navigation
if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"

# ==============================================================================
# NAVIGATION AND SIDEBAR
# ==============================================================================
with st.sidebar:
    st.title("Forecasting Dashboard")
    st.divider()
    
    # Navigation buttons
    if st.button("Home", key="btn_home", width="stretch"):
        st.session_state.current_page = "Home"
    if st.button("Model Performance", key="btn_perf", width="stretch"):
        st.session_state.current_page = "Model Performance"
    if st.button("Future Forecast", key="btn_forecast", width="stretch"):
        st.session_state.current_page = "Future Forecast"
    
    st.divider()
    
    # Dataset selector
    selected_label = st.selectbox(
        "Dataset",
        options=list(DATASETS.values()),
    )
    selected_key = [k for k, v in DATASETS.items() if v == selected_label][0]
    
    # Forecast horizon (only shown on forecast page)
    if st.session_state.current_page == "Future Forecast":
        st.divider()
        horizon = st.slider("Months to forecast", min_value=1, max_value=12, value=6)
    else:
        horizon = 6  # Default value
    
    st.divider()
    st.caption("Data source: ECB Official Website")


# Load all models at startup
all_assets = load_all_assets()

missing = [k for k, v in all_assets.items() if v is None]
if missing:
    st.error(
        f"Model files not found for: {', '.join(missing)}.\n\n"
        "Please run `python3 train_models.py` first to generate the model files."
    )
    st.stop()


# ==============================================================================
# PAGE CONTENT
# ==============================================================================

# HOME PAGE
if st.session_state.current_page == "Home":
    st.markdown("""
    <style>
        .home-title {
            font-size: 3.5em;
            font-weight: bold;
            text-align: center;
            margin-bottom: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .home-subtitle {
            font-size: 1.3em;
            text-align: center;
            color: #666;
            margin-bottom: 40px;
            line-height: 1.6;
        }
        .button-container {
            display: flex;
            gap: 20px;
            justify-content: center;
            margin-top: 50px;
            flex-wrap: wrap;
        }
        .feature-box {
            background: #243447;
            color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            margin: 10px;
            flex: 1;
            min-width: 250px;
            border: 1px solid #3d526b;
        }
        .feature-box strong {
            color: #ffffff;
        }
        .feature-icon {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        .home-description {
            max-width: 900px;
            margin: 0 auto;
            text-align: center;
            line-height: 1.7;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="home-title">Forecasting Dashboard</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="home-subtitle">
    Inflation forecasting powered by<br>
    <strong>Machine Learning</strong> & <strong>XGBoost</strong>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="home-description">
    This dashboard provides data-driven inflation forecasts for the Euro Area across
    multiple categories. Built with machine learning algorithms, it combines
    historical inflation patterns with advanced feature engineering to predict future trends.
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Feature boxes
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="feature-box">
            <strong>4 Categories</strong><br>
            Overall, Energy, Housing, Food
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-box">
            <strong>XGBoost Model</strong><br>
            Advanced ML predictions
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-box">
            <strong>Recursive Forecast</strong><br>
            Up to 12 months ahead
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="feature-box">
            <strong>Real Data</strong><br>
            ECB Official Statistics
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Big action buttons
    st.subheader("Get Started", divider=True)
    
    col_left, col_mid, col_right = st.columns([1, 0.2, 1])
    
    with col_left:
        st.markdown("""
        ### Model Performance
        Review how our models perform on real data. See prediction accuracy metrics (MAE, MAPE) 
        and compare actual vs. predicted inflation rates over the last 12 months.
        """)
        if st.button("View Model Performance", key="home_perf", width="stretch",
                    help="See backtesting results and model accuracy metrics"):
            st.session_state.current_page = "Model Performance"
            st.rerun()
    
    with col_right:
        st.markdown("""
        ### Future Forecast
        Generate predictions for upcoming months. Use our recursive forecasting algorithm 
        to see inflation trends for the next 1-12 months with realistic prediction bounds.
        """)
        if st.button("Generate Forecast", key="home_forecast", width="stretch",
                    help="Forecast future inflation rates"):
            st.session_state.current_page = "Future Forecast"
            st.rerun()
    
    st.divider()
    
    st.markdown("""
    ### How It Works
    
    1. **Data Collection**: Historical inflation data from ECB (Euro Central Bank)
    2. **Feature Engineering**: Lagged changes, rolling averages, seasonal patterns
    3. **Model Training**: XGBoost trained on inflation changes with backtesting
    4. **Recursive Forecasting**: Step-by-step predictions with realistic bounds
    5. **Performance Monitoring**: Continuous evaluation using MAE and MAPE metrics
    """)

# MODEL PERFORMANCE PAGE
elif st.session_state.current_page == "Model Performance":
    # Load selected model and data
    assets  = all_assets[selected_key]
    if assets is None:
        st.error(f"Model for {selected_label} is not available. Please check the training process.")
        st.stop()

    model   = assets["model"]
    stats   = assets["stats"]
    df_raw  = assets["data"]
    
    st.title("Model Performance — Last 12 Months")
    st.markdown(
        f"Comparing the model's predictions against actual Eurostat data "
        f"for _{selected_label}_."
    )
    
    backtest = stats["backtest"]
    mae_val  = stats["mae"]
    mape_val = stats["mape"]
    last_actual = backtest["Actual"].iloc[-1]
    last_pred   = backtest["Predicted"].iloc[-1]
    
    # Show key metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MAE",  f"{mae_val:.4f}")
    col2.metric("MAPE", f"{mape_val:.2f}%")
    col3.metric("Last Actual",    f"{last_actual:.2f}%")
    col4.metric("Last Predicted", f"{last_pred:.2f}%",
                delta=f"{last_pred - last_actual:+.2f}%")
    
    st.divider()
    
    # Plot backtest results
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(backtest.index, backtest["Actual"],
            label="Actual", color="#1f77b4", linewidth=2, marker="o", markersize=5)
    ax.plot(backtest.index, backtest["Predicted"],
            label="Predicted", color="#ff7f0e", linewidth=2,
            linestyle="--", marker="x", markersize=6)
    ax.fill_between(backtest.index, backtest["Actual"], backtest["Predicted"],
                    alpha=0.12, color="#ff7f0e", label="Error band")
    
    style_axis(ax)
    add_zero_line(ax)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45, ha="right")
    ax.legend(fontsize=10)
    ax.set_title(f"{selected_label} — Last 12 Months Performance", fontsize=13)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    
    # Show detailed numbers
    with st.expander("Show raw numbers"):
        display = backtest.copy()
        display.index = display.index.strftime("%b %Y")
        display["Error"] = display["Predicted"] - display["Actual"]
        display = display.round(4)
        st.dataframe(display, width="stretch")
    
    st.info(
        "The model was trained on all data except these 12 months. "
        "This backtest shows how it would have performed on unseen data."
    )


else:
    # FUTURE FORECAST PAGE
    # Load selected model and data
    assets  = all_assets[selected_key]
    if assets is None:
        st.error(f"Model for {selected_label} is not available. Please check the training process.")
        st.stop()

    model   = assets["model"]
    stats   = assets["stats"]
    df_raw  = assets["data"]
    
    st.title("Future Forecast")
    st.markdown(
        f"Recursively forecasting **{horizon} month{'s' if horizon > 1 else ''}** ahead "
        f"for _{selected_label}_."
    )
    
    # Extract bounds that were calculated during training
    lower_bound = stats.get("lower_bound", -10.0)
    upper_bound = stats.get("upper_bound", 15.0)
    
    # Run the forecast
    with st.spinner("Running recursive forecast…"):
        df_forecast = recursive_forecast(df_raw, model, horizon, lower_bound, upper_bound)
    
    # Calculate and display key metrics
    last_known   = df_raw["Inflation_Rate"].iloc[-1]
    last_known_d = df_raw.index[-1]
    first_fc     = df_forecast["Forecast"].iloc[0]
    last_fc      = df_forecast["Forecast"].iloc[-1]
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Last Known Value",
                f"{last_known:.2f}%",
                help=f"As of {last_known_d.strftime('%b %Y')}")
    col2.metric("Next Month Forecast", f"{first_fc:.2f}%",
                delta=f"{first_fc - last_known:+.2f}%")
    col3.metric(f"Month {horizon} Forecast", f"{last_fc:.2f}%",
                delta=f"{last_fc - last_known:+.2f}%")
    
    st.divider()
    
    # Plot historical data + forecast
    HISTORY_MONTHS = 36  # Show 3 years of context
    historical = df_raw["Inflation_Rate"].tail(HISTORY_MONTHS)
    
    fig2, ax2 = plt.subplots(figsize=(13, 5))
    ax2.plot(historical.index, historical,
             label="Historical Data", color="black", linewidth=2, marker="o", markersize=4)
    
    bridge_dates  = [historical.index[-1]] + list(df_forecast.index)
    bridge_values = [historical.iloc[-1]]  + list(df_forecast["Forecast"])
    ax2.plot(bridge_dates, bridge_values,
             label="Forecast", color="#e63946", linewidth=2,
             linestyle="--", marker="s", markersize=6)
    
    # Shade the forecast region
    ax2.axvspan(historical.index[-1], bridge_dates[-1], alpha=0.1, color="#e63946", label="Forecast Period")
    ax2.axvline(historical.index[-1], color="gray", linewidth=1.5, linestyle=":")
    
    # Annotate forecast values
    for date, val in df_forecast["Forecast"].items():
        ax2.annotate(
            f"{val:.2f}%",
            xy=(date, val),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="#e63946",
        )
    
    style_axis(ax2)
    add_zero_line(ax2)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45, ha="right")
    ax2.legend(fontsize=10)
    ax2.set_title(f"{selected_label} — {horizon}-Month Forecast", fontsize=13)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)
    
    # Show forecast table
    with st.expander("Show forecast values"):
        fc_display = df_forecast.copy()
        fc_display.index = fc_display.index.strftime("%b %Y")
        fc_display["Forecast"] = fc_display["Forecast"].round(4)
        st.dataframe(fc_display, width="stretch")
    
    st.divider()
    
    # Alert if bounds were applied
    validation = validate_forecast(df_forecast, lower_bound, upper_bound)
    if validation["has_clipped"]:
        st.warning(
            f"Bounds Applied: {validation['clipped_count']} out of "
            f"{validation['total_count']} forecasts were clipped to "
            f"[{validation['lower']:.2f}%, {validation['upper']:.2f}%] "
            f"to keep predictions realistic. This often means the model is uncertain beyond this horizon."
        )
    
    # Model reliability information
    st.subheader("Model Reliability")
    col_a, col_b = st.columns(2)
    col_a.metric("Historical MAE",  f"{stats['mae']:.4f}",
                 help="Lower is better. Measured on 12 months of test data.")
    col_b.metric("Historical MAPE", f"{stats['mape']:.2f}%",
                 help="Mean Absolute Percentage Error on test data.")
    
    st.caption(
        "Recursive forecasts compound errors over time — the further ahead, "
        "the less reliable. Treat long-term forecasts as directional trends, not precise predictions."
    )
    
    
    
    
    
    
    
    
