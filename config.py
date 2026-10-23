"""
Central configuration for the German Power Futures Decision Support System.

The project analyzes simulated short-term German electricity futures
with delivery horizons from D+1 to D+7.

IMPORTANT:
All generated market prices, weather observations and trading positions
are synthetic and must not be interpreted as actual EEX market data
or Electrohold Trade EAD trading records.
"""

from pathlib import Path

# ============================================================
# 1. PROJECT DIRECTORIES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
SRC_DIR = BASE_DIR / "src"
DASHBOARD_DIR = BASE_DIR / "dashboard"
OUTPUT_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"

SIMULATED_DATA_PATH = DATA_DIR / "simulated_futures.csv"
MODEL_RESULTS_PATH = TABLES_DIR / "forecast_results.csv"
RISK_RESULTS_PATH = TABLES_DIR / "risk_results.csv"

# Create output directories if they do not exist.
for directory in [
    DATA_DIR,
    OUTPUT_DIR,
    FIGURES_DIR,
    TABLES_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. MARKET CONFIGURATION
# ============================================================

MARKET_NAME = "German Power Futures"
MARKET_REGION = "Germany"

CURRENCY = "EUR"
PRICE_UNIT = "EUR/MWh"
VOLUME_UNIT = "MWh"

TIMEZONE = "Europe/Berlin"

# Delivery horizons: D+1, D+2, ..., D+7.
DELIVERY_HORIZONS = list(range(1, 8))

# Initial assumption: daily baseload electricity futures.
PRODUCT_TYPE = "Base"

# Number of hours in a standard delivery day.
# Actual delivery hours must account for DST transitions.
STANDARD_DELIVERY_HOURS = 24


# ============================================================
# 3. HISTORICAL SIMULATION PERIOD
# ============================================================

START_DATE = "2024-01-01"
END_DATE = "2025-12-31"

DATA_FREQUENCY = "D"

# Reproducibility of simulated observations.
RANDOM_SEED = 42

# Simulated data rather than actual exchange prices.
USE_SIMULATED_DATA = True


# ============================================================
# 4. PRICE SIMULATION PARAMETERS
# ============================================================

# Long-run reference price in EUR/MWh.
BASE_PRICE_EUR_MWH = 85.0

# Approximate standard deviation of daily price shocks.
DAILY_PRICE_VOLATILITY = 8.0

# Mean-reversion coefficient for the price level.
MEAN_REVERSION_SPEED = 0.15

# Autoregressive persistence of daily price shocks.
PRICE_AR_COEFFICIENT = 0.35

# Seasonal components in EUR/MWh.
ANNUAL_SEASONAL_AMPLITUDE = 15.0
WEEKLY_SEASONAL_AMPLITUDE = 5.0

# Simulated probability and magnitude of extreme shocks.
EXTREME_EVENT_PROBABILITY = 0.015
EXTREME_EVENT_STD = 45.0

# Minimum simulated futures price.
# Negative prices are possible in electricity markets.
MIN_SIMULATED_PRICE = -150.0

# Maximum simulated futures price.
MAX_SIMULATED_PRICE = 500.0


# ============================================================
# 5. WEATHER SIMULATION PARAMETERS
# ============================================================

WEATHER_VARIABLES = [
    "temperature_c",
    "wind_speed_ms",
    "solar_radiation_wm2",
]

# Approximate reference values.
AVERAGE_TEMPERATURE_C = 10.0
TEMPERATURE_SEASONAL_AMPLITUDE = 12.0

AVERAGE_WIND_SPEED_MS = 6.0
AVERAGE_SOLAR_RADIATION_WM2 = 130.0

# Illustrative coefficients for the synthetic price process.
# They are assumptions, not empirically estimated effects.
TEMPERATURE_PRICE_COEFFICIENT = -0.60
WIND_PRICE_COEFFICIENT = -1.80
SOLAR_PRICE_COEFFICIENT = -0.025


# ============================================================
# 6. FORECASTING MODEL
# ============================================================

MODEL_NAME = "ARX"

# Number of lagged observations for price dynamics.
AR_LAGS = 3

# Variables available to the forecasting model.
EXOGENOUS_VARIABLES = [
    "temperature_c",
    "wind_speed_ms",
    "solar_radiation_wm2",
]

# Chronological training/test split.
TRAIN_RATIO = 0.80

# Probability thresholds for directional signals.
BUY_PROBABILITY_THRESHOLD = 0.60
SELL_PROBABILITY_THRESHOLD = 0.40

# Model evaluation metrics.
REGRESSION_METRICS = [
    "MAE",
    "RMSE",
]

PROBABILITY_METRICS = [
    "Brier Score",
    "Directional Accuracy",
]


# ============================================================
# 7. RISK MANAGEMENT
# ============================================================

# Illustrative position size per futures product.
DEFAULT_POSITION_MW = 5.0

# Fraction of exposure covered by a hedge.
DEFAULT_HEDGE_RATIO = 0.50

HEDGE_RATIOS = [
    0.00,
    0.25,
    0.50,
    0.75,
    1.00,
]

# Value at Risk confidence level.
VAR_CONFIDENCE_LEVEL = 0.95

# Conditional Value at Risk confidence level.
CVAR_CONFIDENCE_LEVEL = 0.95

# Historical observation window for risk estimation.
RISK_LOOKBACK_DAYS = 90

# Illustrative transaction cost in EUR/MWh.
TRANSACTION_COST_EUR_MWH = 0.10


# ============================================================
# 8. STRATEGY SIMULATION
# ============================================================

STRATEGIES = [
    "Unhedged",
    "Fixed Hedge",
    "Adaptive Hedge",
]

# Simplified strategy simulation.
ENABLE_TRANSACTION_COSTS = True

# When True, model decisions are based only on information
# that would have been available at the decision timestamp.
PREVENT_LOOKAHEAD_BIAS = True


# ============================================================
# 9. DASHBOARD CONFIGURATION
# ============================================================

DASHBOARD_TITLE = (
    "German Power Futures - Decision Support System"
)

DASHBOARD_SUBTITLE = (
    "Probabilistic forecasting and price risk analysis "
    "for short-term electricity futures (D+1 to D+7)"
)

DASHBOARD_LAYOUT = "wide"


# ============================================================
# 10. HELPER FUNCTION
# ============================================================

def print_config_summary():
    """Print the main configuration parameters."""

    print("=" * 60)
    print("GERMAN POWER FUTURES DSS")
    print("=" * 60)

    print(f"Market:             {MARKET_NAME}")
    print(f"Product:            {PRODUCT_TYPE}")
    print(f"Delivery horizons:  {DELIVERY_HORIZONS}")
    print(f"Simulation period:  {START_DATE} to {END_DATE}")
    print(f"Forecasting model:  {MODEL_NAME}")
    print(f"Random seed:        {RANDOM_SEED}")
    print(f"Data directory:     {DATA_DIR}")
    print(f"Output directory:   {OUTPUT_DIR}")

    print("=" * 60)


if __name__ == "__main__":
    print_config_summary()