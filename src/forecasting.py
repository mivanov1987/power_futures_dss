"""
ARX forecasting module for German Power Futures DSS.

Forecasting target:
    Next-day change in the quotation of the SAME delivery contract.

Inputs:
    Synthetic quotations and weather forecasts from data_generator.py.

Method:
    Pooled autoregressive regression with exogenous variables (ARX).
    Normal-error approximation for directional probabilities.

Outputs:
    outputs/tables/forecast_results.csv
    outputs/tables/forecast_metrics.csv
    outputs/tables/forecast_coefficients.csv
    outputs/tables/latest_forecasts.csv

IMPORTANT:
    This is a synthetic-data research demonstration.
    D+1 contracts have no subsequent pre-delivery quotation
    in the current dataset, so no next-day forecast is issued.

Run:
    python -m src.forecasting
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from scipy.stats import norm
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    brier_score_loss,
    accuracy_score,
)

import config as cfg


# ============================================================
# 1. MODEL SPECIFICATION
# ============================================================

TARGET = "next_price_change"

CONTINUOUS_FEATURES = [
    "last_price_change",
    "has_previous_change",
    "futures_price_eur_mwh",
    "temperature_c",
    "wind_speed_ms",
    "solar_radiation_wm2",
    "annual_sin",
    "annual_cos",
]

# The current generator provides seven daily horizons.
# Only D+2..D+7 have an observable next-day quotation.
FORECAST_HORIZONS = [
    h for h in cfg.DELIVERY_HORIZONS if h >= 2
]

if not FORECAST_HORIZONS:
    raise ValueError("No valid forecasting horizons configured.")

DUMMY_COLUMNS = [
    f"horizon_{h}" for h in FORECAST_HORIZONS[1:]
]

FEATURE_COLUMNS = CONTINUOUS_FEATURES + DUMMY_COLUMNS


# ============================================================
# 2. LOAD DATA
# ============================================================

def load_data(
    path: Path = cfg.SIMULATED_DATA_PATH,
) -> pd.DataFrame:
    """Load and validate quotation observations."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Missing dataset: {path}\n"
            "Run python -m src.data_generator first."
        )

    df = pd.read_csv(
        path,
        parse_dates=["quotation_date", "delivery_date"],
    )

    required = [
        "quotation_date",
        "delivery_date",
        "contract_id",
        "horizon_days",
        "futures_price_eur_mwh",
        "temperature_c",
        "wind_speed_ms",
        "solar_radiation_wm2",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df[required].isna().any().any():
        raise ValueError("Required columns contain missing values.")

    if df.duplicated(
        ["contract_id", "quotation_date"]
    ).any():
        raise ValueError("Duplicate contract-date observations.")

    if not (
        (df["delivery_date"] - df["quotation_date"]).dt.days
        == df["horizon_days"]
    ).all():
        raise ValueError("Invalid delivery horizons.")

    return df.sort_values(
        ["contract_id", "quotation_date"]
    ).reset_index(drop=True)


# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

def prepare_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create ARX features and next-quotation targets.

    All explanatory variables are available at the current
    quotation date. Future quotations are used exclusively
    to construct the supervised-learning target.
    """

    result = df.copy().sort_values(
        ["contract_id", "quotation_date"]
    )

    grouped = result.groupby("contract_id")

    previous_price = grouped[
        "futures_price_eur_mwh"
    ].shift(1)

    previous_date = grouped[
        "quotation_date"
    ].shift(1)

    previous_gap = (
        result["quotation_date"] - previous_date
    ).dt.days

    observed_change = (
        result["futures_price_eur_mwh"] - previous_price
    )

    valid_previous = (
        previous_price.notna()
        & previous_gap.eq(1)
    )

    result["has_previous_change"] = (
        valid_previous.astype(float)
    )

    result["last_price_change"] = (
        observed_change.where(valid_previous, 0.0)
    )

    day_of_year = result[
        "delivery_date"
    ].dt.dayofyear

    angle = 2 * np.pi * (
        day_of_year - 1
    ) / 365.25

    result["annual_sin"] = np.sin(angle)
    result["annual_cos"] = np.cos(angle)

    # Explicit horizon indicators, with the smallest
    # forecastable horizon as the reference category.
    for horizon in FORECAST_HORIZONS[1:]:
        result[f"horizon_{horizon}"] = (
            result["horizon_days"] == horizon
        ).astype(float)

    # Future values are target-only fields.
    result["target_quotation_date"] = grouped[
        "quotation_date"
    ].shift(-1)

    result["next_price"] = grouped[
        "futures_price_eur_mwh"
    ].shift(-1)

    result["target_gap_days"] = (
        result["target_quotation_date"]
        - result["quotation_date"]
    ).dt.days

    result[TARGET] = (
        result["next_price"]
        - result["futures_price_eur_mwh"]
    )

    result["target_available"] = (
        result["horizon_days"].isin(FORECAST_HORIZONS)
        & result["target_gap_days"].eq(1)
        & result[TARGET].notna()
    )

    result["direction_up"] = np.where(
        result["target_available"],
        (result[TARGET] > 0).astype(float),
        np.nan,
    )

    return result.reset_index(drop=True)


# ============================================================
# 4. CHRONOLOGICAL SPLIT
# ============================================================

def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = cfg.TRAIN_RATIO,
):
    """
    Split by TARGET realization dates, not random rows.

    Each training outcome must be known by the cutoff date.
    This avoids including future outcome information
    in model estimation.
    """

    if not 0.5 <= train_ratio < 1:
        raise ValueError(
            "TRAIN_RATIO must be between 0.5 and 1."
        )

    eligible = df.loc[
        df["target_available"]
    ].copy()

    dates = sorted(
        eligible["target_quotation_date"].unique()
    )

    if len(dates) < 30:
        raise ValueError(
            "Insufficient observations for chronological testing."
        )

    split_index = int(len(dates) * train_ratio)

    cutoff = dates[split_index - 1]

    train = eligible.loc[
        eligible["target_quotation_date"] <= cutoff
    ].copy()

    test = eligible.loc[
        eligible["target_quotation_date"] > cutoff
    ].copy()

    if train.empty or test.empty:
        raise ValueError("Train or test dataset is empty.")

    return train, test, pd.Timestamp(cutoff)


# ============================================================
# 5. ARX REGRESSION
# ============================================================

def design_matrix(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Create a numeric, fixed-column design matrix."""

    X = df[FEATURE_COLUMNS].copy()

    X = X.apply(pd.to_numeric, errors="raise")

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite values in model features.")

    X = sm.add_constant(
        X,
        has_constant="add",
    )

    return X.astype(float)


def fit_arx_model(
    train: pd.DataFrame,
):
    """
    Estimate a pooled ARX model using ordinary least squares.

    The baseline model is deliberately simple to allow
    transparent interpretation in the diploma thesis.
    """

    X = design_matrix(train)

    y = train[TARGET].astype(float)

    model = sm.OLS(y, X).fit()

    fitted = model.predict(X)

    residuals = y - fitted

    # Horizon-specific in-sample residual uncertainty.
    residual_table = train[
        ["horizon_days"]
    ].copy()

    residual_table["residual"] = residuals.to_numpy()

    sigma_by_horizon = residual_table.groupby(
        "horizon_days"
    )["residual"].std(ddof=1).to_dict()

    global_sigma = float(
        residuals.std(ddof=1)
    )

    if not np.isfinite(global_sigma) or global_sigma <= 0:
        raise ValueError(
            "Invalid residual standard deviation."
        )

    return model, sigma_by_horizon, global_sigma


# ============================================================
# 6. PREDICTIONS AND PROBABILITIES
# ============================================================

def predict_arx(
    model,
    df: pd.DataFrame,
    sigma_by_horizon: dict,
    global_sigma: float,
) -> pd.DataFrame:
    """Calculate expected price changes and probabilities."""

    result = df.copy()

    X = design_matrix(result)

    expected_change = model.predict(X).to_numpy()

    sigma = result["horizon_days"].map(
        sigma_by_horizon
    ).to_numpy(dtype=float)

    sigma = np.where(
        np.isfinite(sigma) & (sigma > 1e-8),
        sigma,
        global_sigma,
    )

    result["predicted_change_eur_mwh"] = expected_change

    result["predicted_next_price_eur_mwh"] = (
        result["futures_price_eur_mwh"]
        + expected_change
    )

    result["predicted_sigma_eur_mwh"] = sigma

    # P(change > 0) under a normal-error assumption.
    result["probability_up"] = norm.cdf(
        expected_change / sigma
    )

    result["probability_down"] = (
        1.0 - result["probability_up"]
    )

    result["predicted_direction"] = np.where(
        result["probability_up"] >= 0.5,
        1,
        0,
    )

    result["trading_signal"] = np.select(
        [
            result["probability_up"]
            >= cfg.BUY_PROBABILITY_THRESHOLD,
            result["probability_up"]
            <= cfg.SELL_PROBABILITY_THRESHOLD,
        ],
        [
            "Upward signal",
            "Downward signal",
        ],
        default="Uncertain",
    )

    # Approximate 95% predictive interval.
    z_value = norm.ppf(0.975)

    result["forecast_lower_95"] = (
        expected_change - z_value * sigma
    )

    result["forecast_upper_95"] = (
        expected_change + z_value * sigma
    )

    return result


# ============================================================
# 7. MODEL EVALUATION
# ============================================================

def evaluate_predictions(
    test_predictions: pd.DataFrame,
    train: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate out-of-sample prediction quality."""

    train_base_rate = float(
        train["direction_up"].mean()
    )

    records = []

    horizons = [
        "All"
    ] + sorted(
        test_predictions["horizon_days"].unique()
    )

    for horizon in horizons:

        if horizon == "All":
            subset = test_predictions
        else:
            subset = test_predictions.loc[
                test_predictions["horizon_days"] == horizon
            ]

        if subset.empty:
            continue

        y_true = subset[TARGET].to_numpy()
        y_pred = subset[
            "predicted_change_eur_mwh"
        ].to_numpy()

        actual_direction = (
            subset["direction_up"]
        ).astype(int).to_numpy()

        predicted_direction = (
            subset["predicted_direction"]
        ).astype(int).to_numpy()

        probabilities = subset[
            "probability_up"
        ].to_numpy()

        # Naive reference: no change in futures price.
        baseline_change = np.zeros_like(y_true)

        # Constant directional probability estimated
        # exclusively from the training period.
        baseline_probabilities = np.full(
            len(subset),
            train_base_rate,
        )

        records.append({
            "horizon_days": horizon,
            "observations": len(subset),
            "mae_eur_mwh": mean_absolute_error(
                y_true, y_pred
            ),
            "rmse_eur_mwh": np.sqrt(
                mean_squared_error(y_true, y_pred)
            ),
            "directional_accuracy": accuracy_score(
                actual_direction,
                predicted_direction,
            ),
            "brier_score": brier_score_loss(
                actual_direction,
                probabilities,
            ),
            "baseline_mae_eur_mwh": mean_absolute_error(
                y_true,
                baseline_change,
            ),
            "baseline_rmse_eur_mwh": np.sqrt(
                mean_squared_error(
                    y_true,
                    baseline_change,
                )
            ),
            "baseline_brier_score": brier_score_loss(
                actual_direction,
                baseline_probabilities,
            ),
        })

    return pd.DataFrame(records).round(5)


# ============================================================
# 8. LATEST AVAILABLE FORECASTS
# ============================================================

def generate_latest_forecasts(
    model,
    features: pd.DataFrame,
    sigma_by_horizon: dict,
    global_sigma: float,
) -> pd.DataFrame:
    """
    Generate model forecasts from the latest quotation date.

    D+1 is marked unavailable because the current synthetic
    dataset has no next pre-delivery quotation for that contract.
    """

    latest_date = features[
        "quotation_date"
    ].max()

    latest = features.loc[
        (features["quotation_date"] == latest_date)
        & features["horizon_days"].isin(
            FORECAST_HORIZONS
        )
    ].copy()

    latest = predict_arx(
        model,
        latest,
        sigma_by_horizon,
        global_sigma,
    )

    latest["forecast_for_quotation_date"] = (
        latest["quotation_date"]
        + pd.Timedelta(days=1)
    )

    latest["forecast_status"] = (
        "Available - synthetic model"
    )

    unavailable = features.loc[
        (features["quotation_date"] == latest_date)
        & (features["horizon_days"] == 1)
    ].copy()

    if not unavailable.empty:
        unavailable[
            "forecast_for_quotation_date"
        ] = pd.NaT

        for col in [
            "predicted_change_eur_mwh",
            "predicted_next_price_eur_mwh",
            "predicted_sigma_eur_mwh",
            "probability_up",
            "probability_down",
            "forecast_lower_95",
            "forecast_upper_95",
        ]:
            unavailable[col] = np.nan

        unavailable["predicted_direction"] = np.nan
        unavailable["trading_signal"] = "Not available"
        unavailable["forecast_status"] = (
            "No next pre-delivery quotation for D+1"
        )

        latest = pd.concat(
            [latest, unavailable],
            ignore_index=True,
        )

    return latest.sort_values(
        "horizon_days"
    ).reset_index(drop=True)


# ============================================================
# 9. EXPORT RESULTS
# ============================================================

def export_results(
    model,
    test_predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    latest: pd.DataFrame,
) -> None:
    """Export forecasts, metrics and regression coefficients."""

    cfg.TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    test_columns = [
        "quotation_date",
        "target_quotation_date",
        "delivery_date",
        "contract_id",
        "horizon_days",
        "futures_price_eur_mwh",
        TARGET,
        "predicted_change_eur_mwh",
        "predicted_next_price_eur_mwh",
        "probability_up",
        "probability_down",
        "predicted_direction",
        "direction_up",
        "trading_signal",
        "predicted_sigma_eur_mwh",
        "forecast_lower_95",
        "forecast_upper_95",
    ]

    test_predictions[
        test_columns
    ].sort_values(
        ["quotation_date", "horizon_days"]
    ).to_csv(
        cfg.MODEL_RESULTS_PATH,
        index=False,
    )

    metrics.to_csv(
        cfg.TABLES_DIR / "forecast_metrics.csv",
        index=False,
    )

    coefficients = pd.DataFrame({
        "variable": model.params.index,
        "coefficient": model.params.values,
        "standard_error": model.bse.values,
        "t_statistic": model.tvalues.values,
        "p_value": model.pvalues.values,
    })

    coefficients.to_csv(
        cfg.TABLES_DIR / "forecast_coefficients.csv",
        index=False,
    )

    latest_columns = [
        "quotation_date",
        "forecast_for_quotation_date",
        "delivery_date",
        "contract_id",
        "horizon_days",
        "futures_price_eur_mwh",
        "predicted_change_eur_mwh",
        "predicted_next_price_eur_mwh",
        "probability_up",
        "probability_down",
        "trading_signal",
        "forecast_status",
    ]

    latest[
        latest_columns
    ].to_csv(
        cfg.TABLES_DIR / "latest_forecasts.csv",
        index=False,
    )


# ============================================================
# 10. COMPLETE FORECASTING PIPELINE
# ============================================================

def run_forecasting(
    input_path: Path = cfg.SIMULATED_DATA_PATH,
):
    """Fit, test, and export the ARX forecasting model."""

    print("\n" + "=" * 65)
    print("GERMAN POWER FUTURES DSS - ARX FORECASTING")
    print("=" * 65)

    data = load_data(input_path)

    features = prepare_features(data)

    train, test, cutoff = chronological_split(
        features
    )

    print(f"Training observations: {len(train):,}")
    print(f"Testing observations:  {len(test):,}")
    print(f"Training cutoff:       {cutoff:%Y-%m-%d}")

    model, sigma_by_horizon, global_sigma = (
        fit_arx_model(train)
    )

    test_predictions = predict_arx(
        model,
        test,
        sigma_by_horizon,
        global_sigma,
    )

    metrics = evaluate_predictions(
        test_predictions,
        train,
    )

    # Refit using all outcomes available in the historical
    # dataset before producing the latest illustrative forecast.
    all_labeled = features.loc[
        features["target_available"]
    ].copy()

    final_model, final_sigma, final_global_sigma = (
        fit_arx_model(all_labeled)
    )

    latest = generate_latest_forecasts(
        final_model,
        features,
        final_sigma,
        final_global_sigma,
    )

    export_results(
        model,
        test_predictions,
        metrics,
        latest,
    )

    print("\nOUT-OF-SAMPLE METRICS")
    print(metrics.to_string(index=False))

    print("\nLATEST SYNTHETIC FORECASTS")

    print(
        latest[
            [
                "horizon_days",
                "predicted_change_eur_mwh",
                "probability_up",
                "trading_signal",
            ]
        ].to_string(index=False)
    )

    print(f"\nResults saved to: {cfg.TABLES_DIR}")

    print(
        "\nIMPORTANT: Probabilities rely on a normal-error "
        "approximation and are not externally calibrated."
    )

    return {
        "model": model,
        "final_model": final_model,
        "metrics": metrics,
        "test_predictions": test_predictions,
        "latest_forecasts": latest,
    }


# ============================================================
# 11. COMMAND-LINE ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="ARX forecasting for German power futures."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=cfg.SIMULATED_DATA_PATH,
    )

    args = parser.parse_args()

    run_forecasting(args.input)


if __name__ == "__main__":
    main()
