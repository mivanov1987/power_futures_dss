"""
Synthetic data generator for German Power Futures DSS.

Generates:
- Daily German baseload futures quotations (D+1 to D+7)
- Weather forecasts available on each quotation date
- Seasonality, price persistence and occasional price shocks
- Contract delivery dates and delivery-hour information

All data are SYNTHETIC, not actual EEX quotations.

Run from the project root:
    python -m src.data_generator
"""

from __future__ import annotations

import argparse
import sys
from datetime import timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# Allow execution both as a module and as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as cfg


# ============================================================
# 1. CALENDAR AND DELIVERY CONTRACT UTILITIES
# ============================================================

def get_delivery_hours(delivery_date: pd.Timestamp) -> int:
    """
    Calculate actual delivery-day hours in Europe/Berlin.

    Normally: 24 hours
    Spring DST transition: 23 hours
    Autumn DST transition: 25 hours
    """
    local_timezone = ZoneInfo(cfg.TIMEZONE)

    start = pd.Timestamp(delivery_date).date()
    end = start + timedelta(days=1)

    start_local = pd.Timestamp(
        start, tz=local_timezone
    )
    end_local = pd.Timestamp(
        end, tz=local_timezone
    )

    duration = (
        end_local.tz_convert(timezone.utc)
        - start_local.tz_convert(timezone.utc)
    )

    return int(duration.total_seconds() / 3600)


def get_contract_id(delivery_date: pd.Timestamp) -> str:
    """Create a stable identifier for a daily baseload contract."""
    return f"DE_BASE_{delivery_date:%Y%m%d}"


# ============================================================
# 2. SIMULATE UNDERLYING WEATHER CONDITIONS
# ============================================================

def generate_weather(
    delivery_dates: pd.DatetimeIndex,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Generate synthetic underlying weather conditions.

    These conditions are internal to the simulation.
    They are NOT exported as information available to traders.
    """
    n = len(delivery_dates)

    day_of_year = delivery_dates.dayofyear.to_numpy()
    seasonal_angle = 2 * np.pi * (
        day_of_year - 1
    ) / 365.25

    # Temperature: warm summer, cold winter.
    temperature = (
        cfg.AVERAGE_TEMPERATURE_C
        - cfg.TEMPERATURE_SEASONAL_AMPLITUDE
        * np.cos(seasonal_angle)
        + rng.normal(0, 3.0, n)
    )

    # Wind speed: stochastic and moderately persistent.
    wind = np.zeros(n)
    wind[0] = cfg.AVERAGE_WIND_SPEED_MS

    for i in range(1, n):
        wind[i] = (
            cfg.AVERAGE_WIND_SPEED_MS
            + 0.65 * (
                wind[i - 1]
                - cfg.AVERAGE_WIND_SPEED_MS
            )
            + rng.normal(0, 1.8)
        )

    wind = np.clip(wind, 0, 25)

    # Solar radiation: stronger in summer.
    solar_seasonal = (
        cfg.AVERAGE_SOLAR_RADIATION_WM2
        * (1 + 0.75 * np.cos(
            seasonal_angle - np.pi
        ))
    )

    solar = np.clip(
        solar_seasonal
        + rng.normal(0, 35, n),
        0,
        450,
    )

    return pd.DataFrame({
        "delivery_date": delivery_dates,
        "true_temperature_c": temperature,
        "true_wind_speed_ms": wind,
        "true_solar_radiation_wm2": solar,
    }).set_index("delivery_date")


# ============================================================
# 3. GENERATE WEATHER FORECASTS
# ============================================================

def simulate_weather_forecast(
    true_weather: pd.Series,
    horizon: int,
    rng: np.random.Generator,
) -> dict:
    """
    Simulate forecasts issued on the quotation date.

    Forecast uncertainty increases with the delivery horizon.

    The generator uses latent future weather to construct
    synthetic forecasts, but exports ONLY the forecasts.
    """
    uncertainty = np.sqrt(horizon)

    temperature = (
        true_weather["true_temperature_c"]
        + rng.normal(0, 1.2 * uncertainty)
    )

    wind = max(
        0.0,
        true_weather["true_wind_speed_ms"]
        + rng.normal(0, 0.9 * uncertainty),
    )

    solar = max(
        0.0,
        true_weather["true_solar_radiation_wm2"]
        + rng.normal(0, 15.0 * uncertainty),
    )

    return {
        "temperature_c": float(temperature),
        "wind_speed_ms": float(wind),
        "solar_radiation_wm2": float(solar),
    }


# ============================================================
# 4. DETERMINE SYNTHETIC FAIR VALUE
# ============================================================

def calculate_reference_price(
    delivery_date: pd.Timestamp,
    weather_forecast: dict,
) -> float:
    """
    Calculate a synthetic reference price for a delivery day.

    This is a simulation formula, not an estimated market model.
    """
    day_of_year = delivery_date.dayofyear
    day_of_week = delivery_date.dayofweek

    angle = 2 * np.pi * (
        day_of_year - 1
    ) / 365.25

    annual_component = (
        cfg.ANNUAL_SEASONAL_AMPLITUDE
        * np.cos(angle)
    )

    weekly_component = (
        cfg.WEEKLY_SEASONAL_AMPLITUDE
        * np.cos(2 * np.pi * day_of_week / 7)
    )

    # Deviations from approximate seasonal reference weather.
    seasonal_temperature = (
        cfg.AVERAGE_TEMPERATURE_C
        - cfg.TEMPERATURE_SEASONAL_AMPLITUDE
        * np.cos(angle)
    )

    seasonal_solar = (
        cfg.AVERAGE_SOLAR_RADIATION_WM2
        * (1 - 0.75 * np.cos(angle))
    )

    temperature_effect = (
        cfg.TEMPERATURE_PRICE_COEFFICIENT
        * (
            weather_forecast["temperature_c"]
            - seasonal_temperature
        )
    )

    wind_effect = (
        cfg.WIND_PRICE_COEFFICIENT
        * (
            weather_forecast["wind_speed_ms"]
            - cfg.AVERAGE_WIND_SPEED_MS
        )
    )

    solar_effect = (
        cfg.SOLAR_PRICE_COEFFICIENT
        * (
            weather_forecast["solar_radiation_wm2"]
            - seasonal_solar
        )
    )

    return float(
        cfg.BASE_PRICE_EUR_MWH
        + annual_component
        + weekly_component
        + temperature_effect
        + wind_effect
        + solar_effect
    )


# ============================================================
# 5. SIMULATE FUTURES QUOTATIONS
# ============================================================

def generate_futures_data(
    start_date: str = cfg.START_DATE,
    end_date: str = cfg.END_DATE,
    seed: int = cfg.RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generate a synthetic futures quotation panel.

    Each observation is identified by:
        quotation date + delivery date

    A contract keeps the same contract_id across quotation dates.

    Note:
        This is a calendar-day simulation. It does not reconstruct
        the actual EEX trading calendar or exchange fixing times.
    """
    rng = np.random.default_rng(seed)

    quotation_dates = pd.date_range(
        start_date,
        end_date,
        freq="D",
    )

    if quotation_dates.empty:
        raise ValueError("The selected date range is empty.")

    horizons = sorted(set(cfg.DELIVERY_HORIZONS))

    if not horizons or min(horizons) < 1:
        raise ValueError("Delivery horizons must be positive.")

    # Additional delivery dates are needed beyond the last quote.
    weather_dates = pd.date_range(
        quotation_dates.min() + pd.Timedelta(days=1),
        quotation_dates.max()
        + pd.Timedelta(days=max(horizons)),
        freq="D",
    )

    weather = generate_weather(weather_dates, rng)

    # A common daily market shock introduces cross-contract
    # dependence and short-term persistence.
    market_shocks = {}

    previous_market_shock = 0.0

    for quotation_date in quotation_dates:

        current_shock = (
            cfg.PRICE_AR_COEFFICIENT
            * previous_market_shock
            + rng.normal(
                0,
                cfg.DAILY_PRICE_VOLATILITY * 0.35,
            )
        )

        if rng.random() < cfg.EXTREME_EVENT_PROBABILITY:
            current_shock += rng.normal(
                0,
                cfg.EXTREME_EVENT_STD,
            )

        market_shocks[quotation_date] = current_shock
        previous_market_shock = current_shock

    # State variables for each individual delivery contract.
    previous_contract_price = {}
    previous_reference_price = {}
    previous_contract_change = {}

    records = []

    for quotation_date in quotation_dates:

        common_shock = market_shocks[quotation_date]

        for horizon in horizons:

            delivery_date = (
                quotation_date
                + pd.Timedelta(days=horizon)
            )

            contract_id = get_contract_id(delivery_date)

            true_weather = weather.loc[delivery_date]

            forecast = simulate_weather_forecast(
                true_weather,
                horizon,
                rng,
            )

            reference_price = calculate_reference_price(
                delivery_date,
                forecast,
            )

            if contract_id not in previous_contract_price:

                # First quotation in the generated observation
                # window for this particular delivery contract.
                futures_price = (
                    reference_price
                    + rng.normal(
                        0,
                        cfg.DAILY_PRICE_VOLATILITY,
                    )
                    + common_shock
                )

                price_change = 0.0

            else:

                last_price = previous_contract_price[
                    contract_id
                ]

                previous_reference = (
                    previous_reference_price[contract_id]
                )

                last_change = (
                    previous_contract_change[contract_id]
                )

                # Revisions in weather forecasts influence
                # synthetic quotation revisions.
                reference_revision = (
                    reference_price - previous_reference
                )

                # Persistence, mean reversion and random shock.
                price_change = (
                    0.55 * reference_revision
                    + cfg.PRICE_AR_COEFFICIENT
                    * last_change
                    + cfg.MEAN_REVERSION_SPEED
                    * (reference_price - last_price)
                    + common_shock
                    + rng.normal(
                        0,
                        cfg.DAILY_PRICE_VOLATILITY * 0.45,
                    )
                )

                futures_price = (
                    last_price + price_change
                )

            futures_price = float(np.clip(
                futures_price,
                cfg.MIN_SIMULATED_PRICE,
                cfg.MAX_SIMULATED_PRICE,
            ))

            # Track the actual clipped change.
            if contract_id in previous_contract_price:
                price_change = (
                    futures_price
                    - previous_contract_price[contract_id]
                )

            previous_contract_price[contract_id] = futures_price
            previous_reference_price[contract_id] = reference_price
            previous_contract_change[contract_id] = price_change

            records.append({
                "quotation_date": quotation_date,
                "delivery_date": delivery_date,
                "horizon_days": horizon,
                "contract_id": contract_id,
                "product_type": cfg.PRODUCT_TYPE,
                "delivery_hours": get_delivery_hours(
                    delivery_date
                ),
                "futures_price_eur_mwh": round(
                    futures_price, 4
                ),
                "temperature_c": round(
                    forecast["temperature_c"], 4
                ),
                "wind_speed_ms": round(
                    forecast["wind_speed_ms"], 4
                ),
                "solar_radiation_wm2": round(
                    forecast["solar_radiation_wm2"], 4
                ),
                "weather_forecast_issue_date": (
                    quotation_date
                ),
                "is_synthetic": True,
            })

    df = pd.DataFrame(records)

    df = df.sort_values(
        ["quotation_date", "horizon_days"]
    ).reset_index(drop=True)

    validate_generated_data(df, len(quotation_dates), len(horizons))

    return df


# ============================================================
# 6. VALIDATION
# ============================================================

def validate_generated_data(
    df: pd.DataFrame,
    expected_days: int,
    expected_horizons: int,
) -> None:
    """Perform basic data quality checks."""

    expected_rows = expected_days * expected_horizons

    if len(df) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} rows, got {len(df)}."
        )

    if df.isna().any().any():
        raise ValueError("Generated data contain missing values.")

    if df.duplicated(
        ["quotation_date", "delivery_date"]
    ).any():
        raise ValueError("Duplicate contract quotations detected.")

    observed_horizons = (
        df["delivery_date"] - df["quotation_date"]
    ).dt.days

    if not observed_horizons.equals(
        df["horizon_days"]
    ):
        raise ValueError("Incorrect delivery horizon detected.")

    if not df["delivery_hours"].isin(
        [23, 24, 25]
    ).all():
        raise ValueError("Invalid delivery-day duration detected.")

    if not df["futures_price_eur_mwh"].between(
        cfg.MIN_SIMULATED_PRICE,
        cfg.MAX_SIMULATED_PRICE,
    ).all():
        raise ValueError("Prices exceed configured bounds.")


# ============================================================
# 7. SAVE DATA AND PRINT SUMMARY
# ============================================================

def save_data(
    df: pd.DataFrame,
    output_path: Path = cfg.SIMULATED_DATA_PATH,
) -> Path:
    """Save simulated observations to a UTF-8 CSV file."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    export_df = df.copy()

    for column in [
        "quotation_date",
        "delivery_date",
        "weather_forecast_issue_date",
    ]:
        export_df[column] = export_df[
            column
        ].dt.strftime("%Y-%m-%d")

    export_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )

    return output_path


def print_data_summary(df: pd.DataFrame) -> None:
    """Display essential dataset information."""

    print("\n" + "=" * 65)
    print("GERMAN POWER FUTURES DSS - SYNTHETIC DATA")
    print("=" * 65)

    print(f"Observations:       {len(df):,}")
    print(
        "Quotation period:   "
        f"{df['quotation_date'].min():%Y-%m-%d}"
        " to "
        f"{df['quotation_date'].max():%Y-%m-%d}"
    )

    print(
        "Delivery horizons:  "
        f"{sorted(df['horizon_days'].unique().tolist())}"
    )

    print(
        "Unique contracts:   "
        f"{df['contract_id'].nunique():,}"
    )

    prices = df["futures_price_eur_mwh"]

    print(f"Mean price:         {prices.mean():.2f} EUR/MWh")
    print(f"Minimum price:      {prices.min():.2f} EUR/MWh")
    print(f"Maximum price:      {prices.max():.2f} EUR/MWh")
    print(f"Price std. dev.:    {prices.std():.2f} EUR/MWh")

    print("\nFirst five observations:")

    print(
        df[
            [
                "quotation_date",
                "delivery_date",
                "horizon_days",
                "futures_price_eur_mwh",
            ]
        ].head().to_string(index=False)
    )

    print("=" * 65)


# ============================================================
# 8. MAIN ENTRY POINT
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate simulated German daily power futures "
            "quotations and weather forecasts."
        )
    )

    parser.add_argument(
        "--start",
        default=cfg.START_DATE,
        help="First quotation date (YYYY-MM-DD).",
    )

    parser.add_argument(
        "--end",
        default=cfg.END_DATE,
        help="Last quotation date (YYYY-MM-DD).",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=cfg.RANDOM_SEED,
        help="Random seed for reproducibility.",
    )

    args = parser.parse_args()

    df = generate_futures_data(
        start_date=args.start,
        end_date=args.end,
        seed=args.seed,
    )

    output_path = save_data(df)

    print_data_summary(df)

    print(f"\nSaved dataset to:\n{output_path}")
    print("\nData are synthetic and for research demonstration only.")


if __name__ == "__main__":
    main()