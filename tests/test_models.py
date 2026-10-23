
"""
Automated tests for German Power Futures DSS.

Tests cover:
1. Synthetic data generation and reproducibility
2. Contract identity and delivery calendar
3. Matched-contract price changes
4. ARX forecasting and prevention of look-ahead bias
5. Probabilistic forecast consistency
6. Historical VaR and CVaR
7. Portfolio exposure and P&L calculations
8. Hedging strategies and accounting identities

Run from the project root:

    python -m pytest tests/test_models.py -v

Tests operate on in-memory data and do not overwrite
the project's existing CSV outputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config as cfg

from src import data_generator as generator
from src import market_analytics as market
from src import forecasting as forecast
from src import risk_analytics as risk
from src import strategy_simulator as strategy


# ============================================================
# 1. FIXTURES
# ============================================================

@pytest.fixture(scope="module")
def synthetic_data():
    """Small reproducible synthetic futures dataset."""

    return generator.generate_futures_data(
        start_date="2025-01-01",
        end_date="2025-04-30",
        seed=42,
    )


@pytest.fixture(scope="module")
def forecast_features(synthetic_data):
    """Prepared features for the ARX model."""

    return forecast.prepare_features(
        synthetic_data
    )


@pytest.fixture
def sample_strategy_forecasts():
    """
    Deterministic example of six futures forecasts.

    All contracts have 24-hour delivery days.
    Price changes and probabilities are illustrative.
    """

    quotation_date = pd.Timestamp("2025-06-10")
    target_date = pd.Timestamp("2025-06-11")

    records = []

    changes = {
        2: 10.0,
        3: -5.0,
        4: 8.0,
        5: -12.0,
        6: 3.0,
        7: -2.0,
    }

    probabilities = {
        2: 0.80,
        3: 0.20,
        4: 0.70,
        5: 0.30,
        6: 0.55,
        7: 0.45,
    }

    for horizon in range(2, 8):

        delivery_date = (
            quotation_date
            + pd.Timedelta(days=horizon)
        )

        records.append({
            "quotation_date": quotation_date,
            "target_quotation_date": target_date,
            "delivery_date": delivery_date,
            "contract_id": (
                f"DE_BASE_{delivery_date:%Y%m%d}"
            ),
            "horizon_days": horizon,
            "futures_price_eur_mwh": 85.0,
            "next_price_change": changes[horizon],
            "predicted_change_eur_mwh": (
                changes[horizon] * 0.5
            ),
            "probability_up": probabilities[horizon],
        })

    return pd.DataFrame(records)


# ============================================================
# 2. SYNTHETIC DATA GENERATION
# ============================================================

def test_generator_expected_number_of_rows(
    synthetic_data,
):
    """Seven futures quotations must exist per date."""

    number_of_days = len(
        pd.date_range("2025-01-01", "2025-04-30")
    )

    expected_rows = number_of_days * 7

    assert len(synthetic_data) == expected_rows


def test_generator_reproducibility(synthetic_data):
    """The same random seed must reproduce identical data."""

    repeated = generator.generate_futures_data(
        start_date="2025-01-01",
        end_date="2025-04-30",
        seed=42,
    )

    pd.testing.assert_frame_equal(
        synthetic_data,
        repeated,
    )


def test_different_seeds_change_simulated_prices():
    """Changing the random seed should affect prices."""

    first = generator.generate_futures_data(
        start_date="2025-01-01",
        end_date="2025-01-15",
        seed=42,
    )

    second = generator.generate_futures_data(
        start_date="2025-01-01",
        end_date="2025-01-15",
        seed=123,
    )

    assert not np.array_equal(
        first["futures_price_eur_mwh"].to_numpy(),
        second["futures_price_eur_mwh"].to_numpy(),
    )


def test_unique_contract_quotations(synthetic_data):
    """No contract should have duplicate quotes on one date."""

    duplicates = synthetic_data.duplicated(
        ["quotation_date", "contract_id"]
    )

    assert not duplicates.any()


def test_contract_identity(synthetic_data):
    """Contract identifiers must encode delivery dates."""

    expected_ids = synthetic_data[
        "delivery_date"
    ].dt.strftime("DE_BASE_%Y%m%d")

    assert (
        synthetic_data["contract_id"].to_numpy()
        == expected_ids.to_numpy()
    ).all()


def test_delivery_horizons(synthetic_data):
    """Delivery horizon equals delivery date minus quote date."""

    actual = (
        synthetic_data["delivery_date"]
        - synthetic_data["quotation_date"]
    ).dt.days

    assert np.array_equal(
        actual.to_numpy(),
        synthetic_data["horizon_days"].to_numpy(),
    )

    assert set(
        synthetic_data["horizon_days"]
    ) == set(range(1, 8))


def test_weather_forecasts_are_finite(synthetic_data):
    """All weather forecast variables should be finite."""

    columns = [
        "temperature_c",
        "wind_speed_ms",
        "solar_radiation_wm2",
    ]

    values = synthetic_data[
        columns
    ].to_numpy(dtype=float)

    assert np.isfinite(values).all()

    assert (
        synthetic_data["wind_speed_ms"] >= 0
    ).all()

    assert (
        synthetic_data["solar_radiation_wm2"] >= 0
    ).all()


# ============================================================
# 3. DAYLIGHT SAVING TIME
# ============================================================

@pytest.mark.parametrize(
    "delivery_date,expected_hours",
    [
        ("2025-03-29", 24),
        ("2025-03-30", 23),
        ("2025-03-31", 24),
        ("2025-10-25", 24),
        ("2025-10-26", 25),
        ("2025-10-27", 24),
    ],
)
def test_german_delivery_hours(
    delivery_date,
    expected_hours,
):
    """German delivery days must account for DST."""

    date = pd.Timestamp(delivery_date)

    assert (
        generator.get_delivery_hours(date)
        == expected_hours
    )

    assert (
        risk.calculate_delivery_hours(date)
        == expected_hours
    )

    assert (
        strategy.delivery_hours(date)
        == expected_hours
    )


# ============================================================
# 4. MARKET ANALYTICS
# ============================================================

def test_matched_contract_changes(synthetic_data):
    """Price changes must compare the same contract."""

    changes = market.calculate_price_changes(
        synthetic_data
    )

    assert not changes.empty

    assert (
        changes["observation_gap_days"] == 1
    ).all()

    expected = (
        changes["futures_price_eur_mwh"]
        - changes["previous_price_eur_mwh"]
    )

    np.testing.assert_allclose(
        changes["price_change_eur_mwh"],
        expected,
        rtol=0,
        atol=1e-10,
    )

    assert (
        changes["previous_horizon_days"]
        == changes["horizon_days"] + 1
    ).all()


def test_first_contract_observation_excluded(
    synthetic_data,
):
    """Only quotations with a previous quote may form changes."""

    changes = market.calculate_price_changes(
        synthetic_data
    )

    first_dates = synthetic_data.groupby(
        "contract_id"
    )["quotation_date"].min()

    observed_first_dates = changes[
        "contract_id"
    ].map(first_dates)

    assert (
        changes["quotation_date"]
        > observed_first_dates
    ).all()


def test_extreme_events_are_above_threshold(
    synthetic_data,
):
    """Extreme observations must satisfy their threshold."""

    changes = market.calculate_price_changes(
        synthetic_data
    )

    extremes = market.detect_extreme_events(
        changes,
        quantile=0.95,
    )

    assert not extremes.empty

    assert (
        extremes["absolute_price_change"]
        >= extremes["extreme_threshold_eur_mwh"]
    ).all()


# ============================================================
# 5. FORECASTING
# ============================================================

def test_forecast_targets_use_same_contract(
    forecast_features,
):
    """Next-day targets must refer to identical contracts."""

    eligible = forecast_features.loc[
        forecast_features["target_available"]
    ]

    assert not eligible.empty

    assert (
        eligible["target_gap_days"] == 1
    ).all()

    assert (
        eligible["horizon_days"] >= 2
    ).all()

    expected_change = (
        eligible["next_price"]
        - eligible["futures_price_eur_mwh"]
    )

    np.testing.assert_allclose(
        eligible["next_price_change"],
        expected_change,
    )


def test_d1_has_no_next_quote_target(
    forecast_features,
):
    """D+1 cannot have a next pre-delivery daily quote."""

    d1 = forecast_features.loc[
        forecast_features["horizon_days"] == 1
    ]

    assert not d1["target_available"].any()


def test_chronological_split_prevents_target_leakage(
    forecast_features,
):
    """Training targets must precede testing targets."""

    train, test, cutoff = (
        forecast.chronological_split(
            forecast_features,
            train_ratio=0.80,
        )
    )

    assert not train.empty
    assert not test.empty

    assert (
        train["target_quotation_date"].max()
        <= cutoff
    )

    assert (
        test["target_quotation_date"].min()
        > cutoff
    )

    assert (
        train["target_quotation_date"].max()
        < test["target_quotation_date"].min()
    )


def test_forecasting_probabilities(
    forecast_features,
):
    """Predicted probabilities must be valid and sum to one."""

    train, test, _ = forecast.chronological_split(
        forecast_features
    )

    model, sigma_by_horizon, global_sigma = (
        forecast.fit_arx_model(train)
    )

    predictions = forecast.predict_arx(
        model,
        test,
        sigma_by_horizon,
        global_sigma,
    )

    assert (
        predictions["probability_up"]
        .between(0, 1)
        .all()
    )

    assert (
        predictions["probability_down"]
        .between(0, 1)
        .all()
    )

    np.testing.assert_allclose(
        predictions["probability_up"]
        + predictions["probability_down"],
        1.0,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        predictions["predicted_next_price_eur_mwh"],
        predictions["futures_price_eur_mwh"]
        + predictions["predicted_change_eur_mwh"],
    )


def test_forecast_probability_consistency(
    forecast_features,
):
    """
    Under the normal-error model, a positive predicted
    mean change must imply probability_up > 0.5.
    """

    train, test, _ = forecast.chronological_split(
        forecast_features
    )

    model, sigmas, global_sigma = (
        forecast.fit_arx_model(train)
    )

    result = forecast.predict_arx(
        model,
        test,
        sigmas,
        global_sigma,
    )

    positive = result[
        "predicted_change_eur_mwh"
    ] > 0

    negative = result[
        "predicted_change_eur_mwh"
    ] < 0

    assert (
        result.loc[positive, "probability_up"]
        > 0.5
    ).all()

    assert (
        result.loc[negative, "probability_up"]
        < 0.5
    ).all()


# ============================================================
# 6. VALUE AT RISK AND CVAR
# ============================================================

def test_var_cvar_known_distribution():
    """Verify risk metrics against an exact small example."""

    pnl = np.array(
        [-100.0, -50.0, 0.0, 50.0, 100.0] * 10
    )

    result = risk.calculate_var_cvar(
        pnl,
        confidence=0.80,
    )

    assert result["var_eur"] == pytest.approx(60.0)

    assert result["cvar_eur"] == pytest.approx(100.0)

    assert result["observations"] == 50


def test_cvar_not_below_var():
    """CVaR should not be smaller than reported VaR."""

    pnl = np.array(
        [-500, -250, -100, 0, 50, 100] * 10,
        dtype=float,
    )

    result = risk.calculate_var_cvar(
        pnl,
        confidence=0.95,
    )

    assert (
        result["cvar_eur"]
        >= result["var_eur"]
    )


def test_var_rejects_invalid_confidence():
    """Invalid confidence levels should raise exceptions."""

    pnl = np.arange(50, dtype=float)

    with pytest.raises(ValueError):
        risk.calculate_var_cvar(
            pnl,
            confidence=1.1,
        )


# ============================================================
# 7. PORTFOLIO SCENARIOS
# ============================================================

def test_risk_positions_signs():
    """Signed MW positions must be parsed correctly."""

    positions = risk.parse_positions(
        "2:5,3:-3,4:0,5:5,6:-2,7:5",
        default_mw=5.0,
    )

    assert positions[2] == 5.0
    assert positions[3] == -3.0
    assert positions[4] == 0.0

    assert sum(positions.values()) == pytest.approx(10.0)


def test_zero_position_has_zero_pnl():
    """Zero-volume futures positions must have zero P&L."""

    index = pd.date_range(
        "2025-01-01",
        periods=50,
        freq="D",
    )

    scenario_prices = pd.DataFrame(
        {
            2: np.linspace(-5, 5, 50),
            3: np.linspace(3, -3, 50),
        },
        index=index,
    )

    portfolio = pd.DataFrame({
        "horizon_days": [2, 3],
        "signed_volume_mwh": [0.0, 0.0],
        "absolute_volume_mwh": [0.0, 0.0],
        "predicted_change_eur_mwh": [2.0, -1.0],
        "expected_gross_pnl_eur": [0.0, 0.0],
    })
    
    pnl, updated_portfolio = (
        risk.calculate_scenario_pnl(
            scenario_prices,
            portfolio,
            transaction_cost_eur_mwh=0.1,
        )
    )

    np.testing.assert_allclose(
        pnl["net_portfolio_pnl_eur"],
        0.0,
    )

    assert (
        updated_portfolio["transaction_cost_eur"]
        == 0
    ).all()


def test_joint_scenarios_preserve_aligned_dates():
    """Joint risk scenarios must have complete horizon vectors."""

    dates = pd.date_range(
        "2025-01-01",
        periods=60,
        freq="D",
    )

    records = []

    for date in dates:

        for current_horizon in range(2, 8):

            records.append({
                "quotation_date": date,
                "horizon_days": current_horizon - 1,
                "price_change_eur_mwh": float(
                    current_horizon * np.sin(
                        (date.dayofyear) / 10
                    )
                ),
            })

    changes = pd.DataFrame(records)

    scenarios = risk.build_historical_scenarios(
        changes,
        latest_date=dates[-1],
        lookback_days=90,
    )

    assert list(scenarios.columns) == list(range(2, 8))

    assert len(scenarios) == 60

    assert not scenarios.isna().any().any()


# ============================================================
# 8. HEDGING STRATEGIES
# ============================================================

def test_adaptive_hedge_decision_rule():
    """Probability thresholds must produce correct ratios."""

    probabilities = pd.Series(
        [0.20, 0.40, 0.50, 0.60, 0.80]
    )

    ratios = strategy.adaptive_hedge_ratio(
        probabilities,
        lower_threshold=0.40,
        upper_threshold=0.60,
    )

    expected = np.array(
        [0.0, 0.5, 0.5, 1.0, 1.0]
    )

    np.testing.assert_allclose(
        ratios,
        expected,
    )


def test_zero_fixed_hedge_equals_unhedged(
    sample_strategy_forecasts,
):
    """A zero fixed hedge must reproduce unhedged results."""

    results = strategy.simulate_strategies(
        forecasts=sample_strategy_forecasts,
        exposure_mw=5.0,
        fixed_hedge_ratio=0.0,
        transaction_cost=0.1,
    )

    unhedged = results.loc[
        results["strategy"] == "Unhedged"
    ].sort_values("horizon_days")

    fixed = results.loc[
        results["strategy"] == "Fixed Hedge"
    ].sort_values("horizon_days")

    np.testing.assert_allclose(
        unhedged["net_pnl_eur"],
        fixed["net_pnl_eur"],
    )

    assert fixed["transaction_cost_eur"].sum() == 0


def test_full_hedge_offsets_price_movements(
    sample_strategy_forecasts,
):
    """
    A 100% hedge offsets reference price movements
    under the assumed perfect-basis model.

    Only transaction costs remain.
    """

    results = strategy.simulate_strategies(
        forecasts=sample_strategy_forecasts,
        exposure_mw=5.0,
        fixed_hedge_ratio=1.0,
        transaction_cost=0.1,
    )

    fixed = results.loc[
        results["strategy"] == "Fixed Hedge"
    ]

    np.testing.assert_allclose(
        fixed["exposure_pnl_eur"]
        + fixed["hedge_pnl_eur"],
        0.0,
        atol=1e-10,
    )

    np.testing.assert_allclose(
        fixed["net_pnl_eur"],
        -fixed["transaction_cost_eur"],
        atol=1e-10,
    )


def test_strategy_pnl_accounting_identity(
    sample_strategy_forecasts,
):
    """Net P&L must equal exposure plus hedge less cost."""

    results = strategy.simulate_strategies(
        forecasts=sample_strategy_forecasts,
        exposure_mw=5.0,
        fixed_hedge_ratio=0.5,
        transaction_cost=0.1,
    )

    calculated = (
        results["exposure_pnl_eur"]
        + results["hedge_pnl_eur"]
        - results["transaction_cost_eur"]
    )

    np.testing.assert_allclose(
        results["net_pnl_eur"],
        calculated,
        atol=1e-10,
    )


def test_hedge_ratio_within_bounds(
    sample_strategy_forecasts,
):
    """All illustrative hedge ratios must remain in [0, 1]."""

    results = strategy.simulate_strategies(
        forecasts=sample_strategy_forecasts,
    )

    assert (
        results["hedge_ratio"].between(0, 1)
    ).all()


def test_strategy_cost_never_negative(
    sample_strategy_forecasts,
):
    """Transaction costs must not become negative."""

    results = strategy.simulate_strategies(
        forecasts=sample_strategy_forecasts,
        transaction_cost=0.2,
    )

    assert (
        results["transaction_cost_eur"] >= 0
    ).all()


# ============================================================
# 9. BASIC PROJECT CONFIGURATION
# ============================================================

def test_configuration():
    """Validate basic configuration assumptions."""

    assert cfg.RANDOM_SEED >= 0

    assert 0 < cfg.TRAIN_RATIO < 1

    assert cfg.MIN_SIMULATED_PRICE < (
        cfg.MAX_SIMULATED_PRICE
    )

    assert set(cfg.DELIVERY_HORIZONS) == set(
        range(1, 8)
    )

    assert (
        0
        <= cfg.SELL_PROBABILITY_THRESHOLD
        < cfg.BUY_PROBABILITY_THRESHOLD
        <= 1
    )


# ============================================================
# END OF TEST SUITE
# ============================================================