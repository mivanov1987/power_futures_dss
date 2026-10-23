"""
Portfolio risk analytics for German Power Futures DSS.

Inputs:
    outputs/tables/price_changes.csv
    outputs/tables/latest_forecasts.csv

Method:
    Forecast-adjusted historical simulation using joint
    matched-contract price changes.

The scenario for a contract currently at D+h is sampled
from historical price changes observed at D+(h-1).

Outputs:
    outputs/tables/risk_results.csv
    outputs/tables/risk_portfolio_summary.csv
    outputs/tables/risk_scenarios.csv
    outputs/tables/risk_correlation.csv
    outputs/figures/portfolio_pnl_distribution.png

All results are illustrative and based on synthetic data.

Run:
    python -m src.risk_analytics
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as cfg


# ============================================================
# 1. CONFIGURATION AND INPUT VALIDATION
# ============================================================

CHANGES_PATH = cfg.TABLES_DIR / "price_changes.csv"
FORECASTS_PATH = cfg.TABLES_DIR / "latest_forecasts.csv"

SUPPORTED_HORIZONS = [
    h for h in cfg.DELIVERY_HORIZONS if h >= 2
]


def load_inputs(
    changes_path: Path = CHANGES_PATH,
    forecasts_path: Path = FORECASTS_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load historical changes and current model forecasts."""

    changes_path = Path(changes_path)
    forecasts_path = Path(forecasts_path)

    if not changes_path.exists():
        raise FileNotFoundError(
            f"Missing file: {changes_path}\n"
            "Run: python -m src.market_analytics"
        )

    if not forecasts_path.exists():
        raise FileNotFoundError(
            f"Missing file: {forecasts_path}\n"
            "Run: python -m src.forecasting"
        )

    changes = pd.read_csv(
        changes_path,
        parse_dates=["quotation_date", "delivery_date"],
    )

    forecasts = pd.read_csv(
        forecasts_path,
        parse_dates=["quotation_date", "delivery_date"],
    )

    required_changes = {
        "quotation_date",
        "horizon_days",
        "price_change_eur_mwh",
    }

    required_forecasts = {
        "quotation_date",
        "delivery_date",
        "contract_id",
        "horizon_days",
        "futures_price_eur_mwh",
        "predicted_change_eur_mwh",
        "probability_up",
    }

    missing_changes = required_changes - set(changes.columns)
    missing_forecasts = (
        required_forecasts - set(forecasts.columns)
    )

    if missing_changes:
        raise ValueError(
            f"Missing change columns: {sorted(missing_changes)}"
        )

    if missing_forecasts:
        raise ValueError(
            f"Missing forecast columns: {sorted(missing_forecasts)}"
        )

    if changes.duplicated(
        ["quotation_date", "horizon_days"]
    ).any():
        raise ValueError(
            "Duplicate historical change observations."
        )

    if forecasts.duplicated(["horizon_days"]).any():
        raise ValueError(
            "Multiple latest forecasts for one horizon."
        )

    changes["price_change_eur_mwh"] = pd.to_numeric(
        changes["price_change_eur_mwh"],
        errors="raise",
    )

    if not np.isfinite(
        changes["price_change_eur_mwh"].to_numpy()
    ).all():
        raise ValueError(
            "Historical changes contain non-finite values."
        )

    return changes, forecasts


# ============================================================
# 2. POSITION CONFIGURATION
# ============================================================

def parse_positions(
    specification: str | None,
    default_mw: float,
) -> dict[int, float]:
    """
    Create signed positions by current delivery horizon.

    Positive MW: long futures position.
    Negative MW: short futures position.
    Zero MW: no position.

    Example:
        --positions "2:5,3:-3,4:0,5:5,6:-2,7:5"
    """

    positions = {
        h: float(default_mw)
        for h in SUPPORTED_HORIZONS
    }

    if specification is None:
        return positions

    for item in specification.split(","):
        item = item.strip()

        if not item:
            continue

        if ":" not in item:
            raise ValueError(
                f"Invalid position specification: {item}"
            )

        horizon_text, position_text = item.split(":", 1)

        horizon = int(horizon_text.strip())
        position = float(position_text.strip())

        if horizon not in SUPPORTED_HORIZONS:
            raise ValueError(
                f"Unsupported horizon: D+{horizon}"
            )

        if not np.isfinite(position):
            raise ValueError(
                f"Invalid position for D+{horizon}"
            )

        positions[horizon] = position

    return positions


def calculate_delivery_hours(
    delivery_date: pd.Timestamp,
) -> int:
    """Calculate hours in the German baseload delivery day."""

    start = pd.Timestamp(delivery_date).normalize()

    next_day = start + pd.Timedelta(days=1)

    start_utc = start.tz_localize(
        cfg.TIMEZONE
    ).tz_convert("UTC")

    end_utc = next_day.tz_localize(
        cfg.TIMEZONE
    ).tz_convert("UTC")

    return int(
        (end_utc - start_utc).total_seconds() / 3600
    )


def prepare_portfolio(
    forecasts: pd.DataFrame,
    positions: dict[int, float],
) -> pd.DataFrame:
    """Combine model forecasts with illustrative positions."""

    portfolio = forecasts.loc[
        forecasts["horizon_days"].isin(
            SUPPORTED_HORIZONS
        )
    ].copy()

    portfolio["position_mw"] = (
        portfolio["horizon_days"].map(positions)
    )

    portfolio["delivery_hours"] = (
        portfolio["delivery_date"].apply(
            calculate_delivery_hours
        )
    )

    portfolio["signed_volume_mwh"] = (
        portfolio["position_mw"]
        * portfolio["delivery_hours"]
    )

    portfolio["absolute_volume_mwh"] = (
        portfolio["signed_volume_mwh"].abs()
    )

    required = [
        "position_mw",
        "predicted_change_eur_mwh",
        "signed_volume_mwh",
    ]

    if portfolio[required].isna().any().any():
        raise ValueError(
            "Missing positions or forecast values."
        )

    numeric_values = portfolio[
        required
    ].to_numpy(dtype=float)

    if not np.isfinite(numeric_values).all():
        raise ValueError(
            "Non-finite portfolio or forecast values."
        )

    available = set(portfolio["horizon_days"])

    missing = set(SUPPORTED_HORIZONS) - available

    if missing:
        raise ValueError(
            f"Missing model forecasts: {sorted(missing)}"
        )

    # Market-value sensitivity to EUR 1/MWh price movement.
    portfolio["eur_per_eur_mwh"] = (
        portfolio["signed_volume_mwh"]
    )

    portfolio["expected_gross_pnl_eur"] = (
        portfolio["signed_volume_mwh"]
        * portfolio["predicted_change_eur_mwh"]
    )

    return portfolio.sort_values(
        "horizon_days"
    ).reset_index(drop=True)


# ============================================================
# 3. JOINT HISTORICAL SCENARIOS
# ============================================================

def build_historical_scenarios(
    changes: pd.DataFrame,
    latest_date: pd.Timestamp,
    lookback_days: int,
) -> pd.DataFrame:
    """
    Build synchronized historical observations.

    A current D+2 position is exposed to a D+1 revision
    tomorrow, D+3 to a D+2 revision, and so on.

    Observations are aligned by historical quotation date,
    preserving cross-contract dependence.
    """

    if lookback_days < 30:
        raise ValueError(
            "At least 30 lookback days are required."
        )

    required_change_horizons = [
        h - 1 for h in SUPPORTED_HORIZONS
    ]

    historical = changes.loc[
        (changes["quotation_date"] <= latest_date)
        & (
            changes["quotation_date"]
            > latest_date - pd.Timedelta(days=lookback_days)
        )
        & changes["horizon_days"].isin(
            required_change_horizons
        )
    ].copy()

    pivot = historical.pivot(
        index="quotation_date",
        columns="horizon_days",
        values="price_change_eur_mwh",
    )

    pivot = pivot.reindex(
        columns=required_change_horizons
    ).dropna()

    if len(pivot) < 30:
        raise ValueError(
            "Fewer than 30 complete joint historical "
            "scenarios. Increase --lookback-days."
        )

    # Rename columns to the CURRENT portfolio horizons.
    pivot.columns = [
        h + 1 for h in pivot.columns
    ]

    pivot.columns.name = "current_horizon_days"

    return pivot.sort_index()


# ============================================================
# 4. FORECAST-ADJUSTED SCENARIO P&L
# ============================================================

def calculate_scenario_pnl(
    historical_scenarios: pd.DataFrame,
    portfolio: pd.DataFrame,
    transaction_cost_eur_mwh: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute joint portfolio P&L in EUR.

    Historical changes are centered by horizon and added
    to today's ARX expected price change:

        scenario_change = ARX_mean + historical_deviation

    This is forecast-adjusted historical simulation,
    not a fully calibrated conditional risk model.

    A one-time illustrative transaction cost is charged
    per MWh of absolute exposure.
    """

    if transaction_cost_eur_mwh < 0:
        raise ValueError(
            "Transaction cost must be non-negative."
        )

    result = historical_scenarios.copy()

    selected_horizons = portfolio[
        "horizon_days"
    ].astype(int).tolist()

    historical_means = (
        result[selected_horizons].mean()
    )

    component_pnl = pd.DataFrame(
        index=result.index
    )

    for _, position in portfolio.iterrows():

        horizon = int(position["horizon_days"])

        historical_deviation = (
            result[horizon]
            - historical_means[horizon]
        )

        scenario_change = (
            float(position["predicted_change_eur_mwh"])
            + historical_deviation
        )

        component_pnl[f"pnl_D{horizon}_eur"] = (
            scenario_change
            * float(position["signed_volume_mwh"])
        )

    total_cost = float(
        portfolio["absolute_volume_mwh"].sum()
        * transaction_cost_eur_mwh
    )

    component_pnl["gross_portfolio_pnl_eur"] = (
        component_pnl.sum(axis=1)
    )

    component_pnl["transaction_cost_eur"] = total_cost

    component_pnl["net_portfolio_pnl_eur"] = (
        component_pnl["gross_portfolio_pnl_eur"]
        - total_cost
    )

    component_pnl["portfolio_loss_eur"] = (
        -component_pnl["net_portfolio_pnl_eur"]
    )

    portfolio = portfolio.copy()

    portfolio["transaction_cost_eur"] = (
        portfolio["absolute_volume_mwh"]
        * transaction_cost_eur_mwh
    )

    portfolio["expected_net_pnl_eur"] = (
        portfolio["expected_gross_pnl_eur"]
        - portfolio["transaction_cost_eur"]
    )

    return component_pnl, portfolio


# ============================================================
# 5. RISK MEASURES
# ============================================================

def calculate_var_cvar(
    pnl: pd.Series | np.ndarray,
    confidence: float,
) -> dict:
    """
    Estimate historical VaR and CVaR in EUR.

    VaR is the selected quantile of loss.
    CVaR is mean loss in the quantile-defined tail.

    Both are floored at zero for reporting as
    non-negative loss-risk measures.
    """

    if not 0 < confidence < 1:
        raise ValueError(
            "Confidence must be between 0 and 1."
        )

    values = np.asarray(pnl, dtype=float)

    values = values[np.isfinite(values)]

    if len(values) < 30:
        raise ValueError(
            "Insufficient valid P&L scenarios."
        )

    losses = -values

    raw_var = float(
        np.quantile(losses, confidence)
    )

    tail = losses[losses >= raw_var]

    raw_cvar = float(tail.mean())

    return {
        "var_eur": max(0.0, raw_var),
        "cvar_eur": max(0.0, raw_cvar),
        "raw_loss_quantile_eur": raw_var,
        "mean_pnl_eur": float(values.mean()),
        "pnl_std_eur": float(values.std(ddof=1)),
        "probability_of_loss": float(
            np.mean(values < 0)
        ),
        "worst_pnl_eur": float(values.min()),
        "best_pnl_eur": float(values.max()),
        "observations": len(values),
    }


# ============================================================
# 6. INDIVIDUAL AND PORTFOLIO RISK
# ============================================================

def calculate_position_risks(
    scenarios: pd.DataFrame,
    portfolio: pd.DataFrame,
    confidence: float,
) -> pd.DataFrame:
    """Calculate standalone risks of each open position."""

    records = []

    for _, position in portfolio.iterrows():

        horizon = int(position["horizon_days"])

        pnl_column = f"pnl_D{horizon}_eur"

        net_position_pnl = (
            scenarios[pnl_column]
            - position["transaction_cost_eur"]
        )

        risk = calculate_var_cvar(
            net_position_pnl,
            confidence,
        )

        records.append({
            "horizon_days": horizon,
            "contract_id": position["contract_id"],
            "delivery_date": position["delivery_date"],
            "position_mw": position["position_mw"],
            "delivery_hours": position["delivery_hours"],
            "signed_volume_mwh": position["signed_volume_mwh"],
            "predicted_change_eur_mwh": (
                position["predicted_change_eur_mwh"]
            ),
            "probability_up": position["probability_up"],
            "expected_net_pnl_eur": (
                position["expected_net_pnl_eur"]
            ),
            "var_eur": risk["var_eur"],
            "cvar_eur": risk["cvar_eur"],
            "probability_of_loss": (
                risk["probability_of_loss"]
            ),
            "worst_pnl_eur": risk["worst_pnl_eur"],
            "best_pnl_eur": risk["best_pnl_eur"],
        })

    return pd.DataFrame(records)


def calculate_portfolio_risk(
    scenarios: pd.DataFrame,
    portfolio: pd.DataFrame,
    individual_risks: pd.DataFrame,
    confidence: float,
) -> pd.DataFrame:
    """Calculate joint portfolio VaR and CVaR."""

    risk = calculate_var_cvar(
        scenarios["net_portfolio_pnl_eur"],
        confidence,
    )

    expected_pnl = float(
        portfolio["expected_net_pnl_eur"].sum()
    )

    total_absolute_volume = float(
        portfolio["absolute_volume_mwh"].sum()
    )

    total_transaction_cost = float(
        portfolio["transaction_cost_eur"].sum()
    )

    standalone_var_sum = float(
        individual_risks["var_eur"].sum()
    )

    return pd.DataFrame([{
        "confidence_level": confidence,
        "forecast_date": portfolio[
            "quotation_date"
        ].max(),
        "positions": int(
            (portfolio["position_mw"] != 0).sum()
        ),
        "net_position_mw": float(
            portfolio["position_mw"].sum()
        ),
        "gross_volume_mwh": total_absolute_volume,
        "transaction_cost_eur": total_transaction_cost,
        "expected_net_pnl_eur": expected_pnl,
        "scenario_mean_pnl_eur": risk["mean_pnl_eur"],
        "portfolio_var_eur": risk["var_eur"],
        "portfolio_cvar_eur": risk["cvar_eur"],
        "probability_of_loss": (
            risk["probability_of_loss"]
        ),
        "worst_scenario_pnl_eur": risk["worst_pnl_eur"],
        "best_scenario_pnl_eur": risk["best_pnl_eur"],
        "standalone_var_sum_eur": standalone_var_sum,
        "joint_scenarios": risk["observations"],
    }])


# ============================================================
# 7. VISUALIZATION
# ============================================================

def plot_portfolio_distribution(
    scenarios: pd.DataFrame,
    portfolio_summary: pd.DataFrame,
) -> None:
    """Save the distribution of simulated portfolio P&L."""

    pnl = scenarios[
        "net_portfolio_pnl_eur"
    ].to_numpy()

    var = float(
        portfolio_summary.iloc[0]["portfolio_var_eur"]
    )

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.hist(
        pnl,
        bins=25,
        edgecolor="white",
        alpha=0.85,
    )

    ax.axvline(
        0,
        linestyle="--",
        label="Break-even",
    )

    ax.axvline(
        -var,
        linestyle=":",
        linewidth=2,
        label="Reported VaR threshold",
    )

    ax.set_title(
        "Synthetic Portfolio P&L: Forecast-Adjusted Scenarios"
    )
    ax.set_xlabel("One-day portfolio P&L (EUR)")
    ax.set_ylabel("Historical scenario count")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()

    cfg.FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        cfg.FIGURES_DIR
        / "portfolio_pnl_distribution.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# 8. EXPORT RESULTS
# ============================================================

def export_results(
    individual_risks: pd.DataFrame,
    portfolio_summary: pd.DataFrame,
    scenarios: pd.DataFrame,
    correlation: pd.DataFrame,
) -> None:
    """Export all risk analytics outputs."""

    cfg.TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    individual_risks.to_csv(
        cfg.RISK_RESULTS_PATH,
        index=False,
    )

    portfolio_summary.to_csv(
        cfg.TABLES_DIR / "risk_portfolio_summary.csv",
        index=False,
    )

    scenarios.reset_index().to_csv(
        cfg.TABLES_DIR / "risk_scenarios.csv",
        index=False,
    )

    correlation.to_csv(
        cfg.TABLES_DIR / "risk_correlation.csv",
        index_label="current_horizon_days",
    )


# ============================================================
# 9. COMPLETE RISK PIPELINE
# ============================================================

def run_risk_analysis(
    positions: dict[int, float] | None = None,
    lookback_days: int = cfg.RISK_LOOKBACK_DAYS,
    confidence: float = cfg.VAR_CONFIDENCE_LEVEL,
    transaction_cost: float = cfg.TRANSACTION_COST_EUR_MWH,
) -> dict:
    """Run portfolio risk analytics."""

    print("\n" + "=" * 65)
    print("GERMAN POWER FUTURES DSS - PORTFOLIO RISK")
    print("=" * 65)

    changes, forecasts = load_inputs()

    if positions is None:
        positions = parse_positions(
            None,
            cfg.DEFAULT_POSITION_MW,
        )

    portfolio = prepare_portfolio(
        forecasts,
        positions,
    )

    latest_date = portfolio[
        "quotation_date"
    ].max()

    historical = build_historical_scenarios(
        changes,
        latest_date,
        lookback_days,
    )

    scenarios, portfolio = calculate_scenario_pnl(
        historical,
        portfolio,
        transaction_cost,
    )

    individual_risks = calculate_position_risks(
        scenarios,
        portfolio,
        confidence,
    )

    portfolio_summary = calculate_portfolio_risk(
        scenarios,
        portfolio,
        individual_risks,
        confidence,
    )

    correlation = historical.corr()

    export_results(
        individual_risks,
        portfolio_summary,
        scenarios,
        correlation,
    )

    plot_portfolio_distribution(
        scenarios,
        portfolio_summary,
    )

    print("\nINDIVIDUAL POSITION RISKS (EUR)")
    print(
        individual_risks[
            [
                "horizon_days",
                "position_mw",
                "expected_net_pnl_eur",
                "var_eur",
                "cvar_eur",
            ]
        ].round(2).to_string(index=False)
    )

    print("\nPORTFOLIO RISK SUMMARY")
    summary_display = portfolio_summary.copy()

    numeric_columns = summary_display.select_dtypes(
        include="number"
    ).columns

    summary_display[numeric_columns] = (
        summary_display[numeric_columns].round(2)
    )

    print(summary_display.to_string(index=False))

    print("\nCORRELATION MATRIX")
    print(correlation.round(3).to_string())

    print(
        "\nOutputs saved to:\n"
        f"{cfg.TABLES_DIR}\n{cfg.FIGURES_DIR}"
    )

    print(
        "\nWARNING: Synthetic-data, one-day, "
        "forecast-adjusted risk simulation."
    )

    return {
        "positions": portfolio,
        "historical_scenarios": historical,
        "pnl_scenarios": scenarios,
        "individual_risks": individual_risks,
        "portfolio_summary": portfolio_summary,
        "correlation": correlation,
    }


# ============================================================
# 10. COMMAND LINE
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate portfolio risk for German "
            "daily electricity futures."
        )
    )

    parser.add_argument(
        "--positions",
        type=str,
        default=None,
        help=(
            'Signed MW positions, e.g. '
            '"2:5,3:-3,4:0,5:5,6:-2,7:5"'
        ),
    )

    parser.add_argument(
        "--default-mw",
        type=float,
        default=cfg.DEFAULT_POSITION_MW,
    )

    parser.add_argument(
        "--lookback-days",
        type=int,
        default=cfg.RISK_LOOKBACK_DAYS,
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=cfg.VAR_CONFIDENCE_LEVEL,
    )

    parser.add_argument(
        "--transaction-cost",
        type=float,
        default=cfg.TRANSACTION_COST_EUR_MWH,
    )

    args = parser.parse_args()

    positions = parse_positions(
        args.positions,
        args.default_mw,
    )

    run_risk_analysis(
        positions=positions,
        lookback_days=args.lookback_days,
        confidence=args.confidence,
        transaction_cost=args.transaction_cost,
    )


if __name__ == "__main__":
    main()
