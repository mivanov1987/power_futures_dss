"""
Trading strategy simulator for German Power Futures DSS.

Compares three hedging approaches:

1. Unhedged:
   No derivatives used to hedge the reference price exposure.

2. Fixed Hedge:
   A fixed proportion of exposure is hedged.

3. Adaptive Hedge:
   The hedge ratio depends on out-of-sample ARX probabilities.

Input:
    outputs/tables/forecast_results.csv

Outputs:
    outputs/tables/strategy_trade_results.csv
    outputs/tables/strategy_daily_results.csv
    outputs/tables/strategy_comparison.csv
    outputs/figures/strategy_cumulative_pnl.png
    outputs/figures/strategy_risk_comparison.png
    outputs/figures/adaptive_hedge_ratios.png

All data are synthetic.

Run from project root:
    python -m src.strategy_simulator
"""

from __future__ import annotations

import argparse
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config as cfg


# ============================================================
# 1. INPUTS AND PARAMETERS
# ============================================================

FORECAST_PATH = cfg.MODEL_RESULTS_PATH

STRATEGY_NAMES = [
    "Unhedged",
    "Fixed Hedge",
    "Adaptive Hedge",
]


def load_forecasts(
    path: Path = FORECAST_PATH,
) -> pd.DataFrame:
    """Load out-of-sample ARX forecasts."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Forecast file not found: {path}\n"
            "Run: python -m src.forecasting"
        )

    df = pd.read_csv(
        path,
        parse_dates=[
            "quotation_date",
            "target_quotation_date",
            "delivery_date",
        ],
    )

    required = [
        "quotation_date",
        "target_quotation_date",
        "delivery_date",
        "contract_id",
        "horizon_days",
        "futures_price_eur_mwh",
        "next_price_change",
        "predicted_change_eur_mwh",
        "probability_up",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    if df[required].isna().any().any():
        raise ValueError(
            "Forecast dataset contains missing required values."
        )

    if df.duplicated(
        ["quotation_date", "contract_id"]
    ).any():
        raise ValueError(
            "Duplicate contract forecasts detected."
        )

    if not df["horizon_days"].between(2, 7).all():
        raise ValueError(
            "This simulator supports D+2 to D+7."
        )

    if not (
        (df["delivery_date"] - df["quotation_date"]).dt.days
        == df["horizon_days"]
    ).all():
        raise ValueError(
            "Contract delivery dates are inconsistent."
        )

    if not (
        (df["target_quotation_date"] - df["quotation_date"])
        .dt.days.eq(1)
    ).all():
        raise ValueError(
            "Only consecutive one-day forecasts are supported."
        )

    numeric = [
        "futures_price_eur_mwh",
        "next_price_change",
        "predicted_change_eur_mwh",
        "probability_up",
    ]

    if not np.isfinite(
        df[numeric].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Forecast inputs contain non-finite values."
        )

    if not df["probability_up"].between(0, 1).all():
        raise ValueError(
            "Probabilities must be between zero and one."
        )

    # Ensure a complete balanced panel for daily comparisons.
    counts = df.groupby(
        "quotation_date"
    )["horizon_days"].nunique()

    if not counts.eq(6).all():
        raise ValueError(
            "Expected all six horizons D+2..D+7 on every date."
        )

    expected = set(range(2, 8))

    for _, group in df.groupby("quotation_date"):
        if set(group["horizon_days"]) != expected:
            raise ValueError(
                "An incomplete or repeated horizon was detected."
            )

    return df.sort_values(
        ["quotation_date", "horizon_days"]
    ).reset_index(drop=True)


# ============================================================
# 2. GERMAN DELIVERY HOURS
# ============================================================

def delivery_hours(
    date: pd.Timestamp,
) -> int:
    """
    Actual baseload delivery hours in Europe/Berlin.

    Handles 23-hour and 25-hour DST transition days.
    """
    local_start = pd.Timestamp(
        pd.Timestamp(date).date(),
        tz=cfg.TIMEZONE,
    )

    local_end = pd.Timestamp(
        pd.Timestamp(date).date() + timedelta(days=1),
        tz=cfg.TIMEZONE,
    )

    return int(
        (
            local_end.tz_convert("UTC")
            - local_start.tz_convert("UTC")
        ).total_seconds() / 3600
    )


# ============================================================
# 3. STRATEGY DECISION RULES
# ============================================================

def adaptive_hedge_ratio(
    probability_up: pd.Series,
    lower_threshold: float,
    upper_threshold: float,
) -> pd.Series:
    """
    Select hedge ratio using forecast probabilities.

    Low probability of increase:  0% hedge.
    Intermediate probability:    50% hedge.
    High probability:           100% hedge.

    The rule uses predictions only, not realized changes.
    """

    if not (
        0 <= lower_threshold < upper_threshold <= 1
    ):
        raise ValueError(
            "Thresholds must satisfy 0 <= lower < upper <= 1."
        )

    ratios = np.select(
        [
            probability_up >= upper_threshold,
            probability_up < lower_threshold,
        ],
        [
            1.0,
            0.0,
        ],
        default=0.5,
    )

    return pd.Series(
        ratios,
        index=probability_up.index,
        dtype=float,
    )


# ============================================================
# 4. COMPUTE STRATEGY RESULTS
# ============================================================

def simulate_strategies(
    forecasts: pd.DataFrame,
    exposure_mw: float = cfg.DEFAULT_POSITION_MW,
    fixed_hedge_ratio: float = cfg.DEFAULT_HEDGE_RATIO,
    lower_threshold: float = cfg.SELL_PROBABILITY_THRESHOLD,
    upper_threshold: float = cfg.BUY_PROBABILITY_THRESHOLD,
    transaction_cost: float = cfg.TRANSACTION_COST_EUR_MWH,
) -> pd.DataFrame:
    """
    Simulate changes in the cost of a reference exposure.

    Economic interpretation:
      - The enterprise is exposed to higher electricity prices.
      - A price increase raises the reference procurement cost.
      - A long futures position offsets some of that increase.

    Incremental P&L relative to an unchanged reference price:

        exposure_pnl = -volume * realized_price_change

        hedge_pnl = hedge_ratio * volume * realized_change

        net_pnl = exposure_pnl + hedge_pnl - hedge_cost

    Each observation is a separate one-day mark-to-market
    experiment. Positions are assumed to open and close
    over the observation interval.

    Results do NOT represent an actual physical procurement
    cash-flow statement or a continuously held hedge book.
    """

    if not np.isfinite(exposure_mw) or exposure_mw <= 0:
        raise ValueError("Exposure MW must be positive.")

    if not 0 <= fixed_hedge_ratio <= 1:
        raise ValueError(
            "Fixed hedge ratio must be between 0 and 1."
        )

    if not np.isfinite(transaction_cost) or transaction_cost < 0:
        raise ValueError(
            "Transaction cost must be non-negative."
        )

    base = forecasts.copy()

    base["delivery_hours"] = (
        base["delivery_date"].map(delivery_hours)
    )

    base["exposure_mw"] = float(exposure_mw)

    base["exposure_mwh"] = (
        base["exposure_mw"]
        * base["delivery_hours"]
    )

    base["exposure_pnl_eur"] = (
        -base["exposure_mwh"]
        * base["next_price_change"]
    )

    adaptive_ratios = adaptive_hedge_ratio(
        base["probability_up"],
        lower_threshold,
        upper_threshold,
    )

    strategy_frames = []

    for strategy in STRATEGY_NAMES:

        result = base.copy()

        result["strategy"] = strategy

        if strategy == "Unhedged":
            result["hedge_ratio"] = 0.0

        elif strategy == "Fixed Hedge":
            result["hedge_ratio"] = fixed_hedge_ratio

        else:
            result["hedge_ratio"] = adaptive_ratios

        result["hedge_position_mw"] = (
            result["exposure_mw"]
            * result["hedge_ratio"]
        )

        result["hedged_volume_mwh"] = (
            result["exposure_mwh"]
            * result["hedge_ratio"]
        )

        # Long futures hedge profits when prices increase.
        result["hedge_pnl_eur"] = (
            result["hedged_volume_mwh"]
            * result["next_price_change"]
        )

        # Simplified round-trip trading cost per hedged MWh.
        result["transaction_cost_eur"] = (
            result["hedged_volume_mwh"]
            * transaction_cost
        )

        result["net_pnl_eur"] = (
            result["exposure_pnl_eur"]
            + result["hedge_pnl_eur"]
            - result["transaction_cost_eur"]
        )

        # Positive cost means unfavorable movement
        # relative to the unchanged-price benchmark.
        result["incremental_cost_eur"] = (
            -result["net_pnl_eur"]
        )

        strategy_frames.append(result)

    output = pd.concat(
        strategy_frames,
        ignore_index=True,
    )

    return output.sort_values(
        ["strategy", "quotation_date", "horizon_days"]
    ).reset_index(drop=True)


# ============================================================
# 5. DAILY PORTFOLIO AGGREGATION
# ============================================================

def aggregate_daily_results(
    trades: pd.DataFrame,
) -> pd.DataFrame:
    """Combine six contract observations per decision date."""

    daily = trades.groupby(
        ["strategy", "quotation_date"],
        as_index=False,
    ).agg(
        target_quotation_date=(
            "target_quotation_date", "first"
        ),
        exposure_mwh=("exposure_mwh", "sum"),
        hedged_volume_mwh=("hedged_volume_mwh", "sum"),
        exposure_pnl_eur=("exposure_pnl_eur", "sum"),
        hedge_pnl_eur=("hedge_pnl_eur", "sum"),
        transaction_cost_eur=("transaction_cost_eur", "sum"),
        net_pnl_eur=("net_pnl_eur", "sum"),
        incremental_cost_eur=("incremental_cost_eur", "sum"),
        average_hedge_ratio=("hedge_ratio", "mean"),
    )

    daily = daily.sort_values(
        ["strategy", "quotation_date"]
    ).reset_index(drop=True)

    daily["cumulative_pnl_eur"] = (
        daily.groupby("strategy")["net_pnl_eur"].cumsum()
    )

    return daily


# ============================================================
# 6. RISK METRICS
# ============================================================

def calculate_var_cvar(
    pnl: pd.Series,
    confidence: float,
) -> tuple[float, float]:
    """
    Historical VaR and CVaR from realized daily P&L.

    Losses are defined as -P&L.
    """
    if not 0 < confidence < 1:
        raise ValueError(
            "Confidence must be between zero and one."
        )

    values = pnl.to_numpy(dtype=float)
    losses = -values

    threshold = float(
        np.quantile(losses, confidence)
    )

    tail = losses[losses >= threshold]

    var = max(0.0, threshold)
    cvar = max(0.0, float(tail.mean()))

    return var, cvar


def maximum_drawdown(
    pnl: pd.Series,
) -> float:
    """
    Maximum peak-to-trough loss of cumulative incremental P&L.

    The cumulative series starts at zero.
    """
    cumulative = np.concatenate(
        ([0.0], np.cumsum(pnl.to_numpy(dtype=float)))
    )

    running_maximum = np.maximum.accumulate(cumulative)

    drawdowns = running_maximum - cumulative

    return float(drawdowns.max())


def compare_strategies(
    daily: pd.DataFrame,
    confidence: float = cfg.VAR_CONFIDENCE_LEVEL,
) -> pd.DataFrame:
    """Compare hedging strategies on identical test dates."""

    records = []

    for strategy, group in daily.groupby("strategy"):

        group = group.sort_values("quotation_date")

        pnl = group["net_pnl_eur"]

        var, cvar = calculate_var_cvar(
            pnl,
            confidence,
        )

        records.append({
            "strategy": strategy,
            "days": len(group),
            "total_net_pnl_eur": float(pnl.sum()),
            "mean_daily_pnl_eur": float(pnl.mean()),
            "daily_pnl_std_eur": float(pnl.std(ddof=1)),
            "total_incremental_cost_eur": float(
                group["incremental_cost_eur"].sum()
            ),
            "transaction_cost_eur": float(
                group["transaction_cost_eur"].sum()
            ),
            "average_hedge_ratio": float(
                group["average_hedge_ratio"].mean()
            ),
            "probability_of_loss": float(
                (pnl < 0).mean()
            ),
            "var_95_eur": var,
            "cvar_95_eur": cvar,
            "maximum_drawdown_eur": maximum_drawdown(pnl),
            "worst_daily_pnl_eur": float(pnl.min()),
            "best_daily_pnl_eur": float(pnl.max()),
        })

    result = pd.DataFrame(records)

    # Incremental improvement over unhedged strategy.
    baseline = result.loc[
        result["strategy"] == "Unhedged",
        "total_net_pnl_eur",
    ]

    if baseline.empty:
        raise ValueError("Unhedged benchmark is missing.")

    baseline_pnl = float(baseline.iloc[0])

    result["improvement_vs_unhedged_eur"] = (
        result["total_net_pnl_eur"] - baseline_pnl
    )

    return result.sort_values(
        "strategy"
    ).reset_index(drop=True)


# ============================================================
# 7. PLOTS
# ============================================================

def plot_cumulative_pnl(
    daily: pd.DataFrame,
) -> None:
    """Plot cumulative strategy performance."""

    fig, ax = plt.subplots(figsize=(12, 6))

    for strategy, group in daily.groupby("strategy"):

        group = group.sort_values("quotation_date")

        ax.plot(
            group["quotation_date"],
            group["cumulative_pnl_eur"],
            label=strategy,
            linewidth=1.7,
        )

    ax.axhline(0, linestyle="--", linewidth=0.8)

    ax.set_title(
        "Cumulative Incremental P&L by Hedging Strategy"
    )
    ax.set_xlabel("Decision date")
    ax.set_ylabel("Cumulative incremental P&L (EUR)")
    ax.legend()
    ax.grid(alpha=0.25)

    fig.autofmt_xdate()
    fig.tight_layout()

    fig.savefig(
        cfg.FIGURES_DIR / "strategy_cumulative_pnl.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_risk_comparison(
    comparison: pd.DataFrame,
) -> None:
    """Compare historical VaR and CVaR."""

    indices = np.arange(len(comparison))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(
        indices - width / 2,
        comparison["var_95_eur"],
        width,
        label="VaR 95%",
    )

    ax.bar(
        indices + width / 2,
        comparison["cvar_95_eur"],
        width,
        label="CVaR 95%",
    )

    ax.set_xticks(indices)
    ax.set_xticklabels(comparison["strategy"])
    ax.set_ylabel("Historical loss risk (EUR)")
    ax.set_title("One-Day Risk by Hedging Strategy")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        cfg.FIGURES_DIR / "strategy_risk_comparison.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_hedge_ratios(
    daily: pd.DataFrame,
) -> None:
    """Plot the adaptive strategy's average hedge ratio."""

    adaptive = daily.loc[
        daily["strategy"] == "Adaptive Hedge"
    ].sort_values("quotation_date")

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.step(
        adaptive["quotation_date"],
        adaptive["average_hedge_ratio"],
        where="post",
    )

    ax.set_title("Adaptive Hedging: Average Daily Hedge Ratio")
    ax.set_xlabel("Decision date")
    ax.set_ylabel("Hedged proportion")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.25)

    fig.autofmt_xdate()
    fig.tight_layout()

    fig.savefig(
        cfg.FIGURES_DIR / "adaptive_hedge_ratios.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# 8. EXPORT
# ============================================================

def export_results(
    trades: pd.DataFrame,
    daily: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    """Save all result tables and figures."""

    cfg.TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    cfg.FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    trades.to_csv(
        cfg.TABLES_DIR / "strategy_trade_results.csv",
        index=False,
    )

    daily.to_csv(
        cfg.TABLES_DIR / "strategy_daily_results.csv",
        index=False,
    )

    comparison.to_csv(
        cfg.TABLES_DIR / "strategy_comparison.csv",
        index=False,
    )

    plot_cumulative_pnl(daily)
    plot_risk_comparison(comparison)
    plot_hedge_ratios(daily)


# ============================================================
# 9. MAIN PIPELINE
# ============================================================

def run_strategy_simulation(
    exposure_mw: float = cfg.DEFAULT_POSITION_MW,
    fixed_hedge_ratio: float = cfg.DEFAULT_HEDGE_RATIO,
    lower_threshold: float = cfg.SELL_PROBABILITY_THRESHOLD,
    upper_threshold: float = cfg.BUY_PROBABILITY_THRESHOLD,
    transaction_cost: float = cfg.TRANSACTION_COST_EUR_MWH,
    confidence: float = cfg.VAR_CONFIDENCE_LEVEL,
) -> dict:
    """Run and evaluate all three strategy alternatives."""

    print("\n" + "=" * 70)
    print("GERMAN POWER FUTURES DSS - STRATEGY SIMULATOR")
    print("=" * 70)

    forecasts = load_forecasts()

    print(f"Forecast observations: {len(forecasts):,}")
    print(
        "Test period: "
        f"{forecasts['quotation_date'].min():%Y-%m-%d}"
        " to "
        f"{forecasts['quotation_date'].max():%Y-%m-%d}"
    )

    trades = simulate_strategies(
        forecasts=forecasts,
        exposure_mw=exposure_mw,
        fixed_hedge_ratio=fixed_hedge_ratio,
        lower_threshold=lower_threshold,
        upper_threshold=upper_threshold,
        transaction_cost=transaction_cost,
    )

    daily = aggregate_daily_results(trades)

    comparison = compare_strategies(
        daily,
        confidence,
    )

    # Keep dynamically chosen confidence level reflected
    # in the metadata; column labels remain *_95_eur for
    # compatibility with the initial 95% dashboard design.
    comparison["confidence_level"] = confidence

    export_results(
        trades,
        daily,
        comparison,
    )

    columns = [
        "strategy",
        "days",
        "total_net_pnl_eur",
        "improvement_vs_unhedged_eur",
        "average_hedge_ratio",
        "var_95_eur",
        "cvar_95_eur",
        "maximum_drawdown_eur",
    ]

    print("\nSTRATEGY COMPARISON")
    print(
        comparison[columns].round(3).to_string(index=False)
    )

    print("\nFiles saved to:")
    print(cfg.TABLES_DIR)
    print(cfg.FIGURES_DIR)

    print(
        "\nNOTE: Synthetic-data, out-of-sample, "
        "one-day strategy simulation."
    )
    print(
        "Reported P&L represents incremental price exposure "
        "and simplified hedge settlements, not actual company profits."
    )

    return {
        "trades": trades,
        "daily": daily,
        "comparison": comparison,
    }


# ============================================================
# 10. COMMAND-LINE INTERFACE
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare electricity futures hedging strategies."
        )
    )

    parser.add_argument(
        "--exposure-mw",
        type=float,
        default=cfg.DEFAULT_POSITION_MW,
    )

    parser.add_argument(
        "--fixed-hedge-ratio",
        type=float,
        default=cfg.DEFAULT_HEDGE_RATIO,
    )

    parser.add_argument(
        "--lower-threshold",
        type=float,
        default=cfg.SELL_PROBABILITY_THRESHOLD,
    )

    parser.add_argument(
        "--upper-threshold",
        type=float,
        default=cfg.BUY_PROBABILITY_THRESHOLD,
    )

    parser.add_argument(
        "--transaction-cost",
        type=float,
        default=cfg.TRANSACTION_COST_EUR_MWH,
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=cfg.VAR_CONFIDENCE_LEVEL,
    )

    args = parser.parse_args()

    run_strategy_simulation(
        exposure_mw=args.exposure_mw,
        fixed_hedge_ratio=args.fixed_hedge_ratio,
        lower_threshold=args.lower_threshold,
        upper_threshold=args.upper_threshold,
        transaction_cost=args.transaction_cost,
        confidence=args.confidence,
    )


if __name__ == "__main__":
    main()