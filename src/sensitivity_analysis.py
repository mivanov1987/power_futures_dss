
"""
Sensitivity analysis for German Power Futures DSS.

Examines the robustness of adaptive and fixed hedging
strategies to changes in:

- Forecast probability thresholds
- Transaction costs
- Reference exposure size
- Fixed hedge ratios

Uses out-of-sample ARX predictions previously generated
by src.forecasting.

IMPORTANT:
The same test sample is reused for all parameter scenarios.
This is exploratory sensitivity analysis, not independent
out-of-sample optimization or evidence of live profitability.

Run from project root:
    python -m src.sensitivity_analysis
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as cfg

from src import strategy_simulator as sim


# ============================================================
# 1. SENSITIVITY CONFIGURATION
# ============================================================

LOWER_THRESHOLDS = [0.25, 0.30, 0.35, 0.40, 0.45]
UPPER_THRESHOLDS = [0.55, 0.60, 0.65, 0.70, 0.75]

TRANSACTION_COSTS = [0.10, 0.25, 0.50, 1.00]
EXPOSURE_SIZES_MW = [2.0, 5.0, 10.0]
FIXED_HEDGE_RATIOS = [0.25, 0.50, 0.75, 1.00]

BASE_LOWER = float(cfg.SELL_PROBABILITY_THRESHOLD)
BASE_UPPER = float(cfg.BUY_PROBABILITY_THRESHOLD)
BASE_FIXED_RATIO = float(cfg.DEFAULT_HEDGE_RATIO)
BASE_EXPOSURE_MW = float(cfg.DEFAULT_POSITION_MW)
BASE_TRANSACTION_COST = float(
    cfg.TRANSACTION_COST_EUR_MWH
)

CONFIDENCE = float(cfg.VAR_CONFIDENCE_LEVEL)

TABLE_DIR = cfg.TABLES_DIR
FIGURE_DIR = cfg.FIGURES_DIR


# ============================================================
# 2. INPUT VALIDATION
# ============================================================

def validate_parameters() -> None:
    """Validate the parameter grids."""

    for lower in LOWER_THRESHOLDS:
        for upper in UPPER_THRESHOLDS:
            if not 0 <= lower < upper <= 1:
                raise ValueError(
                    f"Invalid threshold pair: {lower}, {upper}"
                )

    if any(x <= 0 for x in EXPOSURE_SIZES_MW):
        raise ValueError("Exposure sizes must be positive.")

    if any(x < 0 for x in TRANSACTION_COSTS):
        raise ValueError(
            "Transaction costs must be non-negative."
        )

    if any(
        not 0 <= x <= 1 for x in FIXED_HEDGE_RATIOS
    ):
        raise ValueError(
            "Fixed hedge ratios must be between 0 and 1."
        )

    if not BASE_LOWER < BASE_UPPER:
        raise ValueError("Invalid baseline thresholds.")


# ============================================================
# 3. RUN A SINGLE PARAMETER SCENARIO
# ============================================================

def evaluate_scenario(
    forecasts: pd.DataFrame,
    exposure_mw: float,
    transaction_cost: float,
    lower_threshold: float,
    upper_threshold: float,
    fixed_ratio: float,
) -> pd.DataFrame:
    """
    Evaluate the three strategies on the same test sample.

    Reuses the existing, tested strategy simulator.
    Does not alter the original strategy output files.
    """

    trades = sim.simulate_strategies(
        forecasts=forecasts,
        exposure_mw=exposure_mw,
        fixed_hedge_ratio=fixed_ratio,
        lower_threshold=lower_threshold,
        upper_threshold=upper_threshold,
        transaction_cost=transaction_cost,
    )

    daily = sim.aggregate_daily_results(trades)

    comparison = sim.compare_strategies(
        daily,
        confidence=CONFIDENCE,
    )

    comparison = comparison.copy()

    comparison["exposure_mw"] = exposure_mw
    comparison["transaction_cost_eur_mwh"] = transaction_cost
    comparison["lower_threshold"] = lower_threshold
    comparison["upper_threshold"] = upper_threshold
    comparison["fixed_hedge_ratio"] = fixed_ratio

    return comparison


# ============================================================
# 4. FULL ADAPTIVE PARAMETER GRID
# ============================================================

def run_adaptive_grid(
    forecasts: pd.DataFrame,
) -> pd.DataFrame:
    """
    Evaluate all combinations of:

    5 lower thresholds x 5 upper thresholds
    x 4 transaction costs x 3 exposure sizes.

    Total: 300 parameter combinations.

    Each combination evaluates all three strategies.
    """

    records = []

    total = (
        len(LOWER_THRESHOLDS)
        * len(UPPER_THRESHOLDS)
        * len(TRANSACTION_COSTS)
        * len(EXPOSURE_SIZES_MW)
    )

    print(f"Evaluating {total} adaptive parameter scenarios...")

    counter = 0

    for exposure in EXPOSURE_SIZES_MW:
        for cost in TRANSACTION_COSTS:
            for lower in LOWER_THRESHOLDS:
                for upper in UPPER_THRESHOLDS:

                    comparison = evaluate_scenario(
                        forecasts=forecasts,
                        exposure_mw=exposure,
                        transaction_cost=cost,
                        lower_threshold=lower,
                        upper_threshold=upper,
                        fixed_ratio=BASE_FIXED_RATIO,
                    )

                    records.append(comparison)

                    counter += 1

            print(
                f"Completed {counter}/{total}: "
                f"exposure={exposure:g} MW, "
                f"cost={cost:.2f} EUR/MWh"
            )

    return pd.concat(
        records,
        ignore_index=True,
    )


# ============================================================
# 5. FIXED HEDGE RATIO GRID
# ============================================================

def run_fixed_ratio_grid(
    forecasts: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare fixed hedge ratios under different transaction
    costs and exposure sizes.

    The default adaptive thresholds remain unchanged.

    Total: 4 ratios x 4 costs x 3 exposure sizes
           = 48 parameter combinations.
    """

    records = []

    print("Evaluating fixed hedge ratio scenarios...")

    for exposure in EXPOSURE_SIZES_MW:
        for cost in TRANSACTION_COSTS:
            for ratio in FIXED_HEDGE_RATIOS:

                comparison = evaluate_scenario(
                    forecasts=forecasts,
                    exposure_mw=exposure,
                    transaction_cost=cost,
                    lower_threshold=BASE_LOWER,
                    upper_threshold=BASE_UPPER,
                    fixed_ratio=ratio,
                )

                # Store only the fixed strategy for this grid.
                fixed = comparison.loc[
                    comparison["strategy"] == "Fixed Hedge"
                ].copy()

                records.append(fixed)

    return pd.concat(
        records,
        ignore_index=True,
    )


# ============================================================
# 6. ADAPTIVE VS FIXED VS UNHEDGED
# ============================================================

def create_comparison_table(
    adaptive_grid: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reshape results so each scenario includes side-by-side
    adaptive, fixed and unhedged performance.

    Differences are based on identical simulated test dates.
    """

    index_columns = [
        "exposure_mw",
        "transaction_cost_eur_mwh",
        "lower_threshold",
        "upper_threshold",
        "fixed_hedge_ratio",
    ]

    value_columns = [
        "total_net_pnl_eur",
        "var_95_eur",
        "cvar_95_eur",
        "maximum_drawdown_eur",
        "average_hedge_ratio",
    ]

    wide = adaptive_grid.pivot(
        index=index_columns,
        columns="strategy",
        values=value_columns,
    )

    # Flatten the two-level column index.
    wide.columns = [
        f"{measure}_{strategy.lower().replace(' ', '_')}"
        for measure, strategy in wide.columns
    ]

    wide = wide.reset_index()

    wide["adaptive_minus_fixed_eur"] = (
        wide["total_net_pnl_eur_adaptive_hedge"]
        - wide["total_net_pnl_eur_fixed_hedge"]
    )

    wide["adaptive_minus_unhedged_eur"] = (
        wide["total_net_pnl_eur_adaptive_hedge"]
        - wide["total_net_pnl_eur_unhedged"]
    )

    wide["adaptive_var_below_fixed"] = (
        wide["var_95_eur_adaptive_hedge"]
        < wide["var_95_eur_fixed_hedge"]
    )

    wide["adaptive_cvar_below_fixed"] = (
        wide["cvar_95_eur_adaptive_hedge"]
        < wide["cvar_95_eur_fixed_hedge"]
    )

    wide["adaptive_better_pnl_and_cvar"] = (
        (wide["adaptive_minus_fixed_eur"] > 0)
        & wide["adaptive_cvar_below_fixed"]
    )

    return wide


# ============================================================
# 7. ROBUSTNESS SUMMARY
# ============================================================

def summarize_robustness(
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize the fraction of grid combinations in which
    adaptive hedging outperforms the fixed benchmark.

    These are robustness frequencies across the chosen
    parameter grid, NOT probabilities of future success.
    """

    records = []

    for (exposure, cost), group in comparison.groupby(
        ["exposure_mw", "transaction_cost_eur_mwh"]
    ):

        records.append({
            "exposure_mw": exposure,
            "transaction_cost_eur_mwh": cost,
            "threshold_combinations": len(group),
            "share_adaptive_pnl_above_fixed": float(
                (group["adaptive_minus_fixed_eur"] > 0).mean()
            ),
            "share_adaptive_pnl_above_unhedged": float(
                (
                    group["adaptive_minus_unhedged_eur"] > 0
                ).mean()
            ),
            "share_adaptive_cvar_below_fixed": float(
                group["adaptive_cvar_below_fixed"].mean()
            ),
            "share_adaptive_better_pnl_and_cvar": float(
                group["adaptive_better_pnl_and_cvar"].mean()
            ),
            "median_pnl_difference_vs_fixed_eur": float(
                group["adaptive_minus_fixed_eur"].median()
            ),
            "minimum_pnl_difference_vs_fixed_eur": float(
                group["adaptive_minus_fixed_eur"].min()
            ),
            "maximum_pnl_difference_vs_fixed_eur": float(
                group["adaptive_minus_fixed_eur"].max()
            ),
        })

    return pd.DataFrame(records).sort_values(
        ["exposure_mw", "transaction_cost_eur_mwh"]
    ).reset_index(drop=True)


# ============================================================
# 8. HEATMAP: PROBABILITY THRESHOLDS
# ============================================================

def plot_threshold_heatmap(
    comparison: pd.DataFrame,
    exposure_mw: float = BASE_EXPOSURE_MW,
    transaction_cost: float = BASE_TRANSACTION_COST,
) -> None:
    """
    Heatmap of adaptive improvement over fixed hedging.

    The figure shows the EUR difference on the synthetic
    test sample, not an externally validated improvement.
    """

    subset = comparison.loc[
        np.isclose(comparison["exposure_mw"], exposure_mw)
        & np.isclose(
            comparison["transaction_cost_eur_mwh"],
            transaction_cost,
        )
    ].copy()

    if subset.empty:
        raise ValueError(
            "No scenarios match the heatmap configuration."
        )

    matrix = subset.pivot(
        index="lower_threshold",
        columns="upper_threshold",
        values="adaptive_minus_fixed_eur",
    )

    matrix = matrix.sort_index().sort_index(axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))

    values = matrix.to_numpy(dtype=float)

    image = ax.imshow(
        values,
        cmap="RdYlGn",
        aspect="auto",
    )

    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_yticks(np.arange(len(matrix.index)))

    ax.set_xticklabels([
        f"{x:.0%}" for x in matrix.columns
    ])

    ax.set_yticklabels([
        f"{x:.0%}" for x in matrix.index
    ])

    ax.set_xlabel("Upper probability threshold")
    ax.set_ylabel("Lower probability threshold")

    ax.set_title(
        "Adaptive vs Fixed Hedge: Net P&L Difference\n"
        f"Exposure={exposure_mw:g} MW; "
        f"Cost={transaction_cost:.2f} EUR/MWh"
    )

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):

            ax.text(
                j,
                i,
                f"{values[i, j]:,.0f}",
                ha="center",
                va="center",
                fontsize=8,
            )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("P&L improvement (EUR)")

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / "sensitivity_threshold_heatmap.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# 9. TRANSACTION COST SENSITIVITY
# ============================================================

def plot_transaction_cost_sensitivity(
    adaptive_grid: pd.DataFrame,
    exposure_mw: float = BASE_EXPOSURE_MW,
) -> None:
    """
    Compare strategies as transaction costs increase,
    holding adaptive thresholds and exposure constant.
    """

    subset = adaptive_grid.loc[
        np.isclose(
            adaptive_grid["exposure_mw"], exposure_mw
        )
        & np.isclose(
            adaptive_grid["lower_threshold"], BASE_LOWER
        )
        & np.isclose(
            adaptive_grid["upper_threshold"], BASE_UPPER
        )
    ].copy()

    fig, ax = plt.subplots(figsize=(10, 6))

    for name, group in subset.groupby("strategy"):

        group = group.sort_values(
            "transaction_cost_eur_mwh"
        )

        ax.plot(
            group["transaction_cost_eur_mwh"],
            group["total_net_pnl_eur"],
            marker="o",
            linewidth=1.8,
            label=name,
        )

    ax.set_title(
        "Transaction Cost Sensitivity of Hedging Strategies"
    )

    ax.set_xlabel("Transaction cost (EUR/MWh)")
    ax.set_ylabel("Total incremental net P&L (EUR)")
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / "sensitivity_transaction_costs.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# 10. FIXED HEDGE RATIO SENSITIVITY
# ============================================================

def plot_fixed_ratio_sensitivity(
    fixed_grid: pd.DataFrame,
    exposure_mw: float = BASE_EXPOSURE_MW,
    transaction_cost: float = BASE_TRANSACTION_COST,
) -> None:
    """Plot risk and performance for fixed hedge ratios."""

    subset = fixed_grid.loc[
        np.isclose(fixed_grid["exposure_mw"], exposure_mw)
        & np.isclose(
            fixed_grid["transaction_cost_eur_mwh"],
            transaction_cost,
        )
    ].sort_values("fixed_hedge_ratio")

    if subset.empty:
        raise ValueError(
            "No fixed-ratio scenarios match the figure settings."
        )

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        subset["fixed_hedge_ratio"],
        subset["total_net_pnl_eur"],
        marker="o",
        label="Net P&L",
    )

    ax.set_xlabel("Fixed hedge ratio")
    ax.set_ylabel("Total net P&L (EUR)")
    ax.set_title(
        "Sensitivity to the Fixed Hedging Ratio"
    )
    ax.grid(alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / "sensitivity_fixed_hedge_ratio.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# 11. EXPOSURE SIZE SENSITIVITY
# ============================================================

def plot_exposure_sensitivity(
    adaptive_grid: pd.DataFrame,
    transaction_cost: float = BASE_TRANSACTION_COST,
) -> None:
    """Plot the effect of reference exposure size."""

    subset = adaptive_grid.loc[
        np.isclose(
            adaptive_grid["transaction_cost_eur_mwh"],
            transaction_cost,
        )
        & np.isclose(
            adaptive_grid["lower_threshold"], BASE_LOWER
        )
        & np.isclose(
            adaptive_grid["upper_threshold"], BASE_UPPER
        )
    ].copy()

    fig, ax = plt.subplots(figsize=(10, 6))

    for name, group in subset.groupby("strategy"):

        group = group.sort_values("exposure_mw")

        ax.plot(
            group["exposure_mw"],
            group["cvar_95_eur"],
            marker="o",
            label=name,
        )

    ax.set_xlabel("Reference exposure (MW)")
    ax.set_ylabel("CVaR 95% (EUR)")
    ax.set_title(
        "Sensitivity of Tail Risk to Exposure Size"
    )

    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / "sensitivity_exposure_cvar.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# 12. EXPORT RESULTS
# ============================================================

def export_results(
    adaptive_grid: pd.DataFrame,
    fixed_grid: pd.DataFrame,
    comparison: pd.DataFrame,
    robustness: pd.DataFrame,
) -> None:
    """Export tables without modifying existing model outputs."""

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    outputs = {
        "sensitivity_adaptive_grid.csv": adaptive_grid,
        "sensitivity_fixed_grid.csv": fixed_grid,
        "sensitivity_comparison.csv": comparison,
        "sensitivity_robustness.csv": robustness,
    }

    for filename, df in outputs.items():
        df.to_csv(
            TABLE_DIR / filename,
            index=False,
            encoding="utf-8",
        )

    plot_threshold_heatmap(comparison)
    plot_transaction_cost_sensitivity(adaptive_grid)
    plot_fixed_ratio_sensitivity(fixed_grid)
    plot_exposure_sensitivity(adaptive_grid)


# ============================================================
# 13. COMPLETE ANALYSIS
# ============================================================

def run_sensitivity_analysis() -> dict:
    """Run all scenario grids and generate outputs."""

    print("\n" + "=" * 70)
    print("GERMAN POWER FUTURES DSS - SENSITIVITY ANALYSIS")
    print("=" * 70)

    validate_parameters()

    forecasts = sim.load_forecasts()

    print(f"Forecast observations: {len(forecasts):,}")
    print(
        "Test period: "
        f"{forecasts['quotation_date'].min():%Y-%m-%d}"
        " to "
        f"{forecasts['quotation_date'].max():%Y-%m-%d}"
    )

    adaptive_grid = run_adaptive_grid(forecasts)
    fixed_grid = run_fixed_ratio_grid(forecasts)

    comparison = create_comparison_table(
        adaptive_grid
    )

    robustness = summarize_robustness(
        comparison
    )

    export_results(
        adaptive_grid,
        fixed_grid,
        comparison,
        robustness,
    )

    baseline = comparison.loc[
        np.isclose(
            comparison["exposure_mw"],
            BASE_EXPOSURE_MW,
        )
        & np.isclose(
            comparison["transaction_cost_eur_mwh"],
            BASE_TRANSACTION_COST,
        )
        & np.isclose(
            comparison["lower_threshold"],
            BASE_LOWER,
        )
        & np.isclose(
            comparison["upper_threshold"],
            BASE_UPPER,
        )
    ]

    if not baseline.empty:
        b = baseline.iloc[0]

        print("\nBASELINE STRATEGY COMPARISON")

        print(
            f"Adaptive - Fixed: "
            f"{b['adaptive_minus_fixed_eur']:,.2f} EUR"
        )

        print(
            f"Adaptive - Unhedged: "
            f"{b['adaptive_minus_unhedged_eur']:,.2f} EUR"
        )

        print(
            "Adaptive CVaR below Fixed: "
            f"{bool(b['adaptive_cvar_below_fixed'])}"
        )

    print("\nROBUSTNESS BY EXPOSURE AND COST")

    columns = [
        "exposure_mw",
        "transaction_cost_eur_mwh",
        "threshold_combinations",
        "share_adaptive_pnl_above_fixed",
        "share_adaptive_better_pnl_and_cvar",
        "median_pnl_difference_vs_fixed_eur",
    ]

    print(
        robustness[columns].round(3).to_string(
            index=False
        )
    )

    print("\nOutput directories:")
    print(TABLE_DIR)
    print(FIGURE_DIR)

    print(
        "\nNOTE: All results use synthetic data and "
        "the same out-of-sample forecasting test period."
    )

    print(
        "Parameter rankings on this test sample are "
        "exploratory and are not independently validated."
    )

    return {
        "adaptive_grid": adaptive_grid,
        "fixed_grid": fixed_grid,
        "comparison": comparison,
        "robustness": robustness,
    }


# ============================================================
# 14. COMMAND-LINE ENTRY POINT
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze sensitivity of German power futures "
            "hedging strategies."
        )
    )

    parser.parse_args()

    run_sensitivity_analysis()


if __name__ == "__main__":
    main()
