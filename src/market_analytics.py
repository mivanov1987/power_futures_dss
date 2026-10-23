"""
Market analytics for German Power Futures DSS.

Analyzes synthetic daily baseload futures quotations for D+1 to D+7.

Outputs:
    outputs/tables/price_statistics.csv
    outputs/tables/price_changes.csv
    outputs/tables/volatility_statistics.csv
    outputs/tables/extreme_events.csv
    outputs/tables/autocorrelation_statistics.csv

    outputs/figures/futures_price_dynamics.png
    outputs/figures/price_distributions.png
    outputs/figures/volatility_by_horizon.png
    outputs/figures/extreme_price_changes.png

All observations are synthetic.

Run from project root:
    python -m src.market_analytics
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config as cfg


# ============================================================
# 1. LOAD AND VALIDATE DATA
# ============================================================

def load_market_data(
    path: Path = cfg.SIMULATED_DATA_PATH,
) -> pd.DataFrame:
    """Load the synthetic futures quotation dataset."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Run: python -m src.data_generator"
        )

    df = pd.read_csv(
        path,
        parse_dates=[
            "quotation_date",
            "delivery_date",
        ],
    )

    required_columns = [
        "quotation_date",
        "delivery_date",
        "horizon_days",
        "contract_id",
        "futures_price_eur_mwh",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    if df[required_columns].isna().any().any():
        raise ValueError(
            "Missing values detected in required columns."
        )

    if df.duplicated(
        ["quotation_date", "contract_id"]
    ).any():
        raise ValueError(
            "Duplicate quotations for the same contract and date."
        )

    calculated_horizons = (
        df["delivery_date"] - df["quotation_date"]
    ).dt.days

    if not np.array_equal(
        calculated_horizons.to_numpy(),
        df["horizon_days"].to_numpy(),
    ):
        raise ValueError(
            "Delivery horizon does not match the dates."
        )

    df = df.sort_values(
        ["contract_id", "quotation_date"]
    ).reset_index(drop=True)

    return df


# ============================================================
# 2. DESCRIPTIVE STATISTICS
# ============================================================

def calculate_price_statistics(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Descriptive price statistics by delivery horizon."""

    group = df.groupby("horizon_days")[
        "futures_price_eur_mwh"
    ]

    statistics = group.agg(
        observations="count",
        mean_price="mean",
        median_price="median",
        std_price="std",
        min_price="min",
        max_price="max",
    )

    statistics["p05_price"] = group.quantile(0.05)
    statistics["p95_price"] = group.quantile(0.95)

    statistics["coefficient_of_variation"] = (
        statistics["std_price"]
        / statistics["mean_price"].abs().replace(0, np.nan)
    )

    return statistics.reset_index().round(4)


# ============================================================
# 3. MATCHED-CONTRACT PRICE CHANGES
# ============================================================

def calculate_price_changes(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate changes in the quotation of the SAME contract.

    Example:
        DE_BASE_20240108 quoted on January 5 (D+3)
        compared with its quotation on January 6 (D+2).

    These are not changes between different D+3 contracts.

    The first observed quotation of each contract has no
    previous price and is excluded from change statistics.
    """

    result = df.sort_values(
        ["contract_id", "quotation_date"]
    ).copy()

    grouped = result.groupby("contract_id")

    result["previous_quotation_date"] = grouped[
        "quotation_date"
    ].shift(1)

    result["previous_price_eur_mwh"] = grouped[
        "futures_price_eur_mwh"
    ].shift(1)

    result["previous_horizon_days"] = grouped[
        "horizon_days"
    ].shift(1)

    result["observation_gap_days"] = (
        result["quotation_date"]
        - result["previous_quotation_date"]
    ).dt.days

    # The simulated dataset contains daily observations.
    # Restrict analysis to consecutive quotations.
    valid = (
        result["previous_price_eur_mwh"].notna()
        & result["observation_gap_days"].eq(1)
    )

    result = result.loc[valid].copy()

    result["price_change_eur_mwh"] = (
        result["futures_price_eur_mwh"]
        - result["previous_price_eur_mwh"]
    )

    # Relative changes are supplementary; they can become
    # unstable around zero and with negative electricity prices.
    denominator = result[
        "previous_price_eur_mwh"
    ].abs()

    result["relative_change"] = np.where(
        denominator > 1.0,
        result["price_change_eur_mwh"] / denominator,
        np.nan,
    )

    result["absolute_price_change"] = (
        result["price_change_eur_mwh"].abs()
    )

    return result.sort_values(
        ["quotation_date", "horizon_days"]
    ).reset_index(drop=True)


# ============================================================
# 4. VOLATILITY ANALYSIS
# ============================================================

def calculate_volatility_statistics(
    changes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Estimate volatility of matched-contract daily changes.

    Results are grouped by the CURRENT delivery horizon.

    For instance, D+2 includes quotations whose previous
    observations were typically at D+3.

    Standard deviations are expressed in EUR/MWh,
    not annualized percentage volatility.
    """

    group = changes.groupby("horizon_days")[
        "price_change_eur_mwh"
    ]

    stats = group.agg(
        observations="count",
        mean_change="mean",
        volatility_eur_mwh="std",
        minimum_change="min",
        maximum_change="max",
    )

    stats["mean_absolute_change"] = changes.groupby(
        "horizon_days"
    )["absolute_price_change"].mean()

    stats["p95_absolute_change"] = changes.groupby(
        "horizon_days"
    )["absolute_price_change"].quantile(0.95)

    return stats.reset_index().round(4)


# ============================================================
# 5. EXTREME PRICE EVENTS
# ============================================================

def detect_extreme_events(
    changes: pd.DataFrame,
    quantile: float = 0.99,
) -> pd.DataFrame:
    """
    Identify large absolute daily price changes.

    Thresholds are estimated separately for each
    current delivery horizon.

    This is a descriptive tail-event definition,
    not a calibrated market stress model.
    """

    if not 0 < quantile < 1:
        raise ValueError(
            "Extreme-event quantile must be between 0 and 1."
        )

    result = changes.copy()

    thresholds = result.groupby(
        "horizon_days"
    )["absolute_price_change"].transform(
        lambda x: x.quantile(quantile)
    )

    result["extreme_threshold_eur_mwh"] = thresholds

    result["is_extreme"] = (
        result["absolute_price_change"]
        >= result["extreme_threshold_eur_mwh"]
    )

    extremes = result.loc[
        result["is_extreme"]
    ].copy()

    return extremes.sort_values(
        "absolute_price_change",
        ascending=False,
    ).reset_index(drop=True)


# ============================================================
# 6. AUTOCORRELATION
# ============================================================

def calculate_autocorrelation(
    changes: pd.DataFrame,
    max_lag: int = 5,
) -> pd.DataFrame:
    """
    Calculate autocorrelation of matched-contract revisions.

    To avoid correlating revisions from different contracts,
    autocorrelation is computed within each contract.

    Contract-level correlations are then averaged by
    current horizon. Because individual D+1..D+7 contracts
    have few revisions, higher lags may be unavailable.
    """

    records = []

    ordered = changes.sort_values(
        ["contract_id", "quotation_date"]
    )

    # Contract-specific price changes are indexed by
    # their quotation dates.
    for lag in range(1, max_lag + 1):

        pairs = []

        for _, contract in ordered.groupby("contract_id"):

            values = contract[
                "price_change_eur_mwh"
            ].to_numpy(dtype=float)

            if len(values) <= lag + 1:
                continue

            x = values[:-lag]
            y = values[lag:]

            pairs.extend(zip(x, y))

        if len(pairs) >= 3:
            pairs_array = np.asarray(pairs)
            x = pairs_array[:, 0]
            y = pairs_array[:, 1]

            if np.std(x) > 0 and np.std(y) > 0:
                correlation = float(
                    np.corrcoef(x, y)[0, 1]
                )
            else:
                correlation = np.nan
        else:
            correlation = np.nan

        records.append({
            "lag": lag,
            "autocorrelation": correlation,
            "paired_observations": len(pairs),
        })

    return pd.DataFrame(records).round(4)


# ============================================================
# 7. VISUALIZATIONS
# ============================================================

def plot_price_dynamics(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Plot rolling-horizon quotation levels.

    Each line represents a constant time-to-delivery
    series, NOT one continuously held futures contract.
    """

    pivot = df.pivot(
        index="quotation_date",
        columns="horizon_days",
        values="futures_price_eur_mwh",
    )

    fig, ax = plt.subplots(figsize=(12, 6))

    for horizon in pivot.columns:
        ax.plot(
            pivot.index,
            pivot[horizon],
            linewidth=0.9,
            alpha=0.8,
            label=f"D+{horizon}",
        )

    ax.set_title(
        "Synthetic German Power Futures: Rolling Delivery Horizons"
    )
    ax.set_xlabel("Quotation date")
    ax.set_ylabel("Price (EUR/MWh)")
    ax.legend(ncol=4)
    ax.grid(alpha=0.25)

    fig.autofmt_xdate()
    fig.tight_layout()

    fig.savefig(
        output_dir / "futures_price_dynamics.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_price_distributions(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Plot quotation-price distributions by horizon."""

    horizons = sorted(
        df["horizon_days"].unique()
    )

    data = [
        df.loc[
            df["horizon_days"] == horizon,
            "futures_price_eur_mwh",
        ].to_numpy()
        for horizon in horizons
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.boxplot(
        data,
        tick_labels=[
            f"D+{h}" for h in horizons
        ],
        showfliers=True,
    )

    ax.set_title(
        "Synthetic Futures Price Distribution by Delivery Horizon"
    )
    ax.set_xlabel("Delivery horizon")
    ax.set_ylabel("Price (EUR/MWh)")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        output_dir / "price_distributions.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_volatility(
    volatility: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Plot volatility of matched-contract daily revisions."""

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(
        volatility["horizon_days"].astype(str),
        volatility["volatility_eur_mwh"],
    )

    ax.set_title(
        "Standard Deviation of Matched-Contract Price Changes"
    )
    ax.set_xlabel("Current delivery horizon (D+)")
    ax.set_ylabel("Standard deviation (EUR/MWh)")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        output_dir / "volatility_by_horizon.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_extreme_changes(
    changes: pd.DataFrame,
    extremes: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Show matched-contract revisions and extreme events."""

    daily = changes.groupby(
        "quotation_date"
    )["absolute_price_change"].max()

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        daily.index,
        daily.to_numpy(),
        linewidth=0.9,
        label="Largest absolute change per date",
    )

    if not extremes.empty:
        extreme_daily = extremes.groupby(
            "quotation_date"
        )["absolute_price_change"].max()

        ax.scatter(
            extreme_daily.index,
            extreme_daily.to_numpy(),
            s=22,
            label="Extreme observations",
            zorder=3,
        )

    ax.set_title(
        "Extreme Matched-Contract Futures Price Changes"
    )
    ax.set_xlabel("Quotation date")
    ax.set_ylabel("Absolute change (EUR/MWh)")
    ax.legend()
    ax.grid(alpha=0.25)

    fig.autofmt_xdate()
    fig.tight_layout()

    fig.savefig(
        output_dir / "extreme_price_changes.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# 8. RUN COMPLETE ANALYSIS
# ============================================================

def run_analysis(
    input_path: Path = cfg.SIMULATED_DATA_PATH,
    extreme_quantile: float = 0.99,
) -> dict:
    """Run the complete analytical pipeline."""

    cfg.FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    cfg.TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nLoading market data...")

    df = load_market_data(input_path)

    print("Calculating descriptive statistics...")
    price_stats = calculate_price_statistics(df)

    print("Calculating matched-contract changes...")
    changes = calculate_price_changes(df)

    if changes.empty:
        raise ValueError(
            "No consecutive quotations of the same "
            "delivery contract were found."
        )

    print("Estimating price volatility...")
    volatility = calculate_volatility_statistics(
        changes
    )

    print("Detecting extreme market changes...")
    extremes = detect_extreme_events(
        changes,
        quantile=extreme_quantile,
    )

    print("Calculating autocorrelation...")
    autocorrelation = calculate_autocorrelation(
        changes
    )

    # Export tables.
    tables = {
        "price_statistics.csv": price_stats,
        "price_changes.csv": changes,
        "volatility_statistics.csv": volatility,
        "extreme_events.csv": extremes,
        "autocorrelation_statistics.csv": autocorrelation,
    }

    for filename, table in tables.items():
        table.to_csv(
            cfg.TABLES_DIR / filename,
            index=False,
            encoding="utf-8",
        )

    print("Generating figures...")

    plot_price_dynamics(df, cfg.FIGURES_DIR)
    plot_price_distributions(df, cfg.FIGURES_DIR)
    plot_volatility(volatility, cfg.FIGURES_DIR)
    plot_extreme_changes(
        changes,
        extremes,
        cfg.FIGURES_DIR,
    )

    print("\n" + "=" * 65)
    print("GERMAN POWER FUTURES DSS - MARKET ANALYTICS")
    print("=" * 65)

    print(f"Total quotations: {len(df):,}")
    print(f"Matched changes:  {len(changes):,}")
    print(f"Extreme changes:  {len(extremes):,}")

    print("\nPRICE STATISTICS:")
    print(price_stats.to_string(index=False))

    print("\nVOLATILITY STATISTICS:")
    print(volatility.to_string(index=False))

    print(f"\nTables:  {cfg.TABLES_DIR}")
    print(f"Figures: {cfg.FIGURES_DIR}")

    print("\nAll analyses use SYNTHETIC data.")
    print("=" * 65)

    return {
        "market_data": df,
        "price_statistics": price_stats,
        "price_changes": changes,
        "volatility": volatility,
        "extreme_events": extremes,
        "autocorrelation": autocorrelation,
    }


# ============================================================
# 9. COMMAND-LINE INTERFACE
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze synthetic German power futures prices."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=cfg.SIMULATED_DATA_PATH,
        help="Input CSV with simulated futures quotations.",
    )

    parser.add_argument(
        "--extreme-quantile",
        type=float,
        default=0.99,
        help="Quantile for extreme absolute price changes.",
    )

    args = parser.parse_args()

    run_analysis(
        input_path=args.input,
        extreme_quantile=args.extreme_quantile,
    )


if __name__ == "__main__":
    main()