"""
German Power Futures Decision Support System
=============================================

Interactive Streamlit dashboard for a synthetic-data
research prototype of German electricity futures trading.

Modules:
    1. Market Overview
    2. Price Forecasting
    3. Risk Analytics
    4. Hedging Strategies
    5. Methodology & Limitations

Launch from project root:
    streamlit run dashboard/app.py

All prices, positions and backtest results are synthetic.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ------------------------------------------------------------
# PROJECT IMPORTS
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg

from src import risk_analytics as risk
from src import strategy_simulator as strategies


# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------

st.set_page_config(
    page_title="German Power Futures DSS",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2rem;
    }
    div[data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.07);
        padding: 14px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# DATA LOADING
# ------------------------------------------------------------

@st.cache_data(show_spinner=False)
def read_csv_cached(
    path_string: str,
    modified_time: int,
    date_columns: tuple = (),
) -> pd.DataFrame:
    """
    Read CSV data with cache invalidation on modification.

    modified_time is included in the cache key so that
    regenerated CSV files are reloaded automatically.
    """
    return pd.read_csv(
        path_string,
        parse_dates=list(date_columns),
    )


def load_table(
    filename: str,
    date_columns: tuple = (),
) -> pd.DataFrame | None:
    """Load a table, or return None if the file is missing."""

    path = cfg.TABLES_DIR / filename

    if not path.exists():
        return None

    return read_csv_cached(
        str(path),
        path.stat().st_mtime_ns,
        date_columns,
    ).copy()


def load_market() -> pd.DataFrame | None:
    """Load synthetic futures data."""

    path = cfg.SIMULATED_DATA_PATH

    if not path.exists():
        return None

    return read_csv_cached(
        str(path),
        path.stat().st_mtime_ns,
        ("quotation_date", "delivery_date"),
    ).copy()


def missing_data(message: str, command: str) -> None:
    """Display instructions for generating missing results."""

    st.warning(message)
    st.code(command, language="powershell")


def money(value: float) -> str:
    return f"{value:,.2f} EUR"


def heading(text: str, description: str = "") -> None:
    st.subheader(text)

    if description:
        st.caption(description)


# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------

st.sidebar.title("⚡ Power Futures DSS")

st.sidebar.caption(
    "German electricity futures | D+1 to D+7"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Market Overview",
        "Price Forecasting",
        "Risk Analytics",
        "Hedging Strategies",
        "Sensitivity Analysis",
        "Methodology & Limitations",
    ],
)

st.sidebar.divider()

st.sidebar.info(
    "Research prototype using synthetic market data. "
    "Not intended for live trading."
)

st.sidebar.caption(
    f"Model: {cfg.MODEL_NAME} | "
    f"Market: {cfg.MARKET_REGION} | "
    f"Currency: {cfg.CURRENCY}"
)


# ------------------------------------------------------------
# MAIN HEADER
# ------------------------------------------------------------

st.title("German Power Futures DSS")

st.caption(
    "Decision Support System for Short-Term Electricity "
    "Futures Trading and Price Risk Management"
)

st.warning(
    "SYNTHETIC DATA — The dashboard does not display actual "
    "EEX quotations, actual Electrohold Trade EAD positions "
    "or validated live-trading forecasts."
)


# ============================================================
# PAGE 1: MARKET OVERVIEW
# ============================================================

if page == "Market Overview":

    heading(
        "Market Overview",
        "Historical analysis of simulated German daily "
        "baseload electricity futures quotations.",
    )

    market = load_market()

    if market is None:
        missing_data(
            "Synthetic price data were not found.",
            "python -m src.data_generator",
        )
        st.stop()

    horizons = sorted(
        market["horizon_days"].unique().tolist()
    )

    selected_horizons = st.multiselect(
        "Delivery horizons",
        options=horizons,
        default=horizons,
        format_func=lambda x: f"D+{x}",
    )

    if not selected_horizons:
        st.info("Select at least one delivery horizon.")
        st.stop()

    filtered = market.loc[
        market["horizon_days"].isin(selected_horizons)
    ].copy()

    prices = filtered["futures_price_eur_mwh"]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Observations", f"{len(filtered):,}")
    c2.metric("Average price", money(prices.mean()))
    c3.metric("Minimum price", money(prices.min()))
    c4.metric("Maximum price", money(prices.max()))

    st.markdown("#### Futures price dynamics")

    fig = px.line(
        filtered,
        x="quotation_date",
        y="futures_price_eur_mwh",
        color="horizon_days",
        labels={
            "quotation_date": "Quotation date",
            "futures_price_eur_mwh": "EUR/MWh",
            "horizon_days": "D+ horizon",
        },
    )

    fig.update_layout(
        height=460,
        hovermode="x unified",
        legend_title="Delivery horizon",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.caption(
        "Each line represents a rolling time-to-delivery "
        "series, not a continuously held individual contract."
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("#### Price distribution")

        box = px.box(
            filtered,
            x="horizon_days",
            y="futures_price_eur_mwh",
            labels={
                "horizon_days": "Delivery horizon",
                "futures_price_eur_mwh": "EUR/MWh",
            },
        )

        st.plotly_chart(
            box,
            use_container_width=True,
        )

    with col2:

        st.markdown("#### Volatility by horizon")

        volatility = load_table(
            "volatility_statistics.csv"
        )

        if volatility is not None:

            volatility = volatility.loc[
                volatility["horizon_days"].isin(
                    selected_horizons
                )
            ]

            chart = px.bar(
                volatility,
                x="horizon_days",
                y="volatility_eur_mwh",
                labels={
                    "horizon_days": "Current D+ horizon",
                    "volatility_eur_mwh": "Std. dev. (EUR/MWh)",
                },
            )

            st.plotly_chart(
                chart,
                use_container_width=True,
            )

        else:
            missing_data(
                "Volatility statistics are missing.",
                "python -m src.market_analytics",
            )

    st.markdown("#### Price statistics")

    price_stats = load_table(
        "price_statistics.csv"
    )

    if price_stats is not None:

        price_stats = price_stats.loc[
            price_stats["horizon_days"].isin(
                selected_horizons
            )
        ]

        st.dataframe(
            price_stats.round(3),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("#### Extreme price changes")

    extremes = load_table(
        "extreme_events.csv",
        ("quotation_date", "delivery_date"),
    )

    if extremes is not None:

        extremes = extremes.loc[
            extremes["horizon_days"].isin(
                selected_horizons
            )
        ]

        columns = [
            "quotation_date",
            "delivery_date",
            "horizon_days",
            "price_change_eur_mwh",
            "absolute_price_change",
        ]

        st.dataframe(
            extremes[columns].head(30),
            use_container_width=True,
            hide_index=True,
        )

    st.download_button(
        "Download filtered market data",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_market_data.csv",
        mime="text/csv",
    )


# ============================================================
# PAGE 2: PRICE FORECASTING
# ============================================================

elif page == "Price Forecasting":

    heading(
        "ARX Price Forecasting",
        "Probabilistic prediction of the next quotation "
        "change for the same futures delivery contract.",
    )

    latest = load_table(
        "latest_forecasts.csv",
        (
            "quotation_date",
            "delivery_date",
            "forecast_for_quotation_date",
        ),
    )

    metrics = load_table("forecast_metrics.csv")
    results = load_table(
        "forecast_results.csv",
        (
            "quotation_date",
            "target_quotation_date",
            "delivery_date",
        ),
    )

    if latest is None or metrics is None:
        missing_data(
            "Forecasting results are missing.",
            "python -m src.forecasting",
        )
        st.stop()

    latest_date = latest["quotation_date"].max()

    st.caption(
        f"Latest synthetic quotation date: "
        f"{latest_date:%Y-%m-%d}"
    )

    all_metrics = metrics.loc[
        metrics["horizon_days"].astype(str) == "All"
    ]

    if not all_metrics.empty:

        m = all_metrics.iloc[0]

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Directional accuracy",
            f"{m['directional_accuracy']:.2%}",
        )

        c2.metric(
            "MAE",
            f"{m['mae_eur_mwh']:.2f} EUR/MWh",
        )

        c3.metric(
            "RMSE",
            f"{m['rmse_eur_mwh']:.2f} EUR/MWh",
        )

        c4.metric(
            "Brier Score",
            f"{m['brier_score']:.4f}",
        )

    st.markdown("#### Latest directional forecasts")

    display_columns = [
        "horizon_days",
        "delivery_date",
        "futures_price_eur_mwh",
        "predicted_change_eur_mwh",
        "probability_up",
        "probability_down",
        "trading_signal",
        "forecast_status",
    ]

    latest_display = latest[
        display_columns
    ].copy()

    st.dataframe(
        latest_display,
        hide_index=True,
        use_container_width=True,
        column_config={
            "horizon_days": st.column_config.NumberColumn(
                "D+ Horizon",
                format="%d",
            ),
            "probability_up": st.column_config.ProgressColumn(
                "P(Price Increase)",
                min_value=0,
                max_value=1,
                format="%.1f",
            ),
            "probability_down": st.column_config.NumberColumn(
                "P(Price Decrease)",
                format="%.3f",
            ),
        },
    )

    plot_data = latest.dropna(
        subset=["probability_up"]
    ).copy()

    plot_data["product"] = (
        "D+" + plot_data["horizon_days"].astype(str)
    )

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=plot_data["product"],
            y=plot_data["probability_up"],
            name="Probability of increase",
            text=(
                plot_data["probability_up"] * 100
            ).round(1).astype(str) + "%",
            textposition="outside",
        )
    )

    fig.add_hline(
        y=cfg.BUY_PROBABILITY_THRESHOLD,
        line_dash="dash",
        annotation_text="Upper decision threshold",
    )

    fig.add_hline(
        y=cfg.SELL_PROBABILITY_THRESHOLD,
        line_dash="dot",
        annotation_text="Lower decision threshold",
    )

    fig.update_layout(
        title="Probability of a Positive Price Change",
        yaxis_title="Probability",
        yaxis_range=[0, 1.05],
        height=420,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.markdown("#### Out-of-sample model performance")

    st.dataframe(
        metrics,
        use_container_width=True,
        hide_index=True,
    )

    if results is not None:

        st.markdown("#### Actual vs predicted price changes")

        available_horizons = sorted(
            results["horizon_days"].unique()
        )

        selected = st.selectbox(
            "Choose a delivery horizon",
            available_horizons,
            format_func=lambda h: f"D+{h}",
        )

        subset = results.loc[
            results["horizon_days"] == selected
        ].sort_values("quotation_date")

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=subset["quotation_date"],
                y=subset["next_price_change"],
                mode="lines",
                name="Realized synthetic change",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=subset["quotation_date"],
                y=subset["predicted_change_eur_mwh"],
                mode="lines",
                name="ARX prediction",
            )
        )

        fig.update_layout(
            xaxis_title="Forecast origin",
            yaxis_title="EUR/MWh",
            height=450,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.info(
        "D+1 has no next pre-delivery quotation in the "
        "current dataset. Forecast probabilities rely on "
        "a normal-error approximation and have not been "
        "externally calibrated."
    )


# ============================================================
# PAGE 3: RISK ANALYTICS
# ============================================================

elif page == "Risk Analytics":

    heading(
        "Portfolio Risk Analytics",
        "Interactive one-day portfolio exposure and "
        "forecast-adjusted historical risk simulation.",
    )

    st.markdown("#### Portfolio positions")

    st.caption(
        "Positive MW = long futures position. "
        "Negative MW = short futures position."
    )

    position_columns = st.columns(6)

    position_values = {}

    for i, horizon in enumerate(range(2, 8)):

        with position_columns[i]:

            position_values[horizon] = st.number_input(
                f"D+{horizon} (MW)",
                min_value=-100.0,
                max_value=100.0,
                value=float(cfg.DEFAULT_POSITION_MW),
                step=1.0,
                key=f"position_{horizon}",
            )

    with st.expander("Risk parameters", expanded=True):

        r1, r2, r3 = st.columns(3)

        with r1:

            lookback = st.slider(
                "Historical lookback (days)",
                min_value=45,
                max_value=365,
                value=cfg.RISK_LOOKBACK_DAYS,
                step=5,
            )

        with r2:

            confidence = st.selectbox(
                "Confidence level",
                [0.90, 0.95, 0.99],
                index=1,
                format_func=lambda x: f"{x:.0%}",
            )

        with r3:

            trading_cost = st.number_input(
                "Transaction cost (EUR/MWh)",
                min_value=0.0,
                value=float(
                    cfg.TRANSACTION_COST_EUR_MWH
                ),
                step=0.05,
            )

    try:

        changes, forecasts = risk.load_inputs()

        portfolio = risk.prepare_portfolio(
            forecasts,
            position_values,
        )

        historical = risk.build_historical_scenarios(
            changes,
            portfolio["quotation_date"].max(),
            lookback,
        )

        pnl_scenarios, portfolio = (
            risk.calculate_scenario_pnl(
                historical,
                portfolio,
                trading_cost,
            )
        )

        individual = risk.calculate_position_risks(
            pnl_scenarios,
            portfolio,
            confidence,
        )

        summary = risk.calculate_portfolio_risk(
            pnl_scenarios,
            portfolio,
            individual,
            confidence,
        )

        correlation = historical.corr()

    except (FileNotFoundError, ValueError, KeyError) as exc:

        st.error(str(exc))

        st.code(
            "python -m src.market_analytics\n"
            "python -m src.forecasting",
            language="powershell",
        )

        st.stop()

    s = summary.iloc[0]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Expected Net P&L",
        money(s["expected_net_pnl_eur"]),
    )

    c2.metric(
        f"VaR ({confidence:.0%})",
        money(s["portfolio_var_eur"]),
    )

    c3.metric(
        f"CVaR ({confidence:.0%})",
        money(s["portfolio_cvar_eur"]),
    )

    c4.metric(
        "Probability of Loss",
        f"{s['probability_of_loss']:.1%}",
    )

    st.caption(
        f"Joint historical scenarios: "
        f"{int(s['joint_scenarios'])} | "
        f"Gross exposure: {s['gross_volume_mwh']:,.0f} MWh"
    )

    left, right = st.columns(2)

    with left:

        st.markdown("#### Portfolio P&L distribution")

        distribution = px.histogram(
            pnl_scenarios.reset_index(),
            x="net_portfolio_pnl_eur",
            nbins=25,
            labels={
                "net_portfolio_pnl_eur": "P&L (EUR)",
            },
        )

        distribution.add_vline(
            x=-s["portfolio_var_eur"],
            line_dash="dash",
        )

        distribution.update_layout(height=420)

        st.plotly_chart(
            distribution,
            use_container_width=True,
        )

    with right:

        st.markdown("#### Cross-product correlations")

        heatmap = px.imshow(
            correlation,
            zmin=-1,
            zmax=1,
            color_continuous_scale="RdBu_r",
            text_auto=".2f",
            labels={
                "x": "Current D+ horizon",
                "y": "Current D+ horizon",
                "color": "Correlation",
            },
        )

        heatmap.update_layout(height=420)

        st.plotly_chart(
            heatmap,
            use_container_width=True,
        )

    st.markdown("#### Individual position risks")

    columns = [
        "horizon_days",
        "position_mw",
        "signed_volume_mwh",
        "expected_net_pnl_eur",
        "var_eur",
        "cvar_eur",
        "probability_of_loss",
    ]

    st.dataframe(
        individual[columns].round(3),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download current portfolio risk results",
        data=individual.to_csv(index=False).encode("utf-8"),
        file_name="interactive_portfolio_risks.csv",
        mime="text/csv",
    )

    st.info(
        "Portfolio VaR is estimated from synchronized "
        "historical scenarios, adjusted by ARX forecasts. "
        "The method is illustrative, and tail estimates "
        "are sensitive to the number of observations."
    )


# ============================================================
# PAGE 4: HEDGING STRATEGIES
# ============================================================

elif page == "Hedging Strategies":

    heading(
        "Hedging Strategy Comparison",
        "Out-of-sample comparison of three price-risk "
        "management approaches.",
    )

    st.markdown("#### Scenario parameters")

    col1, col2, col3 = st.columns(3)

    with col1:

        exposure = st.number_input(
            "Reference exposure per product (MW)",
            min_value=0.1,
            max_value=100.0,
            value=float(cfg.DEFAULT_POSITION_MW),
            step=1.0,
        )

    with col2:

        fixed_ratio = st.slider(
            "Fixed hedge ratio",
            min_value=0.0,
            max_value=1.0,
            value=float(cfg.DEFAULT_HEDGE_RATIO),
            step=0.05,
        )

    with col3:

        strategy_cost = st.number_input(
            "Transaction cost (EUR/MWh)",
            min_value=0.0,
            max_value=5.0,
            value=float(cfg.TRANSACTION_COST_EUR_MWH),
            step=0.05,
            key="strategy_transaction_cost",
        )

    st.markdown("#### Adaptive strategy thresholds")

    lower, upper = st.columns(2)

    with lower:

        lower_prob = st.slider(
            "Lower probability threshold",
            min_value=0.05,
            max_value=0.49,
            value=float(
                cfg.SELL_PROBABILITY_THRESHOLD
            ),
            step=0.01,
        )

    with upper:

        upper_prob = st.slider(
            "Upper probability threshold",
            min_value=0.51,
            max_value=0.95,
            value=float(
                cfg.BUY_PROBABILITY_THRESHOLD
            ),
            step=0.01,
        )

    st.caption(
        "Below the lower threshold: no hedge. "
        "Between thresholds: 50% hedge. "
        "Above the upper threshold: 100% hedge."
    )

    try:

        forecasts = strategies.load_forecasts()

        trades = strategies.simulate_strategies(
            forecasts=forecasts,
            exposure_mw=exposure,
            fixed_hedge_ratio=fixed_ratio,
            lower_threshold=lower_prob,
            upper_threshold=upper_prob,
            transaction_cost=strategy_cost,
        )

        daily = strategies.aggregate_daily_results(
            trades
        )

        comparison = strategies.compare_strategies(
            daily,
            confidence=cfg.VAR_CONFIDENCE_LEVEL,
        )

    except (FileNotFoundError, ValueError, KeyError) as exc:

        st.error(str(exc))
        st.code(
            "python -m src.forecasting",
            language="powershell",
        )
        st.stop()

    st.markdown("#### Strategy performance")

    show_cols = [
        "strategy",
        "days",
        "total_net_pnl_eur",
        "improvement_vs_unhedged_eur",
        "average_hedge_ratio",
        "var_95_eur",
        "cvar_95_eur",
        "maximum_drawdown_eur",
    ]

    st.dataframe(
        comparison[show_cols].round(2),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### Cumulative incremental P&L")

    cumulative = px.line(
        daily,
        x="quotation_date",
        y="cumulative_pnl_eur",
        color="strategy",
        labels={
            "quotation_date": "Decision date",
            "cumulative_pnl_eur": "Cumulative P&L (EUR)",
        },
    )

    cumulative.update_layout(
        height=470,
        hovermode="x unified",
    )

    st.plotly_chart(
        cumulative,
        use_container_width=True,
    )

    left, right = st.columns(2)

    with left:

        st.markdown("#### VaR and CVaR")

        risk_chart = px.bar(
            comparison,
            x="strategy",
            y=["var_95_eur", "cvar_95_eur"],
            barmode="group",
            labels={
                "value": "Risk (EUR)",
                "strategy": "Strategy",
                "variable": "Risk measure",
            },
        )

        st.plotly_chart(
            risk_chart,
            use_container_width=True,
        )

    with right:

        st.markdown("#### Daily hedge ratio")

        adaptive = daily.loc[
            daily["strategy"] == "Adaptive Hedge"
        ]

        ratios = px.line(
            adaptive,
            x="quotation_date",
            y="average_hedge_ratio",
            labels={
                "quotation_date": "Decision date",
                "average_hedge_ratio": "Hedge ratio",
            },
        )

        ratios.update_yaxes(range=[0, 1.05])

        st.plotly_chart(
            ratios,
            use_container_width=True,
        )

    st.download_button(
        "Download strategy comparison",
        data=comparison.to_csv(index=False).encode("utf-8"),
        file_name="interactive_strategy_comparison.csv",
        mime="text/csv",
    )

    st.info(
        "The backtest uses out-of-sample ARX forecasts "
        "and synthetic price changes. It assumes a "
        "simplified reference exposure and one-day hedge "
        "settlements, not actual physical procurement "
        "cash flows or realized trading profits."
    )



# ============================================================
# PAGE 5: SENSITIVITY ANALYSIS
# ============================================================

elif page == "Sensitivity Analysis":

    heading(
        "Sensitivity Analysis",
        "Robustness of adaptive and fixed hedging "
        "strategies under alternative model parameters.",
    )

    comparison = load_table("sensitivity_comparison.csv")
    adaptive_grid = load_table("sensitivity_adaptive_grid.csv")
    fixed_grid = load_table("sensitivity_fixed_grid.csv")
    robustness = load_table("sensitivity_robustness.csv")

    if any(
        df is None
        for df in [
            comparison,
            adaptive_grid,
            fixed_grid,
            robustness,
        ]
    ):
        missing_data(
            "Sensitivity analysis results are missing.",
            "python -m src.sensitivity_analysis",
        )
        st.stop()

    st.info(
        "Results are based on synthetic electricity futures "
        "data and the same 146-day forecasting test period. "
        "Parameter-grid performance is exploratory, not "
        "independent out-of-sample validation."
    )

    # --------------------------------------------------------
    # 1. INTERACTIVE FILTERS
    # --------------------------------------------------------

    st.markdown("#### Scenario parameters")

    col1, col2 = st.columns(2)

    exposure_options = sorted(
        comparison["exposure_mw"].unique()
    )

    cost_options = sorted(
        comparison["transaction_cost_eur_mwh"].unique()
    )

    with col1:
        selected_exposure = st.selectbox(
            "Reference exposure (MW)",
            options=exposure_options,
            index=(
                exposure_options.index(5.0)
                if 5.0 in exposure_options else 0
            ),
            format_func=lambda x: f"{x:g} MW",
        )

    with col2:
        selected_cost = st.selectbox(
            "Transaction cost (EUR/MWh)",
            options=cost_options,
            index=(
                cost_options.index(0.10)
                if 0.10 in cost_options else 0
            ),
            format_func=lambda x: f"{x:.2f} EUR/MWh",
        )

    scenario = comparison.loc[
        np.isclose(
            comparison["exposure_mw"],
            selected_exposure,
        )
        & np.isclose(
            comparison["transaction_cost_eur_mwh"],
            selected_cost,
        )
    ].copy()

    # --------------------------------------------------------
    # 2. PROBABILITY THRESHOLD HEATMAP
    # --------------------------------------------------------

    st.markdown(
        "#### Adaptive vs Fixed: Probability Thresholds"
    )

    st.caption(
        "Each cell shows the difference in cumulative "
        "net P&L (Adaptive minus Fixed), measured in EUR. "
        "Positive values favor the adaptive strategy."
    )

    pivot = scenario.pivot(
        index="lower_threshold",
        columns="upper_threshold",
        values="adaptive_minus_fixed_eur",
    )

    pivot = pivot.sort_index().sort_index(axis=1)

    heatmap = go.Figure(
        data=go.Heatmap(
            z=pivot.to_numpy(),
            x=[
                f"{x:.0%}"
                for x in pivot.columns
            ],
            y=[
                f"{x:.0%}"
                for x in pivot.index
            ],
            colorscale="RdYlGn",
            colorbar={
                "title": "EUR",
            },
            text=np.round(
                pivot.to_numpy(),
                0,
            ),
            texttemplate="%{text:,.0f}",
            hovertemplate=(
                "Upper threshold: %{x}<br>"
                "Lower threshold: %{y}<br>"
                "P&L advantage: %{z:,.2f} EUR"
                "<extra></extra>"
            ),
        )
    )

    heatmap.update_layout(
        title=(
            "Adaptive Hedging Advantage "
            "Across Probability Thresholds"
        ),
        xaxis_title="Upper probability threshold",
        yaxis_title="Lower probability threshold",
        height=500,
    )

    st.plotly_chart(
        heatmap,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # 3. ROBUSTNESS INDICATORS
    # --------------------------------------------------------

    selected_robustness = robustness.loc[
        np.isclose(
            robustness["exposure_mw"],
            selected_exposure,
        )
        & np.isclose(
            robustness["transaction_cost_eur_mwh"],
            selected_cost,
        )
    ]

    if not selected_robustness.empty:

        r = selected_robustness.iloc[0]

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Adaptive P&L > Fixed",
            f"{r['share_adaptive_pnl_above_fixed']:.0%}",
        )

        c2.metric(
            "Adaptive P&L > Unhedged",
            f"{r['share_adaptive_pnl_above_unhedged']:.0%}",
        )

        c3.metric(
            "Adaptive CVaR < Fixed",
            f"{r['share_adaptive_cvar_below_fixed']:.0%}",
        )

        c4.metric(
            "Better P&L and CVaR",
            f"{r['share_adaptive_better_pnl_and_cvar']:.0%}",
        )

        st.caption(
            "Percentages represent shares of the "
            "25 tested threshold combinations, not "
            "probabilities of future trading success."
        )

    # --------------------------------------------------------
    # 4. TRANSACTION COST SENSITIVITY
    # --------------------------------------------------------

    st.markdown("#### Transaction Cost Sensitivity")

    st.caption(
        "This comparison holds the probability thresholds "
        "at the baseline values of 40% and 60%."
    )

    cost_subset = adaptive_grid.loc[
        np.isclose(
            adaptive_grid["exposure_mw"],
            selected_exposure,
        )
        & np.isclose(
            adaptive_grid["lower_threshold"],
            cfg.SELL_PROBABILITY_THRESHOLD,
        )
        & np.isclose(
            adaptive_grid["upper_threshold"],
            cfg.BUY_PROBABILITY_THRESHOLD,
        )
    ].copy()

    cost_chart = px.line(
        cost_subset,
        x="transaction_cost_eur_mwh",
        y="total_net_pnl_eur",
        color="strategy",
        markers=True,
        labels={
            "transaction_cost_eur_mwh":
                "Transaction cost (EUR/MWh)",
            "total_net_pnl_eur":
                "Cumulative net P&L (EUR)",
            "strategy": "Strategy",
        },
    )

    cost_chart.update_layout(
        height=430,
        hovermode="x unified",
    )

    st.plotly_chart(
        cost_chart,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # 5. EXPOSURE SIZE SENSITIVITY
    # --------------------------------------------------------

    st.markdown("#### Exposure Size and Tail Risk")

    st.caption(
        "Conditional Value at Risk (CVaR 95%) "
        "for different reference exposure sizes."
    )

    exposure_subset = adaptive_grid.loc[
        np.isclose(
            adaptive_grid["transaction_cost_eur_mwh"],
            selected_cost,
        )
        & np.isclose(
            adaptive_grid["lower_threshold"],
            cfg.SELL_PROBABILITY_THRESHOLD,
        )
        & np.isclose(
            adaptive_grid["upper_threshold"],
            cfg.BUY_PROBABILITY_THRESHOLD,
        )
    ].copy()

    exposure_chart = px.line(
        exposure_subset,
        x="exposure_mw",
        y="cvar_95_eur",
        color="strategy",
        markers=True,
        labels={
            "exposure_mw": "Exposure (MW)",
            "cvar_95_eur": "CVaR 95% (EUR)",
            "strategy": "Strategy",
        },
    )

    exposure_chart.update_layout(
        height=430,
        hovermode="x unified",
    )

    st.plotly_chart(
        exposure_chart,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # 6. FIXED HEDGE RATIO SENSITIVITY
    # --------------------------------------------------------

    st.markdown("#### Fixed Hedge Ratio Sensitivity")

    fixed_subset = fixed_grid.loc[
        np.isclose(
            fixed_grid["exposure_mw"],
            selected_exposure,
        )
        & np.isclose(
            fixed_grid["transaction_cost_eur_mwh"],
            selected_cost,
        )
    ].copy()

    fixed_subset = fixed_subset.sort_values(
        "fixed_hedge_ratio"
    )

    fixed_chart = px.line(
        fixed_subset,
        x="fixed_hedge_ratio",
        y="total_net_pnl_eur",
        markers=True,
        labels={
            "fixed_hedge_ratio": "Fixed hedge ratio",
            "total_net_pnl_eur": "Net P&L (EUR)",
        },
    )

    fixed_chart.update_xaxes(tickformat=".0%")

    fixed_chart.update_layout(height=430)

    st.plotly_chart(
        fixed_chart,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # 7. INTERACTIVE SCENARIO DETAIL
    # --------------------------------------------------------

    st.markdown("#### Detailed Scenario Comparison")

    lower_options = sorted(
        scenario["lower_threshold"].unique()
    )

    upper_options = sorted(
        scenario["upper_threshold"].unique()
    )

    lower_col, upper_col = st.columns(2)

    with lower_col:
        selected_lower = st.selectbox(
            "Lower probability threshold",
            options=lower_options,
            index=(
                lower_options.index(0.40)
                if 0.40 in lower_options else 0
            ),
            format_func=lambda x: f"{x:.0%}",
        )

    with upper_col:
        selected_upper = st.selectbox(
            "Upper probability threshold",
            options=upper_options,
            index=(
                upper_options.index(0.60)
                if 0.60 in upper_options else 0
            ),
            format_func=lambda x: f"{x:.0%}",
        )

    selected_scenario = scenario.loc[
        np.isclose(
            scenario["lower_threshold"],
            selected_lower,
        )
        & np.isclose(
            scenario["upper_threshold"],
            selected_upper,
        )
    ]

    if not selected_scenario.empty:

        row = selected_scenario.iloc[0]

        metric1, metric2, metric3 = st.columns(3)

        metric1.metric(
            "Adaptive − Fixed",
            money(row["adaptive_minus_fixed_eur"]),
        )

        metric2.metric(
            "Adaptive − Unhedged",
            money(row["adaptive_minus_unhedged_eur"]),
        )

        metric3.metric(
            "Adaptive CVaR",
            money(row["cvar_95_eur_adaptive_hedge"]),
        )

        detail = pd.DataFrame({
            "Strategy": [
                "Adaptive Hedge",
                "Fixed Hedge",
                "Unhedged",
            ],
            "Net P&L (EUR)": [
                row["total_net_pnl_eur_adaptive_hedge"],
                row["total_net_pnl_eur_fixed_hedge"],
                row["total_net_pnl_eur_unhedged"],
            ],
            "VaR 95% (EUR)": [
                row["var_95_eur_adaptive_hedge"],
                row["var_95_eur_fixed_hedge"],
                row["var_95_eur_unhedged"],
            ],
            "CVaR 95% (EUR)": [
                row["cvar_95_eur_adaptive_hedge"],
                row["cvar_95_eur_fixed_hedge"],
                row["cvar_95_eur_unhedged"],
            ],
            "Maximum Drawdown (EUR)": [
                row["maximum_drawdown_eur_adaptive_hedge"],
                row["maximum_drawdown_eur_fixed_hedge"],
                row["maximum_drawdown_eur_unhedged"],
            ],
            "Average Hedge Ratio": [
                row["average_hedge_ratio_adaptive_hedge"],
                row["average_hedge_ratio_fixed_hedge"],
                row["average_hedge_ratio_unhedged"],
            ],
        })

        st.dataframe(
            detail.round(2),
            use_container_width=True,
            hide_index=True,
        )

    # --------------------------------------------------------
    # 8. EXPORT
    # --------------------------------------------------------

    st.divider()

    st.download_button(
        "Download selected sensitivity scenarios",
        data=scenario.to_csv(index=False).encode("utf-8"),
        file_name="sensitivity_selected_scenarios.csv",
        mime="text/csv",
    )

    st.markdown("#### Interpretation")

    st.info(
        "Sensitivity results describe differences within "
        "the chosen synthetic parameter grid. The strategy "
        "with the highest result was not evaluated on a "
        "separate validation sample after parameter selection. "
        "The percentages shown above therefore measure "
        "robustness across assumptions, not statistical "
        "confidence or real-market success probabilities."
    )


# ============================================================
# PAGE 6: METHODOLOGY AND LIMITATIONS
# ============================================================

elif page == "Methodology & Limitations":

    heading(
        "Methodology & Limitations",
        "Research assumptions and scope of the DSS prototype.",
    )

    st.markdown(
        """
        ### 1. Research objective

        The system demonstrates how quantitative forecasting
        and risk analytics can support decisions involving
        short-term German electricity futures.

        The case study is motivated by electricity trading
        activities, but uses hypothetical exposures and
        synthetic market observations.

        ### 2. Market data

        The research dataset is generated programmatically.

        It includes:

        - Synthetic daily futures prices.
        - Contract quotation and delivery dates.
        - Delivery horizons D+1 to D+7.
        - Simulated weather forecasts.
        - Seasonal effects, autoregressive persistence,
          and occasional extreme price shocks.

        The dataset is not a reconstruction of EEX trading
        history.

        ### 3. Forecasting model

        An autoregressive model with exogenous variables
        (ARX) estimates the next-day price change of the
        same delivery contract.

        Explanatory variables include:

        - Previously observed price changes.
        - Current futures price.
        - Forecast temperature.
        - Forecast wind speed.
        - Forecast solar radiation.
        - Annual seasonal terms.
        - Delivery-horizon indicators.

        Conditional directional probabilities are
        approximated using a normal residual distribution.

        ### 4. Model validation

        A chronological train/test split is applied.

        Performance indicators include:

        - Mean Absolute Error (MAE).
        - Root Mean Squared Error (RMSE).
        - Directional accuracy.
        - Brier Score.

        Forecast accuracy on synthetic data does not
        establish real-market predictive accuracy.

        ### 5. Portfolio risk assessment

        The risk module uses forecast-adjusted historical
        scenarios to estimate:

        - Value at Risk (VaR).
        - Conditional Value at Risk (CVaR).
        - Probability of portfolio loss.
        - Individual position exposures.
        - Cross-product correlations.

        ### 6. Hedging strategy evaluation

        Three illustrative strategies are compared:

        **Unhedged:** no futures hedge.

        **Fixed Hedge:** constant hedge ratio.

        **Adaptive Hedge:** hedge ratio determined by
        forecast probabilities.

        The simulation assumes a simplified reference
        price exposure with one-to-one price sensitivity.

        ### 7. Key limitations

        The current prototype does not explicitly model:

        - Actual exchange order-book liquidity.
        - Bid-ask spreads and market impact.
        - Exchange margin calls and funding liquidity.
        - Detailed electricity procurement contracts.
        - Cross-market basis risk.
        - Power plant outages and transmission constraints.
        - Unexpected geopolitical or regulatory events.
        - Real-time news sentiment or event detection.
        - Structural market regime changes.
        - Actual exchange trading calendars.
        - Independently verified forecast calibration.

        ### 8. Future extensions

        Further development could introduce:

        - Actual historical EEX quotation data.
        - ENTSO-E fundamentals and cross-border flows.
        - Renewable generation and demand forecasts.
        - Fuel and EU ETS allowance prices.
        - News and discrete-event indicators.
        - Probabilistic forecast calibration.
        - Rolling-window model re-estimation.
        - Additional machine-learning benchmarks.
        - Portfolio optimization under risk limits.

        ### 9. Intended use

        This dashboard is an academic decision-support
        prototype and must not be interpreted as a
        production trading recommendation system.
        """
    )

    st.success(
        "The prototype includes sensitivity analysis "
        "across 300 parameter combinations and "
        "three alternative hedging strategies."
    )

    st.divider()

    st.caption(
        "German Power Futures DSS | "
        "Python, Pandas, Statsmodels, Scikit-learn, "
        "Plotly and Streamlit"
    )
