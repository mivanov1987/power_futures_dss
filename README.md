
# German Power Futures Decision Support System

A Python-based Decision Support System (DSS) for short-term
electricity futures trading and price risk management in
the German power market.

The project demonstrates how statistical forecasting,
probabilistic trading signals, portfolio risk analysis,
and hedging strategy simulation can support decisions
under electricity price uncertainty.

**Project status:** Research and educational prototype  
**Market:** German electricity futures  
**Products:** Daily baseload futures, D+1 to D+7  
**Language:** Python 3.11  
**Interface:** Streamlit  
**Data:** Synthetic, reproducible market simulation

> **Important:** This project uses synthetic market data.
> Its forecasts, trading signals, portfolio scenarios, and
> financial results do not represent actual EEX quotations,
> actual company positions, or realized trading profits.
> The software is not intended for live trading.

---

## 1. Project Objectives

The DSS has five principal objectives:

1. Analyze price dynamics and volatility of short-term
   German electricity futures.
2. Forecast next-day changes in futures quotations using
   an autoregressive model with exogenous variables (ARX).
3. Estimate the probability of future price increases
   or decreases.
4. Assess portfolio price risk using Value at Risk (VaR)
   and Conditional Value at Risk (CVaR).
5. Compare unhedged, fixed hedging, and adaptive hedging
   strategies under alternative assumptions.

An additional sensitivity analysis examines the
robustness of the results to model parameters,
transaction costs, and exposure sizes.

---

## 2. System Architecture

The DSS follows a modular analytical workflow:

```text
Synthetic Data Generator
          |
          v
Market Analytics
          |
          v
ARX Forecasting
          |
          +--------------------+
          |                    |
          v                    v
   Risk Analytics       Strategy Simulator
          |                    |
          |                    v
          |              Sensitivity Analysis
          |                    |
          +---------+----------+
                    |
                    v
              CSV Outputs
                    |
                    v
            Streamlit Dashboard
```

The data generator creates reproducible synthetic
futures quotations. The analytical modules then
produce forecasts, risk estimates, strategy results,
tables, and figures.

The dashboard provides an interactive interface to
the analytical outputs.

---

## 3. Repository Structure

```text
german_power_futures_dss/
|
|-- config.py
|-- environment.yml
|-- requirements.txt
|-- README.md
|-- .gitignore
|
|-- src/
|   |-- data_generator.py
|   |-- market_analytics.py
|   |-- forecasting.py
|   |-- risk_analytics.py
|   |-- strategy_simulator.py
|   |-- sensitivity_analysis.py
|
|-- dashboard/
|   |-- app.py
|
|-- tests/
|   |-- test_models.py
|
|-- data/
|   |-- simulated_futures.csv
|
|-- outputs/
|   |-- tables/
|   |-- figures/
```

The `data/` and `outputs/` directories contain
generated files. Their contents can be recreated
by running the corresponding Python modules.

No externally obtained market dataset is required
to reproduce the synthetic demonstration.

---

## 4. Requirements

Recommended software:

- Python 3.11
- Conda (Anaconda or Miniconda), or Python `venv`
- Git
- A web browser
- Visual Studio Code (optional)

Main Python libraries:

- NumPy
- Pandas
- SciPy
- Statsmodels
- Scikit-learn
- Matplotlib
- Plotly
- Streamlit
- PyYAML
- Joblib
- Pytest

The exact package versions of the reference development
environment are recorded in `requirements.txt`.

### 4.1 Installation with Conda

Clone the repository:

```powershell
git clone https://github.com/mivanov1987/power_futures_dss
cd german_power_futures_dss
```

Create the environment:

```powershell
conda env create -f environment.yml
```

Activate it:

```powershell
conda activate power_futures_env
```

Verify the installation:

```powershell
python --version
python -m streamlit --version
python -m pytest --version
```

The Conda environment is defined by `environment.yml`.
It is the recommended setup method for this project.

### 4.2 Alternative Installation with Python venv

If Conda is unavailable, create a virtual environment
using Python 3.11.

On Windows PowerShell:

```powershell
py -3.11 -m venv .venv
```

Install dependencies using the virtual environment's
Python executable:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Check the installation:

```powershell
.\.venv\Scripts\python.exe -m pytest --version
```

For the remaining commands, replace `python` with
`.\.venv\Scripts\python.exe` if the virtual environment
has not been activated.

**Compatibility note:** `requirements.txt` is generated
from the reference environment. Installation on other
operating systems may require platform-specific dependency
resolution.

---

## 5. Configuration

The main project settings are stored in `config.py`.

These include:

- Simulation start and end dates.
- Random seed for reproducibility.
- Delivery horizons D+1 through D+7.
- Price volatility and mean reversion parameters.
- Weather-related price coefficients.
- ARX model specification.
- Training and testing sample ratio.
- Trading signal probability thresholds.
- Portfolio exposure size.
- Hedge ratios.
- VaR and CVaR confidence levels.
- Historical risk lookback period.
- Transaction cost assumptions.

Default research period:

```text
2024-01-01 to 2025-12-31
```

Default random seed:

```text
42
```

The synthetic experiment is reproducible when
the same software environment, configuration,
and generated inputs are used.

Changing configuration parameters may require
regenerating dependent output files.

---

## 6. Running the Complete Analysis

Run the following commands from the project root.

### Step 1: Generate Synthetic Futures Data

```powershell
python -m src.data_generator
```

Generates synthetic daily quotations for German
baseload futures with delivery horizons D+1 to D+7.

The dataset includes:

- Quotation date.
- Delivery date.
- Contract identifier.
- Delivery horizon.
- Futures price in EUR/MWh.
- Delivery hours.
- Simulated weather forecast variables.
- Synthetic-data indicator.

Main output:

```text
data/simulated_futures.csv
```

### Step 2: Market Analytics

```powershell
python -m src.market_analytics
```

Calculates:

- Descriptive price statistics.
- Matched-contract daily price changes.
- Volatility measures.
- Autocorrelation indicators.
- Extreme price movements.

Outputs are saved in:

```text
outputs/tables/
outputs/figures/
```

### Step 3: ARX Forecasting

```powershell
python -m src.forecasting
```

Fits an autoregressive model with exogenous variables.

The forecasting target is the next available daily
quotation change for the **same delivery contract**.

Forecasts are available for D+2 to D+7.

D+1 is excluded from this one-day-ahead prediction
setup because there is no subsequent pre-delivery
quotation in the synthetic daily dataset.

The forecasting module generates:

```text
outputs/tables/forecast_results.csv
outputs/tables/forecast_metrics.csv
outputs/tables/forecast_coefficients.csv
outputs/tables/latest_forecasts.csv
```

Model performance is evaluated using a chronological
training/testing split.

Reported evaluation measures include:

- Mean Absolute Error (MAE).
- Root Mean Squared Error (RMSE).
- Directional accuracy.
- Brier Score.

Price-direction probabilities rely on an
assumed residual distribution.

### Step 4: Portfolio Risk Analytics

```powershell
python -m src.risk_analytics
```

The risk module evaluates hypothetical long and
short futures positions using historical price-change
scenarios adjusted by model forecasts.

Measures include:

- Expected incremental portfolio P&L.
- Value at Risk (VaR).
- Conditional Value at Risk (CVaR).
- Probability of loss.
- Position-level risk.
- Cross-product correlations.

Example scenario with mixed long and short positions:

```powershell
python -m src.risk_analytics --positions "2:5,3:-3,4:0,5:5,6:-2,7:5"
```

Main outputs:

```text
outputs/tables/risk_results.csv
outputs/tables/risk_portfolio_summary.csv
outputs/tables/risk_scenarios.csv
outputs/tables/risk_correlation.csv
```

The analysis uses hypothetical positions.
No actual trading account is accessed.

### Step 5: Hedging Strategy Simulation

```powershell
python -m src.strategy_simulator
```

Compares three approaches:

**Unhedged**

The reference price exposure is left unhedged.

**Fixed Hedge**

A constant proportion of the exposure is hedged.

**Adaptive Hedge**

The hedge ratio changes according to the forecast
probability of a price increase.

Default decision rule:

| Probability of increase | Hedge ratio |
|---|---:|
| Below 40% | 0% |
| From 40% to below 60% | 50% |
| 60% or above | 100% |

The module calculates:

- Cumulative incremental net P&L.
- Daily P&L volatility.
- VaR and CVaR.
- Maximum drawdown.
- Average hedge ratio.
- Transaction costs.
- Performance relative to the unhedged benchmark.

Main outputs:

```text
outputs/tables/strategy_trade_results.csv
outputs/tables/strategy_daily_results.csv
outputs/tables/strategy_comparison.csv
```

**Interpretation:** Strategy P&L reflects simplified
price-exposure changes and simulated hedge settlements,
not actual procurement or trading profits.

### Step 6: Sensitivity Analysis

```powershell
python -m src.sensitivity_analysis
```

Investigates alternative assumptions concerning:

- Lower forecast probability threshold.
- Upper forecast probability threshold.
- Transaction costs.
- Exposure sizes.
- Fixed hedge ratios.

The default parameter grid contains:

- 5 lower probability thresholds.
- 5 upper probability thresholds.
- 4 transaction cost assumptions.
- 3 reference exposure sizes.

This produces 300 scenario configurations and
900 individual strategy evaluations.

An additional fixed-ratio grid contains
48 evaluations.

Main outputs:

```text
outputs/tables/sensitivity_adaptive_grid.csv
outputs/tables/sensitivity_fixed_grid.csv
outputs/tables/sensitivity_comparison.csv
outputs/tables/sensitivity_robustness.csv
```

The module also exports sensitivity charts.

**Important:** Parameter sensitivity is evaluated
on the same forecasting test sample. The observed
best-performing configuration must not be interpreted
as independently validated or optimal for future trading.

---

## 7. Launching the Streamlit Dashboard

Start the application from the project root:

```powershell
python -m streamlit run dashboard/app.py
```

Open the local address displayed by Streamlit,
typically:

http://localhost:8501

The interface includes six sections:

### 7.1 Market Overview

- Historical synthetic futures prices.
- Price distributions by delivery horizon.
- Volatility comparisons.
- Extreme price movements.

### 7.2 Price Forecasting

- ARX model predictions.
- Probability of price increase.
- Forecast accuracy measures.
- Actual versus predicted synthetic price changes.

### 7.3 Risk Analytics

- Editable long and short positions.
- Portfolio VaR and CVaR.
- P&L scenario distributions.
- Correlation heatmaps.

### 7.4 Hedging Strategies

- Unhedged, fixed, and adaptive hedging.
- Adjustable reference exposure and transaction costs.
- Adjustable probability thresholds.
- Cumulative P&L and risk comparisons.

### 7.5 Sensitivity Analysis

- Interactive probability threshold heatmap.
- Transaction cost sensitivity.
- Exposure size sensitivity.
- Fixed hedge ratio comparisons.
- Robustness summaries.

### 7.6 Methodology & Limitations

- Research assumptions.
- Model description.
- Data generation.
- Risk and strategy methodology.
- Scope and limitations.
- Future development opportunities.

The dashboard reads previously generated analytical
CSV outputs. Run the analytical modules again after
changing their inputs or configuration.

---

## 8. Automated Testing

The repository includes tests implemented with pytest.

Run:

```powershell
python -m pytest tests/test_models.py -v
```

Or run the complete test directory:

```powershell
python -m pytest tests/ -v
```

At the documented development checkpoint,
34 automated tests passed.

The tests cover:

- Synthetic data generation and reproducibility.
- Contract and delivery date consistency.
- Daylight saving time adjustments.
- Matched-contract price changes.
- Chronological forecasting splits.
- Forecast probability consistency.
- VaR and CVaR calculations.
- Portfolio exposure identities.
- Hedging ratio boundaries.
- Strategy P&L accounting identities.

Passing these tests supports internal software
consistency. It does not validate the predictive
performance on real market data.

---

## 9. Reproducing the Research Experiment

For consistent results:

1. Use the same project code revision.
2. Install the documented dependencies.
3. Keep the same `config.py` settings.
4. Use the same random seed.
5. Run analytical modules in the prescribed order.
6. Verify generated input file hashes.
7. Record the resulting tables and test outcomes.

Example command for checking the input data hash
on Windows PowerShell:

```powershell
Get-FileHash .\outputs\tables\forecast_results.csv -Algorithm SHA256
```

A file hash identifies the exact file contents used
for an experiment.

Changes to model inputs or outputs should be
documented before comparisons are reported in
academic research.

---

## 10. Methodological Limitations

The DSS is intended to demonstrate the integration
of quantitative methods into electricity futures
risk management.

Principal limitations include:

**Synthetic market data**

Prices and weather variables are generated from an
assumed stochastic process and are not actual exchange
records.

**Forecasting specification**

The baseline model is ARX. Its synthetic-data
performance does not establish forecasting ability
under real market conditions.

**Probability estimation**

Price-direction probabilities are based on assumed
forecast-error behavior. Real-market calibration
has not been established.

**Simplified price exposure**

Hedging strategies represent one-day incremental
price exposure and simplified futures settlements.

**Limited execution modeling**

Actual bid-ask spreads, liquidity constraints,
market impact, margin requirements, and funding
costs are not fully represented.

**Market fundamentals**

The model does not explicitly incorporate all
electricity-market fundamentals, outages,
regulatory events, or news-driven shocks.

**Parameter sensitivity**

Sensitivity analysis uses a shared historical test
sample and does not constitute an independent
validation after parameter selection.

**Trading calendars**

The demonstration uses simulated daily observations.
Actual exchange trading calendars, holidays, and
contract-specific settlement rules would require
additional modeling.

---

## 11. Future Development

Possible extensions include:

- Integration of historical EEX quotations.
- ENTSO-E market and generation data.
- Actual load and renewable energy forecasts.
- Fuel and carbon allowance prices.
- Rolling-window model re-estimation.
- Probability calibration diagnostics.
- Additional statistical and ML models.
- Realistic execution and liquidity modeling.
- Portfolio optimization under risk constraints.
- Scenario stress testing.
- Database persistence.
- User authentication and access control.
- Automated data pipelines and monitoring.

Any future integration of proprietary market
or position data should use appropriate access
controls and data governance procedures.

---

## 12. Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python 3.11 |
| Data processing | Pandas, NumPy |
| Statistical modeling | Statsmodels, SciPy |
| Machine learning utilities | Scikit-learn |
| Static visualization | Matplotlib |
| Interactive visualization | Plotly |
| Dashboard | Streamlit |
| Automated testing | Pytest |
| Environment management | Conda / Python venv |
| Version control | Git / GitHub |

---

## 13. Academic and Research Use

This repository accompanies a research-oriented
decision-support prototype concerning the management
of electricity price risk through short-term futures
trading.

The system is designed for demonstration, education,
and methodological evaluation.

All examples and financial results must be interpreted
as hypothetical unless independently verified
against real market data.

---

## 14. Repository

GitHub:

https://github.com/mivanov1987/power_futures_dss

---

## 15. Disclaimer

This software is provided for research and educational
purposes only.

It does not constitute investment advice, financial
advice, or a recommendation to trade electricity
futures or other derivatives.

The authors make no representation that simulated
strategy performance can be replicated under actual
market conditions.
