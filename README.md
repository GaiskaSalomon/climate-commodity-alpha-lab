# Climate Commodity Alpha Lab

A quantitative research project for commodity return forecasting using price dynamics, weather anomalies, and extreme climate-risk signals.

## Research Question

> Can weather and climate-risk variables improve commodity return forecasts under a robust walk-forward validation framework?

## Motivation

Commodity markets are physically constrained: energy supply disrupts when hurricanes reach the Gulf of Mexico, grain yields fall under drought, and demand for natural gas spikes with temperature anomalies. This project transforms that physical domain knowledge into systematic, testable predictive signals using rigorous time-series validation.

## Results (summary)

Walk-forward validation (500-day train / 21-day test / 21-day step) over 7 commodity ETFs, 2007–2026, net of 5 bps one-way transaction costs.

- Climate + price features beat price-only baselines on **5 of 7 assets** by Information Coefficient (IC).
- Best assets: **SOYB (IC 0.121)** and **GLD (IC 0.113, net Sharpe 1.38)**.
- Climate features add **+0.05 to +0.07 IC** on agricultural and gold tickers; the Gulf hurricane-disruption proxy ranks top-5 by XGBoost gain importance for energy.
- Equal-weight portfolio: **net Sharpe ≈ 0.82** after costs.

![Equity curve](reports/figures/equity_curve.png)

> **Scope & honesty:** this is a *research study* on whether climate / alternative data adds predictive signal (measured by IC), **not** a deployable trading strategy. Drawdowns are large and some assets (e.g., UNG) are not forecastable with this feature set and are excluded from portfolio views. All results use strictly temporal validation (no look-ahead) and are reported net of costs. Full details in [`reports/quant_research_note.md`](reports/quant_research_note.md) and [`reports/backtest_report.md`](reports/backtest_report.md).

## Architecture

```mermaid
flowchart LR
    A[Commodity Prices] --> D[Aligned Research Panel]
    B[Weather Variables] --> D
    C[Hurricane / Extreme Event Data] --> D

    D --> E[Feature Engineering]
    E --> F[Return Targets]
    F --> G[Walk-Forward Modeling]

    G --> H[Forecast Scores]
    H --> I[Signal Ranking]
    I --> J[Portfolio Construction]

    J --> K[Backtesting]
    K --> L[Risk Metrics]
    K --> M[Research Report]

    L --> N[Sharpe / Drawdown / Turnover]
    M --> O[Quant Research Note]
```

## Assets

| Proxy | Commodity | Climate Relationship |
|-------|-----------|----------------------|
| UNG | Natural Gas | Temperature anomalies, hurricane disruptions |
| USO | Crude Oil | Gulf hurricane risk, supply shocks |
| DBA | Agriculture broad | Temperature, precipitation |
| CORN | Corn | Precipitation, heat stress |
| WEAT | Wheat | Drought, temperature extremes |
| SOYB | Soybeans | Precipitation, growing season climate |
| GLD | Gold | Macro/risk-off control |

## Data Sources

| Dataset | Source | Purpose |
|---------|--------|---------|
| ETF prices | Yahoo Finance (yfinance) | Commodity price proxies |
| Weather data | NOAA GHCND | Temperature, precipitation |
| Hurricane tracks | IBTrACS v4 | Tropical cyclone risk features |
| Macro variables | FRED / public sources | Optional regime context |

## Methodology

1. **Price feature engineering** — returns, momentum, volatility, drawdown, skewness
2. **Weather anomaly construction** — day-of-year climatology, rolling anomalies, drought proxies
3. **Hurricane risk scoring** — distance-weighted wind intensity for key infrastructure regions
4. **Forward return targets** — 1d, 5d, 10d, 20d horizons; directional and regression targets
5. **Walk-forward model training** — strictly temporal, no shuffling, no random splits
6. **Signal generation** — cross-sectional ranking and time-series forecasts
7. **Portfolio backtesting** — with transaction costs and turnover constraints
8. **Risk and robustness analysis** — regime breakdown, stress tests, drawdown analysis

## Models

| Model | Role |
|-------|------|
| Ridge Regression | Baseline, regularized |
| Logistic Regression | Directional baseline |
| Random Forest | Non-linear reference |
| XGBoost / LightGBM | Primary forecasting model |
| Bayesian Regression (PyMC) | Uncertainty quantification |
| Volatility / Climate Regime Detection | Conditional performance analysis |

## Evaluation Framework

| Metric | Purpose |
|--------|---------|
| Information Coefficient (IC) | Predictive quality of scores |
| Hit Rate | Directional accuracy |
| Sharpe Ratio (net) | Risk-adjusted performance |
| Max Drawdown | Downside risk |
| Turnover | Cost efficiency |
| Performance by regime | Robustness under different conditions |

## Repository Structure

```
climate-commodity-alpha-lab/
├── src/
│   ├── data/           # Data loading and panel construction
│   ├── features/       # Price, weather, hurricane and regime features
│   ├── models/         # Baseline, tree, Bayesian models + walk-forward
│   ├── backtesting/    # Signal-to-position, portfolio, performance
│   ├── risk/           # Drawdown, VaR/CVaR, stress tests
│   └── utils/          # Metrics, date utilities, leakage validation
├── notebooks/          # Ordered research notebooks (01–08)
├── reports/            # Quant research note, backtest report, figures
├── tests/              # Unit tests for all modules
└── data/               # Raw, interim and processed data (not tracked)
```

## Notebooks

| Notebook | Content |
|----------|---------|
| `01_price_data_eda.ipynb` | Price data, returns, feature distributions |
| `02_weather_climate_features.ipynb` | Weather anomalies, drought proxies |
| `03_hurricane_risk_features.ipynb` | Storm tracks, risk scoring, energy exposure |
| `04_signal_engineering.ipynb` | Feature selection, cross-asset correlations |
| `05_walk_forward_modeling.ipynb` | Model comparison under temporal validation |
| `06_backtesting_strategy.ipynb` | Signal-to-portfolio, equity curve, metrics |
| `07_regime_detection.ipynb` | Volatility and climate regime analysis |
| `08_research_report.ipynb` | Full findings as a quant research note |

> Notebooks `01` and `03`–`08` run end to end on the committed processed data (Python 3.10).
> Notebook `02` (ERA5 weather features) is **optional**: it requires downloading ERA5 data via the Copernicus CDS API (and the `cfgrib`/`eccodes` stack). The pipeline runs without it — ERA5 features are skipped automatically when the raw files are not present.

## Why this project matters

Physical climate-risk modeling and systematic commodity research share the same core challenge: transforming noisy, non-stationary time-series data into robust predictive signals. This project applies hydrometeorological domain knowledge — tropical cyclone tracks, precipitation, temperature — directly to commodity return forecasting, with the same rigor expected in production quant systems: temporal validation, realistic transaction costs, and structured reporting.

## Setup

```bash
git clone https://github.com/GaiskaSalomon/climate-commodity-alpha-lab.git
cd climate-commodity-alpha-lab
pip install -r requirements.txt
pip install -e .
cp .env.example .env  # fill in API keys
pytest tests/
```

## CV Summary

> Built a systematic commodity forecasting pipeline using price dynamics, weather anomalies and tropical cyclone risk features. Implemented walk-forward validation, XGBoost/LightGBM models, signal generation, transaction-cost-aware backtesting and risk reporting. Designed to test whether climate-risk alternative data improves commodity return forecasts.

## License

MIT.
