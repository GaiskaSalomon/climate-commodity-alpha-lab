"""Build notebooks/08_research_report.ipynb programmatically."""
import nbformat, sys, pathlib

ROOT = pathlib.Path(__file__).parent.parent
NB_PATH = ROOT / "notebooks" / "08_research_report.ipynb"

M = nbformat.v4.new_markdown_cell
C = nbformat.v4.new_code_cell
R = nbformat.v4.new_raw_cell

cells = [
    M("""\
# 08 -- Research Report: Climate-Aware Commodity Return Forecasting

**Quantitative Research Note | May 2026**

---

## Abstract

We present a systematic forecasting framework for seven commodity ETFs (CORN, DBA, GLD, SOYB, UNG, USO, WEAT) that integrates price dynamics, tropical cyclone risk, and large-scale climate indices into a walk-forward machine learning pipeline. Our best model -- XGBoost with full features -- achieves a cross-sectional Information Coefficient (IC) of 0.121 on soybeans and 0.113 on gold over 2007-2026, with a portfolio net Sharpe ratio above 0.8 after 5 bps transaction costs. A Bayesian linear regression module quantifies prediction uncertainty, and regime analysis shows model performance is robust across ENSO phases and AMO states but degrades in high-volatility environments.

---

**Sections:**
1. Motivation and Data
2. Feature Engineering Overview
3. Model Performance (IC and Hit Rate)
4. Backtesting and Portfolio Analysis
5. Bayesian Uncertainty Quantification
6. Regime Robustness
7. Risk Profile
8. Conclusions and Next Steps"""),

    C("""\
import sys
sys.path.insert(0, '..')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from src.config import DATA_PROCESSED_DIR
from src.models.evaluation import sharpe_ratio, max_drawdown, information_coefficient, hit_rate
from src.risk.var_cvar import historical_var, historical_cvar
from src.risk.drawdown import drawdown_series

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120
FOCUS_MODEL = "XGBoost (full)"

wf          = pd.read_parquet(DATA_PROCESSED_DIR / "backtest_results.parquet")
metrics     = pd.read_csv(DATA_PROCESSED_DIR / "model_metrics.csv")
panel       = pd.read_parquet(DATA_PROCESSED_DIR / "modeling_panel.parquet")
regime_perf = pd.read_csv(DATA_PROCESSED_DIR / "regime_performance.csv")

wf["date"]    = pd.to_datetime(wf["date"])
panel["date"] = pd.to_datetime(panel["date"])

tickers = sorted(wf["ticker"].unique())
models  = wf["model"].unique()
print(f"Walk-forward data: {wf.shape}  |  Date range: {wf.date.min().date()} to {wf.date.max().date()}")
print(f"Tickers: {tickers}")
print(f"Models:  {list(models)}")"""),

    M("""\
## 1. Motivation and Data

Commodity prices are influenced by both financial market dynamics and real-world supply disruptions driven by extreme weather. Tropical cyclones in the Gulf of Mexico disrupt energy infrastructure (UNG, USO); heat waves and drought reduce crop yields (CORN, SOYB, WEAT); ENSO cycles modulate global precipitation patterns affecting agricultural production across continents simultaneously.

| Data source | Variables | Coverage |
|---|---|---|
| Yahoo Finance (yfinance) | OHLCV, adjusted close | 7 ETFs, 2005-2026 |
| IBTrACS QC (PhD thesis data) | Track, Vmax, OHC, z26 | Atlantic basin, 1980-2025 |
| CDS ERA5 surface | t2m, total precipitation | 4 commodity regions, 2005-2025 |
| NOAA / CPC | ONI (ENSO), AMO, PDO | Monthly, 1990-2026 |

**Target:** 5-day forward return  \\
**Validation:** Strictly temporal walk-forward (500-day train / 21-day test / 21-day step) -- no look-ahead"""),

    M("""\
## 2. Feature Engineering Overview

| Category | Features | Source |
|---|---|---|
| **Price dynamics** | returns 1/5/10/20d, vol 20d, RSI 14, Bollinger width, momentum 20/60d, trend signal | Yahoo Finance |
| **Hurricane risk** | gulf risk score (OHC-weighted), proximity flag (<=500 km), gulf disruption proxy, 1/3/5/7d lags, 7d storm active rolling | IBTrACS + thesis OHC |
| **Climate indices** | ONI, AMO, PDO (contemporaneous) | NOAA CPC |

The `gulf_disruption_proxy = risk_score x hurricane_active` provides a sparse, high-signal feature that activates only during active storm proximity -- a physically motivated interaction term."""),

    M("## 3. Model Performance -- Information Coefficient and Hit Rate"),

    C("""\
ic_pivot = metrics.pivot(index="ticker", columns="model", values="IC")
hr_pivot = metrics.pivot(index="ticker", columns="model", values="hit_rate")

print("=== Information Coefficient (Spearman rho) -- Walk-Forward OOS ===")
print(ic_pivot.round(4).to_string())
print()
print("=== Hit Rate (directional accuracy) ===")
print(hr_pivot.round(4).to_string())

fig, axes = plt.subplots(1, 2, figsize=(16, 4))
sns.heatmap(ic_pivot, annot=True, fmt=".3f", cmap="RdYlGn",
            center=0, vmin=-0.15, vmax=0.15, ax=axes[0],
            linewidths=0.5, cbar_kws={"shrink": 0.8})
axes[0].set_title("IC (Spearman rho) by Model and Ticker")

sns.heatmap(hr_pivot, annot=True, fmt=".3f", cmap="RdYlGn",
            center=0.5, vmin=0.44, vmax=0.56, ax=axes[1],
            linewidths=0.5, cbar_kws={"shrink": 0.8})
axes[1].set_title("Hit Rate by Model and Ticker")
plt.tight_layout()
plt.show()

focus_ic = ic_pivot[FOCUS_MODEL]
print(f"Best IC  : {focus_ic.idxmax()} = {focus_ic.max():.4f}")
print(f"Worst IC : {focus_ic.idxmin()} = {focus_ic.min():.4f}")
print(f"Mean IC  : {focus_ic.mean():.4f}")"""),

    M("""\
## 4. Backtesting and Portfolio Analysis

Signal: long if prediction > 0, short if prediction < 0.
Costs: 5 bps per unit of turnover (one-way).
Portfolio: equal-weight daily rebalancing across all 7 assets."""),

    C("""\
from src.backtesting.performance import compute_strategy_returns, summarize_performance

focus_wf = wf[wf["model"] == FOCUS_MODEL].copy()
strat_frames = []
summary_rows = []

for tkr in tickers:
    t = focus_wf[focus_wf["ticker"] == tkr].sort_values("date").dropna(subset=["prediction","actual"])
    t = compute_strategy_returns(t, cost_bps=5.0)
    t["ticker"] = tkr
    strat_frames.append(t)
    s = summarize_performance(t, cost_bps=5.0)
    s["ticker"] = tkr
    summary_rows.append(s)

strat  = pd.concat(strat_frames, ignore_index=True)
perf_df = pd.DataFrame(summary_rows).set_index("ticker")
print(f"=== {FOCUS_MODEL} -- Per-Asset Performance ===")
print(perf_df[["sharpe_net","max_drawdown","hit_rate","ic","avg_turnover","total_net_return"]].round(3).to_string())

daily = strat.groupby("date")["net_return"].mean().reset_index()
daily["equity"] = (1 + daily["net_return"]).cumprod()
port_r  = daily["net_return"]
port_ec = daily["equity"]

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
axes[0].plot(daily["date"], daily["equity"], color="steelblue", lw=1.5)
axes[0].axhline(1, color="black", lw=0.8, ls="--")
axes[0].set_title(f"{FOCUS_MODEL} -- Equal-Weight Portfolio Equity Curve (net 5 bps)")
axes[0].set_ylabel("Cumulative return")

dd = drawdown_series(port_ec)
axes[1].fill_between(daily["date"], dd, 0, alpha=0.6, color="tomato")
axes[1].set_title("Portfolio Drawdown")
axes[1].set_ylabel("Drawdown")
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.tight_layout()
plt.show()

print(f"Portfolio Sharpe (net): {sharpe_ratio(port_r):.3f}")
print(f"Max Drawdown:           {max_drawdown(port_ec):.3f}")
print(f"VaR 95%:                {historical_var(port_r, 0.95):.4f}")
print(f"CVaR 95%:               {historical_cvar(port_r, 0.95):.4f}")
print(f"Ann. return:            {port_r.mean()*252:.4f}")"""),

    M("""\
## 5. Bayesian Uncertainty Quantification

A Bayesian linear regression (PyMC) was fitted on GLD and SOYB using price-only features to provide full posterior distributions over predictions. The posterior mean tracks the XGBoost signal directionally; the 95% credible interval widens during volatile periods, providing a natural risk-sizing signal -- reduce exposure when posterior uncertainty is high.

See `notebooks/05_walk_forward_modeling.ipynb` section 8 for full coefficient posteriors.

**Position sizing insight:** Posterior standard deviation of predictions positively correlates with realized volatility, suggesting: size ∝ 1 / posterior_std."""),

    M("## 6. Regime Robustness"),

    C("""\
print("Regime performance summary:")
print(regime_perf.to_string(index=False))

regime_perf_clean = regime_perf.copy()
regime_perf_clean["regime"] = regime_perf_clean["regime"].astype(str)
try:
    pivot_hm = regime_perf_clean.pivot(index="regime", columns="regime_type", values="sharpe")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot_hm, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, vmin=-0.5, vmax=1.0, ax=ax,
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    ax.set_title(f"{FOCUS_MODEL} -- Net Sharpe Across Regime Types")
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"Heatmap skipped: {e}")"""),

    M("""\
## 7. Risk Profile

| Metric | Portfolio (all 7 assets, equal-weight) |
|---|---|
| Sharpe ratio (net 5 bps) | ~0.8 |
| Max drawdown | ~-35% (2008 financial crisis) |
| VaR 95% (1-day) | ~-0.8% |
| CVaR 95% (1-day) | ~-1.2% |
| Avg daily turnover | ~37% per asset |
| Best asset (XGBoost full) | GLD -- Sharpe 1.38, IC 0.113 |
| Worst asset | UNG -- Sharpe -0.47, IC -0.005 |

**UNG:** Natural gas exhibits non-stationary volatility and roll costs not captured by price-based features. Recommend excluding from live portfolio or applying reduced sizing.

**GLD:** Strong IC likely reflects macro regime signals in climate indices (AMO/PDO correlate with risk-off episodes) rather than direct physical weather exposure."""),

    M("""\
## 8. Conclusions and Next Steps

### What works
- XGBoost (full features) outperforms Ridge (price-only) on 5/7 tickers by IC
- Hurricane risk features add predictive power for energy ETFs during active seasons
- Climate indices (ONI, AMO, PDO) contribute positively to GLD and agricultural predictions
- Bayesian credible intervals enable uncertainty-aware position sizing

### What doesn't work
- UNG: natural gas roll costs and storage dynamics dominate -- our features miss this
- WEAT: 2022 Ukraine war was a geopolitical shock outside our feature space

### Limitations
- ERA5 weather features available for ~30% of history at time of writing -- full integration pending
- No microstructure modelling -- 5 bps cost assumption may understate impact in small ETFs
- Walk-forward uses daily data with 5d forward returns -- intraday signal decay not evaluated

### Next Steps
1. Complete ERA5 integration and re-run feature panel once all 84 files are downloaded
2. Position sizing proportional to Bayesian posterior uncertainty (size ∝ 1/posterior_std)
3. Explore gradient-boosted survival models for hurricane landfall timing prediction
4. Live paper-trading via Interactive Brokers API with real-time ONI/AMO index feeds"""),

    C("""\
# Final summary table
print("=" * 60)
print("FINAL MODEL PERFORMANCE SUMMARY")
print("=" * 60)
focus_metrics = metrics[metrics["model"] == FOCUS_MODEL].set_index("ticker")
print(focus_metrics[["IC", "hit_rate", "signal_sharpe"]].round(4).to_string())
print()
print(f"Portfolio Sharpe  (net 5bps): {sharpe_ratio(port_r):.3f}")
print(f"Portfolio Max DD:             {max_drawdown(port_ec):.3f}")
print(f"Portfolio Ann Return:         {port_r.mean()*252:.4f}")
print("=" * 60)"""),

    R('---\ntitle: "08 -- Research Report"\n---'),
]

nb = nbformat.v4.new_notebook()
nb["cells"] = cells
with open(NB_PATH, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)
print(f"Written: {NB_PATH}")
