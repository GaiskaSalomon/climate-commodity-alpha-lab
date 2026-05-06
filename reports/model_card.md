# Model Card — Climate-Aware Commodity Return Forecasting

**Model name:** XGBoost Climate-Commodity Forecaster  
**Version:** 1.0  
**Date:** May 2026  
**Author:** Gaiska Salomon  
**Purpose:** Quantitative research / portfolio signal generation

---

## Model Description

An XGBoost gradient-boosted tree regressor trained to predict 5-day forward returns of commodity ETFs using a combination of price momentum features, tropical cyclone risk scores, and large-scale climate teleconnection indices.

**Task:** Regression — predict 5-day forward log return  
**Input:** 24 features (price dynamics × 11, hurricane risk × 10, climate indices × 3)  
**Output:** Point prediction of 5-day forward return per ticker per day  

**Framework:** scikit-learn compatible via `make_xgb_regressor()` in `src/models/tree_models.py`

---

## Training Details

| Parameter | Value |
|---|---|
| Algorithm | XGBoost (gradient-boosted trees) |
| n_estimators | 300 |
| learning_rate | 0.05 |
| max_depth | 4 |
| subsample | 0.8 |
| reg_alpha (L1) | 0.1 |
| Validation | Walk-forward temporal (500/21/21) |
| Target | `future_return_5d` |
| Missing values | Dropped rows with NaN in features or target |

---

## Feature Set

### Price dynamics (11 features)
- `return_1d`, `return_5d`, `return_10d`, `return_20d`
- `volatility_20d`
- `rsi_14`
- `bollinger_width`
- `momentum_20d`, `momentum_60d`
- `trend_signal`
- `cross_sectional_rank`

### Hurricane risk (10 features, region-matched per ticker)
- `hurricane_risk_score` — Vmax / (1 + distance_km), weighted by OHC
- `hurricane_active` — binary: storm within 500 km
- `gulf_disruption_proxy` — risk_score × hurricane_active (energy assets)
- `hurricane_risk_lag_1d`, `_lag_3d`, `_lag_5d`, `_lag_7d`
- `storm_active_7d` — rolling 7-day active window
- `max_risk_5d`, `max_risk_14d` — rolling max risk exposure

### Climate indices (3 features)
- `oni` — Oceanic Nino Index (ENSO)
- `amo` — Atlantic Multidecadal Oscillation
- `pdo` — Pacific Decadal Oscillation

---

## Performance Summary

| Ticker | IC (Spearman) | Hit Rate | Sharpe (net 5 bps) |
|---|---|---|---|
| GLD | 0.113 | 0.550 | 1.384 |
| SOYB | 0.121 | 0.534 | 1.222 |
| DBA | 0.061 | 0.532 | 0.602 |
| USO | 0.021 | 0.513 | 0.425 |
| CORN | 0.046 | 0.504 | 0.310 |
| WEAT | 0.029 | 0.504 | -0.115 |
| UNG | -0.005 | 0.498 | -0.474 |

**Portfolio (equal-weight):** Sharpe ~0.82, Max DD ~-35%

---

## Intended Use

- **Intended users:** Quantitative analysts and portfolio managers at commodity-focused funds
- **Intended use cases:** Signal generation for medium-frequency (weekly rebalancing) commodity strategies; regime-conditional allocation overlays
- **Out-of-scope use cases:** Intraday trading, options pricing, markets outside the 7-ETF universe, periods of market stress where model was shown to underperform

---

## Limitations

1. **UNG fails** — natural gas not forecastable with this feature set; exclude from production deployment
2. **ERA5 integration partial** — weather anomaly features not yet fully integrated; IC may improve with full coverage
3. **Crisis regimes** — model degrades significantly in high-volatility regimes (2008, 2020)
4. **No macro regime conditioning** — FOMC decisions, rate shocks, and geopolitical events are not in the feature set
5. **Temporal distribution shift** — model trained on 2007-2024 data; ENSO patterns may be affected by ongoing climate change

---

## Ethical Considerations

- This model is intended for use in regulated financial markets by qualified institutional investors
- No personal data is used; all inputs are market prices and publicly available climate indices
- The model does not make investment recommendations; it generates probabilistic signals that require human oversight

---

## Bayesian Extension

A companion Bayesian linear regression (`src/models/bayesian_models.py`) provides:
- Full posterior distributions over predictions (mean + standard deviation)
- Coefficient credible intervals (94% HDI via ArviZ)
- Uncertainty-aware position sizing: size proportional to 1 / posterior_std

The Bayesian module is fitted on a static train/test split (75%/25%) rather than walk-forward due to MCMC computation time.

---

## Reproducibility

```bash
# Install dependencies
pip install -r requirements.txt

# Download price data
python scripts/download_prices.py

# Download ERA5 weather data (requires CDS API key)
python scripts/download_era5.py

# Run notebooks in order
jupyter nbconvert --to notebook --execute notebooks/01_price_data_eda.ipynb
...
jupyter nbconvert --to notebook --execute notebooks/08_research_report.ipynb
```

See `README.md` for full setup instructions.
