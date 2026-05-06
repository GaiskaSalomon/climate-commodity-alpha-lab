"""
Reusable plotting functions for the climate-commodity-alpha-lab pipeline.

All functions return (fig, ax) or (fig, axes) so callers can save or further
customize the figure before displaying.
"""
from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

_PALETTE = {
    "equity":   "steelblue",
    "drawdown": "tomato",
    "positive": "steelblue",
    "negative": "tomato",
    "neutral":  "gray",
}


# ---------------------------------------------------------------------------
# Equity curve + drawdown
# ---------------------------------------------------------------------------

def plot_equity_curve(
    dates: pd.Series,
    equity: pd.Series,
    drawdown: pd.Series | None = None,
    title: str = "Portfolio Equity Curve",
    label: str = "Strategy",
    figsize: tuple = (14, 7),
) -> tuple:
    """
    Plot cumulative equity curve with optional drawdown panel.

    Parameters
    ----------
    dates:    date index aligned to equity/drawdown series
    equity:   cumulative return series (starts at 1.0)
    drawdown: optional drawdown series (values <= 0); adds a subplot below
    """
    n_rows = 2 if drawdown is not None else 1
    fig, axes = plt.subplots(n_rows, 1, figsize=figsize, sharex=True)
    if n_rows == 1:
        axes = [axes]

    ax0 = axes[0]
    ax0.plot(dates, equity, color=_PALETTE["equity"], lw=1.5, label=label)
    ax0.axhline(1, color="black", lw=0.8, ls="--", alpha=0.5)
    ax0.set_title(title, fontsize=12)
    ax0.set_ylabel("Cumulative return")
    ax0.legend(fontsize=9)
    ax0.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    if drawdown is not None:
        ax1 = axes[1]
        ax1.fill_between(dates, drawdown, 0, alpha=0.65, color=_PALETTE["drawdown"])
        ax1.axhline(0, color="black", lw=0.8)
        ax1.set_title("Drawdown", fontsize=11)
        ax1.set_ylabel("Drawdown")
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# IC heatmap (model × ticker)
# ---------------------------------------------------------------------------

def plot_ic_heatmap(
    ic_df: pd.DataFrame,
    title: str = "Information Coefficient (Spearman rho)",
    vmin: float = -0.15,
    vmax: float = 0.15,
    figsize: tuple = (12, 4),
) -> tuple:
    """
    Plot a heatmap of Information Coefficient values.

    Parameters
    ----------
    ic_df: DataFrame with tickers as index, models as columns, IC values as cells.
           (output of metrics.pivot(index='ticker', columns='model', values='IC'))
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        ic_df,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        center=0,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("")
    plt.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Monthly IC bar chart
# ---------------------------------------------------------------------------

def plot_monthly_ic(
    monthly_ic: pd.Series,
    ticker: str,
    model: str = "XGBoost (full)",
    figsize: tuple = (14, 3),
) -> tuple:
    """
    Bar chart of monthly Information Coefficient values.

    Parameters
    ----------
    monthly_ic: Series with datetime index (monthly frequency) and IC values.
    """
    fig, ax = plt.subplots(figsize=figsize)
    colors = [_PALETTE["positive"] if v >= 0 else _PALETTE["negative"] for v in monthly_ic.values]
    ax.bar(monthly_ic.index, monthly_ic.values, color=colors, alpha=0.8, width=20)
    ax.axhline(0, color="black", lw=0.8)
    mean_ic = monthly_ic.mean()
    ax.axhline(mean_ic, color="navy", lw=1.2, ls="--", label=f"Mean IC = {mean_ic:.3f}")
    ax.set_title(f"{ticker} — Monthly IC ({model})")
    ax.set_ylabel("IC (Spearman rho)")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Feature importance bar chart
# ---------------------------------------------------------------------------

def plot_feature_importance(
    importances: pd.Series,
    price_features: list[str],
    climate_features: list[str],
    title: str = "XGBoost Feature Importance (mean across tickers)",
    figsize: tuple = (9, 6),
) -> tuple:
    """
    Horizontal bar chart for feature importances with color coding by category.

    Parameters
    ----------
    importances:     Series with feature names as index, importance values as values.
                     Should already be sorted ascending for a left-to-right reading.
    price_features:  Feature names classified as price/momentum.
    climate_features: Feature names classified as climate indices.
                     Anything else is treated as hurricane/weather.
    """
    colors = []
    for f in importances.index:
        if f in price_features:
            colors.append(_PALETTE["positive"])       # blue
        elif f in climate_features:
            colors.append("darkorange")
        else:
            colors.append(_PALETTE["drawdown"])       # red/tomato

    fig, ax = plt.subplots(figsize=figsize)
    importances.plot.barh(ax=ax, color=colors, alpha=0.85)
    ax.set_xlabel("Mean importance (gain)")
    ax.set_title(title, fontsize=11)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=_PALETTE["positive"],  label="Price features"),
        Patch(facecolor=_PALETTE["drawdown"],  label="Hurricane / weather"),
        Patch(facecolor="darkorange",           label="Climate indices"),
    ]
    ax.legend(handles=legend_elements, fontsize=9)
    plt.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Regime performance comparison
# ---------------------------------------------------------------------------

def plot_regime_sharpe(
    regime_df: pd.DataFrame,
    regime_col: str,
    sharpe_col: str = "sharpe",
    title: str = "Net Sharpe by Regime",
    figsize: tuple = (10, 4),
) -> tuple:
    """
    Bar chart comparing Sharpe ratios across regimes.

    Parameters
    ----------
    regime_df:  DataFrame with one row per regime, containing regime labels and Sharpe values.
    regime_col: Column name for regime labels.
    sharpe_col: Column name for Sharpe ratio values.
    """
    fig, ax = plt.subplots(figsize=figsize)
    values = regime_df[sharpe_col].values
    labels = regime_df[regime_col].astype(str).values
    colors = [_PALETTE["positive"] if v >= 0 else _PALETTE["negative"] for v in values]
    ax.bar(labels, values, color=colors, alpha=0.8)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("Net Sharpe ratio")
    ax.set_xlabel(regime_col)
    plt.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Return distribution by regime
# ---------------------------------------------------------------------------

def plot_return_distributions(
    data: pd.DataFrame,
    return_col: str,
    regime_col: str,
    title: str = "Return Distribution by Regime",
    figsize: tuple = (14, 4),
) -> tuple:
    """
    One histogram per regime, laid out horizontally.
    """
    regimes = sorted(data[regime_col].dropna().unique())
    n = len(regimes)
    cmap = plt.cm.get_cmap("tab10", n)
    fig, axes = plt.subplots(1, n, figsize=figsize, sharey=True)
    if n == 1:
        axes = [axes]

    for ax, regime, color_idx in zip(axes, regimes, range(n)):
        r = data[data[regime_col] == regime][return_col].dropna()
        ax.hist(r, bins=60, color=cmap(color_idx), alpha=0.8, edgecolor="none")
        mean_sh = r.mean() * np.sqrt(252) / r.std() if r.std() > 0 else np.nan
        ax.set_title(f"{regime}\nSharpe={mean_sh:.2f}  n={len(r):,}")
        ax.set_xlabel("Net return")
        ax.axvline(0, color="black", lw=0.8)

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    return fig, axes
