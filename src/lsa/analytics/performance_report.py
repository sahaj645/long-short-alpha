"""
Performance Report — composite views built from `metrics` and `trade_statistics`.

This module provides the higher-level analytics that aggregate single-number
metrics, time-grouped tables, and trade-level summaries into a single
`PerformanceReport` object suitable for committee decks, monitoring
dashboards, or post-mortem reviews.

Functions in this module:
  - `monthly_return_table` — year × month grid of compounded monthly returns
  - `yearly_summary` — per-year Sharpe, return, vol, max DD
  - `rolling_sharpe` — rolling-window Sharpe time series
  - `benchmark_comparison` — side-by-side strategy vs benchmark metrics
  - `generate_performance_report` — bundle of everything

All functions are pure and operate on a daily-return Series (decimal).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .metrics import (
    DEFAULT_FREQ,
    PerformanceMetrics,
    _equity_from_returns,
    compute_all_metrics,
    drawdown_series,
    sharpe_ratio,
    max_drawdown,
)
from .trade_statistics import (
    TradeStatistics,
    compute_trade_statistics,
    compute_turnover_statistics,
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Dataclass
# -----------------------------------------------------------------------------


@dataclass
class PerformanceReport:
    """Composite performance report."""

    summary_stats: PerformanceMetrics
    equity_curve: pd.Series
    drawdown_series: pd.Series
    rolling_sharpe: pd.Series
    monthly_returns_table: pd.DataFrame
    yearly_summary: pd.DataFrame
    benchmark_comparison: Optional[pd.DataFrame] = None
    trade_stats: Optional[TradeStatistics] = None
    metadata: dict = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Time-grouped tables
# -----------------------------------------------------------------------------


def _to_period_returns(returns: pd.Series, rule: str) -> pd.Series:
    """Resample daily returns to a higher-frequency period via compounded product."""
    r = returns.dropna()
    if r.empty:
        return r
    if not isinstance(r.index, pd.DatetimeIndex):
        raise TypeError("returns Series must have a DatetimeIndex for resampling")
    return (1.0 + r).resample(rule).prod() - 1.0


def monthly_return_table(returns: pd.Series) -> pd.DataFrame:
    """Year × month pivot of compounded monthly returns, plus YTD column.

    Returns an empty DataFrame on empty input.
    """
    monthly = _to_period_returns(returns, "ME")
    if monthly.empty:
        return pd.DataFrame()

    table = pd.DataFrame({
        "year": monthly.index.year,
        "month": monthly.index.month,
        "ret": monthly.values,
    })
    pivot = table.pivot(index="year", columns="month", values="ret")
    pivot.columns = [pd.Timestamp(month=m, day=1, year=2000).strftime("%b")
                     for m in pivot.columns]

    yearly = _to_period_returns(returns, "YE")
    if not yearly.empty:
        ytd = pd.Series(yearly.values, index=yearly.index.year, name="YTD")
        pivot = pivot.join(ytd, how="left")
    return pivot


def yearly_summary(returns: pd.Series, freq: int = DEFAULT_FREQ) -> pd.DataFrame:
    """Per-year summary: total return, volatility, Sharpe, max DD, n_days."""
    r = returns.dropna()
    if r.empty:
        return pd.DataFrame()
    if not isinstance(r.index, pd.DatetimeIndex):
        raise TypeError("returns Series must have a DatetimeIndex")

    rows = []
    for year, ry in r.groupby(r.index.year):
        equity = (1.0 + ry).cumprod()
        sd = float(ry.std(ddof=1)) if len(ry) > 1 else 0.0
        rows.append({
            "year": int(year),
            "n_days": int(len(ry)),
            "total_return": float(equity.iloc[-1] - 1.0) if len(equity) else float("nan"),
            "ann_volatility": float(sd * np.sqrt(freq)) if sd > 0 else float("nan"),
            "sharpe": (
                float((ry.mean() / sd) * np.sqrt(freq))
                if sd > 0 else float("nan")
            ),
            "max_drawdown": float((equity / equity.cummax() - 1.0).min())
                              if len(equity) else float("nan"),
            "hit_rate": float((ry > 0).mean()) if len(ry) else float("nan"),
        })
    return pd.DataFrame(rows).set_index("year")


# -----------------------------------------------------------------------------
# Rolling Sharpe
# -----------------------------------------------------------------------------


def rolling_sharpe(
    returns: pd.Series,
    *,
    window: int = 252,
    freq: int = DEFAULT_FREQ,
    min_periods: Optional[int] = None,
) -> pd.Series:
    """Rolling-window Sharpe time series.

    Each point is `mean(window) / std(window) × √freq`. With defaults, a
    252-day window provides a one-year Sharpe estimate updated daily.
    """
    r = returns.dropna()
    if r.empty:
        return pd.Series(dtype="float64", name="rolling_sharpe")
    if min_periods is None:
        min_periods = max(window // 2, 2)
    rolling_mean = r.rolling(window, min_periods=min_periods).mean()
    rolling_std = r.rolling(window, min_periods=min_periods).std(ddof=1)
    # Guard against zero std
    out = (rolling_mean / rolling_std.replace(0.0, np.nan)) * np.sqrt(freq)
    return out.rename("rolling_sharpe")


# -----------------------------------------------------------------------------
# Benchmark comparison
# -----------------------------------------------------------------------------


def benchmark_comparison(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    *,
    risk_free_rate: float = 0.0,
    freq: int = DEFAULT_FREQ,
) -> pd.DataFrame:
    """Side-by-side metrics for strategy and benchmark, plus active-return stats.

    Returns a DataFrame with rows = metric names and columns =
    ['strategy', 'benchmark', 'active'].
    """
    s = strategy_returns.dropna()
    b = benchmark_returns.dropna()
    aligned = pd.concat([s.rename("s"), b.rename("b")], axis=1, join="inner").dropna()
    if aligned.empty:
        return pd.DataFrame()

    sm = compute_all_metrics(aligned["s"], risk_free_rate=risk_free_rate, freq=freq)
    bm = compute_all_metrics(aligned["b"], risk_free_rate=risk_free_rate, freq=freq)
    active = aligned["s"] - aligned["b"]
    active_sharpe = (
        float((active.mean() / active.std(ddof=1)) * np.sqrt(freq))
        if active.std(ddof=1) > 0 else float("nan")
    )

    return pd.DataFrame({
        "strategy": [
            sm.cagr, sm.annualized_volatility, sm.sharpe_ratio,
            sm.max_drawdown, sm.max_drawdown_duration_days, sm.hit_rate,
            sm.skewness, sm.kurtosis,
        ],
        "benchmark": [
            bm.cagr, bm.annualized_volatility, bm.sharpe_ratio,
            bm.max_drawdown, bm.max_drawdown_duration_days, bm.hit_rate,
            bm.skewness, bm.kurtosis,
        ],
        "active": [
            float(active.mean() * freq),
            float(active.std(ddof=1) * np.sqrt(freq)) if active.std(ddof=1) > 0 else float("nan"),
            active_sharpe,
            float("nan"),
            int(0),
            float((active > 0).mean()),
            float(active.skew()) if len(active) >= 3 else float("nan"),
            float(active.kurtosis()) if len(active) >= 4 else float("nan"),
        ],
    }, index=[
        "CAGR", "Annualized Vol", "Sharpe Ratio", "Max Drawdown",
        "Max DD Duration", "Hit Rate", "Skewness", "Kurtosis",
    ])


# -----------------------------------------------------------------------------
# Composite report
# -----------------------------------------------------------------------------


def generate_performance_report(
    returns: pd.Series,
    *,
    benchmark_returns: Optional[pd.Series] = None,
    episodes_with_pnl: Optional[pd.DataFrame] = None,
    turnover_series: Optional[pd.Series] = None,
    risk_free_rate: float = 0.0,
    freq: int = DEFAULT_FREQ,
    rolling_window: int = 252,
    metadata: Optional[dict] = None,
) -> PerformanceReport:
    """Bundle every analytics view into one structured report.

    Parameters
    ----------
    returns
        Daily decimal return Series indexed by date.
    benchmark_returns
        Optional benchmark for IR and side-by-side comparison.
    episodes_with_pnl
        Output of `trade_statistics.compute_episode_pnl`. Required to populate
        the report's `trade_stats` field; pass None to skip trade-level stats.
    turnover_series
        Daily one-way turnover series. Folded into `trade_stats` if both that
        and `episodes_with_pnl` are provided; otherwise computed standalone
        into the metadata.
    """
    summary = compute_all_metrics(
        returns,
        benchmark_returns=benchmark_returns,
        risk_free_rate=risk_free_rate,
        freq=freq,
    )

    equity = _equity_from_returns(returns).rename("equity")
    dd = drawdown_series(returns)
    rs = rolling_sharpe(returns, window=rolling_window, freq=freq)
    monthly_table = monthly_return_table(returns)
    yearly_table = yearly_summary(returns, freq=freq)

    bench_comp = None
    if benchmark_returns is not None:
        bench_comp = benchmark_comparison(
            returns, benchmark_returns,
            risk_free_rate=risk_free_rate, freq=freq,
        )

    trade_stats = None
    if episodes_with_pnl is not None:
        trade_stats = compute_trade_statistics(
            episodes_with_pnl, turnover_series=turnover_series, freq=freq,
        )

    meta = dict(metadata or {})
    if turnover_series is not None and trade_stats is None:
        meta["turnover"] = compute_turnover_statistics(turnover_series, freq=freq)

    return PerformanceReport(
        summary_stats=summary,
        equity_curve=equity,
        drawdown_series=dd,
        rolling_sharpe=rs,
        monthly_returns_table=monthly_table,
        yearly_summary=yearly_table,
        benchmark_comparison=bench_comp,
        trade_stats=trade_stats,
        metadata=meta,
    )
