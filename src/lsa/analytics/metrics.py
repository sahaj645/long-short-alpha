"""
Performance metrics — single-number summaries of a return series.

Every function in this module:
  - Takes a daily return Series (decimal returns).
  - Drops NaN observations internally before computation.
  - Returns NaN on degenerate inputs (empty series, zero variance) rather
    than raising. Contract violations (missing columns, wrong dtype) still
    raise.

The default annualization factor is 252 trading days per year. Pass a
different `freq` if working with weekly / monthly returns.

Conventions
-----------
- Returns are simple returns: r_t = (P_t - P_{t-1}) / P_{t-1}.
- Excess return = r - rf/freq.
- Drawdown is computed on the equity curve, not directly on returns,
  because cumulative-product compounding is the correct accounting.
- Kurtosis is **excess** kurtosis (Gaussian = 0), matching pandas default.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


DEFAULT_FREQ: int = 252


# -----------------------------------------------------------------------------
# Dataclass
# -----------------------------------------------------------------------------


@dataclass
class PerformanceMetrics:
    """All headline metrics from `compute_all_metrics`."""

    n_periods: int = 0
    n_periods_dropped_na: int = 0

    # Return metrics
    cagr: float = float("nan")
    annualized_arithmetic_return: float = float("nan")
    annualized_geometric_return: float = float("nan")

    # Risk metrics
    annualized_volatility: float = float("nan")
    max_drawdown: float = float("nan")
    max_drawdown_duration_days: int = 0

    # Risk-adjusted
    sharpe_ratio: float = float("nan")
    sortino_ratio: float = float("nan")
    calmar_ratio: float = float("nan")
    information_ratio: Optional[float] = None

    # Distribution
    skewness: float = float("nan")
    kurtosis: float = float("nan")
    hit_rate: float = float("nan")

    # Metadata
    freq: int = DEFAULT_FREQ
    risk_free_rate: float = 0.0


# -----------------------------------------------------------------------------
# Helpers (private)
# -----------------------------------------------------------------------------


def _clean(returns: pd.Series) -> pd.Series:
    """Drop NaN and infinity. Return an empty Series on bad input."""
    if not isinstance(returns, pd.Series):
        raise TypeError(f"Expected pd.Series, got {type(returns).__name__}")
    if returns.empty:
        return returns
    r = returns.replace([np.inf, -np.inf], np.nan).dropna()
    return r


def _equity_from_returns(returns: pd.Series) -> pd.Series:
    """Cumulative-product equity curve from a clean return series."""
    r = _clean(returns)
    if r.empty:
        return pd.Series(dtype="float64")
    return (1.0 + r).cumprod()


# -----------------------------------------------------------------------------
# Return metrics
# -----------------------------------------------------------------------------


def annualized_arithmetic_return(returns: pd.Series, freq: int = DEFAULT_FREQ) -> float:
    """Arithmetic mean × annualization factor.

    Faster but biased upward versus geometric annualization at high volatility.
    """
    r = _clean(returns)
    if r.empty:
        return float("nan")
    return float(r.mean() * freq)


def annualized_geometric_return(returns: pd.Series, freq: int = DEFAULT_FREQ) -> float:
    """Geometric annualization: (1 + mean(r))^freq - 1.

    Differs from CAGR in that it uses the mean return, not the cumulative
    compound. Useful when the holding period < 1 year.
    """
    r = _clean(returns)
    if r.empty:
        return float("nan")
    mean_r = float(r.mean())
    if mean_r <= -1.0:
        return float("nan")
    return float((1.0 + mean_r) ** freq - 1.0)


def cagr(returns: pd.Series, freq: int = DEFAULT_FREQ) -> float:
    """Compound annual growth rate from cumulative compounded return.

    (V_end / V_start)^(freq / n) - 1. Returns NaN if the cumulative product
    is non-positive (full loss) or the time horizon is too short to annualize.
    """
    r = _clean(returns)
    if len(r) < 2:
        return float("nan")
    equity = _equity_from_returns(r)
    total = float(equity.iloc[-1])
    if total <= 0.0:
        return float("nan")
    years = len(r) / freq
    if years <= 0.0:
        return float("nan")
    return float(total ** (1.0 / years) - 1.0)


# -----------------------------------------------------------------------------
# Risk metrics
# -----------------------------------------------------------------------------


def annualized_volatility(returns: pd.Series, freq: int = DEFAULT_FREQ) -> float:
    """Sample standard deviation × √freq."""
    r = _clean(returns)
    if len(r) < 2:
        return float("nan")
    return float(r.std(ddof=1) * np.sqrt(freq))


def drawdown_series(returns_or_equity: pd.Series, *, from_returns: bool = True) -> pd.Series:
    """Per-period drawdown from running maximum.

    Pass `from_returns=False` if `returns_or_equity` is already an equity curve.
    """
    if from_returns:
        equity = _equity_from_returns(returns_or_equity)
    else:
        equity = returns_or_equity.dropna()
    if equity.empty:
        return pd.Series(dtype="float64", name="drawdown")
    running_max = equity.cummax()
    return ((equity - running_max) / running_max).rename("drawdown")


def max_drawdown(returns: pd.Series) -> float:
    """Maximum (most negative) drawdown over the equity curve.

    Returns 0.0 when the strategy never dips below its running maximum.
    Returns NaN on empty input.
    """
    dd = drawdown_series(returns)
    if dd.empty:
        return float("nan")
    return float(dd.min())


def max_drawdown_duration(returns: pd.Series) -> int:
    """Longest contiguous run of underwater periods.

    Counts the number of consecutive periods where the equity is strictly
    below its prior maximum. Returns 0 if the strategy never goes underwater
    or input is empty.
    """
    equity = _equity_from_returns(returns)
    if equity.empty or len(equity) < 2:
        return 0
    running_max = equity.cummax()
    underwater = (equity < running_max)
    if not underwater.any():
        return 0
    # Group consecutive runs by detecting boundary changes
    run_id = (underwater != underwater.shift()).cumsum()
    run_lengths = underwater.groupby(run_id).sum()
    return int(run_lengths.max())


# -----------------------------------------------------------------------------
# Risk-adjusted metrics
# -----------------------------------------------------------------------------


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    freq: int = DEFAULT_FREQ,
) -> float:
    """Annualized Sharpe: mean(excess) / std(excess) × √freq.

    `risk_free_rate` is annualized; per-period rate is `rf / freq`.
    Returns NaN if excess returns have zero variance or input is empty.
    """
    r = _clean(returns)
    if len(r) < 2:
        return float("nan")
    excess = r - risk_free_rate / freq
    sd = float(excess.std(ddof=1))
    if sd <= 0 or not np.isfinite(sd):
        return float("nan")
    return float((excess.mean() / sd) * np.sqrt(freq))


def sortino_ratio(
    returns: pd.Series,
    minimum_acceptable_return: float = 0.0,
    freq: int = DEFAULT_FREQ,
) -> float:
    """Annualized Sortino: mean(excess) / downside_deviation × √freq.

    Downside deviation = sqrt(mean(min(0, excess)²)). MAR is annualized.

    Edge cases:
      - All excess returns non-negative → returns +inf if mean > 0 else NaN.
      - Empty / single observation → NaN.
    """
    r = _clean(returns)
    if len(r) < 2:
        return float("nan")
    excess = r - minimum_acceptable_return / freq
    downside = excess.clip(upper=0.0)
    dd_var = float((downside ** 2).mean())
    if dd_var <= 0:
        return float("inf") if excess.mean() > 0 else float("nan")
    dd_dev = np.sqrt(dd_var)
    return float((excess.mean() / dd_dev) * np.sqrt(freq))


def calmar_ratio(returns: pd.Series, freq: int = DEFAULT_FREQ) -> float:
    """CAGR / |Max Drawdown|. Returns NaN when MDD is zero or NaN."""
    growth = cagr(returns, freq=freq)
    mdd = max_drawdown(returns)
    if not np.isfinite(growth) or not np.isfinite(mdd) or mdd >= 0:
        return float("nan")
    return float(growth / abs(mdd))


def information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    freq: int = DEFAULT_FREQ,
) -> float:
    """Annualized IR of active return vs benchmark.

    Aligns the two series on common dates (inner join) before computing.
    Returns NaN if active returns have zero variance.
    """
    r = _clean(returns)
    b = _clean(benchmark_returns)
    if r.empty or b.empty:
        return float("nan")
    aligned = pd.concat([r.rename("r"), b.rename("b")], axis=1, join="inner").dropna()
    if len(aligned) < 2:
        return float("nan")
    active = aligned["r"] - aligned["b"]
    sd = float(active.std(ddof=1))
    if sd <= 0 or not np.isfinite(sd):
        return float("nan")
    return float((active.mean() / sd) * np.sqrt(freq))


# -----------------------------------------------------------------------------
# Distribution metrics
# -----------------------------------------------------------------------------


def skewness(returns: pd.Series) -> float:
    """Sample skewness (unbiased Fisher-Pearson). Returns NaN if n < 3."""
    r = _clean(returns)
    if len(r) < 3:
        return float("nan")
    return float(r.skew())


def kurtosis(returns: pd.Series) -> float:
    """Sample EXCESS kurtosis (Gaussian = 0). Returns NaN if n < 4."""
    r = _clean(returns)
    if len(r) < 4:
        return float("nan")
    return float(r.kurtosis())


def hit_rate(returns: pd.Series) -> float:
    """Fraction of periods with strictly positive return."""
    r = _clean(returns)
    if r.empty:
        return float("nan")
    return float((r > 0).mean())


# -----------------------------------------------------------------------------
# Composite
# -----------------------------------------------------------------------------


def compute_all_metrics(
    returns: pd.Series,
    *,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_rate: float = 0.0,
    freq: int = DEFAULT_FREQ,
) -> PerformanceMetrics:
    """Compute every metric in one pass and return a `PerformanceMetrics`."""
    if not isinstance(returns, pd.Series):
        raise TypeError(f"Expected pd.Series for returns, got {type(returns).__name__}")

    n_input = len(returns)
    clean = _clean(returns)
    n_clean = len(clean)

    ir = None
    if benchmark_returns is not None:
        ir = information_ratio(clean, benchmark_returns, freq=freq)

    return PerformanceMetrics(
        n_periods=n_clean,
        n_periods_dropped_na=n_input - n_clean,
        cagr=cagr(clean, freq=freq),
        annualized_arithmetic_return=annualized_arithmetic_return(clean, freq=freq),
        annualized_geometric_return=annualized_geometric_return(clean, freq=freq),
        annualized_volatility=annualized_volatility(clean, freq=freq),
        max_drawdown=max_drawdown(clean),
        max_drawdown_duration_days=max_drawdown_duration(clean),
        sharpe_ratio=sharpe_ratio(clean, risk_free_rate=risk_free_rate, freq=freq),
        sortino_ratio=sortino_ratio(clean, freq=freq),
        calmar_ratio=calmar_ratio(clean, freq=freq),
        information_ratio=ir,
        skewness=skewness(clean),
        kurtosis=kurtosis(clean),
        hit_rate=hit_rate(clean),
        freq=freq,
        risk_free_rate=risk_free_rate,
    )
