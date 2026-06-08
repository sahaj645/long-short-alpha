"""
Residual momentum baseline strategy.

This module implements the within-GICS-Sub-Industry residual momentum strategy
that serves as the comparison baseline against the H6 (Sub-Industry
Leader-Follower) work. The mechanism is the residual-momentum effect
documented in Blitz, Huij & Martens (2011) and Asness et al.: after removing
sector-ETF beta, the cross-section of remaining (idiosyncratic) momentum
yields a cleaner factor than raw return momentum.

Pipeline
--------
1. Map each name to its GICS-Sector SPDR ETF.
2. Estimate rolling beta of name return against matched sector ETF return.
3. Residual return = name_return − beta × etf_return.
4. Cumulative residual over a 6-month lookback, skipping the most recent
   month (standard skip-month momentum construction).
5. Cross-sectional percentile rank within Sub-Industry per date.
6. Long the top quintile, short the bottom quintile, equal-weighted within
   each side and normalized to gross 2.0 (long +1, short -1), giving a
   dollar-neutral L/S portfolio by construction.
7. Strategy returns realized on date t+1 using weights set at close of t.

Public API
----------
- sector_to_etf_mapping()
- compute_rolling_betas(...)
- compute_residual_returns(...)
- compute_residual_momentum(...)
- rank_within_subindustry(...)
- build_long_short_weights(...)
- verify_dollar_neutrality(...)
- compute_strategy_returns(...)
- build_residual_momentum_strategy(...)   # orchestrator
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Default sector ETF mapping
# -----------------------------------------------------------------------------

# GICS Level-1 Sector → State Street SPDR Sector ETF. XLC launched 2018-06-19;
# rows whose date precedes the ETF's first valid observation receive NaN beta
# and are dropped from the signal — conservative handling without an explicit
# pre-launch proxy.
DEFAULT_SECTOR_TO_ETF: dict[str, str] = {
    "Communication Services": "XLC",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Industrials": "XLI",
    "Information Technology": "XLK",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
}


def sector_to_etf_mapping() -> dict[str, str]:
    """Return a copy of the standard GICS Sector → SPDR ETF mapping."""
    return DEFAULT_SECTOR_TO_ETF.copy()


# -----------------------------------------------------------------------------
# Rolling beta and residual return
# -----------------------------------------------------------------------------


def compute_rolling_betas(
    panel: pd.DataFrame,
    etf_returns_wide: pd.DataFrame,
    sector_to_etf: Mapping[str, str] | None = None,
    *,
    id_col: str = "ID",
    date_col: str = "DATE",
    return_col: str = "ret",
    sector_col: str = "GICS_Sector",
    window: int = 126,
    min_periods: int = 63,
    out_col: str = "beta",
) -> pd.DataFrame:
    """Rolling beta of each name's return against its matched sector ETF.

    Uses the closed-form β = Cov(r, m) / Var(m), computed via pandas'
    `rolling.cov(other_series)` which vectorizes across all columns
    (i.e., all names in a sector simultaneously). Significantly faster than
    a per-name OLS loop and numerically equivalent.

    Parameters
    ----------
    panel
        Long-form return panel. Must contain `id_col`, `date_col`,
        `return_col`, `sector_col`.
    etf_returns_wide
        Wide DataFrame indexed by date, columns are ETF tickers. Values are
        daily returns (typically Close.pct_change() or Adj_Close.pct_change()).
    sector_to_etf
        Optional override of the default GICS → SPDR mapping.
    window, min_periods
        Rolling window parameters (default 126 / 63 ≈ 6 months / 3 months).
    out_col
        Name of the output beta column.

    Returns
    -------
    pd.DataFrame
        Long-form with columns [id_col, date_col, out_col]. NaN beta is
        returned for (a) sectors with no ETF mapping, (b) dates where the
        rolling window is not yet warm.
    """
    mapping = dict(sector_to_etf) if sector_to_etf is not None else DEFAULT_SECTOR_TO_ETF

    out_chunks: list[pd.DataFrame] = []
    for sector, etf in mapping.items():
        if etf not in etf_returns_wide.columns:
            logger.warning("ETF %s not in ETF returns; skipping sector %s", etf, sector)
            continue

        sub = panel.loc[panel[sector_col] == sector, [id_col, date_col, return_col]]
        if sub.empty:
            continue

        # Cross-tier migrations can produce duplicate (ID, DATE) rows in the
        # merged panel (e.g., simultaneous-membership edge cases). Dedupe
        # before pivoting; returns are identical across duplicate rows.
        sub = sub.drop_duplicates(subset=[id_col, date_col], keep="first")

        wide = sub.pivot(index=date_col, columns=id_col, values=return_col)
        m = etf_returns_wide[etf]
        wide, m_aligned = wide.align(m, axis=0, join="left")

        cov = wide.rolling(window, min_periods=min_periods).cov(m_aligned)
        var = m_aligned.rolling(window, min_periods=min_periods).var()
        beta_wide = cov.div(var, axis=0)

        # PIT-safety: pandas .rolling(window) ends AT the current observation
        # (inclusive). Without this shift, residual_t would use beta_t which
        # itself was estimated using return_t, creating a subtle lookahead.
        # Lag by 1 day so residual_t = ret_t - beta_{t-1} * etf_t.
        beta_wide = beta_wide.shift(1)

        long_form = (beta_wide.stack().rename(out_col).reset_index())
        out_chunks.append(long_form)

    if not out_chunks:
        return pd.DataFrame(columns=[id_col, date_col, out_col])

    betas = pd.concat(out_chunks, ignore_index=True)
    logger.info("Computed %d rolling beta observations across %d sectors",
                len(betas), len(out_chunks))
    return betas


def compute_residual_returns(
    panel: pd.DataFrame,
    betas: pd.DataFrame,
    etf_returns_wide: pd.DataFrame,
    sector_to_etf: Mapping[str, str] | None = None,
    *,
    id_col: str = "ID",
    date_col: str = "DATE",
    return_col: str = "ret",
    sector_col: str = "GICS_Sector",
    beta_col: str = "beta",
    out_col: str = "residual",
) -> pd.DataFrame:
    """Compute residual returns: r_resid = r_name − beta × r_etf.

    Joins beta (per ID, date) and the matched sector-ETF return onto the panel,
    then subtracts. Rows without a beta (sector not mapped, rolling window not
    warm) produce NaN residuals.
    """
    mapping = dict(sector_to_etf) if sector_to_etf is not None else DEFAULT_SECTOR_TO_ETF

    out = panel.merge(betas, on=[id_col, date_col], how="left")
    out["_etf_ticker"] = out[sector_col].map(mapping)

    etf_long = (etf_returns_wide.stack()
                .rename("_etf_ret").reset_index())
    etf_long.columns = [date_col, "_etf_ticker", "_etf_ret"]

    out = out.merge(etf_long, on=[date_col, "_etf_ticker"], how="left")
    out[out_col] = out[return_col] - out[beta_col] * out["_etf_ret"]
    return out.drop(columns=["_etf_ticker", "_etf_ret"])


# -----------------------------------------------------------------------------
# Residual momentum signal
# -----------------------------------------------------------------------------


def compute_residual_momentum(
    panel: pd.DataFrame,
    *,
    id_col: str = "ID",
    date_col: str = "DATE",
    residual_col: str = "residual",
    lookback_days: int = 126,
    skip_days: int = 21,
    out_col: str = "signal",
) -> pd.DataFrame:
    """Cumulative residual return over [t − lookback, t − skip − 1].

    Standard "skip-month" momentum construction. With defaults (lookback=126,
    skip=21), this is the 6-month residual momentum excluding the most recent
    month — the construction avoids the well-documented 1-month reversal.

    Implementation
    --------------
    Per-ID running cumulative sum of residuals, then `signal_t = cum_{t-skip}
    − cum_{t-lookback}`. This equals the sum of residuals over the window
    [t − lookback, t − skip − 1]. NaNs in the residual series are treated as
    zero contribution to keep the cumsum well-defined across small gaps.
    """
    if lookback_days <= skip_days:
        raise ValueError(f"lookback_days ({lookback_days}) must exceed skip_days ({skip_days})")

    out = panel.sort_values([id_col, date_col]).reset_index(drop=True).copy()
    grp = out.groupby(id_col, sort=False, observed=True)

    out["_cum"] = grp[residual_col].cumsum()
    out[out_col] = grp["_cum"].shift(skip_days) - grp["_cum"].shift(lookback_days)
    return out.drop(columns=["_cum"])


# -----------------------------------------------------------------------------
# Cross-sectional ranking and portfolio construction
# -----------------------------------------------------------------------------


def rank_within_subindustry(
    panel: pd.DataFrame,
    *,
    signal_col: str = "signal",
    date_col: str = "DATE",
    subind_col: str = "GICS_Sub_Industry",
    min_cohort_size: int = 4,
    out_col: str = "rank_pct",
) -> pd.DataFrame:
    """Per (date, sub-industry) percentile rank of the signal.

    Sub-industries with fewer than `min_cohort_size` valid signals on a given
    date receive NaN rank. Average-method ranking; pct=True produces ranks in
    [0, 1] where 1 is the highest signal.
    """
    out = panel.copy()
    keys = [date_col, subind_col]
    cohort = out.groupby(keys, observed=True)[signal_col].transform("count")
    valid = cohort >= min_cohort_size

    ranks = pd.Series(np.nan, index=out.index, dtype="float64")
    sub = out.loc[valid]
    ranks.loc[valid] = (sub.groupby(keys, observed=True)[signal_col]
                        .rank(pct=True, method="average").values)
    out[out_col] = ranks
    return out


def build_long_short_weights(
    panel: pd.DataFrame,
    *,
    rank_col: str = "rank_pct",
    date_col: str = "DATE",
    top_pct: float = 0.20,
    bottom_pct: float = 0.20,
    out_col: str = "weight",
) -> pd.DataFrame:
    """Construct dollar-neutral L/S weights from cross-sectional ranks.

    Long names: rank_pct ≥ 1 − top_pct (top quintile by default).
    Short names: rank_pct ≤ bottom_pct (bottom quintile).
    Within each side, weights are equal so the side totals to +1 (long) and
    −1 (short). Gross = 2.0, net = 0.0 per date — dollar-neutral by
    construction.
    """
    if not 0 < top_pct <= 0.5 or not 0 < bottom_pct <= 0.5:
        raise ValueError("top_pct and bottom_pct must lie in (0, 0.5]")

    out = panel.copy()
    long_mask = out[rank_col] >= (1.0 - top_pct)
    short_mask = out[rank_col] <= bottom_pct

    grp = out.groupby(date_col, observed=True)
    n_long = grp[rank_col].transform(lambda s: (s >= (1.0 - top_pct)).sum())
    n_short = grp[rank_col].transform(lambda s: (s <= bottom_pct).sum())

    weight = pd.Series(0.0, index=out.index)
    weight = weight.where(~long_mask, 1.0 / n_long.where(n_long > 0))
    weight = weight.where(~short_mask, -1.0 / n_short.where(n_short > 0))
    out[out_col] = weight.fillna(0.0)
    return out


def verify_dollar_neutrality(
    panel: pd.DataFrame,
    *,
    weight_col: str = "weight",
    date_col: str = "DATE",
    tolerance: float = 1e-6,
) -> pd.Series:
    """Return the per-date sum of weights and log any tolerance violations."""
    sums = panel.groupby(date_col, observed=True)[weight_col].sum()
    n_viol = int((sums.abs() > tolerance).sum())
    if n_viol > 0:
        logger.warning("%d dates violate dollar neutrality at tolerance %g",
                       n_viol, tolerance)
    return sums


# -----------------------------------------------------------------------------
# Strategy return
# -----------------------------------------------------------------------------


def compute_strategy_returns(
    panel: pd.DataFrame,
    *,
    id_col: str = "ID",
    date_col: str = "DATE",
    weight_col: str = "weight",
    return_col: str = "ret",
    holding_days: int = 1,
) -> pd.Series:
    """Daily strategy returns from weights × forward returns.

    Weights set at the close of date t are held for `holding_days` and earn
    the forward return r_{t+1} (default). This is the canonical execution
    assumption: signal observable at close of t, traded at the open or close
    of t+1, return realized over the holding period.
    """
    if holding_days < 1:
        raise ValueError("holding_days must be >= 1")

    out = panel.sort_values([id_col, date_col]).copy()
    grp = out.groupby(id_col, sort=False, observed=True)
    out["_fwd_ret"] = grp[return_col].shift(-holding_days)
    out["_contrib"] = out[weight_col] * out["_fwd_ret"]
    daily = (out.groupby(date_col, observed=True)["_contrib"]
             .sum(min_count=1)
             .rename("strategy_return"))
    return daily


# -----------------------------------------------------------------------------
# Orchestrator
# -----------------------------------------------------------------------------


@dataclass
class StrategyResult:
    """Container for the end-to-end residual momentum strategy run."""

    panel: pd.DataFrame
    daily_returns: pd.Series
    betas: pd.DataFrame
    diagnostics: dict = field(default_factory=dict)


def build_residual_momentum_strategy(
    panel: pd.DataFrame,
    etf_returns_wide: pd.DataFrame,
    *,
    window: int = 126,
    min_periods: int = 63,
    lookback_days: int = 126,
    skip_days: int = 21,
    top_pct: float = 0.20,
    bottom_pct: float = 0.20,
    min_cohort_size: int = 4,
    holding_days: int = 1,
    sector_to_etf: Mapping[str, str] | None = None,
) -> StrategyResult:
    """End-to-end residual momentum strategy.

    Steps
    -----
    1. compute_rolling_betas
    2. compute_residual_returns
    3. compute_residual_momentum
    4. rank_within_subindustry
    5. build_long_short_weights
    6. verify_dollar_neutrality (diagnostic)
    7. compute_strategy_returns

    Returns
    -------
    StrategyResult
        Final panel with all intermediate columns, daily strategy return
        series, beta DataFrame for diagnostics, and a dict of parameter +
        neutrality diagnostics.
    """
    betas = compute_rolling_betas(
        panel, etf_returns_wide, sector_to_etf=sector_to_etf,
        window=window, min_periods=min_periods,
    )

    panel = compute_residual_returns(panel, betas, etf_returns_wide,
                                      sector_to_etf=sector_to_etf)
    panel = compute_residual_momentum(panel, lookback_days=lookback_days,
                                       skip_days=skip_days)
    panel = rank_within_subindustry(panel, min_cohort_size=min_cohort_size)
    panel = build_long_short_weights(panel, top_pct=top_pct, bottom_pct=bottom_pct)

    sums = verify_dollar_neutrality(panel)
    daily = compute_strategy_returns(panel, holding_days=holding_days)

    return StrategyResult(
        panel=panel,
        daily_returns=daily,
        betas=betas,
        diagnostics={
            "params": {
                "window": window, "min_periods": min_periods,
                "lookback_days": lookback_days, "skip_days": skip_days,
                "top_pct": top_pct, "bottom_pct": bottom_pct,
                "min_cohort_size": min_cohort_size, "holding_days": holding_days,
            },
            "dollar_neutrality_max_abs": float(sums.abs().max()) if len(sums) else 0.0,
            "dollar_neutrality_violations": int((sums.abs() > 1e-6).sum()),
        },
    )
              "top_pct": top_pct, "bottom_pct": bottom_pct,
                "min_cohort_size": min_cohort_size, "holding_days": holding_days,
            },
            "dollar_neutrality_max_abs": float(sums.abs().max()) if len(sums) else 0.0,
            "dollar_neutrality_violations": int((sums.abs() > 1e-6).sum()),
        },
    )
