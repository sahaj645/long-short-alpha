"""
Portfolio Builder — convert daily cross-sectional alpha scores into target weights.

The builder is the pure transformation layer of the portfolio stack. It does
not enforce risk constraints (that is `constraints.py`) and it does not own
state (that is `rebalancer.py`). It accepts a signal panel and returns the
target weights for each (ID, date) row.

Design
------
- Pure functions with no side effects.
- Configuration via `PortfolioBuildConfig` dataclass.
- Per-date construction, then concatenation, via groupby. Single-date logic is
  unit-testable in isolation.
- Industry neutrality is implemented as a per-industry demean of weights
  followed by gross-renormalization. Slightly perturbs equal-weight purity
  inside long and short books to enforce industry net exposure ≈ 0 by
  construction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PortfolioBuildConfig:
    """Parameters for `build_target_weights_*` functions."""

    # Selection
    n_long: int = 50
    n_short: int = 50

    # Column conventions
    score_col: str = "signal_rank"
    id_col: str = "ID"
    date_col: str = "DATE"
    industry_col: str = "GICS_Industry"
    sector_col: str = "GICS_Sector"

    # Construction
    neutralize_industry: bool = True
    gross_exposure: float = 2.0           # long 1.0, short -1.0 → gross 2.0
    selection_mode: Literal["count", "percentile"] = "count"
    long_pct: float = 0.20                # used only if selection_mode == "percentile"
    short_pct: float = 0.20

    # Output
    weight_col: str = "weight"


# -----------------------------------------------------------------------------
# Primitive helpers (private)
# -----------------------------------------------------------------------------


def _pick_long_short_count(
    scores: pd.Series, n_long: int, n_short: int
) -> tuple[pd.Index, pd.Index]:
    """Return (long_index, short_index) by top-N / bottom-N of `scores`."""
    if scores.empty:
        return scores.index[:0], scores.index[:0]
    ranked = scores.rank(method="first", ascending=False)
    n = len(scores)
    long_idx = ranked[ranked <= n_long].index
    short_idx = ranked[ranked > n - n_short].index
    return long_idx, short_idx


def _pick_long_short_percentile(
    scores: pd.Series, long_pct: float, short_pct: float
) -> tuple[pd.Index, pd.Index]:
    """Return (long_index, short_index) by top-pct / bottom-pct of `scores`."""
    if scores.empty:
        return scores.index[:0], scores.index[:0]
    ranks = scores.rank(method="average", pct=True)
    long_idx = ranks[ranks >= (1.0 - long_pct)].index
    short_idx = ranks[ranks <= short_pct].index
    return long_idx, short_idx


def _equal_weights(
    universe: pd.DataFrame, long_idx: pd.Index, short_idx: pd.Index
) -> pd.Series:
    """+1/n_long on longs, -1/n_short on shorts, 0 elsewhere."""
    w = pd.Series(0.0, index=universe.index, dtype="float64")
    n_long = len(long_idx)
    n_short = len(short_idx)
    if n_long > 0:
        w.loc[long_idx] = 1.0 / n_long
    if n_short > 0:
        w.loc[short_idx] = -1.0 / n_short
    return w


def _industry_demean(weights: pd.Series, industries: pd.Series) -> pd.Series:
    """Demean weights per industry. Forces per-industry sum to ~ 0."""
    if industries.isna().all():
        return weights
    industry_mean = weights.groupby(industries).transform("mean")
    return weights - industry_mean.fillna(0.0)


def _renormalize_gross(weights: pd.Series, target_gross: float) -> pd.Series:
    """Scale so the absolute weight sum equals `target_gross`. No-op when zero."""
    gross = float(weights.abs().sum())
    if gross > 0:
        return weights * (target_gross / gross)
    return weights


# -----------------------------------------------------------------------------
# Public API — single date
# -----------------------------------------------------------------------------


def build_target_weights_single_date(
    signal_panel: pd.DataFrame, config: PortfolioBuildConfig
) -> pd.DataFrame:
    """Build a single date's target weights.

    Parameters
    ----------
    signal_panel
        Slice of the signal panel for ONE date. Must contain `score_col`,
        `id_col`. Industry/sector columns optional; required only if
        `neutralize_industry=True` or downstream sector-exposure tests run.
    config
        Build parameters.

    Returns
    -------
    pd.DataFrame
        Copy of `signal_panel` with a new `weight_col` populated. Names not
        selected for either book have weight = 0.
    """
    df = signal_panel.dropna(subset=[config.score_col]).copy()
    if df.empty:
        return df.assign(**{config.weight_col: pd.Series(dtype="float64")})

    if config.selection_mode == "count":
        long_idx, short_idx = _pick_long_short_count(
            df[config.score_col], config.n_long, config.n_short
        )
    elif config.selection_mode == "percentile":
        long_idx, short_idx = _pick_long_short_percentile(
            df[config.score_col], config.long_pct, config.short_pct
        )
    else:
        raise ValueError(f"Unknown selection_mode: {config.selection_mode!r}")

    w = _equal_weights(df, long_idx, short_idx)

    if config.neutralize_industry and config.industry_col in df.columns:
        w = _industry_demean(w, df[config.industry_col])

    w = _renormalize_gross(w, config.gross_exposure)
    df[config.weight_col] = w
    return df


def build_target_weights_panel(
    signal_panel: pd.DataFrame, config: PortfolioBuildConfig
) -> pd.DataFrame:
    """Build target weights across all dates by applying single-date construction."""
    if signal_panel.empty:
        return signal_panel.assign(**{config.weight_col: pd.Series(dtype="float64")})

    out = (
        signal_panel.groupby(config.date_col, observed=True, group_keys=False)
        .apply(lambda g: build_target_weights_single_date(g, config))
        .reset_index(drop=True)
    )
    logger.info(
        "Built target weights across %d dates; total rows %d",
        signal_panel[config.date_col].nunique(),
        len(out),
    )
    return out
