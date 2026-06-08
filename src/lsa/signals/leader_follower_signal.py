"""
Sub-Industry Leader-Follower Signal Construction (H6).

This module builds the trading signal for the Sub-Industry Leader-Follower
Lead-Lag strategy. It takes a residualized panel (output of
`lsa.signals.residual_momentum.compute_residual_returns`) and produces a
per-(follower, date) signal equal to the cap-weighted leader's residual
return at the configured lag.

Pipeline
--------
1. `identify_leaders` — tag the cap-weighted top-1 name in each
   (Sub-Industry, date) using a trailing-window mean of market cap for
   stability. Apply universe filters (min member count, leader cap-share band).

2. `identify_followers` — tag eligible non-leader rows. A row is a follower
   iff (a) it is not the leader on that (Sub-Industry, date), and (b) the
   cell has a valid leader (filters passed).

3. `compute_leader_follower_signal` — for each follower row at date t, the
   raw signal is the leader's residual return at date (t − lag_days),
   optionally aggregated over `signal_horizon` days.

4. `normalize_signal` — cross-sectional z-score (default) or rank, computed
   on the unique (Sub-Industry, date) signal values so each Sub-Industry
   has equal weight regardless of follower cohort size.

5. `rank_signal_cross_sectional` — percentile rank across all Sub-Industries
   per date. Followers within the same Sub-Industry share the rank by
   construction.

6. `signal_diagnostics` — coverage, distribution, autocorrelation, rank
   stability. Returns a `SignalDiagnostics` dataclass.

7. `extract_trading_signal` — clean (ID, DATE, signal, rank) DataFrame for
   handoff to portfolio construction.

The orchestrator `build_leader_follower_signal` runs the full pipeline.

Conventions
-----------
- All functions are pure: input frames are not mutated; new frames are
  returned with added columns.
- Leader rows always have NaN signal — leaders do not trade their own
  signal.
- Ineligible cells (filter violations) have NaN in all signal-related
  columns, propagating cleanly to portfolio construction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Default parameters (mirror configs/base.yaml)
# -----------------------------------------------------------------------------

DEFAULT_MC_SMOOTHING_DAYS: int = 21
DEFAULT_MIN_MEMBERS: int = 4
DEFAULT_LEADER_SHARE_MIN: float = 0.20
DEFAULT_LEADER_SHARE_MAX: float = 0.70
DEFAULT_LAG_DAYS: int = 1
DEFAULT_SIGNAL_HORIZON: int = 1
DEFAULT_NORMALIZATION_METHOD: str = "zscore"
DEFAULT_NORMALIZATION_CAP_SIGMA: float = 3.0
DEFAULT_AUTOCORR_LAGS: tuple[int, ...] = (1, 2, 3, 5, 10)


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------


@dataclass
class SignalDiagnostics:
    """Diagnostic statistics for a constructed signal panel."""

    n_signal_observations: int
    mean_daily_coverage: float
    mean_subindustries_with_signal: float
    signal_distribution: pd.DataFrame
    signal_autocorr: pd.Series
    rank_stability_lag1: float
    summary: dict = field(default_factory=dict)


@dataclass
class LeaderFollowerSignalResult:
    """Return value of `build_leader_follower_signal`."""

    panel: pd.DataFrame
    diagnostics: SignalDiagnostics
    params: dict = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Step 1 — Leader identification
# -----------------------------------------------------------------------------


def identify_leaders(
    panel: pd.DataFrame,
    *,
    id_col: str = "ID",
    date_col: str = "DATE",
    subind_col: str = "GICS_Sub_Industry",
    market_cap_col: str = "Market_Cap",
    mc_smoothing_days: int = DEFAULT_MC_SMOOTHING_DAYS,
    min_members: int = DEFAULT_MIN_MEMBERS,
    leader_share_min: float = DEFAULT_LEADER_SHARE_MIN,
    leader_share_max: float = DEFAULT_LEADER_SHARE_MAX,
    leader_flag_col: str = "is_leader",
) -> pd.DataFrame:
    """Tag the cap-weighted leader per (Sub-Industry, date) and emit cell metadata.

    A name is the leader on (Sub-Industry s, date t) iff:
      1. The cell (s, t) has at least `min_members` distinct IDs.
      2. The cell's leader cap-share lies in `[leader_share_min, leader_share_max]`.
      3. The name has the highest trailing-`mc_smoothing_days` mean market cap
         within the cell.

    Smoothing the market cap with a trailing mean prevents single-day price
    spikes from flipping the leader. Ties (rare in practice with floats) are
    broken by alphabetical ID order.

    Parameters
    ----------
    panel
        Long-form panel. Must contain `id_col`, `date_col`, `subind_col`,
        and `market_cap_col`.
    mc_smoothing_days
        Trailing window for market cap smoothing (default 21).
    min_members, leader_share_min, leader_share_max
        Universe filter parameters. Must match the values used in
        `apply_universe_filters` for end-to-end consistency.

    Returns
    -------
    pd.DataFrame
        Input panel with added columns:
        - `mc_smooth` : trailing-window mean of market cap per ID
        - `n_members` : count of distinct IDs in the (Sub-Industry, date) cell
        - `leader_share` : top cap-share within the cell
        - `is_leader` : boolean leader flag

    Raises
    ------
    KeyError
        If a required column is missing from `panel`.
    """
    required = {id_col, date_col, subind_col, market_cap_col}
    missing = required - set(panel.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    out = panel.sort_values([id_col, date_col]).reset_index(drop=True).copy()

    out["mc_smooth"] = (
        out.groupby(id_col, sort=False, observed=True)[market_cap_col]
           .transform(lambda s: s.rolling(mc_smoothing_days,
                                           min_periods=max(mc_smoothing_days // 2, 1))
                                  .mean())
    )

    cell_grp = out.groupby([date_col, subind_col], observed=True)
    out["n_members"] = cell_grp[id_col].transform("nunique")
    cell_total_mc = cell_grp["mc_smooth"].transform("sum")
    cell_max_mc = cell_grp["mc_smooth"].transform("max")
    out["leader_share"] = cell_max_mc / cell_total_mc.replace(0, np.nan)

    eligible_cell = (
        (out["n_members"] >= min_members)
        & out["leader_share"].between(leader_share_min, leader_share_max)
    )

    # Rank within (date, sub-industry) by smoothed cap; method='first' breaks ties.
    rank_in_cell = (
        cell_grp["mc_smooth"]
        .rank(ascending=False, method="first")
    )
    out[leader_flag_col] = eligible_cell & (rank_in_cell == 1)

    n_leader_rows = int(out[leader_flag_col].sum())
    n_eligible_cells = int(out.loc[eligible_cell, [date_col, subind_col]]
                             .drop_duplicates().shape[0])
    logger.info(
        "Identified %d leader rows across %d eligible (Sub-Industry, date) cells",
        n_leader_rows, n_eligible_cells,
    )
    return out


# -----------------------------------------------------------------------------
# Step 2 — Follower identification
# -----------------------------------------------------------------------------


def identify_followers(
    panel: pd.DataFrame,
    *,
    date_col: str = "DATE",
    subind_col: str = "GICS_Sub_Industry",
    leader_flag_col: str = "is_leader",
    follower_flag_col: str = "is_follower",
) -> pd.DataFrame:
    """Tag eligible follower rows.

    A row is a follower iff:
      - It is NOT the leader on its (Sub-Industry, date), AND
      - The cell has a leader (i.e., the universe filters passed for that cell).

    Rows in ineligible cells have `is_follower = False`.
    """
    if leader_flag_col not in panel.columns:
        raise KeyError(
            f"`{leader_flag_col}` column missing — call identify_leaders first."
        )

    out = panel.copy()
    cell_has_leader = (
        out.groupby([date_col, subind_col], observed=True)[leader_flag_col]
           .transform("any")
    )
    out[follower_flag_col] = (~out[leader_flag_col]) & cell_has_leader.astype(bool)
    n_followers = int(out[follower_flag_col].sum())
    logger.info("Identified %d follower rows", n_followers)
    return out


# -----------------------------------------------------------------------------
# Step 3 — Signal computation
# -----------------------------------------------------------------------------


def compute_leader_follower_signal(
    panel: pd.DataFrame,
    *,
    date_col: str = "DATE",
    subind_col: str = "GICS_Sub_Industry",
    residual_col: str = "residual",
    leader_flag_col: str = "is_leader",
    lag_days: int = DEFAULT_LAG_DAYS,
    signal_horizon: int = DEFAULT_SIGNAL_HORIZON,
    out_col: str = "signal",
) -> pd.DataFrame:
    """Compute the per-row signal = leader's residual return at the right lag.

    For each row at (Sub-Industry s, date t):
      signal_{s,t} = sum_{k=lag_days}^{lag_days + signal_horizon - 1}
                       leader_residual_{s, t - k}

    With defaults (`lag_days=1, signal_horizon=1`), this is simply the leader's
    residual return on the previous trading day. With `signal_horizon=3`, the
    signal is the sum of the leader's residuals over the prior three days.

    Leader rows have NaN signal — leaders do not trade their own signal.
    Rows where the cell lacks a leader (ineligible) also have NaN signal.

    Parameters
    ----------
    panel
        Panel with leader flag and residual columns. Output of
        `identify_leaders` followed by residualization (or vice versa).
    lag_days
        Trading-day lag between leader observation and follower signal.
        Must be >= 1.
    signal_horizon
        Number of leader days summed into the signal. Must be >= 1.

    Returns
    -------
    pd.DataFrame
        Input panel with `out_col` added.

    Raises
    ------
    ValueError
        If `lag_days < 1` or `signal_horizon < 1`.
    """
    if lag_days < 1:
        raise ValueError(f"lag_days must be >= 1, got {lag_days}")
    if signal_horizon < 1:
        raise ValueError(f"signal_horizon must be >= 1, got {signal_horizon}")

    leaders = (
        panel.loc[panel[leader_flag_col].fillna(False),
                  [subind_col, date_col, residual_col]]
        .sort_values([subind_col, date_col])
        .reset_index(drop=True)
        .copy()
    )

    grp_by_sub = leaders.groupby(subind_col, sort=False, observed=True)

    if signal_horizon == 1:
        leaders["_signal"] = grp_by_sub[residual_col].shift(lag_days)
    else:
        # Sum of leader residuals over the window [t - lag, t - lag - horizon + 1]
        rolled = (
            grp_by_sub[residual_col]
            .rolling(signal_horizon, min_periods=signal_horizon)
            .sum()
            .reset_index(level=0, drop=True)
        )
        leaders["_signal_window"] = rolled.values
        leaders["_signal"] = (
            leaders.groupby(subind_col, sort=False, observed=True)["_signal_window"]
                   .shift(lag_days)
        )
        leaders = leaders.drop(columns=["_signal_window"])

    out = panel.merge(
        leaders[[subind_col, date_col, "_signal"]].rename(columns={"_signal": out_col}),
        on=[subind_col, date_col],
        how="left",
    )
    out.loc[out[leader_flag_col].fillna(False), out_col] = np.nan

    logger.info(
        "Computed signal: lag=%d, horizon=%d → %d non-null rows",
        lag_days, signal_horizon, int(out[out_col].notna().sum()),
    )
    return out


# -----------------------------------------------------------------------------
# Step 4 — Normalization
# -----------------------------------------------------------------------------


def normalize_signal(
    panel: pd.DataFrame,
    *,
    signal_col: str = "signal",
    date_col: str = "DATE",
    subind_col: str = "GICS_Sub_Industry",
    method: str = DEFAULT_NORMALIZATION_METHOD,
    cap_sigma: float | None = DEFAULT_NORMALIZATION_CAP_SIGMA,
    out_col: str = "signal_norm",
) -> pd.DataFrame:
    """Normalize the signal cross-sectionally per date.

    Because every follower within a Sub-Industry shares the same leader signal,
    normalization is computed on the unique (Sub-Industry, date) signal values
    and then broadcast back to all rows. This weighting ensures each
    Sub-Industry contributes equally to the normalization parameters,
    independent of how many followers it contains.

    Methods
    -------
    - `zscore` : (signal − cross-sectional mean) / cross-sectional std
    - `rank`   : percentile rank in [0, 1]
    - `none`   : copy of input

    Parameters
    ----------
    cap_sigma
        For `zscore`, clip the normalized signal to [-cap_sigma, +cap_sigma].
        Pass None to disable. Ignored for `rank` and `none`.

    Raises
    ------
    ValueError
        If `method` is unrecognized.
    """
    if method not in {"zscore", "rank", "none"}:
        raise ValueError(f"Unknown normalization method: {method!r}")

    if method == "none":
        return panel.assign(**{out_col: panel[signal_col]})

    unique = (
        panel.dropna(subset=[signal_col])
             .drop_duplicates(subset=[date_col, subind_col])
             [[date_col, subind_col, signal_col]]
             .copy()
    )

    if method == "zscore":
        grp = unique.groupby(date_col, observed=True)[signal_col]
        mean = grp.transform("mean")
        std = grp.transform("std").replace(0, np.nan)
        unique[out_col] = (unique[signal_col] - mean) / std
        if cap_sigma is not None:
            unique[out_col] = unique[out_col].clip(lower=-cap_sigma, upper=cap_sigma)
    elif method == "rank":
        unique[out_col] = (
            unique.groupby(date_col, observed=True)[signal_col]
                  .rank(method="average", pct=True)
        )

    out = panel.merge(
        unique[[date_col, subind_col, out_col]],
        on=[date_col, subind_col], how="left",
    )
    return out


# -----------------------------------------------------------------------------
# Step 5 — Cross-sectional ranking
# -----------------------------------------------------------------------------


def rank_signal_cross_sectional(
    panel: pd.DataFrame,
    *,
    signal_col: str = "signal_norm",
    date_col: str = "DATE",
    subind_col: str = "GICS_Sub_Industry",
    method: str = "average",
    out_col: str = "signal_rank",
) -> pd.DataFrame:
    """Percentile-rank the signal across Sub-Industries per date.

    Computes the rank on the unique (Sub-Industry, date) signal values and
    broadcasts back to all rows. Followers within the same Sub-Industry share
    the rank by construction.

    Returns
    -------
    pd.DataFrame
        Input panel with `out_col` added; ranks in [0, 1] where 1 is the
        most positive signal (leader had largest positive residual).
    """
    unique = (
        panel.dropna(subset=[signal_col])
             .drop_duplicates(subset=[date_col, subind_col])
             [[date_col, subind_col, signal_col]]
             .copy()
    )
    unique[out_col] = (
        unique.groupby(date_col, observed=True)[signal_col]
              .rank(method=method, pct=True)
    )
    out = panel.merge(
        unique[[date_col, subind_col, out_col]],
        on=[date_col, subind_col], how="left",
    )
    return out


# -----------------------------------------------------------------------------
# Step 6 — Diagnostics
# -----------------------------------------------------------------------------


def signal_diagnostics(
    panel: pd.DataFrame,
    *,
    signal_col: str = "signal",
    rank_col: str = "signal_rank",
    date_col: str = "DATE",
    subind_col: str = "GICS_Sub_Industry",
    autocorr_lags: Sequence[int] = DEFAULT_AUTOCORR_LAGS,
) -> SignalDiagnostics:
    """Compute coverage, distribution, autocorrelation, and rank-stability stats.

    Coverage
        Fraction of universe rows per date that carry a valid signal.

    Distribution
        Per-date `mean`, `std`, `min`, `max`, count of unique Sub-Industries.

    Autocorrelation
        For each lag in `autocorr_lags`, the average across Sub-Industries of
        the time-series autocorrelation of the signal.

    Rank stability
        Mean across Sub-Industries of the lag-1 autocorrelation of
        `rank_col` (if present). High stability means a sub-industry that
        ranks high today tends to rank high tomorrow — informative for
        choosing the rebalance frequency.

    Returns
    -------
    SignalDiagnostics
    """
    unique = (
        panel.dropna(subset=[signal_col])
             .drop_duplicates(subset=[date_col, subind_col])
             .copy()
    )

    coverage = (
        panel.groupby(date_col, observed=True)[signal_col]
             .apply(lambda s: s.notna().mean())
             .rename("coverage")
    )
    subind_per_date = (
        unique.groupby(date_col, observed=True)[subind_col].nunique()
              .rename("n_subindustries")
    )

    distribution = (
        unique.groupby(date_col, observed=True)[signal_col]
              .agg(["mean", "std", "count", "min", "max"])
    )

    wide_signal = unique.pivot(index=date_col, columns=subind_col, values=signal_col)
    autocorr_series = pd.Series(
        {lag: float(wide_signal.apply(lambda c: c.autocorr(lag=lag)).mean())
         for lag in autocorr_lags},
        name="autocorrelation",
    )

    if rank_col in panel.columns:
        rank_unique = (
            panel.dropna(subset=[rank_col])
                 .drop_duplicates(subset=[date_col, subind_col])
        )
        rank_wide = rank_unique.pivot(index=date_col, columns=subind_col, values=rank_col)
        rank_stab = float(rank_wide.apply(lambda c: c.autocorr(lag=1)).mean())
    else:
        rank_stab = float("nan")

    summary = {
        "n_observations": int(panel[signal_col].notna().sum()),
        "mean_daily_coverage": float(coverage.mean()),
        "mean_subindustries_with_signal": float(subind_per_date.mean()),
        "autocorr_lag1": float(autocorr_series.iloc[0]) if len(autocorr_series) else float("nan"),
        "rank_stability_lag1": rank_stab,
        "signal_mean_global": float(unique[signal_col].mean()),
        "signal_std_global": float(unique[signal_col].std()),
    }

    logger.info("Signal diagnostics: %s", summary)
    return SignalDiagnostics(
        n_signal_observations=summary["n_observations"],
        mean_daily_coverage=summary["mean_daily_coverage"],
        mean_subindustries_with_signal=summary["mean_subindustries_with_signal"],
        signal_distribution=distribution,
        signal_autocorr=autocorr_series,
        rank_stability_lag1=rank_stab,
        summary=summary,
    )


# -----------------------------------------------------------------------------
# Step 7 — Handoff to portfolio construction
# -----------------------------------------------------------------------------


def extract_trading_signal(
    panel: pd.DataFrame,
    *,
    id_col: str = "ID",
    date_col: str = "DATE",
    subind_col: str = "GICS_Sub_Industry",
    sector_col: str = "GICS_Sector",
    follower_flag_col: str = "is_follower",
    signal_col: str = "signal",
    signal_norm_col: str = "signal_norm",
    rank_col: str = "signal_rank",
) -> pd.DataFrame:
    """Extract a clean follower-only DataFrame for portfolio construction.

    Returns one row per (follower ID, date) where the signal is valid. Drops
    leader rows, ineligible cells, and NaN signals. The output columns are
    the minimum required for portfolio construction.
    """
    if follower_flag_col not in panel.columns:
        raise KeyError(
            f"`{follower_flag_col}` column missing — call identify_followers first."
        )

    cols = [c for c in (id_col, date_col, subind_col, sector_col,
                        signal_col, signal_norm_col, rank_col)
            if c in panel.columns]

    mask = panel[follower_flag_col] & panel[signal_col].notna()
    out = panel.loc[mask, cols].reset_index(drop=True).copy()
    logger.info("Extracted %d trading-ready rows for portfolio construction", len(out))
    return out


# -----------------------------------------------------------------------------
# Orchestrator
# -----------------------------------------------------------------------------


def build_leader_follower_signal(
    panel: pd.DataFrame,
    *,
    # Column conventions
    id_col: str = "ID",
    date_col: str = "DATE",
    subind_col: str = "GICS_Sub_Industry",
    market_cap_col: str = "Market_Cap",
    residual_col: str = "residual",
    # Leader identification
    mc_smoothing_days: int = DEFAULT_MC_SMOOTHING_DAYS,
    min_members: int = DEFAULT_MIN_MEMBERS,
    leader_share_min: float = DEFAULT_LEADER_SHARE_MIN,
    leader_share_max: float = DEFAULT_LEADER_SHARE_MAX,
    # Signal
    lag_days: int = DEFAULT_LAG_DAYS,
    signal_horizon: int = DEFAULT_SIGNAL_HORIZON,
    # Normalization
    normalization_method: str = DEFAULT_NORMALIZATION_METHOD,
    normalization_cap_sigma: float | None = DEFAULT_NORMALIZATION_CAP_SIGMA,
    # Diagnostics
    autocorr_lags: Sequence[int] = DEFAULT_AUTOCORR_LAGS,
    run_diagnostics: bool = True,
) -> LeaderFollowerSignalResult:
    """End-to-end leader-follower signal construction.

    Pipeline order
    --------------
    1. identify_leaders
    2. identify_followers
    3. compute_leader_follower_signal
    4. normalize_signal
    5. rank_signal_cross_sectional
    6. signal_diagnostics (optional)

    Parameters
    ----------
    panel
        Residualized panel. Must contain at minimum `id_col`, `date_col`,
        `subind_col`, `market_cap_col`, `residual_col`.
    lag_days, signal_horizon
        Lead-lag and aggregation horizon for the leader's signal.
    normalization_method
        One of `zscore`, `rank`, `none`.
    normalization_cap_sigma
        Clip threshold for z-score normalization. None disables clipping.
    run_diagnostics
        If False, skip the diagnostics step (useful in tight backtest loops).

    Returns
    -------
    LeaderFollowerSignalResult
        Container with the final panel (containing all intermediate columns),
        a `SignalDiagnostics` object, and the parameter dict.
    """
    p = identify_leaders(
        panel,
        id_col=id_col, date_col=date_col, subind_col=subind_col,
        market_cap_col=market_cap_col,
        mc_smoothing_days=mc_smoothing_days,
        min_members=min_members,
        leader_share_min=leader_share_min,
        leader_share_max=leader_share_max,
    )
    p = identify_followers(p, date_col=date_col, subind_col=subind_col)
    p = compute_leader_follower_signal(
        p, date_col=date_col, subind_col=subind_col,
        residual_col=residual_col,
        lag_days=lag_days, signal_horizon=signal_horizon,
    )
    p = normalize_signal(
        p, date_col=date_col, subind_col=subind_col,
        method=normalization_method, cap_sigma=normalization_cap_sigma,
    )
    p = rank_signal_cross_sectional(
        p, date_col=date_col, subind_col=subind_col,
    )

    if run_diagnostics:
        diag = signal_diagnostics(
            p, date_col=date_col, subind_col=subind_col,
            autocorr_lags=autocorr_lags,
        )
    else:
        diag = SignalDiagnostics(
            n_signal_observations=int(p["signal"].notna().sum()),
            mean_daily_coverage=float("nan"),
            mean_subindustries_with_signal=float("nan"),
            signal_distribution=pd.DataFrame(),
            signal_autocorr=pd.Series(dtype=float),
            rank_stability_lag1=float("nan"),
            summary={},
        )

    params = {
        "mc_smoothing_days": mc_smoothing_days,
        "min_members": min_members,
        "leader_share_min": leader_share_min,
        "leader_share_max": leader_share_max,
        "lag_days": lag_days,
        "signal_horizon": signal_horizon,
        "normalization_method": normalization_method,
        "normalization_cap_sigma": normalization_cap_sigma,
    }
    return LeaderFollowerSignalResult(panel=p, diagnostics=diag, params=params)
