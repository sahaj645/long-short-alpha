"""
Trade-level statistics — episode extraction, win rate, profit factor,
long/short attribution, and turnover summaries.

A *position episode* is a contiguous run of non-zero weight for a single ID.
Episodes open when a name's weight goes from zero to non-zero and close when
it returns to zero. For a daily-rebalanced strategy, episodes are typically
multi-day windows during which the signal kept the name in the book.

The trade statistics in this module are computed at the episode level. This
is the institutional convention because (a) it gives an honest hit rate per
position decision rather than per-day micro-fluctuation, and (b) it allows
long-side vs short-side attribution by direction.

Pipeline
--------
1. `reconstruct_weights_panel` — assemble per-(ID, DATE) weights from a
   sequence of trade deltas (the backtester's `rebalance_history`).
2. `extract_position_episodes` — find contiguous holding periods per ID.
3. `compute_episode_pnl` — for each episode, sum daily contributions
   (weight × return).
4. `compute_trade_statistics` — aggregate episode PnLs into the headline
   `TradeStatistics` summary.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


DEFAULT_FREQ: int = 252
WEIGHT_EPSILON: float = 1e-10           # numerical zero for weight comparisons


# -----------------------------------------------------------------------------
# Dataclass
# -----------------------------------------------------------------------------


@dataclass
class TradeStatistics:
    """Aggregate trade-level performance summary."""

    # Episode counts
    n_episodes: int = 0
    n_long_episodes: int = 0
    n_short_episodes: int = 0

    # Win / loss
    win_rate: float = float("nan")
    loss_rate: float = float("nan")

    # Trade PnL distribution
    avg_winning_pnl: float = float("nan")
    avg_losing_pnl: float = float("nan")
    largest_winning_pnl: float = float("nan")
    largest_losing_pnl: float = float("nan")
    profit_factor: float = float("nan")

    # Holding period
    avg_holding_period_days: float = float("nan")
    median_holding_period_days: float = float("nan")

    # Side attribution
    long_side_total_pnl: float = 0.0
    short_side_total_pnl: float = 0.0
    long_side_win_rate: float = float("nan")
    short_side_win_rate: float = float("nan")

    # Turnover (set by `compute_turnover_statistics` if a turnover series is supplied)
    avg_daily_turnover: float = float("nan")
    median_daily_turnover: float = float("nan")
    max_daily_turnover: float = float("nan")
    annualized_turnover: float = float("nan")


# -----------------------------------------------------------------------------
# Weights reconstruction (from backtester rebalance history)
# -----------------------------------------------------------------------------


def reconstruct_weights_panel(
    rebalance_history: Iterable,
    *,
    id_col: str = "ID",
    date_col: str = "DATE",
    weight_col: str = "weight",
) -> pd.DataFrame:
    """Build a long-form (ID, DATE, weight) panel from backtest rebalance history.

    Each `ExecutionReport` in the history carries a `trades` Series of Δweight
    per ID for one date. Per-ID cumulative summation across dates reconstructs
    the weight time series.

    Parameters
    ----------
    rebalance_history
        Iterable of `ExecutionReport`-like objects with `.date` and `.trades`
        attributes. Compatible with `BacktestResult.rebalance_history`.

    Returns
    -------
    pd.DataFrame
        Long-form panel with columns (`id_col`, `date_col`, `weight_col`)
        for every (ID, DATE) where weight was non-zero.
    """
    rows = []
    for report in rebalance_history:
        date = getattr(report, "date", None)
        trades = getattr(report, "trades", None)
        if trades is None or date is None or len(trades) == 0:
            continue
        for id_val, delta in trades.items():
            if not np.isfinite(delta) or delta == 0.0:
                continue
            rows.append({id_col: id_val, date_col: date, "_delta": float(delta)})

    if not rows:
        return pd.DataFrame(columns=[id_col, date_col, weight_col])

    df = pd.DataFrame(rows).sort_values([id_col, date_col])
    df[weight_col] = df.groupby(id_col, observed=True)["_delta"].cumsum()
    # Filter to non-zero (within tolerance)
    df = df.loc[df[weight_col].abs() > WEIGHT_EPSILON]
    return df[[id_col, date_col, weight_col]].reset_index(drop=True)


# -----------------------------------------------------------------------------
# Episode extraction
# -----------------------------------------------------------------------------


def extract_position_episodes(
    weights_panel: pd.DataFrame,
    *,
    id_col: str = "ID",
    date_col: str = "DATE",
    weight_col: str = "weight",
) -> pd.DataFrame:
    """Identify contiguous holding episodes per ID.

    An episode is a run of consecutive trading dates where the ID had a
    non-zero weight. The episode's direction is the sign of the average
    weight during the episode (long or short).

    Returns
    -------
    pd.DataFrame
        One row per episode with columns:
        (id_col, episode_id, start_date, end_date, n_days, direction,
         avg_weight). `episode_id` is unique across the full panel.
    """
    required = {id_col, date_col, weight_col}
    missing = required - set(weights_panel.columns)
    if missing:
        raise KeyError(f"weights_panel missing required columns: {sorted(missing)}")

    if weights_panel.empty:
        return pd.DataFrame(
            columns=[id_col, "episode_id", "start_date", "end_date",
                     "n_days", "direction", "avg_weight"]
        )

    df = weights_panel.sort_values([id_col, date_col]).reset_index(drop=True).copy()
    df["_held"] = df[weight_col].abs() > WEIGHT_EPSILON

    # Detect calendar-day gaps within a held run: treat a gap > 5 calendar days
    # as a break (handles intra-period delistings and re-additions).
    df["_calendar_gap"] = (df.groupby(id_col, observed=True)[date_col]
                             .diff().dt.days.fillna(0) > 5)
    df["_block_id"] = (
        (df[id_col] != df[id_col].shift())
        | (df["_held"] != df["_held"].shift())
        | df["_calendar_gap"]
    ).cumsum()

    held = df[df["_held"]]
    if held.empty:
        return pd.DataFrame(
            columns=[id_col, "episode_id", "start_date", "end_date",
                     "n_days", "direction", "avg_weight"]
        )

    grp = held.groupby([id_col, "_block_id"], observed=True)
    episodes = grp.agg(
        start_date=(date_col, "min"),
        end_date=(date_col, "max"),
        n_days=(date_col, "count"),
        avg_weight=(weight_col, "mean"),
    ).reset_index()
    episodes["direction"] = np.where(episodes["avg_weight"] > 0, "long", "short")
    episodes = episodes.rename(columns={"_block_id": "episode_id"})

    return episodes[[id_col, "episode_id", "start_date", "end_date",
                     "n_days", "direction", "avg_weight"]]


# -----------------------------------------------------------------------------
# Episode PnL
# -----------------------------------------------------------------------------


def compute_episode_pnl(
    episodes: pd.DataFrame,
    weights_panel: pd.DataFrame,
    returns_panel: pd.DataFrame,
    *,
    id_col: str = "ID",
    date_col: str = "DATE",
    weight_col: str = "weight",
    return_col: str = "ret",
) -> pd.DataFrame:
    """Aggregate per-day contribution (weight × return) into total PnL per episode.

    Returns
    -------
    pd.DataFrame
        The input `episodes` DataFrame with a new `pnl` column.
    """
    if episodes.empty:
        return episodes.assign(pnl=pd.Series(dtype="float64"))

    joined = weights_panel.merge(returns_panel, on=[id_col, date_col], how="left")
    joined = joined[joined[weight_col].abs() > WEIGHT_EPSILON].copy()
    joined["_contrib"] = joined[weight_col] * joined[return_col].fillna(0.0)

    # Attach episode_id by joining episodes' (start_date, end_date, ID) onto joined
    # using interval matching.
    episodes_sorted = episodes.sort_values([id_col, "start_date"]).reset_index(drop=True)
    pnl_rows = []
    for ep in episodes_sorted.itertuples(index=False):
        mask = (
            (joined[id_col] == getattr(ep, id_col))
            & (joined[date_col] >= ep.start_date)
            & (joined[date_col] <= ep.end_date)
        )
        pnl_rows.append({
            id_col: getattr(ep, id_col),
            "episode_id": ep.episode_id,
            "pnl": float(joined.loc[mask, "_contrib"].sum()),
        })
    pnl_df = pd.DataFrame(pnl_rows)
    out = episodes.merge(pnl_df, on=[id_col, "episode_id"], how="left")
    return out


# -----------------------------------------------------------------------------
# Trade statistics aggregator
# -----------------------------------------------------------------------------


def _profit_factor(wins_sum: float, losses_sum: float) -> float:
    """sum(wins) / |sum(losses)| with degenerate-case handling."""
    if wins_sum > 0 and losses_sum >= 0:
        return float("inf")
    if wins_sum <= 0 and losses_sum >= 0:
        return float("nan")
    return float(wins_sum / abs(losses_sum))


def compute_trade_statistics(
    episodes_with_pnl: pd.DataFrame,
    *,
    turnover_series: Optional[pd.Series] = None,
    freq: int = DEFAULT_FREQ,
) -> TradeStatistics:
    """Aggregate episode-level PnL into a `TradeStatistics` summary."""
    if episodes_with_pnl.empty or "pnl" not in episodes_with_pnl.columns:
        stats = TradeStatistics()
    else:
        df = episodes_with_pnl.dropna(subset=["pnl"]).copy()
        n = len(df)
        wins = df[df["pnl"] > 0]
        losses = df[df["pnl"] < 0]

        long_ep = df[df["direction"] == "long"]
        short_ep = df[df["direction"] == "short"]

        stats = TradeStatistics(
            n_episodes=n,
            n_long_episodes=len(long_ep),
            n_short_episodes=len(short_ep),
            win_rate=float(len(wins) / n) if n > 0 else float("nan"),
            loss_rate=float(len(losses) / n) if n > 0 else float("nan"),
            avg_winning_pnl=float(wins["pnl"].mean()) if len(wins) else float("nan"),
            avg_losing_pnl=float(losses["pnl"].mean()) if len(losses) else float("nan"),
            largest_winning_pnl=float(wins["pnl"].max()) if len(wins) else float("nan"),
            largest_losing_pnl=float(losses["pnl"].min()) if len(losses) else float("nan"),
            profit_factor=_profit_factor(
                float(wins["pnl"].sum()), float(losses["pnl"].sum())
            ),
            avg_holding_period_days=float(df["n_days"].mean()) if n else float("nan"),
            median_holding_period_days=float(df["n_days"].median()) if n else float("nan"),
            long_side_total_pnl=float(long_ep["pnl"].sum()) if len(long_ep) else 0.0,
            short_side_total_pnl=float(short_ep["pnl"].sum()) if len(short_ep) else 0.0,
            long_side_win_rate=(
                float((long_ep["pnl"] > 0).mean()) if len(long_ep) else float("nan")
            ),
            short_side_win_rate=(
                float((short_ep["pnl"] > 0).mean()) if len(short_ep) else float("nan")
            ),
        )

    if turnover_series is not None:
        t_stats = compute_turnover_statistics(turnover_series, freq=freq)
        stats.avg_daily_turnover = t_stats["avg_daily_turnover"]
        stats.median_daily_turnover = t_stats["median_daily_turnover"]
        stats.max_daily_turnover = t_stats["max_daily_turnover"]
        stats.annualized_turnover = t_stats["annualized_turnover"]

    return stats


def compute_turnover_statistics(
    turnover_series: pd.Series, *, freq: int = DEFAULT_FREQ
) -> dict:
    """Aggregate a daily one-way turnover series into headline numbers."""
    t = turnover_series.dropna()
    if t.empty:
        return {
            "avg_daily_turnover": float("nan"),
            "median_daily_turnover": float("nan"),
            "max_daily_turnover": float("nan"),
            "annualized_turnover": float("nan"),
        }
    return {
        "avg_daily_turnover": float(t.mean()),
        "median_daily_turnover": float(t.median()),
        "max_daily_turnover": float(t.max()),
        "annualized_turnover": float(t.mean() * freq),
    }
