"""
Accounting — portfolio NAV, daily PnL, exposures, and long/short attribution.

The Accountant owns the only mutable PnL state in the backtest. Every day it
receives the weights held during the day, the realized returns on those
names, the new weights set at end of day, and the realized transaction cost,
and returns a `DailyRecord` that captures the full bookkeeping.

The Accountant deliberately does not know anything about signals, portfolio
construction, or execution mechanics. Its contract is narrow: given weights
and returns, produce honest PnL and exposure numbers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Per-day record
# -----------------------------------------------------------------------------


@dataclass
class DailyRecord:
    """One day's accounting outcome. All percentages are decimal fractions."""

    date: pd.Timestamp

    nav_open: float = 1.0
    nav_close: float = 1.0

    gross_pnl_pct: float = 0.0           # w_{t-1} · r_t
    long_book_pnl_pct: float = 0.0       # sum of long positions' contribution
    short_book_pnl_pct: float = 0.0      # sum of short positions' contribution

    transaction_cost_pct: float = 0.0    # cost paid at end-of-day rebalance
    net_pnl_pct: float = 0.0             # gross − cost

    gross_exposure_after: float = 0.0    # |new weights|.sum()
    net_exposure_after: float = 0.0      # new weights.sum()
    n_long_after: int = 0
    n_short_after: int = 0

    one_way_turnover: float = 0.0        # from prior to new weights

    n_returns_missing: int = 0           # diagnostic: count of NaN returns for held names

    notes: list[str] = field(default_factory=list)


# -----------------------------------------------------------------------------
# State container
# -----------------------------------------------------------------------------


@dataclass
class AccountingState:
    """Mutable container; touched only via `Accountant`."""

    initial_nav: float = 1.0
    nav: float = 1.0
    history: list[DailyRecord] = field(default_factory=list)

    def reset(self, initial_nav: float = 1.0) -> None:
        self.initial_nav = initial_nav
        self.nav = initial_nav
        self.history = []


# -----------------------------------------------------------------------------
# Accountant
# -----------------------------------------------------------------------------


class Accountant:
    """Owns NAV and the per-day PnL ledger.

    Methods:
      - `book_day(...)` records one trading day's outcome.
      - `equity_curve` returns a NAV time series.
      - `daily_returns` returns the period-on-period net return series.
      - `to_frame()` materializes the full history as a DataFrame.
      - `summary_stats()` reports headline performance numbers.

    The Accountant does not compute weights or trades — those are the
    backtester's responsibility. It only books what it is told.
    """

    def __init__(self, initial_nav: float = 1.0) -> None:
        self.state = AccountingState(initial_nav=initial_nav, nav=initial_nav)

    # ------------------------------------------------------------------
    # Per-day bookkeeping
    # ------------------------------------------------------------------

    def book_day(
        self,
        date: pd.Timestamp,
        *,
        weights_held: pd.Series,
        returns: pd.Series,
        new_weights: pd.Series,
        transaction_cost_pct: float,
        one_way_turnover: float,
    ) -> DailyRecord:
        """Book one trading day and return a `DailyRecord`.

        Parameters
        ----------
        date
            The trading day t.
        weights_held
            Weights held DURING day t. Set at close of day t-1.
        returns
            Single-day returns on day t. Series indexed by ID. Names without
            a return are treated as 0 contribution (logged at INFO).
        new_weights
            Weights established at close of day t.
        transaction_cost_pct
            Cost paid for the rebalance at close of t, as a fraction of NAV.
        one_way_turnover
            Σ|trades| / 2 of the day's trading.
        """
        nav_open = self.state.nav

        # Realize PnL from weights_held × returns
        if weights_held.empty:
            gross_pnl_pct = 0.0
            long_pct = 0.0
            short_pct = 0.0
            n_missing = 0
        else:
            common = weights_held.index.intersection(returns.index)
            n_missing = int(len(weights_held) - len(common))

            r = returns.reindex(weights_held.index, fill_value=np.nan)
            r_filled = r.fillna(0.0)

            contributions = weights_held * r_filled
            gross_pnl_pct = float(contributions.sum())

            long_mask = weights_held > 0
            short_mask = weights_held < 0
            long_pct = float(contributions[long_mask].sum())
            short_pct = float(contributions[short_mask].sum())

        net_pnl_pct = gross_pnl_pct - transaction_cost_pct
        nav_close = nav_open * (1.0 + net_pnl_pct)

        record = DailyRecord(
            date=date,
            nav_open=nav_open,
            nav_close=nav_close,
            gross_pnl_pct=gross_pnl_pct,
            long_book_pnl_pct=long_pct,
            short_book_pnl_pct=short_pct,
            transaction_cost_pct=transaction_cost_pct,
            net_pnl_pct=net_pnl_pct,
            gross_exposure_after=float(new_weights.abs().sum()),
            net_exposure_after=float(new_weights.sum()),
            n_long_after=int((new_weights > 0).sum()),
            n_short_after=int((new_weights < 0).sum()),
            one_way_turnover=float(one_way_turnover),
            n_returns_missing=n_missing,
        )
        self.state.history.append(record)
        self.state.nav = nav_close
        return record

    # ------------------------------------------------------------------
    # Materialization and stats
    # ------------------------------------------------------------------

    def to_frame(self) -> pd.DataFrame:
        """Materialize the accounting history as a DataFrame indexed by date."""
        if not self.state.history:
            return pd.DataFrame()
        rows = [
            {
                "date": r.date,
                "nav_open": r.nav_open,
                "nav_close": r.nav_close,
                "gross_pnl_pct": r.gross_pnl_pct,
                "long_book_pnl_pct": r.long_book_pnl_pct,
                "short_book_pnl_pct": r.short_book_pnl_pct,
                "transaction_cost_pct": r.transaction_cost_pct,
                "net_pnl_pct": r.net_pnl_pct,
                "gross_exposure_after": r.gross_exposure_after,
                "net_exposure_after": r.net_exposure_after,
                "n_long_after": r.n_long_after,
                "n_short_after": r.n_short_after,
                "one_way_turnover": r.one_way_turnover,
                "n_returns_missing": r.n_returns_missing,
            }
            for r in self.state.history
        ]
        return pd.DataFrame(rows).set_index("date").sort_index()

    @property
    def equity_curve(self) -> pd.Series:
        """NAV time series, indexed by date."""
        df = self.to_frame()
        if df.empty:
            return pd.Series(dtype="float64", name="NAV")
        return df["nav_close"].rename("NAV")

    @property
    def daily_returns(self) -> pd.Series:
        """Net daily return series (decimal)."""
        df = self.to_frame()
        if df.empty:
            return pd.Series(dtype="float64", name="net_return")
        return df["net_pnl_pct"].rename("net_return")

    def summary_stats(self, freq: int = 252) -> dict:
        """Headline performance summary."""
        r = self.daily_returns.dropna()
        if r.empty:
            return {}
        equity = (1.0 + r).cumprod()
        max_dd = (equity / equity.cummax() - 1.0).min()
        ann_ret = r.mean() * freq
        ann_vol = r.std() * np.sqrt(freq)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else float("nan")
        return {
            "n_days": int(len(r)),
            "ann_return": float(ann_ret),
            "ann_vol": float(ann_vol),
            "sharpe": float(sharpe),
            "max_drawdown": float(max_dd),
            "hit_rate": float((r > 0).mean()),
            "skew": float(r.skew()),
            "kurtosis": float(r.kurtosis()),
            "total_return": float(equity.iloc[-1] - 1.0),
        }
