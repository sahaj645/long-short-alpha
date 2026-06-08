"""
Backtester — event-driven daily portfolio simulation.

The backtester is the only stateful orchestrator in the strategy stack. It
takes a PIT signal panel and a returns panel and replays the strategy day
by day. The execution sequence is:

    Step A.  Realize PnL from weights set at close of day t-1, multiplied
             by the returns on day t. No signal_t is touched yet — using it
             would be lookahead.
    Step B.  Build new target weights from signal_t (computed using data
             through close of day t).
    Step C.  Apply the constraint pipeline, with the previous day's weights
             as turnover context.
    Step D.  Compute trades = w_t − w_{t-1} and estimate transaction cost.
    Step E.  Book the day with the Accountant: nav update, PnL attribution.
    Step F.  Roll w_{t-1} ← w_t for the next iteration.

The backtester does NOT compute signals or do residualization. It expects a
`signal_panel` with the signal already lagged to be PIT-safe at date t. This
contract is enforced socially (the signal layer is responsible) rather than
mechanically — the backtester cannot verify lookahead in the signal itself.
The backtester DOES guarantee that within its own loop, signal_t is used only
to set w_t (not to compute PnL for day t).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from lsa.portfolio import (
    ConstraintConfig,
    PortfolioBuildConfig,
    apply_constraint_pipeline,
    build_target_weights_single_date,
)

from .accounting import Accountant, DailyRecord
from .execution_model import ExecutionConfig, ExecutionModel, ExecutionReport

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Configuration & result objects
# -----------------------------------------------------------------------------


@dataclass
class BacktestConfig:
    """Top-level backtest configuration."""

    portfolio: PortfolioBuildConfig = field(default_factory=PortfolioBuildConfig)
    constraints: ConstraintConfig = field(default_factory=ConstraintConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)

    initial_nav: float = 1.0

    # Column conventions for the returns panel
    return_col: str = "ret"

    # Diagnostics / logging
    progress_log_every_n_days: int = 252
    fail_on_excessive_missing_returns: bool = False


@dataclass
class BacktestResult:
    """Container for everything produced by `Backtester.run`."""

    accounting_frame: pd.DataFrame
    equity_curve: pd.Series
    daily_returns: pd.Series
    summary_stats: dict

    rebalance_history: list[ExecutionReport] = field(default_factory=list)
    daily_records: list[DailyRecord] = field(default_factory=list)


# -----------------------------------------------------------------------------
# Backtester
# -----------------------------------------------------------------------------


class Backtester:
    """Event-driven daily backtester.

    Construct once with a `BacktestConfig`. Call `.run(signal_panel,
    returns_panel)` to simulate. The instance retains the most recent run's
    state (`accountant`, `execution_model`, `current_weights`) for
    post-mortem inspection; re-running clears the state automatically.
    """

    def __init__(self, config: BacktestConfig) -> None:
        self.config = config
        self.execution_model = ExecutionModel(config.execution)
        self.accountant = Accountant(initial_nav=config.initial_nav)

        self._current_weights: pd.Series = pd.Series(dtype="float64")
        self._rebalance_history: list[ExecutionReport] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset to a fresh state. Called automatically at the start of `run`."""
        self.accountant.state.reset(initial_nav=self.config.initial_nav)
        self._current_weights = pd.Series(dtype="float64")
        self._rebalance_history = []

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(
        self,
        signal_panel: pd.DataFrame,
        returns_panel: pd.DataFrame,
        *,
        adv_panel: Optional[pd.DataFrame] = None,
        tier_panel: Optional[pd.DataFrame] = None,
    ) -> BacktestResult:
        """Replay the strategy over all dates in `signal_panel`.

        Parameters
        ----------
        signal_panel
            Long-form panel with at minimum (id_col, date_col, score_col,
            industry_col, sector_col). Signal at date t must already be
            lagged to be PIT-safe.
        returns_panel
            Long-form panel with (id_col, date_col, return_col). Provides the
            close-to-close return on each name on each date.
        adv_panel
            Optional. Long-form (id_col, date_col, 'ADV') — daily trailing
            dollar volume. Required by the sqrt_impact cost model and the
            liquidity constraint.
        tier_panel
            Optional. Long-form (id_col, date_col, 'tier') — SP500 / SP400 /
            SP600 label per name per date. Used by the tiered cost model.

        Returns
        -------
        BacktestResult
        """
        self.reset()
        port_cfg = self.config.portfolio
        ret_col = self.config.return_col

        # Sort once, index for fast slicing
        signal_idx = signal_panel.sort_values(port_cfg.date_col).set_index(port_cfg.date_col)
        returns_idx = returns_panel.sort_values(port_cfg.date_col).set_index(port_cfg.date_col)
        dates = sorted(signal_idx.index.unique())

        if not dates:
            logger.warning("Empty signal panel; nothing to backtest")
            return self._build_result()

        logger.info("Backtest over %d dates: %s → %s",
                    len(dates), dates[0], dates[-1])

        for i, date in enumerate(dates):
            # ------- Step A: realize PnL from prior weights × today's returns -----
            try:
                today_returns_df = returns_idx.loc[[date]]
            except KeyError:
                today_returns_df = returns_idx.iloc[0:0]
            today_returns = (
                today_returns_df.set_index(port_cfg.id_col)[ret_col]
                if not today_returns_df.empty else pd.Series(dtype="float64")
            )

            # ------- Step B: build target weights from today's signal -----------
            try:
                today_signal_df = signal_idx.loc[[date]].reset_index()
            except KeyError:
                today_signal_df = signal_panel.iloc[0:0]

            if today_signal_df.empty:
                # No signal today → hold prior weights
                target_weights = self._current_weights.copy()
                industry_map = None
                sector_map = None
            else:
                target_panel = build_target_weights_single_date(today_signal_df, port_cfg)
                target_weights = (
                    target_panel.set_index(port_cfg.id_col)[port_cfg.weight_col]
                    .astype("float64")
                )
                industry_map = (
                    target_panel.set_index(port_cfg.id_col)[port_cfg.industry_col]
                    if port_cfg.industry_col in target_panel.columns else None
                )
                sector_map = (
                    target_panel.set_index(port_cfg.id_col)[port_cfg.sector_col]
                    if port_cfg.sector_col in target_panel.columns else None
                )

            # ------- Step C: constraints ----------------------------------------
            adv_today = self._slice_series(adv_panel, date, "ADV", port_cfg)
            tier_today = self._slice_series(tier_panel, date, "tier", port_cfg)

            new_weights, _ = apply_constraint_pipeline(
                target_weights,
                self.config.constraints,
                industry_map=industry_map,
                sector_map=sector_map,
                adv=adv_today,
                portfolio_nav=self.accountant.state.nav,
                prior_weights=self._current_weights,
            )

            # ------- Step D: trades and execution cost --------------------------
            exec_report = self.execution_model.execute(
                target_weights=new_weights,
                prior_weights=self._current_weights,
                date=date,
                tier_map=tier_today,
                adv=adv_today,
                nav=self.accountant.state.nav,
            )
            self._rebalance_history.append(exec_report)

            # ------- Step E: book the day --------------------------------------
            self.accountant.book_day(
                date=date,
                weights_held=self._current_weights,    # weights DURING day t
                returns=today_returns,                  # returns on day t
                new_weights=new_weights,                # weights set at close of t
                transaction_cost_pct=exec_report.transaction_cost_pct,
                one_way_turnover=exec_report.one_way_turnover,
            )

            # ------- Step F: roll state ----------------------------------------
            self._current_weights = new_weights.loc[new_weights != 0.0].copy()

            # ------- Progress logging ------------------------------------------
            if (i + 1) % self.config.progress_log_every_n_days == 0:
                logger.info(
                    "Backtest progress: %d/%d days. NAV=%.4f",
                    i + 1, len(dates), self.accountant.state.nav,
                )

        return self._build_result()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _slice_series(
        panel: Optional[pd.DataFrame],
        date: pd.Timestamp,
        value_col: str,
        port_cfg: PortfolioBuildConfig,
    ) -> Optional[pd.Series]:
        """Slice an optional panel to a single date and return a Series."""
        if panel is None:
            return None
        sub = panel.loc[panel[port_cfg.date_col] == date]
        if sub.empty or value_col not in sub.columns:
            return None
        return sub.set_index(port_cfg.id_col)[value_col]

    def _build_result(self) -> BacktestResult:
        return BacktestResult(
            accounting_frame=self.accountant.to_frame(),
            equity_curve=self.accountant.equity_curve,
            daily_returns=self.accountant.daily_returns,
            summary_stats=self.accountant.summary_stats(),
            rebalance_history=list(self._rebalance_history),
            daily_records=list(self.accountant.state.history),
        )
