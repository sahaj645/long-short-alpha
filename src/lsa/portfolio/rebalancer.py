"""
Portfolio Rebalancer — stateful daily orchestrator.

Owns:
  - Current weights and portfolio NAV.
  - Per-rebalance audit history (`RebalanceReport` objects).
  - Per-tier transaction cost model.

Per-rebalance flow:
  1. Build target weights from the day's signal panel (portfolio_builder).
  2. Apply the constraint pipeline with prior weights as turnover context.
  3. Compute trades = constrained - prior.
  4. Estimate transaction cost per the tier-aware model.
  5. Update NAV ← NAV × (1 − relative_cost).
  6. Record a `RebalanceReport` and append to history.

The drift step (`apply_drift`) is exposed separately so the caller can
update weights overnight using realized returns and rebalance from the
drifted positions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .constraints import (
    ConstraintConfig,
    ConstraintReport,
    apply_constraint_pipeline,
)
from .portfolio_builder import (
    PortfolioBuildConfig,
    build_target_weights_single_date,
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass
class TransactionCostConfig:
    """Per-tier transaction cost in basis points (one-way per side)."""

    sp500_bps: float = 3.0
    sp400_bps: float = 7.0
    sp600_bps: float = 12.0
    default_bps: float = 8.0

    def per_tier_bps(self, tier: str | None) -> float:
        """Look up bps for a tier label, falling back to default."""
        if tier is None:
            return self.default_bps
        return {
            "SP500": self.sp500_bps,
            "SP400": self.sp400_bps,
            "SP600": self.sp600_bps,
        }.get(tier, self.default_bps)


# -----------------------------------------------------------------------------
# Report objects
# -----------------------------------------------------------------------------


@dataclass
class RebalanceReport:
    """Snapshot of one rebalance day."""

    date: pd.Timestamp
    n_long: int = 0
    n_short: int = 0
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    turnover_one_way: float = 0.0
    transaction_cost_bps: float = 0.0
    nav_before: float = 1.0
    nav_after: float = 1.0
    constraint_report: Optional[ConstraintReport] = None


# -----------------------------------------------------------------------------
# State
# -----------------------------------------------------------------------------


@dataclass
class PortfolioState:
    """Mutable portfolio state. Touched only via `Rebalancer` methods."""

    weights: pd.Series = field(default_factory=lambda: pd.Series(dtype="float64"))
    nav: float = 1.0
    last_rebalance_date: Optional[pd.Timestamp] = None
    history: list[RebalanceReport] = field(default_factory=list)

    def reset(self, initial_nav: float = 1.0) -> None:
        """Clear positions and history."""
        self.weights = pd.Series(dtype="float64")
        self.nav = initial_nav
        self.last_rebalance_date = None
        self.history = []


# -----------------------------------------------------------------------------
# Rebalancer
# -----------------------------------------------------------------------------


class Rebalancer:
    """Daily portfolio rebalancer.

    Composes `portfolio_builder.build_target_weights_single_date` with the
    constraint pipeline and adds state, transaction cost, and drift handling.

    The rebalancer is constructed once per backtest and reused across all
    dates. Call `state.reset()` between independent runs.
    """

    def __init__(
        self,
        portfolio_config: PortfolioBuildConfig,
        constraint_config: ConstraintConfig,
        tc_config: TransactionCostConfig,
    ) -> None:
        self.portfolio_config = portfolio_config
        self.constraint_config = constraint_config
        self.tc_config = tc_config
        self.state = PortfolioState()

    # ------------------------------------------------------------------
    # Cost estimation
    # ------------------------------------------------------------------

    def estimate_transaction_cost(
        self,
        trades: pd.Series,
        tier_map: Optional[pd.Series] = None,
    ) -> float:
        """Estimate the relative-NAV cost of executing `trades`.

        For tier-aware costing, supply `tier_map` (ID -> {SP500, SP400, SP600}).
        Trades to names not in the map use the default rate.

        Returns
        -------
        float
            Total cost as a fraction of NAV (e.g., 0.0015 = 15 bps).
        """
        if trades.empty:
            return 0.0
        if tier_map is None:
            return float(trades.abs().sum() * self.tc_config.default_bps / 10_000)

        tier_aligned = tier_map.reindex(trades.index)
        per_name_bps = tier_aligned.map(self.tc_config.per_tier_bps).astype("float64")
        per_name_bps = per_name_bps.fillna(self.tc_config.default_bps)
        cost = float((trades.abs() * per_name_bps / 10_000).sum())
        return cost

    # ------------------------------------------------------------------
    # Drift (between rebalance days)
    # ------------------------------------------------------------------

    def apply_drift(
        self,
        prior_weights: pd.Series,
        returns: pd.Series,
        *,
        renormalize: bool = False,
    ) -> pd.Series:
        """Update weights overnight by applying realized returns.

        With `renormalize=False` (default), weights change because each
        position's value drifts independently. With `renormalize=True`,
        weights are renormalized to maintain the prior gross exposure
        (useful for visualization, not for accurate PnL tracking).
        """
        common = prior_weights.index.intersection(returns.index)
        drifted = prior_weights.copy()
        drifted.loc[common] = prior_weights.loc[common] * (
            1.0 + returns.loc[common].fillna(0.0)
        )
        if renormalize:
            gross = float(drifted.abs().sum())
            prior_gross = float(prior_weights.abs().sum())
            if gross > 0 and prior_gross > 0:
                drifted = drifted * (prior_gross / gross)
        return drifted

    # ------------------------------------------------------------------
    # Rebalance
    # ------------------------------------------------------------------

    def rebalance(
        self,
        signal_panel_today: pd.DataFrame,
        date: pd.Timestamp,
        *,
        adv: Optional[pd.Series] = None,
        tier_map: Optional[pd.Series] = None,
    ) -> RebalanceReport:
        """Run one rebalance day.

        Parameters
        ----------
        signal_panel_today
            Slice of the signal panel for `date`.
        date
            The rebalance date (used for reporting and state update).
        adv
            ID -> trailing dollar volume Series, for the liquidity constraint.
        tier_map
            ID -> {SP500, SP400, SP600} Series, for tier-aware cost estimation.
        """
        # 1. Build target weights from signal
        target_panel = build_target_weights_single_date(
            signal_panel_today, self.portfolio_config
        )
        cfg = self.portfolio_config
        if target_panel.empty:
            report = RebalanceReport(date=date, nav_before=self.state.nav, nav_after=self.state.nav)
            self.state.history.append(report)
            logger.warning("Empty signal panel on %s; no rebalance executed", date)
            return report

        target_weights = (
            target_panel.set_index(cfg.id_col)[cfg.weight_col]
            .astype("float64")
        )
        industry_map = (
            target_panel.set_index(cfg.id_col)[cfg.industry_col]
            if cfg.industry_col in target_panel.columns else None
        )
        sector_map = (
            target_panel.set_index(cfg.id_col)[cfg.sector_col]
            if cfg.sector_col in target_panel.columns else None
        )

        # 2. Apply constraints
        constrained, constraint_report = apply_constraint_pipeline(
            target_weights,
            self.constraint_config,
            industry_map=industry_map,
            sector_map=sector_map,
            adv=adv,
            portfolio_nav=self.state.nav,
            prior_weights=self.state.weights,
        )

        # 3. Compute trades
        all_ids = constrained.index.union(self.state.weights.index)
        new_aligned = constrained.reindex(all_ids, fill_value=0.0)
        old_aligned = self.state.weights.reindex(all_ids, fill_value=0.0)
        trades = new_aligned - old_aligned

        # 4. Estimate transaction cost
        cost = self.estimate_transaction_cost(trades, tier_map=tier_map)

        # 5. Update NAV
        nav_before = self.state.nav
        self.state.nav = nav_before * (1.0 - cost)
        nav_after = self.state.nav

        # 6. Update state
        # Drop zero-weight entries to keep the position book sparse
        nonzero = constrained.loc[constrained != 0.0].copy()
        self.state.weights = nonzero
        self.state.last_rebalance_date = date

        # 7. Report
        report = RebalanceReport(
            date=date,
            n_long=int((constrained > 0).sum()),
            n_short=int((constrained < 0).sum()),
            gross_exposure=float(constrained.abs().sum()),
            net_exposure=float(constrained.sum()),
            turnover_one_way=float(trades.abs().sum() / 2.0),
            transaction_cost_bps=cost * 10_000,
            nav_before=nav_before,
            nav_after=nav_after,
            constraint_report=constraint_report,
        )
        self.state.history.append(report)
        return report

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def history_as_frame(self) -> pd.DataFrame:
        """Materialize the rebalance history as a DataFrame."""
        if not self.state.history:
            return pd.DataFrame()
        return pd.DataFrame([
            {
                "date": r.date,
                "n_long": r.n_long,
                "n_short": r.n_short,
                "gross_exposure": r.gross_exposure,
                "net_exposure": r.net_exposure,
                "turnover_one_way": r.turnover_one_way,
                "transaction_cost_bps": r.transaction_cost_bps,
                "nav_before": r.nav_before,
                "nav_after": r.nav_after,
                "constraint_n_position_clips": (
                    r.constraint_report.n_position_clips
                    if r.constraint_report else 0
                ),
                "constraint_n_industry_adjustments": (
                    r.constraint_report.n_industry_adjustments
                    if r.constraint_report else 0
                ),
                "constraint_turnover_alpha": (
                    r.constraint_report.turnover_blend_alpha
                    if r.constraint_report else 1.0
                ),
            }
            for r in self.state.history
        ]).set_index("date").sort_index()


# -----------------------------------------------------------------------------
# Simulation driver
# -----------------------------------------------------------------------------


def run_simulation(
    signal_panel: pd.DataFrame,
    portfolio_config: PortfolioBuildConfig,
    constraint_config: ConstraintConfig,
    tc_config: TransactionCostConfig,
    *,
    adv_per_date: Optional[pd.DataFrame] = None,
    tier_per_date: Optional[pd.DataFrame] = None,
    initial_nav: float = 1.0,
) -> tuple[Rebalancer, pd.DataFrame]:
    """Run a full historical simulation over the dates in `signal_panel`.

    Parameters
    ----------
    signal_panel
        Multi-date signal panel (output of the signals layer).
    portfolio_config, constraint_config, tc_config
        Configuration objects.
    adv_per_date
        Optional long-form ADV table with columns (ID, DATE, ADV). Joined
        per rebalance day on (ID, DATE).
    tier_per_date
        Optional long-form tier table with columns (ID, DATE, tier).
        Used for tier-aware transaction costs.
    initial_nav
        Starting NAV (any consistent base unit).

    Returns
    -------
    (rebalancer, history_dataframe)
    """
    rebalancer = Rebalancer(portfolio_config, constraint_config, tc_config)
    rebalancer.state.reset(initial_nav=initial_nav)

    date_col = portfolio_config.date_col
    dates = sorted(pd.unique(signal_panel[date_col]))
    logger.info("Running simulation across %d dates", len(dates))

    for d in dates:
        today = signal_panel.loc[signal_panel[date_col] == d].copy()
        if today.empty:
            continue

        adv_series = None
        if adv_per_date is not None:
            adv_today = adv_per_date.loc[adv_per_date[date_col] == d]
            if not adv_today.empty:
                adv_series = adv_today.set_index(portfolio_config.id_col)["ADV"]

        tier_series = None
        if tier_per_date is not None:
            tier_today = tier_per_date.loc[tier_per_date[date_col] == d]
            if not tier_today.empty:
                tier_series = tier_today.set_index(portfolio_config.id_col)["tier"]

        rebalancer.rebalance(today, d, adv=adv_series, tier_map=tier_series)

    return rebalancer, rebalancer.history_as_frame()
