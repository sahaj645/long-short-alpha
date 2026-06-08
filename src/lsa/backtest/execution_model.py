"""
Execution Model — trade generation and transaction cost estimation.

Stateless module: given a target weight Series and a prior weight Series,
returns the trades and an estimate of the cost of executing them. Three cost
models are supported:

- **flat**: a single `flat_bps` applied to gross trade magnitude.
- **tiered**: per-name basis-point cost looked up via a `tier_map`
  (SP500 / SP400 / SP600 → bps), with a default for unknown names.
- **sqrt_impact**: `cost_bps = impact_alpha × √(participation_rate)`,
  where `participation_rate = |trade_dollars| / ADV`. Models the canonical
  square-root impact function.

Optional `slippage_bps` is added on top of whichever cost model is chosen.

The model returns an `ExecutionReport` rather than raw numbers so the
backtester can record full audit details per day.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass
class ExecutionConfig:
    """Execution-model parameters."""

    cost_model: Literal["flat", "tiered", "sqrt_impact"] = "tiered"

    # 'flat' parameters
    flat_bps: float = 8.0

    # 'tiered' parameters (round-trip → divide by 2 if you want per-side)
    sp500_bps: float = 3.0
    sp400_bps: float = 7.0
    sp600_bps: float = 12.0
    default_bps: float = 8.0

    # 'sqrt_impact' parameters: cost(bps) = impact_alpha × √(participation)
    impact_alpha: float = 10.0   # ~10 bps at 1% participation, ~32 bps at 10%

    # Common
    slippage_bps: float = 0.0
    minimum_trade_dollars: float = 0.0   # 0 = no minimum trade size

    def tier_bps(self, tier: Optional[str]) -> float:
        """Look up cost in bps for a tier label."""
        if tier is None:
            return self.default_bps
        return {
            "SP500": self.sp500_bps,
            "SP400": self.sp400_bps,
            "SP600": self.sp600_bps,
        }.get(tier, self.default_bps)


# -----------------------------------------------------------------------------
# Report
# -----------------------------------------------------------------------------


@dataclass
class ExecutionReport:
    """Per-day execution outcome."""

    date: pd.Timestamp
    trades: pd.Series
    n_trades: int = 0
    one_way_turnover: float = 0.0
    gross_traded: float = 0.0
    transaction_cost_pct: float = 0.0       # fraction of NAV
    transaction_cost_bps: float = 0.0       # NAV bps
    cost_breakdown_bps: dict = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------


class ExecutionModel:
    """Stateless trade generator and cost estimator.

    A single instance is constructed at backtest start and used across all
    days. No internal state — all parameters live in `config`.
    """

    def __init__(self, config: ExecutionConfig):
        self.config = config

    # ------------------------------------------------------------------
    # Trade generation
    # ------------------------------------------------------------------

    def compute_trades(
        self, target_weights: pd.Series, prior_weights: pd.Series
    ) -> pd.Series:
        """Return Δw = target − prior over the union of the two indices.

        Names appearing in only one side default to 0 in the other. The result
        is the change in weight required to move from prior to target. A
        positive value is a buy; a negative value is a sell or short add.
        """
        all_idx = target_weights.index.union(prior_weights.index)
        target_aligned = target_weights.reindex(all_idx, fill_value=0.0)
        prior_aligned = prior_weights.reindex(all_idx, fill_value=0.0)
        trades = target_aligned - prior_aligned

        # Apply minimum-trade-size filter
        if self.config.minimum_trade_dollars > 0:
            # Assumes weights are normalized to gross 2.0; trades in weight space
            min_weight = self.config.minimum_trade_dollars  # caller responsible for scale
            trades = trades.where(trades.abs() >= min_weight, 0.0)

        return trades

    # ------------------------------------------------------------------
    # Cost estimation
    # ------------------------------------------------------------------

    def estimate_cost(
        self,
        trades: pd.Series,
        *,
        tier_map: Optional[pd.Series] = None,
        adv: Optional[pd.Series] = None,
        nav: float = 1.0,
    ) -> tuple[float, dict]:
        """Estimate cost as a fraction of NAV.

        Parameters
        ----------
        trades
            Δw series (output of `compute_trades`). Assumed in weight units.
        tier_map
            ID → tier name. Used by the 'tiered' cost model.
        adv
            ID → trailing dollar volume. Required by 'sqrt_impact'.
        nav
            Portfolio NAV in the same dollar units as ADV. Used by 'sqrt_impact'.

        Returns
        -------
        (cost_pct, breakdown_bps)
            cost_pct is the fraction of NAV consumed. breakdown_bps is a
            dict of cost components in NAV-bps for the audit trail.
        """
        breakdown: dict[str, float] = {}
        if trades.empty:
            return 0.0, breakdown

        abs_trades = trades.abs()

        # --- model cost ---
        if self.config.cost_model == "flat":
            cost_pct = float((abs_trades * self.config.flat_bps / 10_000).sum())
            breakdown["model"] = cost_pct * 10_000

        elif self.config.cost_model == "tiered":
            if tier_map is None:
                logger.warning(
                    "tiered cost model with no tier_map; falling back to default_bps"
                )
                per_name_bps = pd.Series(
                    self.config.default_bps, index=abs_trades.index
                )
            else:
                per_name_bps = (
                    tier_map.reindex(abs_trades.index)
                    .map(self.config.tier_bps)
                    .astype("float64")
                    .fillna(self.config.default_bps)
                )
            cost_pct = float((abs_trades * per_name_bps / 10_000).sum())
            breakdown["model"] = cost_pct * 10_000

        elif self.config.cost_model == "sqrt_impact":
            if adv is None:
                raise ValueError("sqrt_impact cost model requires `adv` series")
            adv_aligned = adv.reindex(abs_trades.index)
            with np.errstate(divide="ignore", invalid="ignore"):
                participation = (abs_trades * nav) / adv_aligned
            participation = participation.fillna(0.0).clip(lower=0.0)
            cost_bps = self.config.impact_alpha * np.sqrt(participation)
            cost_pct = float((abs_trades * cost_bps / 10_000).sum())
            breakdown["model"] = cost_pct * 10_000

        else:
            raise ValueError(f"Unknown cost_model: {self.config.cost_model!r}")

        # --- slippage adder ---
        slip_pct = 0.0
        if self.config.slippage_bps > 0:
            slip_pct = float((abs_trades * self.config.slippage_bps / 10_000).sum())
            breakdown["slippage"] = slip_pct * 10_000

        total_pct = cost_pct + slip_pct
        breakdown["total"] = total_pct * 10_000
        return total_pct, breakdown

    # ------------------------------------------------------------------
    # Full execute
    # ------------------------------------------------------------------

    def execute(
        self,
        target_weights: pd.Series,
        prior_weights: pd.Series,
        date: pd.Timestamp,
        *,
        tier_map: Optional[pd.Series] = None,
        adv: Optional[pd.Series] = None,
        nav: float = 1.0,
    ) -> ExecutionReport:
        """Generate trades, estimate cost, package as ExecutionReport."""
        trades = self.compute_trades(target_weights, prior_weights)
        cost_pct, breakdown = self.estimate_cost(
            trades, tier_map=tier_map, adv=adv, nav=nav
        )
        return ExecutionReport(
            date=date,
            trades=trades,
            n_trades=int((trades.abs() > 1e-12).sum()),
            one_way_turnover=float(trades.abs().sum() / 2.0),
            gross_traded=float(trades.abs().sum()),
            transaction_cost_pct=cost_pct,
            transaction_cost_bps=cost_pct * 10_000,
            cost_breakdown_bps=breakdown,
        )
