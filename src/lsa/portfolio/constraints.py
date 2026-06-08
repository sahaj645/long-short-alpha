"""
Portfolio Constraints — enforce position, exposure, liquidity, turnover and
dollar-neutrality limits on proposed weights.

Six primitive constraint functions, plus the `apply_constraint_pipeline`
orchestrator that runs them in the canonical order and returns a structured
`ConstraintReport`.

Order in the pipeline
---------------------
1. `clip_max_position` — per-name absolute weight cap.
2. `enforce_group_exposure_cap(industry)` — per-industry net exposure cap.
3. `enforce_group_exposure_cap(sector)` — per-sector net exposure cap.
4. `enforce_liquidity_cap` — per-name participation rate cap vs ADV.
5. `apply_turnover_cap` — blend toward prior weights if daily turnover exceeds cap.
6. `enforce_dollar_neutrality` — bring |Σweights| within tolerance.

Each constraint is a pure function: takes a Series of weights (plus context),
returns a Series of weights. The pipeline accumulates a `ConstraintReport`
with pre / post statistics so the rebalancer (or a unit test) can audit
what changed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ConstraintConfig:
    """Constraint thresholds. Mirrors configs/base.yaml when wired."""

    # Per-name caps
    max_position: float = 0.05                  # max |weight| per name (5% of gross)

    # Group exposure caps (signed; |Σweights per group| must be <= cap)
    max_industry_net_exposure: float = 0.05     # 5% of gross
    max_sector_net_exposure: float = 0.10       # 10% of gross

    # Dollar neutrality
    dollar_neutrality_tolerance: float = 1e-4   # |Σweights| <= tolerance

    # Liquidity
    max_participation_rate: float = 0.10        # % of trailing ADV per name
    enable_liquidity_cap: bool = True

    # Turnover
    max_daily_turnover: float = 0.50            # 50% one-way turnover per day
    enable_turnover_cap: bool = True


@dataclass
class ConstraintReport:
    """Audit trail of one constraint pipeline pass."""

    # Position
    n_position_clips: int = 0
    pre_max_position: float = 0.0
    post_max_position: float = 0.0

    # Industry
    n_industry_adjustments: int = 0
    max_industry_net_pre: float = 0.0
    max_industry_net_post: float = 0.0

    # Sector
    n_sector_adjustments: int = 0
    max_sector_net_pre: float = 0.0
    max_sector_net_post: float = 0.0

    # Liquidity
    n_liquidity_clips: int = 0

    # Turnover
    pre_turnover: float = 0.0
    post_turnover: float = 0.0
    turnover_blend_alpha: float = 1.0           # 1.0 = full rebalance; < 1.0 = blended

    # Gross / net
    pre_gross: float = 0.0
    post_gross: float = 0.0
    pre_net: float = 0.0
    post_net: float = 0.0

    notes: list[str] = field(default_factory=list)


# -----------------------------------------------------------------------------
# Constraint primitives
# -----------------------------------------------------------------------------


def clip_max_position(
    weights: pd.Series, max_position: float
) -> tuple[pd.Series, int, float, float]:
    """Clip each |weight| to `max_position`.

    Returns
    -------
    (clipped, n_clipped, pre_max_abs, post_max_abs)
    """
    if weights.empty:
        return weights, 0, 0.0, 0.0
    abs_w = weights.abs()
    pre_max = float(abs_w.max())
    over = abs_w > max_position
    n_clip = int(over.sum())
    if n_clip == 0:
        return weights, 0, pre_max, pre_max
    out = weights.clip(lower=-max_position, upper=max_position)
    return out, n_clip, pre_max, float(out.abs().max())


def enforce_group_exposure_cap(
    weights: pd.Series, group: pd.Series, max_net_exposure: float
) -> tuple[pd.Series, int, float, float]:
    """Cap |Σweights| within each group at `max_net_exposure`.

    Algorithm
    ---------
    For each group whose absolute net exposure exceeds the cap, scale the
    dominant side of the group's weights inward until the cap binds.
    Specifically, if a group is net long X > cap, scale its long weights by
    (cap + |shorts|) / sum(longs). The symmetric rule applies for net-short
    groups. Groups with only longs or only shorts get all positions scaled
    toward zero by the cap-to-excess ratio.
    """
    if weights.empty or group.empty:
        return weights, 0, 0.0, 0.0

    df = pd.DataFrame({"w": weights, "g": group})
    df = df.dropna(subset=["g"])
    if df.empty:
        return weights, 0, 0.0, 0.0

    group_net = df.groupby("g")["w"].sum()
    pre_max = float(group_net.abs().max())

    out = weights.copy()
    n_adjust = 0
    for g, net in group_net.items():
        if abs(net) <= max_net_exposure:
            continue
        n_adjust += 1
        idx = df[df["g"] == g].index
        sub = df.loc[idx, "w"]
        long_idx = sub[sub > 0].index
        short_idx = sub[sub < 0].index
        long_sum = float(sub[sub > 0].sum())
        short_sum = float(sub[sub < 0].sum())

        if net > max_net_exposure and len(long_idx) > 0:
            # Reduce longs: target_long_sum = max_net_exposure + |short_sum|
            target = max_net_exposure + abs(short_sum)
            scale = target / long_sum if long_sum > 0 else 1.0
            out.loc[long_idx] *= scale
        elif net < -max_net_exposure and len(short_idx) > 0:
            # Boost (less negative) shorts: target_short_sum = -(max + long_sum)
            target = -(max_net_exposure + long_sum)
            scale = target / short_sum if short_sum < 0 else 1.0
            out.loc[short_idx] *= scale
        else:
            # Single-side group exceeding cap: scale toward cap
            scale = max_net_exposure / abs(net)
            out.loc[idx] *= scale

    post_net = pd.DataFrame({"w": out, "g": group}).dropna().groupby("g")["w"].sum().abs()
    post_max = float(post_net.max()) if len(post_net) else 0.0
    return out, n_adjust, pre_max, post_max


def enforce_liquidity_cap(
    weights: pd.Series, adv: pd.Series, portfolio_nav: float, max_participation_rate: float
) -> tuple[pd.Series, int]:
    """Clip |weight × NAV| ≤ max_participation_rate × ADV per name.

    Names with NaN ADV (unknown liquidity) are left untouched. The caller can
    choose to exclude unknown-ADV names upstream if desired.
    """
    if weights.empty or adv is None or portfolio_nav <= 0:
        return weights, 0

    adv_aligned = adv.reindex(weights.index)
    max_dollar_position = max_participation_rate * adv_aligned
    current_dollar = weights.abs() * portfolio_nav
    over = (current_dollar > max_dollar_position) & adv_aligned.notna()
    n_clip = int(over.sum())
    if n_clip == 0:
        return weights, 0

    max_weight = (max_dollar_position / portfolio_nav).fillna(np.inf)
    out = np.sign(weights) * np.minimum(weights.abs(), max_weight)
    return pd.Series(out, index=weights.index, name=weights.name), n_clip


def apply_turnover_cap(
    new_weights: pd.Series, old_weights: pd.Series, max_turnover: float
) -> tuple[pd.Series, float, float, float]:
    """If realized one-way turnover > cap, blend new and old: w = (1−α)·old + α·new.

    α = cap / raw_turnover when raw_turnover > cap; α = 1 otherwise.

    Returns
    -------
    (blended_weights, pre_turnover, post_turnover, alpha)
    """
    all_idx = new_weights.index.union(old_weights.index)
    new_aligned = new_weights.reindex(all_idx, fill_value=0.0)
    old_aligned = old_weights.reindex(all_idx, fill_value=0.0)

    delta = new_aligned - old_aligned
    raw_turnover = float(delta.abs().sum() / 2.0)
    if raw_turnover <= max_turnover or raw_turnover == 0:
        return new_weights, raw_turnover, raw_turnover, 1.0

    alpha = max_turnover / raw_turnover
    blended = old_aligned + alpha * delta
    post_turnover = float((blended - old_aligned).abs().sum() / 2.0)
    return blended.reindex(new_weights.index), raw_turnover, post_turnover, alpha


def enforce_dollar_neutrality(
    weights: pd.Series, tolerance: float
) -> tuple[pd.Series, float, float]:
    """Bring |Σweights| within `tolerance` by subtracting the per-row mean.

    Returns
    -------
    (adjusted_weights, pre_net, post_net)
    """
    if weights.empty:
        return weights, 0.0, 0.0
    pre_net = float(weights.sum())
    if abs(pre_net) <= tolerance:
        return weights, pre_net, pre_net
    out = weights - (pre_net / len(weights))
    return out, pre_net, float(out.sum())


# -----------------------------------------------------------------------------
# Pipeline orchestrator
# -----------------------------------------------------------------------------


def apply_constraint_pipeline(
    proposed_weights: pd.Series,
    config: ConstraintConfig,
    *,
    industry_map: Optional[pd.Series] = None,
    sector_map: Optional[pd.Series] = None,
    adv: Optional[pd.Series] = None,
    portfolio_nav: float = 1.0,
    prior_weights: Optional[pd.Series] = None,
) -> tuple[pd.Series, ConstraintReport]:
    """Apply all enabled constraints in the canonical order.

    Parameters
    ----------
    proposed_weights
        Output of `portfolio_builder.build_target_weights_*`.
    config
        Threshold parameters.
    industry_map, sector_map
        Series mapping ID -> industry/sector. Required for the group-exposure
        constraints; if None, those constraints are skipped.
    adv
        Series mapping ID -> trailing dollar-volume. Required for liquidity
        constraint; if None, that constraint is skipped.
    portfolio_nav
        Notional NAV in USD (or any consistent base). Used for liquidity sizing.
    prior_weights
        Previous-day weights, indexed by ID. Required for turnover cap.

    Returns
    -------
    (constrained_weights, ConstraintReport)
    """
    report = ConstraintReport()
    w = proposed_weights.copy()
    report.pre_gross = float(w.abs().sum())
    report.pre_net = float(w.sum())

    # 1. Per-name max position
    w, n_clip, pre_max, post_max = clip_max_position(w, config.max_position)
    report.n_position_clips = n_clip
    report.pre_max_position = pre_max
    report.post_max_position = post_max

    # 2. Per-industry net exposure cap
    if industry_map is not None:
        w, n_adj, pre, post = enforce_group_exposure_cap(
            w, industry_map.reindex(w.index), config.max_industry_net_exposure
        )
        report.n_industry_adjustments = n_adj
        report.max_industry_net_pre = pre
        report.max_industry_net_post = post

    # 3. Per-sector net exposure cap
    if sector_map is not None:
        w, n_adj, pre, post = enforce_group_exposure_cap(
            w, sector_map.reindex(w.index), config.max_sector_net_exposure
        )
        report.n_sector_adjustments = n_adj
        report.max_sector_net_pre = pre
        report.max_sector_net_post = post

    # 4. Liquidity cap
    if config.enable_liquidity_cap and adv is not None:
        w, n_clip = enforce_liquidity_cap(
            w, adv, portfolio_nav, config.max_participation_rate
        )
        report.n_liquidity_clips = n_clip

    # 5. Turnover cap
    if config.enable_turnover_cap and prior_weights is not None and not prior_weights.empty:
        w, pre_t, post_t, alpha = apply_turnover_cap(
            w, prior_weights, config.max_daily_turnover
        )
        report.pre_turnover = pre_t
        report.post_turnover = post_t
        report.turnover_blend_alpha = alpha

    # 6. Dollar neutrality
    w, pre_net, post_net = enforce_dollar_neutrality(
        w, config.dollar_neutrality_tolerance
    )

    report.post_gross = float(w.abs().sum())
    report.post_net = float(w.sum())

    logger.debug(
        "Constraints: pos_clips=%d, ind_adj=%d, sec_adj=%d, liq_clips=%d, "
        "turnover=%.3f→%.3f (α=%.3f), net=%.6f→%.6f",
        report.n_position_clips, report.n_industry_adjustments,
        report.n_sector_adjustments, report.n_liquidity_clips,
        report.pre_turnover, report.post_turnover, report.turnover_blend_alpha,
        report.pre_net, report.post_net,
    )
    return w, report
