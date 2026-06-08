"""Signal construction: residual momentum baseline, lead-lag (H6), and friends."""

from .residual_momentum import (
    DEFAULT_SECTOR_TO_ETF,
    StrategyResult,
    build_long_short_weights,
    build_residual_momentum_strategy,
    compute_residual_momentum,
    compute_residual_returns,
    compute_rolling_betas,
    compute_strategy_returns,
    rank_within_subindustry,
    sector_to_etf_mapping,
    verify_dollar_neutrality,
)

__all__ = [
    "DEFAULT_SECTOR_TO_ETF",
    "StrategyResult",
    "build_long_short_weights",
    "build_residual_momentum_strategy",
    "compute_residual_momentum",
    "compute_residual_returns",
    "compute_rolling_betas",
    "compute_strategy_returns",
    "rank_within_subindustry",
    "sector_to_etf_mapping",
    "verify_dollar_neutrality",
]
