"""Performance analytics: metrics, trade statistics, composite reports."""

from .metrics import (
    DEFAULT_FREQ,
    PerformanceMetrics,
    annualized_arithmetic_return,
    annualized_geometric_return,
    annualized_volatility,
    cagr,
    calmar_ratio,
    compute_all_metrics,
    drawdown_series,
    hit_rate,
    information_ratio,
    kurtosis,
    max_drawdown,
    max_drawdown_duration,
    sharpe_ratio,
    skewness,
    sortino_ratio,
)
from .performance_report import (
    PerformanceReport,
    benchmark_comparison,
    generate_performance_report,
    monthly_return_table,
    rolling_sharpe,
    yearly_summary,
)
from .trade_statistics import (
    TradeStatistics,
    compute_episode_pnl,
    compute_trade_statistics,
    compute_turnover_statistics,
    extract_position_episodes,
    reconstruct_weights_panel,
)

__all__ = [
    # metrics
    "DEFAULT_FREQ",
    "PerformanceMetrics",
    "annualized_arithmetic_return",
    "annualized_geometric_return",
    "annualized_volatility",
    "cagr",
    "calmar_ratio",
    "compute_all_metrics",
    "drawdown_series",
    "hit_rate",
    "information_ratio",
    "kurtosis",
    "max_drawdown",
    "max_drawdown_duration",
    "sharpe_ratio",
    "skewness",
    "sortino_ratio",
    # trade_statistics
    "TradeStatistics",
    "compute_episode_pnl",
    "compute_trade_statistics",
    "compute_turnover_statistics",
    "extract_position_episodes",
    "reconstruct_weights_panel",
    # performance_report
    "PerformanceReport",
    "benchmark_comparison",
    "generate_performance_report",
    "monthly_return_table",
    "rolling_sharpe",
    "yearly_summary",
]
