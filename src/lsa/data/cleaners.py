"""
Data cleaning operations for PIT equity panels and ETF OHLCV data.

All functions are pure: they accept and return new objects without mutating
their inputs. PIT integrity is preserved by:
  - Never imputing prices (no forward-fill).
  - Not crossing intra-ID gaps when computing returns.
  - Always emitting an audit trail (CleaningReport) alongside the data.

Module conventions
------------------
df          : input DataFrame
id_col      : identifier column name (default "ID")
date_col    : observation-date column name (default "DATE")
price_col   : close price column (default "Price")
volume_col  : share-volume column (default "Volume")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Quarantine thresholds — daily returns outside this band are clipped and flagged.
# Rationale: the data audit identified vendor adjustment artifacts (FSLR 2022-12-01,
# EQT 2022-10-03) producing > 100% one-day returns. A ±50% threshold catches them
# without clipping any real market event observed in the 2016-2025 sample.
DEFAULT_RETURN_LOWER: float = -0.50
DEFAULT_RETURN_UPPER: float = 0.50

# Maximum calendar-day gap between consecutive (ID, DATE) rows for a return to be
# computed across them. Longer gaps indicate non-membership periods (entered or
# left the index, suspended) and the implied return is not economically meaningful.
DEFAULT_MAX_GAP_DAYS: int = 5


@dataclass
class CleaningReport:
    """Audit trail emitted by clean_pit_panel and related orchestrators."""

    n_input_rows: int = 0
    n_output_rows: int = 0
    n_duplicates_dropped: int = 0
    n_non_trading_dropped: int = 0
    n_returns_winsorized: int = 0
    quarantine: pd.DataFrame = field(default_factory=pd.DataFrame)

    def summary(self) -> dict[str, int]:
        return {
            "n_input_rows": self.n_input_rows,
            "n_output_rows": self.n_output_rows,
            "n_duplicates_dropped": self.n_duplicates_dropped,
            "n_non_trading_dropped": self.n_non_trading_dropped,
            "n_returns_winsorized": self.n_returns_winsorized,
        }


def normalize_dates(
    df: pd.DataFrame,
    date_cols: Sequence[str],
    *,
    drop_tz: bool = True,
) -> pd.DataFrame:
    """Convert listed columns to tz-naive pandas Timestamps normalized to midnight.

    Parameters
    ----------
    df
        Input frame; not mutated.
    date_cols
        Names of columns to normalize. Missing columns raise KeyError.
    drop_tz
        If True (default), drop any timezone info; if False, preserve.

    Returns
    -------
    pd.DataFrame
        Copy of df with the specified columns normalized.
    """
    out = df.copy()
    for col in date_cols:
        if col not in out.columns:
            raise KeyError(f"Date column not found: {col!r}")
        out[col] = pd.to_datetime(out[col], errors="raise")
        if drop_tz and getattr(out[col].dt, "tz", None) is not None:
            out[col] = out[col].dt.tz_localize(None)
        out[col] = out[col].dt.normalize()
    return out


def drop_duplicate_observations(
    df: pd.DataFrame,
    key_cols: Sequence[str] = ("ID", "DATE"),
    *,
    keep: str = "last",
) -> tuple[pd.DataFrame, int]:
    """Drop duplicates on `key_cols` and return (deduped_df, n_dropped)."""
    n_before = len(df)
    out = df.drop_duplicates(subset=list(key_cols), keep=keep).reset_index(drop=True)
    n_dropped = n_before - len(out)
    if n_dropped > 0:
        logger.warning("Dropped %d duplicate rows on %s", n_dropped, tuple(key_cols))
    return out, n_dropped


def filter_to_trading_rows(
    df: pd.DataFrame,
    *,
    price_col: str = "Price",
) -> tuple[pd.DataFrame, int]:
    """Drop rows where `price_col` is NaN.

    NaN price covers weekends, market holidays, and any non-trading day for the
    security (pre-IPO, post-delisting, intraday suspension). This is the right
    filter to apply at load time; do NOT forward-fill — that would impute trades
    that did not occur.
    """
    n_before = len(df)
    out = df.dropna(subset=[price_col]).reset_index(drop=True)
    n_dropped = n_before - len(out)
    logger.info("Filtered %d non-trading rows of %d (%.1f%%)",
                n_dropped, n_before, 100.0 * n_dropped / max(n_before, 1))
    return out, n_dropped


def winsorize_returns(
    returns: pd.Series,
    *,
    lower: float = DEFAULT_RETURN_LOWER,
    upper: float = DEFAULT_RETURN_UPPER,
) -> tuple[pd.Series, pd.Series]:
    """Clip returns to [lower, upper]; return (clipped_series, flag_series).

    The flag_series is a boolean Series aligned with `returns` that is True
    where the value was clipped (used as the quarantine list for downstream
    diagnostics).
    """
    flags = (returns < lower) | (returns > upper)
    clipped = returns.clip(lower=lower, upper=upper)
    n_flagged = int(flags.fillna(False).sum())
    if n_flagged > 0:
        logger.warning("Winsorized %d returns outside [%g, %g]", n_flagged, lower, upper)
    return clipped, flags


def compute_returns(
    df: pd.DataFrame,
    *,
    id_col: str = "ID",
    date_col: str = "DATE",
    price_col: str = "Price",
    max_gap_days: int = DEFAULT_MAX_GAP_DAYS,
    out_col: str = "ret",
) -> pd.DataFrame:
    """Per-ID daily returns from Price, with gap-aware NaN insertion.

    Algorithm
    ---------
    1. Sort by (id_col, date_col) ascending.
    2. Within each ID, compute Price.pct_change() across consecutive rows.
    3. Where the consecutive (date_col) gap exceeds `max_gap_days`, set the
       return to NaN — prevents computing spurious returns across long
       non-membership intervals.

    The returned DataFrame is sorted by (id_col, date_col); the original index
    is reset. Caller may re-sort if a different ordering is required.
    """
    required = {id_col, date_col, price_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    out = df.sort_values([id_col, date_col]).reset_index(drop=True).copy()
    grp = out.groupby(id_col, sort=False, observed=True)
    rets = grp[price_col].pct_change()
    gap_days = grp[date_col].diff().dt.days
    rets = rets.mask(gap_days > max_gap_days)
    out[out_col] = rets
    return out


def compute_dollar_volume(
    df: pd.DataFrame,
    *,
    price_col: str = "Price",
    volume_col: str = "Volume",
    out_col: str = "dollar_volume",
) -> pd.DataFrame:
    """Add `out_col` = Price × Volume to a copy of df.

    NaN propagates from either input — this is intentional. Dollar volume on a
    non-trading row should be NaN, not zero.
    """
    out = df.copy()
    out[out_col] = out[price_col] * out[volume_col]
    return out


def clean_pit_panel(
    df: pd.DataFrame,
    *,
    id_col: str = "ID",
    date_col: str = "DATE",
    price_col: str = "Price",
    volume_col: str = "Volume",
    date_cols_to_normalize: Sequence[str] = ("DATE", "ID_DATE"),
    drop_pit_member_date: bool = True,
    winsorize_lower: float = DEFAULT_RETURN_LOWER,
    winsorize_upper: float = DEFAULT_RETURN_UPPER,
    max_gap_days: int = DEFAULT_MAX_GAP_DAYS,
) -> tuple[pd.DataFrame, CleaningReport]:
    """End-to-end cleaning pipeline for a single PIT panel.

    Pipeline (order matters)
    ------------------------
    1. Normalize date columns (DATE, ID_DATE).
    2. Drop the redundant PIT_Member_Date column if present.
    3. Drop duplicates on (id_col, date_col).
    4. Filter to trading rows (Price not NaN).
    5. Compute returns with gap-aware logic.
    6. Winsorize returns; preserve a quarantine sub-frame in the report.
    7. Compute dollar volume.

    Returns
    -------
    cleaned : pd.DataFrame
        Cleaned panel sorted by (id_col, date_col), with new columns
        `ret` and `dollar_volume`.
    report : CleaningReport
        Audit trail including the quarantine of winsorized observations.
    """
    report = CleaningReport(n_input_rows=len(df))

    present_date_cols = [c for c in date_cols_to_normalize if c in df.columns]
    df = normalize_dates(df, present_date_cols)

    if drop_pit_member_date and "PIT_Member_Date" in df.columns:
        df = df.drop(columns=["PIT_Member_Date"])

    df, n_dups = drop_duplicate_observations(df, (id_col, date_col))
    report.n_duplicates_dropped = n_dups

    df, n_filtered = filter_to_trading_rows(df, price_col=price_col)
    report.n_non_trading_dropped = n_filtered

    df = compute_returns(
        df,
        id_col=id_col,
        date_col=date_col,
        price_col=price_col,
        max_gap_days=max_gap_days,
    )

    raw_returns = df["ret"].copy()
    clipped, flags = winsorize_returns(raw_returns, lower=winsorize_lower, upper=winsorize_upper)
    df["ret"] = clipped

    flag_mask = flags.fillna(False).to_numpy()
    if flag_mask.any():
        quarantine = df.loc[flag_mask, [id_col, date_col, "ret", price_col]].copy()
        quarantine["raw_return"] = raw_returns.to_numpy()[flag_mask]
        report.quarantine = quarantine.reset_index(drop=True)
    report.n_returns_winsorized = int(flag_mask.sum())

    df = compute_dollar_volume(df, price_col=price_col, volume_col=volume_col)

    report.n_output_rows = len(df)
    logger.info("clean_pit_panel summary: %s", report.summary())
    return df, report
