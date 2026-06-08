"""Unit tests for lsa.data.cleaners."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lsa.data import cleaners


def test_normalize_dates_strips_time_and_tz():
    df = pd.DataFrame({"DATE": ["2023-01-02 09:30:00", "2023-01-03 16:00:00"]})
    out = cleaners.normalize_dates(df, ["DATE"])
    assert out["DATE"].dt.hour.eq(0).all()
    assert out["DATE"].dt.tz is None


def test_drop_duplicate_observations_counts_and_keeps_last():
    df = pd.DataFrame({
        "ID": ["A", "A", "B"],
        "DATE": pd.to_datetime(["2023-01-02", "2023-01-02", "2023-01-02"]),
        "Price": [100.0, 101.0, 50.0],
    })
    out, n_dropped = cleaners.drop_duplicate_observations(df)
    assert n_dropped == 1
    assert len(out) == 2
    # keep='last' should retain the 101.0 row
    assert out.loc[out["ID"] == "A", "Price"].iloc[0] == 101.0


def test_filter_to_trading_rows_drops_nan_prices():
    df = pd.DataFrame({"Price": [10.0, np.nan, 12.0]})
    out, n_dropped = cleaners.filter_to_trading_rows(df)
    assert n_dropped == 1
    assert len(out) == 2


def test_winsorize_returns_flags_and_clips():
    rets = pd.Series([0.01, 0.60, -0.55, np.nan])
    clipped, flags = cleaners.winsorize_returns(rets, lower=-0.5, upper=0.5)
    assert clipped.iloc[1] == 0.5
    assert clipped.iloc[2] == -0.5
    assert flags.iloc[0] is np.False_ or flags.iloc[0] == False  # noqa: E712
    assert flags.iloc[1] == True   # noqa: E712
    assert flags.iloc[2] == True   # noqa: E712


def test_compute_returns_masks_long_gaps():
    df = pd.DataFrame({
        "ID": ["A", "A", "A"],
        "DATE": pd.to_datetime(["2023-01-02", "2023-01-03", "2023-07-03"]),
        "Price": [100.0, 101.0, 200.0],
    })
    out = cleaners.compute_returns(df, max_gap_days=5)
    # Day 2 -> Day 3: normal 1-day gap; return computed
    assert out["ret"].iloc[1] == pytest.approx(0.01)
    # Day 3 -> Day 4 (181 calendar days later): too long, return masked to NaN
    assert pd.isna(out["ret"].iloc[2])


def test_compute_dollar_volume_propagates_nan():
    df = pd.DataFrame({"Price": [10.0, np.nan], "Volume": [1000.0, 2000.0]})
    out = cleaners.compute_dollar_volume(df)
    assert out["dollar_volume"].iloc[0] == 10_000
    assert pd.isna(out["dollar_volume"].iloc[1])


def test_clean_pit_panel_orchestrator_runs_end_to_end(tiny_pit_panel):
    cleaned, report = cleaners.clean_pit_panel(tiny_pit_panel)
    assert "ret" in cleaned.columns
    assert "dollar_volume" in cleaned.columns
    assert "PIT_Member_Date" not in cleaned.columns
    assert report.n_input_rows >= report.n_output_rows
    assert report.n_duplicates_dropped == 0
