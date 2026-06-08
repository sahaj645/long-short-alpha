"""Shared pytest fixtures for the lsa test suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tiny_pit_panel() -> pd.DataFrame:
    """A minimal valid PIT panel: two IDs, two snapshots, business days only."""
    dates = pd.date_range("2023-01-02", "2023-02-28", freq="B")
    snapshots = pd.to_datetime(["2023-01-31", "2023-02-28"])

    rows = []
    for snap in snapshots:
        for sid in ("AAA UN Equity", "BBB UN Equity"):
            for d in dates:
                if d.month != snap.month:
                    continue
                rows.append({
                    "ID": sid,
                    "DATE": d,
                    "CURRENCY": "USD",
                    "ID_DATE": snap,
                    "Price": 100.0 + np.random.RandomState(hash((sid, d)) % 2**31).randn(),
                    "Volume": 1_000_000.0,
                    "Market_Cap": 1e10,
                    "Shares_Out": 1e8,
                    "GICS_Sector": "Information Technology",
                    "GICS_Ind_Group": "Software & Services",
                    "GICS_Industry": "Software",
                    "GICS_Sub_Industry": "Application Software",
                    "Index": "S&P 500",
                    "Index_Ticker": "SPX Index",
                    "PIT_Member_Date": snap,
                })
    return pd.DataFrame(rows)
