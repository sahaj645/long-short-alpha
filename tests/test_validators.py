"""Unit tests for lsa.data.validators."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lsa.data import validators


def test_schema_detects_missing_column():
    df = pd.DataFrame({"ID": ["A"], "DATE": pd.to_datetime(["2023-01-02"])})
    report = validators.validate_schema(df, validators.PIT_PANEL_SCHEMA)
    assert not report.passed  # Missing many critical columns


def test_schema_passes_on_complete_panel(tiny_pit_panel):
    report = validators.validate_schema(tiny_pit_panel, validators.PIT_PANEL_SCHEMA)
    assert report.passed


def test_missing_values_flags_critical_nan_in_id():
    df = pd.DataFrame({"ID": ["A", None], "DATE": pd.to_datetime(["2023-01-02", "2023-01-03"])})
    report = validators.check_missing_values(df)
    assert not report.passed


def test_industry_hierarchy_detects_unstable_gics(tiny_pit_panel):
    # Force one ID to have two different sectors
    df = tiny_pit_panel.copy()
    mask = df["ID"] == "AAA UN Equity"
    df.loc[df.index[mask][:5], "GICS_Sector"] = "Financials"
    report = validators.check_industry_hierarchy(df)
    codes = [i.code for i in report.issues]
    assert "unstable_gics" in codes


def test_market_cap_rejects_non_positive():
    df = pd.DataFrame({
        "ID": ["A", "A"],
        "DATE": pd.to_datetime(["2023-01-02", "2023-01-03"]),
        "Market_Cap": [1e9, -1.0],
    })
    report = validators.check_market_cap(df)
    assert not report.passed


def test_volume_flags_negative():
    df = pd.DataFrame({"Volume": [1000.0, -50.0], "Price": [10.0, 10.0]})
    report = validators.check_volume(df)
    assert not report.passed


def test_validate_pit_panel_returns_list(tiny_pit_panel):
    reports = validators.validate_pit_panel(tiny_pit_panel)
    assert isinstance(reports, list)
    assert len(reports) == 5
    assert validators.all_reports_passed(reports)
