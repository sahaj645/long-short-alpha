"""Unit tests for lsa.data.universe_builder."""

from __future__ import annotations

import pandas as pd
import pytest

from lsa.data import universe_builder as ub


def test_get_lagged_snapshot_returns_strict_prior():
    snaps = pd.to_datetime(["2023-01-31", "2023-02-28", "2023-03-31"])
    # Trade on 2023-02-15: most recent strictly prior snap is 2023-01-31
    assert ub.get_lagged_snapshot(snaps, pd.Timestamp("2023-02-15")) == pd.Timestamp("2023-01-31")
    # Trade on 2023-02-28: STRICTLY prior excludes 2023-02-28; should be 2023-01-31
    assert ub.get_lagged_snapshot(snaps, pd.Timestamp("2023-02-28")) == pd.Timestamp("2023-01-31")


def test_get_lagged_snapshot_returns_none_before_first():
    snaps = pd.to_datetime(["2023-01-31"])
    assert ub.get_lagged_snapshot(snaps, pd.Timestamp("2022-12-15")) is None


def test_build_universe_at_date_pit_safe():
    panel = pd.DataFrame({
        "ID": ["A", "B", "C"],
        "ID_DATE": pd.to_datetime(["2023-01-31", "2023-01-31", "2023-02-28"]),
        "index_label": ["SP500", "SP500", "SP500"],
    })
    # Trade on Feb 15: use Jan 31 snapshot → {A, B}
    assert ub.build_universe_at_date(panel, pd.Timestamp("2023-02-15")) == {"A", "B"}
    # Trade on Feb 28: strictly prior is Jan 31 → {A, B}, NOT {C}
    assert ub.build_universe_at_date(panel, pd.Timestamp("2023-02-28")) == {"A", "B"}


def test_merge_pit_panels_concatenates():
    p1 = pd.DataFrame({"ID": ["A"], "index_label": ["SP500"]})
    p2 = pd.DataFrame({"ID": ["B"], "index_label": ["SP400"]})
    merged = ub.merge_pit_panels({"SP500": p1, "SP400": p2})
    assert len(merged) == 2
    assert set(merged["index_label"]) == {"SP500", "SP400"}


def test_find_simultaneous_overlap_detects_duplication():
    df = pd.DataFrame({
        "ID": ["A", "A", "B"],
        "ID_DATE": pd.to_datetime(["2023-01-31", "2023-01-31", "2023-01-31"]),
        "index_label": ["SP500", "SP400", "SP500"],
    })
    overlap = ub.find_simultaneous_overlap(df)
    assert len(overlap) == 1
    assert overlap.iloc[0]["ID"] == "A"


def test_dedupe_dual_class_keeps_higher_dollar_volume():
    df = pd.DataFrame({
        "ID": ["GOOG UW Equity", "GOOGL UW Equity"],
        "DATE": pd.to_datetime(["2023-01-02", "2023-01-02"]),
        "dollar_volume": [1e9, 5e9],
    })
    out = ub.dedupe_dual_class(df)
    assert len(out) == 1
    assert out["ID"].iloc[0] == "GOOGL UW Equity"


def test_dedupe_dual_class_requires_dollar_volume_column():
    df = pd.DataFrame({"ID": ["A"], "DATE": pd.to_datetime(["2023-01-02"])})
    with pytest.raises(KeyError):
        ub.dedupe_dual_class(df)
