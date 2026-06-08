"""
Research universe construction from the three PIT panels (SP400/500/600).

This module is the single source of truth for the PIT-safe universe-at-date
function. The cardinal rule it enforces:

    For any decision made on trade date D, the universe of tradable names is
    M(t-1), where M(t-1) is the membership snapshot whose ID_DATE is the most
    recent month-end STRICTLY before D.

Using the same-month snapshot would leak future information because S&P
membership for month M is only revealed at month-end M.

The module also handles:
  - Merging the three index panels into a single long-form frame.
  - Detecting simultaneous (ID, ID_DATE) membership across indices — these
    should be near zero by S&P design.
  - De-duplicating dual-class shares (e.g., GOOG/GOOGL, FOX/FOXA) in favor of
    the more-liquid share class per issuer per date.
  - Applying the three research-universe filters: liquidity floor, Sub-Industry
    member count, and Sub-Industry leader cap-share band.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd

from . import cleaners, validators

logger = logging.getLogger(__name__)


# Default filenames for the three PIT panels in the data directory.
INDEX_FILES: dict[str, tuple[str, str]] = {
    "SP400": ("sp400_pit_20160301_20251231.csv", "S&P 400"),
    "SP500": ("sp500_pit_20160301_20251231.csv", "S&P 500"),
    "SP600": ("sp600_pit_20160301_20251231.csv", "S&P 600"),
}

# Default research-universe parameter values (matching configs/base.yaml).
DEFAULT_DOLLAR_VOLUME_FLOOR: float = 5_000_000.0
DEFAULT_MIN_SUBINDUSTRY_MEMBERS: int = 4
DEFAULT_LEADER_CAP_SHARE_MIN: float = 0.20
DEFAULT_LEADER_CAP_SHARE_MAX: float = 0.70
DEFAULT_ADV_WINDOW_DAYS: int = 21


@dataclass
class UniverseBuildResult:
    """Return value of `build_research_universe`."""

    panel: pd.DataFrame
    overlaps: pd.DataFrame
    validation_reports: list[validators.ValidationReport] = field(default_factory=list)
    n_rows_after_merge: int = 0
    n_rows_after_dedupe: int = 0
    n_rows_after_filters: int = 0


# -----------------------------------------------------------------------------
# Loading and merging
# -----------------------------------------------------------------------------


def load_pit_panel(path: Path, index_name: str) -> pd.DataFrame:
    """Load a single PIT panel from CSV and tag rows with `index_label`.

    The CSV is expected to follow the schema in `validators.PIT_PANEL_SCHEMA`.
    DATE and ID_DATE are parsed as datetimes at load time for downstream
    efficiency.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PIT panel not found at {path}")
    logger.info("Loading PIT panel: index=%s path=%s", index_name, path)
    df = pd.read_csv(path, parse_dates=["DATE", "ID_DATE"])
    if "PIT_Member_Date" in df.columns:
        df["PIT_Member_Date"] = pd.to_datetime(df["PIT_Member_Date"])
    df["index_label"] = index_name
    return df


def merge_pit_panels(panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Concatenate the three (or more) index PIT panels into a single frame.

    Panels must already be cleaned (passed through `cleaners.clean_pit_panel`).
    `index_label` is preserved for each row to identify the source index.
    """
    if not panels:
        raise ValueError("merge_pit_panels requires at least one panel")
    out = pd.concat(list(panels.values()), ignore_index=True, sort=False)
    logger.info("Merged %d panels into %d rows", len(panels), len(out))
    return out


def find_simultaneous_overlap(
    merged: pd.DataFrame,
    *,
    id_col: str = "ID",
    snapshot_col: str = "ID_DATE",
    index_col: str = "index_label",
) -> pd.DataFrame:
    """Identify (ID, ID_DATE) pairs that appear in two or more indices at once.

    S&P design forbids a security being a member of more than one of SP400,
    SP500, SP600 at the same snapshot date. In practice the data audit
    detected a single edge case (AMTM 2025-05) over the 10-year window. This
    function returns a long-form frame listing every overlap so it can be
    quarantined or hand-reviewed.
    """
    pairs = merged[[id_col, snapshot_col, index_col]].drop_duplicates()
    counts = pairs.groupby([id_col, snapshot_col])[index_col].nunique()
    overlap = counts[counts > 1].reset_index(name="n_indices")
    if not overlap.empty:
        logger.warning("Detected %d simultaneous-membership overlaps", len(overlap))
    return overlap


# -----------------------------------------------------------------------------
# PIT-safe universe-at-date
# -----------------------------------------------------------------------------


def get_lagged_snapshot(
    snapshots: Iterable[pd.Timestamp] | pd.Series | pd.Index,
    trade_date: pd.Timestamp,
) -> pd.Timestamp | None:
    """Return the most recent snapshot STRICTLY before `trade_date`, or None.

    `snapshots` is any iterable of timestamps (typically a panel's ID_DATE
    column). The function returns the largest snapshot value < trade_date,
    which is the correct "most recently revealed" snapshot for a PIT-safe
    universe lookup.
    """
    eligible = pd.DatetimeIndex(pd.unique(pd.DatetimeIndex(snapshots)))
    eligible = eligible[eligible < pd.Timestamp(trade_date)]
    if len(eligible) == 0:
        return None
    return eligible.max()


def build_universe_at_date(
    panel: pd.DataFrame,
    trade_date: pd.Timestamp,
    *,
    id_col: str = "ID",
    snapshot_col: str = "ID_DATE",
    index_label: str | None = None,
) -> set[str]:
    """PIT-safe set of IDs in the universe as of `trade_date`.

    Parameters
    ----------
    panel
        Merged PIT panel (output of `merge_pit_panels`).
    trade_date
        The date on which a trading decision is being made.
    index_label
        Optional restriction to a single index ("SP500" / "SP400" / "SP600").
        If None (default), the union universe across all indices in the panel
        is returned.

    Returns
    -------
    set[str]
        IDs that were members of the relevant index(es) as of the most recent
        snapshot strictly before `trade_date`. Empty set if `trade_date`
        precedes the first available snapshot.
    """
    snap = get_lagged_snapshot(panel[snapshot_col], trade_date)
    if snap is None:
        return set()

    mask = panel[snapshot_col] == snap
    if index_label is not None:
        mask &= panel["index_label"] == index_label
    return set(panel.loc[mask, id_col].unique())


# -----------------------------------------------------------------------------
# Dual-class de-duplication
# -----------------------------------------------------------------------------


def dedupe_dual_class(
    panel: pd.DataFrame,
    *,
    id_col: str = "ID",
    date_col: str = "DATE",
    dollar_volume_col: str = "dollar_volume",
) -> pd.DataFrame:
    """Retain only the more-liquid share class per issuer per date.

    Bloomberg IDs in this dataset look like "GOOGL UW Equity". We approximate
    the issuer root by taking the first space-separated token (the ticker
    stem) and stripping a single trailing 'A' or 'B' character (which Bloomberg
    uses for share-class suffixes — e.g., GOOGL→GOOG[L]→GOOG, FOXA→FOX).
    The dedupe is per (date, issuer_root): keep the row with the highest
    `dollar_volume_col`.

    Note: this is a heuristic. Names without an A/B share suffix collapse to
    themselves and are unaffected. For production trading, a curated mapping
    of share-class equivalences would replace this heuristic.
    """
    if dollar_volume_col not in panel.columns:
        raise KeyError(
            f"Required column {dollar_volume_col!r} missing — "
            "call cleaners.compute_dollar_volume first."
        )

    df = panel.copy()
    stem = df[id_col].str.split(" ", n=1).str[0]
    issuer_root = stem.str.replace(r"[AB]$", "", regex=True)
    df["_issuer_root"] = issuer_root

    keep_idx = (
        df.groupby([date_col, "_issuer_root"])[dollar_volume_col]
          .idxmax()
          .dropna()
          .values
    )
    out = df.loc[keep_idx].drop(columns=["_issuer_root"]).reset_index(drop=True)
    n_dropped = len(df) - len(out)
    if n_dropped > 0:
        logger.info("Dedupe dual-class: dropped %d rows in favor of higher dollar-volume", n_dropped)
    return out


# -----------------------------------------------------------------------------
# Research-universe filters
# -----------------------------------------------------------------------------


def apply_universe_filters(
    panel: pd.DataFrame,
    *,
    dollar_volume_floor: float = DEFAULT_DOLLAR_VOLUME_FLOOR,
    min_subindustry_members: int = DEFAULT_MIN_SUBINDUSTRY_MEMBERS,
    leader_cap_share_min: float = DEFAULT_LEADER_CAP_SHARE_MIN,
    leader_cap_share_max: float = DEFAULT_LEADER_CAP_SHARE_MAX,
    adv_window_days: int = DEFAULT_ADV_WINDOW_DAYS,
    id_col: str = "ID",
    date_col: str = "DATE",
    subind_col: str = "GICS_Sub_Industry",
    dollar_volume_col: str = "dollar_volume",
    market_cap_col: str = "Market_Cap",
) -> pd.DataFrame:
    """Apply the three research-universe filters from configs/base.yaml.

    Filters (applied in order; each can only contract the universe):
      1. Liquidity floor: rolling `adv_window_days`-day median dollar volume
         per name must be >= `dollar_volume_floor`.
      2. Sub-Industry member count: the name's Sub-Industry must contain at
         least `min_subindustry_members` eligible names on that date.
      3. Leader cap-share band: the Sub-Industry's leader (max market cap)
         share of total Sub-Industry cap must lie within
         [leader_cap_share_min, leader_cap_share_max].

    The function never alters which (ID, DATE) rows are kept based on future
    information; the rolling ADV uses a trailing window only.
    """
    df = panel.sort_values([id_col, date_col]).reset_index(drop=True).copy()

    # Filter 1 — liquidity floor on trailing-window median dollar volume.
    df["adv_med"] = (
        df.groupby(id_col, sort=False, observed=True)[dollar_volume_col]
          .transform(lambda s: s.rolling(adv_window_days, min_periods=adv_window_days // 2).median())
    )
    df = df[df["adv_med"] >= dollar_volume_floor]

    # Filter 2 — Sub-Industry member-count floor.
    member_cnt = (
        df.groupby([date_col, subind_col], observed=True)[id_col]
          .transform("nunique")
    )
    df = df[member_cnt >= min_subindustry_members]

    # Filter 3 — leader cap-share band.
    grp_subind = df.groupby([date_col, subind_col], observed=True)[market_cap_col]
    df["_subind_total_mc"] = grp_subind.transform("sum")
    df["_subind_leader_mc"] = grp_subind.transform("max")
    leader_share = df["_subind_leader_mc"] / df["_subind_total_mc"]
    df = df[(leader_share >= leader_cap_share_min) & (leader_share <= leader_cap_share_max)]

    df = df.drop(columns=["adv_med", "_subind_total_mc", "_subind_leader_mc"]).reset_index(drop=True)
    logger.info("Universe filters retained %d of %d rows (%.1f%%)",
                len(df), len(panel), 100.0 * len(df) / max(len(panel), 1))
    return df


# -----------------------------------------------------------------------------
# Orchestrator
# -----------------------------------------------------------------------------


def build_research_universe(
    data_dir: Path,
    *,
    index_files: dict[str, tuple[str, str]] | None = None,
    apply_filters: bool = True,
    dollar_volume_floor: float = DEFAULT_DOLLAR_VOLUME_FLOOR,
    min_subindustry_members: int = DEFAULT_MIN_SUBINDUSTRY_MEMBERS,
    leader_cap_share_min: float = DEFAULT_LEADER_CAP_SHARE_MIN,
    leader_cap_share_max: float = DEFAULT_LEADER_CAP_SHARE_MAX,
    run_validators: bool = True,
) -> UniverseBuildResult:
    """End-to-end research-universe build.

    Steps:
      1. Load each PIT panel via `load_pit_panel`.
      2. Clean each panel via `cleaners.clean_pit_panel`.
      3. (Optional) Run the validators on each cleaned panel.
      4. Merge into a single long-form panel.
      5. Detect simultaneous-membership overlaps for diagnostic logging.
      6. De-duplicate dual-class shares.
      7. Apply research-universe filters (if requested).

    Returns
    -------
    UniverseBuildResult
        Contains the final panel, overlap diagnostics, validation reports,
        and row counts at each stage.
    """
    data_dir = Path(data_dir)
    files = index_files if index_files is not None else INDEX_FILES

    panels: dict[str, pd.DataFrame] = {}
    all_reports: list[validators.ValidationReport] = []

    for index_name, (filename, _label) in files.items():
        raw = load_pit_panel(data_dir / filename, index_name)
        cleaned, _ = cleaners.clean_pit_panel(raw)
        if run_validators:
            all_reports.extend(validators.validate_pit_panel(cleaned, name=index_name))
        panels[index_name] = cleaned

    merged = merge_pit_panels(panels)
    n_after_merge = len(merged)

    overlaps = find_simultaneous_overlap(merged)
    deduped = dedupe_dual_class(merged)
    n_after_dedupe = len(deduped)

    if apply_filters:
        final = apply_universe_filters(
            deduped,
            dollar_volume_floor=dollar_volume_floor,
            min_subindustry_members=min_subindustry_members,
            leader_cap_share_min=leader_cap_share_min,
            leader_cap_share_max=leader_cap_share_max,
        )
    else:
        final = deduped

    return UniverseBuildResult(
        panel=final,
        overlaps=overlaps,
        validation_reports=all_reports,
        n_rows_after_merge=n_after_merge,
        n_rows_after_dedupe=n_after_dedupe,
        n_rows_after_filters=len(final),
    )
