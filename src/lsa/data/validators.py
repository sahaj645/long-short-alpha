"""
Schema and data-quality validators for PIT equity panels and ETF OHLCV.

Each validator returns a ValidationReport — a structured collection of issues
with severity, code, message, count, and arbitrary detail. Reports never
mutate input. Issues are emitted to the module logger at the corresponding
severity level when added, so a caller can rely on logging alone for
operational monitoring.

Severity policy
---------------
CRITICAL : downstream tooling must abort; data is unsafe to use.
WARNING  : surfaces an issue, but data remains usable with caution.
INFO     : informational; appears in run summaries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Severity(str, Enum):
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    code: str
    severity: Severity
    message: str
    count: int = 0
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Collected validation issues from a single check."""

    name: str
    issues: list[ValidationIssue] = field(default_factory=list)

    _LOG = {
        Severity.CRITICAL: logger.error,
        Severity.WARNING: logger.warning,
        Severity.INFO: logger.info,
    }

    def add(self, issue: ValidationIssue) -> None:
        """Append issue and emit a log record at the matching severity."""
        self.issues.append(issue)
        self._LOG[issue.severity](
            "[%s] %s: %s (n=%d)", self.name, issue.code, issue.message, issue.count
        )

    @property
    def passed(self) -> bool:
        """True when no CRITICAL issues are present."""
        return not any(i.severity is Severity.CRITICAL for i in self.issues)

    def summary(self) -> dict[str, int]:
        counts = {s.value: 0 for s in Severity}
        for i in self.issues:
            counts[i.severity.value] += 1
        return counts


# -----------------------------------------------------------------------------
# Expected schemas (dtype string match is prefix-based; e.g. "float64" matches
# both float64 and float64[pyarrow]).
# -----------------------------------------------------------------------------

PIT_PANEL_SCHEMA: dict[str, str] = {
    "ID": "object",
    "DATE": "datetime64",
    "CURRENCY": "object",
    "ID_DATE": "datetime64",
    "Price": "float64",
    "Volume": "float64",
    "Market_Cap": "float64",
    "Shares_Out": "float64",
    "GICS_Sector": "object",
    "GICS_Ind_Group": "object",
    "GICS_Industry": "object",
    "GICS_Sub_Industry": "object",
    "Index": "object",
    "Index_Ticker": "object",
}

ETF_PANEL_SCHEMA: dict[str, str] = {
    "Ticker": "object",
    "ETF_Group": "object",
    "Date": "datetime64",
    "Open": "float64",
    "High": "float64",
    "Low": "float64",
    "Close": "float64",
    "Adj_Close": "float64",
    "Volume": "float64",
}

GICS_HIERARCHY_COLS: tuple[str, str, str, str] = (
    "GICS_Sector",
    "GICS_Ind_Group",
    "GICS_Industry",
    "GICS_Sub_Industry",
)


def _dtype_matches(actual: str, expected: str) -> bool:
    """Prefix match on dtype strings, tolerant of numpy/pyarrow variants."""
    return actual.startswith(expected.split("[")[0])


def validate_schema(
    df: pd.DataFrame,
    schema: dict[str, str],
    *,
    report_name: str = "schema",
    allow_extra_cols: bool = True,
) -> ValidationReport:
    """Verify column presence and dtype against an expected schema."""
    report = ValidationReport(name=report_name)
    actual = {c: str(t) for c, t in df.dtypes.items()}

    for col, expected_dtype in schema.items():
        if col not in actual:
            report.add(ValidationIssue(
                code="missing_column",
                severity=Severity.CRITICAL,
                message=f"Required column missing: {col!r}",
                count=1,
                details={"column": col, "expected_dtype": expected_dtype},
            ))
            continue
        if not _dtype_matches(actual[col], expected_dtype):
            report.add(ValidationIssue(
                code="dtype_mismatch",
                severity=Severity.WARNING,
                message=f"Column {col!r} expected {expected_dtype}, got {actual[col]}",
                count=1,
                details={"column": col, "expected": expected_dtype, "actual": actual[col]},
            ))

    if not allow_extra_cols:
        extras = sorted(set(actual) - set(schema))
        if extras:
            report.add(ValidationIssue(
                code="extra_columns",
                severity=Severity.INFO,
                message=f"Unexpected columns: {extras}",
                count=len(extras),
                details={"columns": extras},
            ))
    return report


def check_missing_values(
    df: pd.DataFrame,
    *,
    critical_cols: Sequence[str] = ("ID", "DATE"),
    warning_cols: Sequence[str] = ("GICS_Sector", "Market_Cap"),
    report_name: str = "missing_values",
) -> ValidationReport:
    """Categorize NaN frequency: critical columns must be complete; warning columns logged."""
    report = ValidationReport(name=report_name)

    for col in critical_cols:
        if col not in df.columns:
            continue
        n_na = int(df[col].isna().sum())
        if n_na > 0:
            report.add(ValidationIssue(
                code="missing_critical",
                severity=Severity.CRITICAL,
                message=f"Critical column {col!r} has {n_na} NaN",
                count=n_na,
                details={"column": col},
            ))

    for col in warning_cols:
        if col not in df.columns:
            continue
        n_na = int(df[col].isna().sum())
        if n_na > 0:
            report.add(ValidationIssue(
                code="missing_warning",
                severity=Severity.WARNING,
                message=f"Column {col!r} has {n_na} NaN ({100 * n_na / len(df):.2f}%)",
                count=n_na,
                details={"column": col, "fraction": n_na / max(len(df), 1)},
            ))
    return report


def check_industry_hierarchy(
    df: pd.DataFrame,
    *,
    id_col: str = "ID",
    hierarchy_cols: Sequence[str] = GICS_HIERARCHY_COLS,
    report_name: str = "gics_hierarchy",
) -> ValidationReport:
    """Verify GICS taxonomy integrity.

    Two checks:
      (a) Stability: each ID has a single (sector, ind_group, industry,
          sub_industry) tuple across its history.
      (b) Consistency: each Sub-Industry maps to one Industry, each Industry
          to one Ind_Group, each Ind_Group to one Sector.
    """
    report = ValidationReport(name=report_name)
    cols = [c for c in hierarchy_cols if c in df.columns]
    if not cols:
        return report

    sub = df.dropna(subset=cols)[[id_col, *cols]].drop_duplicates()

    # (a) Per-ID stability of GICS classification
    tuples_per_id = sub.groupby(id_col).size()
    unstable_ids = tuples_per_id[tuples_per_id > 1].index.tolist()
    if unstable_ids:
        report.add(ValidationIssue(
            code="unstable_gics",
            severity=Severity.WARNING,
            message=f"{len(unstable_ids)} IDs carry inconsistent GICS over time",
            count=len(unstable_ids),
            details={"sample_ids": unstable_ids[:5]},
        ))

    # (b) Child-to-parent consistency
    pairs = [
        ("GICS_Sub_Industry", "GICS_Industry"),
        ("GICS_Industry", "GICS_Ind_Group"),
        ("GICS_Ind_Group", "GICS_Sector"),
    ]
    for child, parent in pairs:
        if child not in cols or parent not in cols:
            continue
        cnt = sub.groupby(child)[parent].nunique()
        ambiguous = cnt[cnt > 1].index.tolist()
        if ambiguous:
            report.add(ValidationIssue(
                code="ambiguous_hierarchy",
                severity=Severity.WARNING,
                message=f"{len(ambiguous)} {child} values map to multiple {parent}",
                count=len(ambiguous),
                details={"child": child, "parent": parent, "sample": ambiguous[:3]},
            ))
    return report


def check_market_cap(
    df: pd.DataFrame,
    *,
    id_col: str = "ID",
    date_col: str = "DATE",
    mc_col: str = "Market_Cap",
    max_daily_change_ratio: float = 10.0,
    report_name: str = "market_cap",
) -> ValidationReport:
    """Validate Market_Cap positivity, finiteness, and absence of extreme jumps."""
    report = ValidationReport(name=report_name)
    if mc_col not in df.columns:
        return report

    mc = df[mc_col]
    n_non_pos = int(((mc <= 0) & mc.notna()).sum())
    n_inf = int(np.isinf(mc.fillna(0.0)).sum())
    if n_non_pos:
        report.add(ValidationIssue(
            code="non_positive_market_cap",
            severity=Severity.CRITICAL,
            message=f"{n_non_pos} rows with Market_Cap <= 0",
            count=n_non_pos,
        ))
    if n_inf:
        report.add(ValidationIssue(
            code="infinite_market_cap",
            severity=Severity.CRITICAL,
            message=f"{n_inf} rows with infinite Market_Cap",
            count=n_inf,
        ))

    if not {id_col, date_col, mc_col}.issubset(df.columns):
        return report

    df_sorted = df[[id_col, date_col, mc_col]].dropna().sort_values([id_col, date_col])
    ratio = df_sorted.groupby(id_col, sort=False)[mc_col].transform(
        lambda s: s / s.shift()
    )
    jumps_mask = ratio > max_daily_change_ratio
    n_jumps = int(jumps_mask.sum())
    if n_jumps:
        sample = df_sorted.loc[jumps_mask, [id_col, date_col, mc_col]].head(5).to_dict("records")
        report.add(ValidationIssue(
            code="market_cap_jump",
            severity=Severity.WARNING,
            message=f"{n_jumps} (ID, DATE) pairs show Market_Cap jump > {max_daily_change_ratio}x",
            count=n_jumps,
            details={"sample": sample, "threshold": max_daily_change_ratio},
        ))
    return report


def check_volume(
    df: pd.DataFrame,
    *,
    vol_col: str = "Volume",
    price_col: str = "Price",
    report_name: str = "volume",
) -> ValidationReport:
    """Volume must be non-negative; zero allowed only on non-trading rows."""
    report = ValidationReport(name=report_name)
    if vol_col not in df.columns:
        return report

    v = df[vol_col]
    n_neg = int(((v < 0) & v.notna()).sum())
    if n_neg:
        report.add(ValidationIssue(
            code="negative_volume",
            severity=Severity.CRITICAL,
            message=f"{n_neg} rows with Volume < 0",
            count=n_neg,
        ))

    if price_col in df.columns:
        bad_zero = (v == 0) & df[price_col].notna()
        n_bad = int(bad_zero.sum())
        if n_bad:
            report.add(ValidationIssue(
                code="zero_volume_with_price",
                severity=Severity.WARNING,
                message=f"{n_bad} rows with Price present but Volume = 0",
                count=n_bad,
            ))
    return report


def validate_pit_panel(df: pd.DataFrame, *, name: str = "pit_panel") -> list[ValidationReport]:
    """Run the complete validation suite for a PIT panel.

    Returns a list of ValidationReports; the caller decides whether to abort
    on the presence of any CRITICAL issue (use `all(r.passed for r in reports)`).
    """
    return [
        validate_schema(df, PIT_PANEL_SCHEMA, report_name=f"{name}.schema"),
        check_missing_values(df, report_name=f"{name}.missing"),
        check_industry_hierarchy(df, report_name=f"{name}.gics"),
        check_market_cap(df, report_name=f"{name}.market_cap"),
        check_volume(df, report_name=f"{name}.volume"),
    ]


def all_reports_passed(reports: Sequence[ValidationReport]) -> bool:
    """Convenience: True iff no CRITICAL issues exist across any report."""
    return all(r.passed for r in reports)
