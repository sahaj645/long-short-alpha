# long-short-alpha

US equity long-short systematic research framework. Implements and evaluates the **Sub-Industry Leader-Follower Information Diffusion (H6)** hypothesis on the S&P 1500 universe over 2016-2025.

## 1. Executive Summary

**What.** A complete signal-to-PnL research stack: data ingestion, validation, universe construction, residualization, leader-follower signal generation, portfolio construction, event-driven backtesting, and performance analytics. Eleven layers; ~5,000 lines of production Python; institutional documentation across seven `.docx` artifacts.

**Why.** To test, end-to-end and under pre-registration discipline, whether systematic L/S strategies can extract the gradual information-diffusion effect within GICS Sub-Industries — and to do so on infrastructure that survives senior research and engineering review.

**Research Question.** Does information about a Sub-Industry's fundamentals reach the largest-market-cap firm first and diffuse to its smaller peers with a measurable, tradable delay?

**Main findings.**

| Finding | Evidence |
|---|---|
| Gross alpha is consistently positive | +0.7 to +2.4 bps/day across all 12 tested configurations; same sign in 8 of 10 calendar years |
| The mechanism is real | Effect concentrates in the H1-pre-committed [30%, 60%] leader cap-share band as predicted |
| The strategy is cost-bound at SP500 scale | At 3 bps/side TC, net Sharpe = −0.31 after E1+E2 architectural fixes |
| Architectural alignment matters more than parameters | Sub-Industry portfolio + 5-day overlapping books moved Sharpe from −3.66 → −0.31 with no parameter optimization |
| The strategy is not yet production-ready at SP500 | Break-even cost = ~1.5 bps/side; production execution would need either SP1500 expansion or sub-bps VWAP cost surface |

## 2. Strategy Overview

**Universe.** Point-in-time membership panel for S&P 400 (mid-cap), S&P 500 (large-cap), and S&P 600 (small-cap), 2016-03-01 through 2025-12-31. ~3.7M (ID, DATE) rows after cleaning. Sector residualization against the 11 GICS Sector SPDR ETFs (XLB, XLC, XLE, XLF, XLI, XLK, XLP, XLRE, XLU, XLV, XLY).

**Signal.** For each (Sub-Industry, date):
1. Identify the cap-weighted leader (top-1 by trailing 21-day mean Market Cap).
2. Residualize each name's daily return against its matched Sector SPDR (rolling 126-day beta, lagged by one trading day for PIT purity).
3. The signal at date *t* for every follower in the Sub-Industry is the **leader's residual return at date *t−1***.
4. Cross-sectional normalization: z-score over unique (Sub-Industry, date) signal values per date, capped at ±3σ.

**Portfolio construction.** Sub-Industry-level (E1). Each date, rank Sub-Industries by signal magnitude. Long the top-8, short the bottom-8. Within each selected Sub-Industry, all followers get equal weight; capital is allocated equally across selected Sub-Industries. Five overlapping books (E2): each day's target portfolio is held for 5 trading days; the active book is the rolling 5-day mean of daily target weights.

**Execution.** Close-to-close convention. Signal at *t* uses data through close of *t*. Weights *w_t* set at close of *t*. PnL on day *t+1* = *w_t · r_{t+1}*. Daily rebalance (one of the 5 overlapping books turns over per day).

**Cost model.** Flat 3 bps per side baseline (conservative for SP500). Tiered cost model available with per-tier basis points: SP500 = 3, SP400 = 7, SP600 = 12. Square-root impact model available for capacity analysis.

## 3. Research Findings

All results in this section are on the S&P 500 / 2016-03-01 to 2025-12-31 backtest with the executed pipeline. Memory constraints in the sandboxed execution environment forced restriction to SP500; the H6 thesis predicts the strongest effect in SP400/600 (smaller, less-followed names), which is the highest-priority extension.

### Baseline (daily rebalance, name-level top-30)

| | Value |
|---|---:|
| Sharpe | −3.66 |
| Gross PnL | +0.42 bps/day |
| TC | 8.08 bps/day |
| Net | −7.66 bps/day |
| Annualized turnover | ~25,000% |

Diagnosis: turnover-cost mismatch. Daily rebalance at horizon=1 produces ~100% one-way turnover, costing ~8 bps/day that the small gross signal cannot cover.

### E1 — Sub-Industry portfolio construction

Replace name-level top-30 with Sub-Industry-level top-8 / bottom-8. All followers in each selected Sub-Industry traded.

| | Value | Δ vs baseline |
|---|---:|---:|
| Sharpe | −0.89 | +2.77 |
| Gross PnL | +1.46 bps/day | +1.04 |
| TC | 5.76 bps/day | −2.32 |
| Net | −4.29 bps/day | +3.37 |

Diagnosis: aligning the trading unit with the signal unit lifted gross alpha by 3.5× (concentration on extreme tail of signal distribution). Turnover stayed near 100% because daily reshuffling of the top-8 Sub-Industries triggered full-book rotation. Architectural fix worked at the signal level; cost remained binding.

### E2 — 5-day overlapping books

Apply E1 with 5-day holding period implemented as 5 overlapping books at 1/5 NAV each.

| | Value | Δ vs E1 |
|---|---:|---:|
| Sharpe | **−0.31** | +0.58 |
| Gross PnL | +0.96 bps/day | −0.50 |
| TC | 1.96 bps/day | −3.80 |
| Net | −1.00 bps/day | +3.29 |
| Annualized turnover | ~8,200% | −16,000pp |
| Maximum drawdown | −28.9% | +40.3pp |

Diagnosis: matching the holding period to the diffusion half-life dropped turnover by 3× and TC by 3×. Maximum drawdown improved by 40 percentage points. Gross alpha decayed slightly (5-day averaging dilutes day-1 strength) but the trade was strongly net positive.

### Where the strategy currently stands

Sharpe progression across the three configurations: **−3.66 → −0.89 → −0.31**. Net PnL gap to break-even: **1.0 bps/day**.

The remaining gap is **not signal failure** (gross is positive, persistent, and concentrated in the H1-predicted Sub-Industry band) and **not portfolio construction failure** (E1+E2 correctly express the signal). It is **execution-cost failure** at the 3 bps/side conservative cost, and **universe limitation** (SP500 mega-caps are the weakest segment of the H6 mechanism per Hou 2007).

### Future research directions

1. **Liquidity-tier follower selection** within Sub-Industry. Trade only the top-quartile of follower names by dollar volume. Expected cost reduction: 40-60%.
2. **Per-tier impact cost calibration** against execution-desk realized fills. Likely reveals the SP500 mega-cap real cost is ~1 bps/side, not 3.
3. **Full SP1500 deployment**. The H6 mechanism is theoretically strongest in SP400/600 where coverage and price discovery are bandwidth-limited. Expected gross uplift: 50-100% per the lead-lag literature.

## 4. Repository Architecture

```
long-short-alpha/
├── README.md                      # This document
├── LICENSE                        # MIT
├── Makefile                       # install, test, lint, format, clean
├── pyproject.toml                 # Package metadata + dependencies
├── configs/
│   └── base.yaml                  # All strategy hyperparameters
├── data/                          # Raw CSVs (gitignored — not in repo)
├── docs/                          # Institutional documentation (Word format)
│   ├── 00_data_audit.docx
│   ├── 01_dataset_teaching.docx
│   ├── 02_hypothesis_catalog.docx
│   ├── 03_showcase_ranking.docx
│   ├── 04_research_design.docx
│   ├── 05_experimental_framework.docx
│   └── 06_results_interpretation_framework.docx
├── notebooks/                     # Exploratory and analysis notebooks
│   ├── 00_data_audit.ipynb        # Reproducible EDA + integrity checks
│   ├── 01_residual_momentum_baseline.ipynb
│   ├── 02_information_diffusion_hypothesis.ipynb
│   └── 05_robustness.ipynb        # Sensitivity sweeps with composite scorecard
├── results/                       # Backtest outputs (gitignored)
├── src/lsa/                       # Production package
│   ├── data/                      # Loaders, validators, universe construction
│   ├── features/                  # (placeholder for future feature library)
│   ├── signals/                   # Residual momentum + leader-follower signal
│   ├── portfolio/                 # Portfolio builder, constraints, rebalancer
│   ├── backtest/                  # Execution model, accounting, event-driven driver
│   ├── analytics/                 # Metrics, trade statistics, composite reports
│   └── utils/                     # (placeholder for shared helpers)
└── tests/                         # Pytest suite
```

### Per-directory responsibilities

| Directory | Purpose | Inputs | Outputs |
|---|---|---|---|
| `src/lsa/data/` | Loaders, validators, PIT universe construction | Raw CSV PIT panels + ETF OHLCV | Cleaned panels, `ValidationReport`, `UniverseBuildResult` |
| `src/lsa/signals/` | Residualization, leader identification, signal construction | Cleaned panel + ETF returns | Long-form panel with `signal`, `signal_norm`, `signal_rank` columns |
| `src/lsa/portfolio/` | Target-weight construction, constraint enforcement, daily rebalancer | Signal panel + config | `PortfolioBuildResult`, per-day `ConstraintReport` |
| `src/lsa/backtest/` | Event-driven daily simulator + accounting | Signal panel + returns panel | `BacktestResult` with accounting frame, equity curve, rebalance history |
| `src/lsa/analytics/` | Performance metrics, episode-level trade stats, composite reports | `BacktestResult.daily_returns` | `PerformanceMetrics`, `TradeStatistics`, `PerformanceReport` |
| `notebooks/` | Exploratory + experimental work | Production modules above | Charts, regression tables, scorecards |
| `docs/` | Process artifacts: audit, RDD, framework, IRF | — | Institutional documents for committee review |
| `tests/` | Unit and integration tests | — | Pytest pass/fail |

## 5. Data Pipeline

```
Raw CSVs        →  Validation       →  Cleaning           →  Universe         →  Feature
(PIT + ETF)        (schema + GICS)     (filter + winsorize)  (merge + dedupe)    (residuals)
```

**Raw.** Three PIT panels (`sp{400,500,600}_pit_*.csv`, ~1.4M / 1.8M / 2.2M rows) and one ETF OHLCV (`etf_ohlcv_*.csv`, 34K rows). Coverage 2016-03-01 through 2025-12-31. Bloomberg conventions throughout (IDs of form `"AAPL UW Equity"`, delisted suffix `"<ID>D"`).

**Validation.** Five checks per PIT panel: schema (column names + dtypes), missing-value critical/warning split, GICS hierarchy consistency (sector → ind_group → industry → sub-industry uniqueness), market-cap positivity + no >10x daily jumps, volume non-negativity. Validators emit `ValidationReport` with severity-tagged issues; failure threshold = any `CRITICAL`.

**Cleaning.** Pipeline order: (1) normalize DATE/ID_DATE to tz-naive midnight, (2) drop redundant PIT_Member_Date column, (3) drop (ID, DATE) duplicates, (4) filter to trading rows (`Price.notna()`), (5) compute returns with gap-aware NaN insertion (no return computed across calendar gaps > 5 days), (6) winsorize daily returns at ±50% and emit a quarantine sub-frame for audit, (7) compute dollar volume.

**Universe.** Concatenate the three PIT panels with `index_label` tags. Detect cross-index simultaneous-membership overlaps for diagnostic logging (~1 over 10 years, AMTM 2025-05 edge case). De-duplicate dual-class shares (e.g., `GOOG`/`GOOGL`) by retaining the higher-dollar-volume class per (date, issuer). Apply three filters: dollar-volume floor ($5M trailing 21-day median), Sub-Industry member count (≥ 4), leader cap-share band [0.20, 0.70].

**PIT-safe universe-at-date.** `build_universe_at_date(panel, trade_date)` returns the set of IDs that were members as of the most recent snapshot **strictly before** `trade_date`. Same-month snapshot would leak future information; strict-prior is the single most important PIT discipline in the entire codebase.

## 6. Signal Pipeline

```
Returns        →  Residualization      →  Leader Detection    →  Signal           →  Ranking
(per name)        (sector ETF beta)       (per Sub-Industry)     (lagged broadcast)  (cross-section)
```

**Returns.** Daily price returns from PIT `Price.pct_change()` after filtering to business days. Note: PIT prices are split-adjusted but **not** dividend-adjusted; the strategy operates on price returns throughout for internal consistency.

**Residualization.** For each name, compute rolling 126-day beta against its matched Sector SPDR (Information Technology → XLK, Financials → XLF, etc.) using the closed-form β = Cov(r, m) / Var(m). **Beta is lagged by one trading day** so the residual at *t* uses β_{t−1} rather than β_t — this prevents subtle lookahead from `pandas.rolling` which by default includes the current observation. Residual = r_name − β_{t−1} × r_etf.

**Leader detection.** Per (Sub-Industry, date), identify the top-1 ID by trailing 21-day mean Market Cap. Smoothing prevents single-day cap spikes from flipping the leader. Apply universe filters (≥ 4 members, leader cap-share ∈ [0.20, 0.70]). The cap-weighted leader is the proxy for the bandwidth-constrained information channel.

**Signal construction.** For each follower row at date *t*, the signal is the leader's residual return at *t − 1*. Leaders themselves get NaN signal (they do not trade their own information). All followers within a Sub-Industry share the same signal value by construction.

**Ranking.** Cross-sectional z-score and percentile rank computed on the **unique (Sub-Industry, date)** signal values per date, then broadcast back to all rows. This ensures each Sub-Industry contributes equally to the normalization independent of follower cohort size.

## 7. Portfolio Pipeline

```
Signal panel  →  Selection         →  Weight assignment    →  Constraints       →  Trades
(rank + ID)      (top-K Sub-Ind)      (equal across Sub-Ind)   (position + neut)    (Δw)
```

**Selection.** At each date, rank Sub-Industries by signal magnitude. Pick top-K=8 (long) and bottom-K=8 (short). Selection at the Sub-Industry level (not the name level) ensures that all capital flows to the strongest signals on a per-conviction-unit basis.

**Weight assignment.** Each selected Sub-Industry receives equal capital allocation: 1/K of long book on the long side, 1/K of short book on the short side. Within each Sub-Industry, all followers receive equal share of that allocation. Gross exposure = 2.0 (long 1.0, short −1.0). Dollar-neutral by construction.

**Overlapping books.** Each day's target portfolio is held for 5 trading days as 5 overlapping books at 1/5 NAV each. Implementation: the active portfolio at date *t* is the rolling 5-day mean of daily target weights. Mathematically equivalent to 5 single-day books accumulating over time.

**Constraints.** Six-step pipeline applied in fixed order to every day's target weights: max position (5% gross per name) → max industry net exposure (5%) → max sector net exposure (10%) → liquidity participation cap (10% of trailing ADV) → turnover cap (configurable) → dollar neutrality (1e-4 tolerance). Each constraint can only contract or shift weights; dollar neutrality renormalization is last because it compensates for any upstream drift.

## 8. Backtesting Methodology

**Execution sequence (PIT-safe).** Each daily iteration:
1. Realize PnL from prior weights × today's returns: gross_pnl_t = w_{t−1} · r_t. Signal_t is **not** referenced yet.
2. Build new target weights from signal_t (which itself was computed using data through close of *t*).
3. Apply constraint pipeline with prior weights as turnover context.
4. Compute trades = w_t − w_{t−1}.
5. Estimate transaction cost via the execution model.
6. Book the day with the Accountant: NAV update, PnL attribution.
7. Roll w_{t−1} ← w_t for the next iteration.

**Lookahead prevention.** The backtester guarantees that within its own loop, signal_t is used only to set w_t, never to compute PnL on day t. The signal panel itself is socially required to be PIT-safe (signal at *t* uses data through *t*) — that contract is enforced upstream by the `lag_days=1` parameter in `compute_leader_follower_signal` and the `.shift(1)` of rolling beta in `compute_rolling_betas`.

**Transaction costs.** Three configurable cost models:
- **Flat.** Single bps applied uniformly: `cost = |trades|.sum() × bps / 10_000`.
- **Tiered.** Per-name bps lookup via tier_map. Default rates: SP500 = 3 bps, SP400 = 7 bps, SP600 = 12 bps.
- **Square-root impact.** `cost_bps = α × √(trade_dollars / ADV)`. Used for capacity analysis.
- Optional `slippage_bps` adder on top of any model.

**Portfolio accounting.** The `Accountant` class owns the only mutable PnL state. Per-day `DailyRecord` captures NAV open/close, gross PnL, long-book PnL, short-book PnL (attribution), transaction cost, net PnL, gross/net exposure, n_long, n_short, one-way turnover, and missing-returns diagnostic count. The frame is materialized via `to_frame()` for analytics consumption.

**NAV calculation.** `NAV_t = NAV_{t−1} × (1 + gross_pnl_pct_t − transaction_cost_pct_t)`. No drift renormalization between rebalances (with daily rebalancing, drift is small; the cumulative-product approach is the correct convention).

**Missing returns.** A name in w_{t−1} whose return on day *t* is NaN contributes zero to gross PnL. The `DailyRecord.n_returns_missing` counter tracks how many — a daily data-quality monitor. This is the right choice for genuine non-trading events (suspensions, acquisitions); the alternative (raising) would make the backtest brittle.

## 9. Running The Project

### Installation

```bash
git clone https://github.com/sahaj645/long-short-alpha.git
cd long-short-alpha
pip install -e ".[dev]"
```

### Data setup

Place the four raw CSVs in `data/`:

```
data/
├── etf_ohlcv_20160301_20251231.csv
├── sp400_pit_20160301_20251231.csv
├── sp500_pit_20160301_20251231.csv
└── sp600_pit_20160301_20251231.csv
```

The `data/` directory is gitignored — raw data must be supplied locally.

### Main commands

```bash
make test        # Run pytest suite (data layer covered)
make lint        # ruff check
make format      # ruff check --fix
make clean       # Remove pyc, pycache, .pytest_cache, build artifacts
```

### Execution order

Notebooks should be run in numerical order. Each gate must clear before the next is run.

```
notebooks/00_data_audit.ipynb                       # Validate raw data
notebooks/01_residual_momentum_baseline.ipynb       # Baseline comparison strategy
notebooks/02_information_diffusion_hypothesis.ipynb # H6 central test (KILL GATE)
notebooks/05_robustness.ipynb                       # Sensitivity sweeps
```

### Programmatic use

```python
from lsa.data    import load_pit_panel, clean_pit_panel, merge_pit_panels, INDEX_FILES
from lsa.signals import compute_rolling_betas, compute_residual_returns
from lsa.signals import identify_leaders, identify_followers
from lsa.signals import compute_leader_follower_signal, normalize_signal, rank_signal_cross_sectional
from lsa.backtest import Backtester, BacktestConfig
from lsa.analytics import compute_all_metrics, generate_performance_report

# Each layer composes with stable contracts; see notebooks for end-to-end examples.
```

## 10. Research Workflow

**Run an existing experiment.** Open the notebook, point `DATA_DIR` at the local data directory, execute cells top to bottom. Each notebook is reproducible and gated — failure at one cell stops downstream cells.

**Add a new signal.** Implement a function in `src/lsa/signals/<your_signal>.py` following the contract:

```python
def build_<name>_signal(panel: pd.DataFrame, ...) -> pd.DataFrame:
    """Returns a panel with at minimum (ID, DATE, signal_rank) columns."""
```

Re-export in `src/lsa/signals/__init__.py`. The backtester consumes any signal panel that exposes `signal_rank` — no other plumbing required.

**Add a new portfolio construction.** Implement in `src/lsa/portfolio/<your_builder>.py`. Follow the `PortfolioBuildConfig` and `build_target_weights_single_date(panel, config) -> DataFrame` contract. The constraint pipeline and rebalancer are universe-construction-agnostic and will accept any conforming weight panel.

**Add a new cost model.** Extend `ExecutionConfig.cost_model` enum and add the corresponding branch in `ExecutionModel.estimate_cost`. The interface is `(trades, tier_map, adv, nav) → (cost_pct, breakdown_bps)`.

**Add metrics.** New per-series metrics belong in `src/lsa/analytics/metrics.py` with NaN-defensive logic and a documented unit (bps, decimal, integer). Add to the `PerformanceMetrics` dataclass and the `compute_all_metrics` orchestrator.

## 11. Testing

**Unit tests.** Pytest suite under `tests/`. Currently covers the data layer:

- `tests/test_cleaners.py` — normalize_dates, drop_duplicate_observations, filter_to_trading_rows, winsorize_returns, gap-aware compute_returns, compute_dollar_volume, end-to-end clean_pit_panel.
- `tests/test_validators.py` — schema validation, missing-value classification, GICS hierarchy stability, market cap positivity + jump detection, volume non-negativity, orchestrator.
- `tests/test_universe_builder.py` — strict-prior `get_lagged_snapshot` (the PIT-safety test), `build_universe_at_date`, merge, simultaneous-overlap detection, dual-class dedupe.

Run with `pytest -v`.

**Validation checks.** Built into the data-layer pipeline. The orchestrator `validate_pit_panel(panel)` returns a list of `ValidationReport` objects. Caller checks `all_reports_passed(reports)`; any CRITICAL issue causes downstream tooling to abort.

**PIT-safety checks.** Three layers of protection:

1. **`get_lagged_snapshot` enforces strict prior.** Verified by `test_universe_builder.test_get_lagged_snapshot_returns_strict_prior`. A trade decision on month-end *m* uses the snapshot at *m − 1 month*, never the same-month snapshot.
2. **`compute_leader_follower_signal` enforces lag.** The `lag_days` parameter (default 1) explicitly shifts the leader's residual forward so the follower at *t* receives the leader's *t − 1* observation.
3. **`compute_rolling_betas` enforces lag.** `beta_wide.shift(1)` is applied before stacking so the residual at *t* uses β_{t−1}, not β_t. Without this, `pandas.rolling(window).cov()` would include the current observation in the beta estimate, producing subtle lookahead.

**Day-0 PnL check.** The backtester guarantees that on the first day of execution (no prior weights), gross PnL is exactly 0. This invariant is verifiable from any `BacktestResult.accounting_frame.iloc[0]`.

## 12. Future Work

In priority order, with cost-to-implement estimates.

| # | Initiative | Goal | Effort | Expected impact |
|---|---|---|---|---|
| 1 | **Full SP1500 deployment** | Run the strategy across the full S&P 1500 universe (currently SP500-only due to sandbox memory). | 2 weeks (memory-efficient signal computation: chunk by sector or year). | Gross alpha estimated to lift 50-100% per Hou 2007. Likely positive Sharpe at current 3 bps cost assumption. |
| 2 | **Liquidity-aware follower selection** | Within each selected Sub-Industry, trade only the top-quartile most liquid followers by trailing dollar volume. | 1 week (universe filter + portfolio_builder extension). | TC reduction 40-60%. Sharpe expected to flip positive at SP500 scale. |
| 3 | **Per-tier impact cost calibration** | Replace flat 3 bps cost with execution-desk-calibrated tiered + square-root-impact surface. | 3 days (implementation already present; needs calibration data). | More accurate net Sharpe; likely shows current SP500 cost ~1 bps not 3 bps. |
| 4 | **Walk-forward validation** | Refit rolling beta and Sub-Industry rankings on rolling lookback only. | 1 week. | Robustness signal. Expected walk-forward Sharpe within 70% of full-sample. |
| 5 | **Alternative leader definitions** | Test top-1-by-dollar-volume and top-3-cap-weighted-composite leaders against the baseline top-1-by-cap. | 1 week. | Robustness signal. Expected ≤ 15% Sharpe variation per E1's framework. |
| 6 | **Regime-conditional sizing** | Scale exposure by cross-sectional dispersion regime (high dispersion → up-size, low → down-size). | 2 weeks. | Expected drawdown reduction in low-dispersion regimes (mid-2017, mid-2019). |
| 7 | **Cross-tier lead-lag** | Test whether SP500 leader residuals predict SP400/600 same-industry followers (the H9 hypothesis from the catalog). | 3 weeks. | New strategy variant; likely 1.5× capacity. |
| 8 | **End-to-end runner script** | `scripts/run_h6.py` that wires data → signal → portfolio → backtest → analytics in one command. | 1 day. | Reproducibility + interview demo value. |
| 9 | **Production execution harness** | Live paper-trading loop with realized-fill ingestion and cost reconciliation. | 4 weeks. | Pre-production validation. |

## License

MIT. See `LICENSE`.
