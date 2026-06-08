# long-short-alpha

**US Equity Long-Short Strategy — Sub-Industry Leader-Follower Lead-Lag (H6)**

Research repository for an institutional-quality systematic strategy that trades the gradual information diffusion from cap-weighted leaders to their followers within each GICS Sub-Industry.

## Hypothesis

After residualizing single-name returns against their matched GICS-Sector SPDR ETF, the idiosyncratic return of the largest-market-cap name within each GICS Sub-Industry positively predicts the equal-weighted idiosyncratic return of its sub-industry peers over the subsequent 1–5 trading days.

**Economic mechanism**: bandwidth-limited information diffusion. Sell-side coverage, passive flow, and dealer activity concentrate in the leader; smaller peers re-rate with a measurable lag.

## Dataset

| File | Description |
|---|---|
| `etf_ohlcv_*.csv` | Daily OHLCV for 14 ETFs (SPY, QQQ, RSP, 11 GICS sector SPDRs) |
| `sp400_pit_*.csv` | S&P 400 PIT membership panel |
| `sp500_pit_*.csv` | S&P 500 PIT membership panel |
| `sp600_pit_*.csv` | S&P 600 PIT membership panel |

Coverage: 2016-03-01 → 2025-12-31. Raw data lives in `data/` (gitignored).

## Documents (`docs/`)

| # | Document | Purpose |
|---|---|---|
| 00 | Data Audit | Structural integrity, biases, quarantine list |
| 01 | Dataset Teaching | First-principles walkthrough of the four files |
| 02 | Hypothesis Catalog | 25 candidate hypotheses across four research surfaces |
| 03 | Showcase Ranking | Re-rank under the interview-submission lens; H6 fund-first |
| 04 | Research Design | Pre-registered investment-committee proposal |
| 05 | Experimental Framework | Nine experiments with pass/fail criteria |

## Experiments (`notebooks/`)

The nine experiments from the framework, executed in dependency order. Experiment 3 (Lead-Lag Validation) is the binary kill-gate; Experiments 1, 2 are its foundation.

| # | Notebook | Gate |
|---|---|---|
| 00 | Data audit (reproduction) | — |
| 01 | Leader identification validation | Foundation |
| 02 | Residualization validation | Foundation |
| 03 | Lead-lag relationship validation | **Kill gate** |
| 04 | Signal decay analysis | Horizon |
| 05 | Liquidity analysis | Capacity floor |
| 06 | Sector neutrality analysis | Realized exposure |
| 07 | Time stability analysis | Regime robustness |
| 08 | Capacity analysis | AUM ceiling |
| 09 | Transaction cost analysis | Net economics |

## Source layout (`src/lsa/`)

```
lsa/
├── data/        # loaders, PIT integrity rules, calendar
├── features/    # residualization, leader identification, GICS utilities
├── signals/     # leader-follower signal construction
├── portfolio/   # construction, neutralization, sizing
├── backtest/    # simulation engine, cost model
└── utils/       # logging, configs, stats helpers
```

## Reproducing the work

1. Place raw data in `data/`.
2. `pip install -e .`
3. Run notebooks in numerical order. Each notebook's gate must clear before the next is run.
4. Final outputs land in `results/` (gitignored).

## Status

Pre-research phase. Documents complete; implementation pending Phase 1 of the approved RDD.
