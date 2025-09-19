ARCHIVED — Obsolete Alpha/Motif References

This document references motif-centered alpha and optimizer workflows that have been removed. The canonical path is allocator-lite for weights and guard/engine for trade admission. See `docs/01_System_Concepts.md` and `docs/02_Operations_Runbook.md`.

7.  Alpha Generation Playbook

Purpose
This playbook defines a rigorous, iterative process to maximize alpha from motif-driven trading in SEP. It addresses ranking and selection of motifs, portfolio allocation, execution policy (TP/SL), and validation via a modern backtesting methodology. It includes concrete experiments, metrics, and prioritized engineering tasks.

Baseline Snapshot (System Behavior Today)

- Allocation: Composite Top‑K (default K=3) with weights published to Valkey key `risk:allocation_weights` by optimizer-batch.
- Budget: Dynamic target ~30% margin utilization via `RISK_ALLOC_TARGET_PCT` only. No separate exposure scale env.
- Entries: MarginAwareMotif (blocks new entries ≥30% margin; kill at ≥80%). Motif admission requires score and quality (|μ1| ≥ ~0.5 bps, p_lcb1 and samples thresholds).
- Brackets: TP/SL attached on fill, RR fixed (default `AUTO_TP_RR=1.5`).
- Optimizer MQ thresholds: `OPT_MQ_SAMPLES_MIN=12`, `OPT_MQ_P_MIN=0.70` (strict) used to compute the MQ component in composite ranking.

Observed Pain Points

- Concentration: Top‑K + strict MQ thresholds can reduce diversification and overfit to stale composite components (Opt/Roots) when MQ filters admit too few motifs.
- TP/SL rigidity: Fixed RR can misalign with actual distribution of motif outcomes/volatility, clipping winners or over-stopping noise.
- Margin stall: Live utilization can float above 30% due to move/slippage; strict gate pauses new entries, harming opportunity capture and skewing PnL.
- Score calibration: Current motif score = |μ1| × p_lcb1 × log(samples+1) lacks shrinkage and volatility context; sensitive to sampling noise and regime shifts.

Goals

1. Improve motif selection precision/recall per instrument and time-of-day while maintaining diversification.
2. Align TP/SL with realized distribution of motif outcomes to increase P/L expectancy and reduce drawdowns.
3. Validate improvements with robust, walk-forward methodology and reality checks; prevent overfitting.

Methodology 2.0 (Backtesting and Validation)
Data Sources

- Motif stats: Valkey `strand:motifs:{instrument}:*` (mu1, p_lcb1, samples1, score).
- Market data: OANDA M1/M5 candles via `/api/oanda/candles` or cached `md:candles:{instrument}:{granularity}`.
- Live decisions: Ledger mirror `sep:ledger:events` via `/api/ledger/history` and motif performance endpoints from docs.

Reconstruction Targets

- Entry policy replay: For each candle, compute admission for selected motifs and produce a hypothetical entry stream (BUY/SELL or skip), including bracket distances.
- Portfolio replay: Apply Top‑K weights and dynamic budget to size per-instrument units, respecting margin caps.

Validation Protocol

- Walk-forward: Rolling train/test windows (e.g., 24h train, 6h test) across all 8 majors; repeat by hour-of-day strata.
- Nested CV for key hyperparams: p_lcb1 min, samples min, score shrinkage lambda, half-life decay, ATR k, RR grid.
- Robust significance: Bootstrapped CIs for expectancy; White’s Reality Check for multiple testing.
- Stress checks: Slippage grid (+0.1–1.5 pips), spread widening, delayed fill; sensitivity to Top‑K.

Metrics (report all by instrument and portfolio)

- Expectancy (bps/trade) and hourly realized bps.
- Win rate, avg win/avg loss, payoff ratio, Sharpe/Calmar, max drawdown.
- Trade count and motif coverage (diversification), time-in-trade.
- Margin utilization profile (median, p95), entry stall rate, kill‑switch events.

Motif Ranking and Selection
Scoring Improvements

- Shrinkage for μ: mu_shrunk = (n/(n+λ))·μ1 with λ per instrument (default 8–16). Reduces small-sample inflation.
- Variance-aware score: score = |mu_shrunk| / σ_hat × p_lcb1 × log(n+1), where σ_hat is robust volatility proxy (e.g., median absolute deviation of forward returns or ATR-normalized bps).
- Half-life decay: Weight motif performance by recency with exponential decay (τ=1–3 days on M1/M5 contexts).
- Time-of-day conditioning: Fit separate μ, p_lcb1 per 4–6 hour blocks; score uses matching block at inference.
- Instrument priors: Calibrate λ and p_lcb1 floors by instrument (e.g., stricter for JPY crosses if tails are heavier).

Admission Policy

- Multi-Motif Sampling: Admit up to N motifs per instrument (N=2–3), allocate sub-weights proportional to score, constrained by per-instrument cap. Reduces dependence on a single microstructure.
- Dynamic thresholds: Start with MIN_P_LCB1=0.60 (from 0.55) and MIN_SAMPLES=10; relax if portfolio exposure < 50% of target for >2 cycles to maintain coverage.
- Coherence-context guard: Tighten entry when rupture high or coherence low; relax when coherence and stability co-move (from gate metrics).

Execution Policy (TP/SL)

- ATR-anchored brackets: Compute ATR on M5/M15; SL = k·ATR (k∈[1.2–2.2]), TP = RR·SL with RR∈[1.0–1.8]. Grid search by instrument/time-of-day.
- Quantile-based alternative: Estimate motif outcome quantiles over lookahead window; set SL to q10 loss and TP near q65–q75 gain for asymmetric motifs.
- Trailing/step-out: Optional partial take at 1.0·SL, trail remainder by 0.75·ATR; validate via walk-forward.
- Spread-aware suppression: Suppress entries when spread/ATR > θ (e.g., >12%) or during scheduled illiquidity.

Portfolio and Sizing

- Rebalance Top‑K weekly with daily drift adjustments; add small residual weights (e.g., 1–2%) to non‑Top‑K for minimal diversification if clamp allows.
- Exposure sizing uses a fixed internal inverse‑leverage proxy; portfolio budget controls utilization. No `RISK_EXPOSURE_SCALE` env.
- Margin threshold hysteresis: Raise `MARGIN_AWARE_THRESHOLD` to 0.35 with a 5% hysteresis band to avoid oscillating stalls.

Optimizer Enhancements

- MQ computation: Use improved score (shrinkage/variance-aware) over motifs passing adaptive filters; lower `OPT_MQ_P_MIN` from 0.70 to 0.60 initially to restore signal diversity, then re-tighten per results.
- Composite weights: Start with Opt=0.45, Roots=0.20, MQ=0.35 to give MQ larger influence under live adaptation; revert if backtest dominance proves superior out-of-sample.
- Top‑K auto-tune: Evaluate K∈{2,3,4}; choose K with best risk-adjusted bps in last 72h, constrained by margin target.

Experiments (Two-Week Plan)
Phase 1: Instrumentation and Baselines (Day 1–2)

- Enable/validate endpoints: `/api/performance/motif-entries`, `/api/performance/motif-summary`, `/api/ledger/history` (limit 2000).
- Snapshot: `/api/risk/allocation-status`, `/api/oanda/account`, `/api/oanda/open-trades` hourly; persist JSON in `output/`.

Phase 2: Backtest Harness (Day 2–5)

Phase 3: Live A/B and Rollout (Day 6–14)

- Choose 2–3 policy variants; alternate hourly or by instrument.
- Guardrails: Keep kill active on entry expansion; allow only planned increases; auto-drain excess exposure.
- Promote winning variant if p95 out-of-sample expectancy > baseline and drawdown < baseline by at least 15%.

Immediate Param Recommendations (Safe Defaults)

- `OPT_MQ_P_MIN=0.60` (from 0.70) and `OPT_MQ_SAMPLES_MIN=10` to increase eligible motifs for MQ.
- `ALLOC_TOP_K=3` (keep) but consider residual clamp 1–2% if you enable `RISK_ALLOC_ENABLE_CLAMP=1`.
- `MARGIN_AWARE_THRESHOLD=0.35` with 0.05 hysteresis.
- Pilot `AUTO_TP_RR=1.2` for EUR_USD/GBP_USD during London/NY hours; adopt ATR-anchored brackets in code below.

Concrete Engineering Tasks 2) Motif Score Rework

- In `scripts/trading/strategies/margin_aware_motif.py` and optimizer-batch sources:
  - Compute mu_shrunk and σ_hat; recency weight with half-life decay; add time-of-day conditioning.
  - Replace score formula; expose env overrides: `SCORE_SHRINK_LAMBDA`, `SCORE_USE_SIGMA=1`, `SCORE_HALFLIFE_H`.

3. ATR‑Anchored Brackets

- In `scripts/oanda_client.py` (order construction) and `scripts/trading/api_orders.py`:
  - Add ATR fetch (M5/M15) and compute SL = k·ATR, TP = RR·SL; fall back to fixed RR when ATR unavailable.
  - Env: `AUTO_TP_MODE=atr|fixed`, `AUTO_TP_ATR_K=1.8`, `AUTO_TP_ATR_GRAN=M5`.

4. Margin Gate Hysteresis

- In `PortfolioManager` and `MarginAwareMotifStrategy`:
  - Add hysteresis: entries allowed < 0.30, blocked > 0.35; smooths oscillations.
  - Env: `MARGIN_HYST_LOW=0.30`, `MARGIN_HYST_HIGH=0.35`.

5. Multi‑Motif Sampling

- New util: `scripts/trading/motif_sampler.py` to select top N motifs with allocation proportions; share with PortfolioManager.
- Env: `MM_N=3`, `MM_MIN_SCORE`, `MM_MIN_SAMPLES`, `MM_MIN_P_LCB1`.

6. Optimizer Adjustments

- Switch MQ calculation to variance-aware mu_shrunk.
- Expose `OPT_WEIGHT_OPT/ROOTS/MQ` via env and wire into UI.
- Optional: implement Top‑K auto-tune over trailing 72h realized bps.

7. Telemetry and Reports

- Ensure performance endpoints are implemented:
  - `GET /api/performance/motif-entries` (recent ledger entries with motif stats)
  - `GET /api/performance/motif-summary?hours=24` (grouped counts by motif and direction)
  - `GET /api/ledger/history?limit=2000`
- Add `/api/performance/policy-leaderboard` to surface backtest variants.

How To Run (Operator Checklist)

1. Baseline snapshot
   curl -s /api/risk/allocation-status | jq > output/snap_alloc.json
   curl -s /api/oanda/account | jq > output/snap_account.json
   curl -s /api/oanda/open-trades | jq > output/snap_trades.json

2. Review leaderboard and promote
   Inspect output/backtests/metrics.json and `/api/performance/policy-leaderboard`.

Contingencies and Risk Controls

- Keep kill-switch ON during parameter swaps; allow only exposure reductions.
- Disable new TP/SL mode per instrument if outage in ATR feed or spread exceeds threshold.
- Maintain used-margin <35% p75; drain-overcap script for fast reductions.

Expected Impact

- Higher motif coverage with improved precision through shrinkage and recency weighting.
- Better expectancy with ATR-anchored brackets tailored to volatility/time-of-day.
- Reduced drawdowns via diversification across motifs and smoother margin gating.

Appendix A: Example Score
score = (|mu_shrunk| / σ_hat) × p_lcb1 × log(n+1) × w_recency
where mu_shrunk = (n/(n+λ))·μ1, σ_hat = robust volatility proxy, w_recency = exp(-Δt/τ).

Appendix B: Parameter Starting Grid

- λ∈{8,12,16}; τ∈{24h,48h,72h}; ATR-k∈{1.4,1.8,2.2}; RR∈{1.0,1.2,1.5}; K∈{2,3,4}.
- MIN_P_LCB1∈{0.55,0.60,0.65}; MIN_SAMPLES∈{8,10,12}.

Next Steps

- Implement backtest harness and ATR brackets.
- Reduce MQ strictness to 0.60/10 and observe 24–48h live impact.
- Rebalance composite weights to elevate MQ influence; monitor via optimizer scoreboard.

Version: 1.0 (2025-09-11)
