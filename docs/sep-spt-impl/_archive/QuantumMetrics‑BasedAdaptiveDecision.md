Quantum Metrics–Based Adaptive Decision Framework

**Abstract**
- Markets and other complex systems generate high‑velocity time series where spurious patterns can masquerade as signal. This whitepaper formalizes an operational framework used in SEP’s live trading system that measures directional agreement (coherence), persistence (stability), structural breaks (rupture), and randomness (entropy) on sliding windows, then gates and allocates only when strict, adaptive criteria are met. The approach emphasizes event‑time backtesting, strict span integrity, per‑pair hysteresis with cooldown, single‑source gating for allocation, and observability. Results show stable selection with reduced churn and improved live alignment. The framework generalizes to any decision problem driven by streaming telemetry.

**1. Motivation**
- Data abundance creates false positives. Indicators overfit or respond to noise.
- We target “moments of interest”: intervals with statistically meaningful coordination or disruption.
- Requirements: scale across time windows, resist oscillation, degrade safely, and surface decisions with auditability.

**2. Core Metrics (Intuition)**
- Coherence (C): directional agreement across recent samples; higher implies aligned motion and a usable slope.
- Stability (S): persistence of structure across adjacent windows; higher implies repeatability.
- Rupture (R): strength of structural breaks (change‑points) that reset regimes; lower is safer for continuation.
- Entropy (H): unpredictability of the stream; lower suggests more regularity and less noise.
- These are computed on sliding windows over an encoded bitstream (QBSA) and summarized into daily “manifolds”.

**3. Methodology and Architecture**
- Signal encoding and metrics: QBSA encodes price delta/sign; QFH computes C/S/R/H on windows; daily manifolds persist per instrument.
- Strict spans: selection uses a single long run, and all “series slices” for 14/30/60d are derived from the same run to ensure monotonic counts and comparable metrics.
- Event‑time backtesting: the harness replays historical events with injected timestamps so engine cooldowns elapse identically to live.
- Rolling evaluator (30‑day): recomputes strict backtests every few minutes, records summaries, and updates eligibility gates.
  - Update (2025‑09): evaluator runs every 60s; session‑gated (skip closed, block late adds), and writes consolidated `opt:rolling:gates_blob`.
- Adaptive gates (conservative floors + percentiles):
  - MIN_TRADES = max(floor, p50(trades)) with floor typically 10.
  - CALMAR_MIN = max(floor, p50(calmar) + 0.10·IQR).
  - PNL_MIN = max(floor, p40(pnl)).
  - Hysteresis: per‑pair buffer, two consecutive passes/fails to flip; cooldown after 4 misses for 2h; re‑entry requires fresh passes. Buffers are clamped to [0.00, 0.30] with pair‑specific defaults (CHF wider) and env overrides (`HYSTERESIS_DEFAULT`, `HYSTERESIS_PAIR_JSON`).
- Single source of truth for allocation: after all instruments are evaluated, the service writes `opt:rolling:gates_blob` with a complete snapshot and timestamp.
  - JSON schema: `{ ts: ms, gates: {inst:0|1}, cooldowns: {inst:0|1}, bufs: {inst:0.10} }`.
  - The allocator reads only this blob; if missing/stale, it holds previous weights and logs a warning instead of improvising.

**3.1 Data→Metrics→Signals→Decision (End‑to‑End)**
- Ingest: append new sample `x_t` to a per‑instrument ring buffer.
- Encode: transform to compact features `b_t` (e.g., sign of delta, volatility‑scaled step, spread‑aware mask).
- Window: select `W_t = {b_{t−w+1}..b_t}`; compute C/S/R/H over `W_t` and adjacent windows.
- Signal: form a feature vector `f_t = {C_t, S_t, R_t, H_t}` and a quality score.
- Gate: apply strict thresholds + per‑pair hysteresis and cooldown to produce `eligible_t ∈ {0,1}`.
- Allocate: write a single `gates_blob` for all instruments; allocator ranks by scores, applies Top‑K under risk caps.
- Telemetry: export `sep_rolling_gate{instrument, eligible, buf}` and selection/cooldown gauges; persist a summary at `bt:rolling:summary:{instrument}`.

Example (toy series, 5‑sample window):
- Price deltas: `+,+,+,+,−,−,−,+,+` → encode to `b_t ∈ {−1,+1}`.
- At t=9: coherence `C=0.67` (majority alignment), stability `S=0.60` (recent windows similar), rupture `R=0.10` (no break), entropy `H=0.30` (moderate regularity).
- Gate floors: `min_trades=10` (met in rolling backtest), `calmar ≥ 0.15` (met), per‑pair buffer `buf=0.10`.
- Hysteresis: second consecutive pass flips `eligible: 0→1`.

**3.2 Implementation Crosswalk (Repo Evidence)**
- Encoding + kernel (C++):
  - `src/core/qfh.h/.cpp`: `QFHBasedProcessor::analyze` computes coherence, stability, entropy, rupture ratio; emits `QFHResult` with events/aggregates.
  - `src/app/manifold_generator_main.cpp`: loads candles (file/Valkey), encodes to bitstream, runs QFH, writes manifold JSON (signals). Built as `bin/manifold_generator`.
- Historical signals + indexing (Python):
  - `scripts/ops/prime_qfh_history.py`: runs `manifold_generator`, indexes per‑signal hashes `sep:signal:{instrument}:{ts_ns}` and ZSET `sep:signal_index:{instrument}`; optional daily manifold `manifold:{instrument}:{YYYY-MM-DD}`.
- Event‑time strict backtests:
  - `scripts/backtest_engine.py`: StrategyEngine backtests use event‑time to honor cooldowns identically to live.
- Rolling evaluator + adaptive gates:
  - `scripts/rolling_backtest_evaluator.py`: `_series_gate` (percentiles + floors), pure helpers `evaluate_and_gate_once` and `apply_cooldown_logic`, per‑pair hysteresis `get_hysteresis(pair)` with clamps [0.00, 0.30], consolidated `opt:rolling:gates_blob`.
  - Session gating and repetition: evaluator filters to `open_instruments(now_utc)` and forces ineligible within `SESSION_EXIT_MINUTES`; optional repetition gate requires coarse pattern match ≥ N in trailing minutes.
- Allocator + staleness guard:
  - `scripts/allocator_lite.py`: scoring `_score`, eligibility `_eligible_map` reads only blob, `should_publish` blocks stale/missing; publishes `risk:allocation_weights`; `/status` endpoint.
- Backend exporter:
  - `scripts/trading/http_api.py`: `render_prometheus_metrics` emits `sep_rolling_gate{instrument,eligible,buf}`, `sep_allocator_selected`, `sep_allocator_cooldown`, span integrity and diagnostics.
- UI (frontend):
  - `apps/frontend/src/components/OandaCandleChart.tsx`: overlays C/S/H/R per candle (coherence color, stability dash, entropy alpha, rupture line).
  - `apps/frontend/src/context/WebSocketContext.tsx`: subscribes to `manifold/signals/performance` and extracts coherence/stability/entropy.

**3.3 Decision Grammar (Concrete Rules)**
- Floors from `_series_gate` → `min_trades`, `calmar_min`, `pnl_min`.
- Hysteresis buffer: `buf = get_hysteresis(pair) ∈ [0.00, 0.30]` (env overrides via `HYSTERESIS_PAIR_JSON`).
- Flip enable (prev=0): require two consecutive passes with metrics ≥ floor·(1+buf).
- Flip disable (prev=1): require two consecutive fails with metrics ≤ floor·(1−buf).
- Cooldown: arm after 4 consecutive misses (TTL 2h); while active, force `prev=0`, reset counters; re‑entry needs fresh passes.
- Consolidation: after all instruments, write a single `gates_blob`; allocator must not mix per‑key gates.
 - Session rules: only open instruments are considered; `ExitBeforeClose` flattens positions before session close; pulse trader places a small budget‑aware entry every ~240–300s when gates are green.

**4. Allocation and Risk Management**
- Scoring: weighted sum W = 0.6·C + 0.2·S + 0.15·(1–R) + 0.05·(1–H); rank and select Top‑K.
- Budget: clamp to 30% portfolio target, ~10% per instrument; never exceed total cap.
 - Margin gating: hold entries when `marginUsed/(marginUsed+marginAvailable) ≥ MARGIN_HYST_HIGH` (default 0.30).
 - Budget is dynamic: `max_total_exposure ≈ min(balance, marginAvailable) × RISK_ALLOC_TARGET_PCT (0.30)`; inspect `/api/risk/budget`.
- Guards at entry: live coherence ≥ 0.55; ATR‑based TP/SL on every open; kill‑switch allows only closes.
- Cooldown scope is (strategy, instrument) and respects event‑time during backtests.

**5. Observability and Controls**
- Prometheus metrics (text exposition on backend):
  - `sep_rolling_gate{instrument,eligible="0|1",buf="0.10"} 1` (one series per instrument; eligible count via `eligible="1"`).
  - `sep_allocator_selected{instrument}` (gauge 0/1) and `sep_allocator_cooldown{instrument}`.
  - Additional ops gauges: span integrity, WS drops, diagnostics ratios.
- Alerts (Prometheus rules):
  - Fewer than 3 eligible for 10m (page), cooldown lockout > 4 for 30m (warn), NAV down 7d while backtest Calmar positive (investigate).
- CI and safety:
  - Unit tests for hysteresis transitions and cooldown grace (pure functions). Gate staleness checks (`should_publish`). Exporter presence test enforces `buf` label.
  - GitHub Actions matrix (3.10/3.11) runs focused suites; coverage artifacts published.
  - Make targets: `test-gates`, `test-allocator`, `test-metrics`, `test-ci`, `coverage`.

**5.1 Example Artifacts (Live/Simulated)**
- Valkey blob (fresh snapshot):
```json
{
  "ts": 1726339200123,
  "gates": {"EUR_USD": 1, "USD_CHF": 0, "USD_JPY": 1},
  "cooldowns": {"EUR_USD": 0, "USD_CHF": 0, "USD_JPY": 0},
  "bufs": {"EUR_USD": 0.10, "USD_CHF": 0.15, "USD_JPY": 0.10}
}
```
- Backend `/metrics` lines (excerpt):
```
sep_rolling_gate{instrument="EUR_USD",eligible="1",buf="0.10"} 1
sep_rolling_gate{instrument="USD_CHF",eligible="0",buf="0.15"} 1
sep_allocator_selected{instrument="EUR_USD"} 1
sep_allocator_cooldown{instrument="USD_CHF"} 0
```
- Allocator status:
```json
{"ts": 1726339201, "top_k": 3, "weights": {"EUR_USD": 0.34, "USD_JPY": 0.33, "GBP_USD": 0.33}}
```
- Rolling summary payload:
```json
{"ts":1726339200, "params": {"c_min":0.55, "direction_lookback":12, "dir_threshold_bps":2.0},
 "summary": {"trades":18, "calmar":0.22, "pnl_total":0.008},
 "gates": {"min_trades": 12, "calmar_min": 0.18, "pnl_min": 0.002},
 "eligible": 1}
```

**5.2 Live Snapshots (from repo outputs)**
- Manifold metrics (USD_JPY, 2025‑09‑09): count=890; metrics={coherence=0.5197, stability=0.8898, entropy=0.8769, rupture=0.1102}.
  - Source: `output/manifolds/USD_JPY/2025-09-09.json` (field `metrics`, `count`).
- Manifold metrics (EUR_USD, 2025‑09‑10): count=1431; metrics={coherence=0.4837, stability=0.8583, entropy=0.8954, rupture=0.1417}.
  - Source: `output/manifolds/EUR_USD/2025-09-10.json`.
- Manifold metrics (USD_JPY, 2025‑08‑31): count=158; metrics={coherence=0.4996, stability=0.8661, entropy=0.9015, rupture=0.1339}.
  - Source: `output/manifolds/USD_JPY/2025-08-31.json`.
- Allocator weights snapshot (Top‑K proportions):
```json
{"weights": {"GBP_USD": 0.3091, "USD_CHF": 0.2852, "EUR_USD": 0.2220, "USD_JPY": 0.1836}}
```
  - Source: `output/snap_alloc.json`.

These artifacts are included as direct evidence of the running pipeline: native manifolds (kernel outputs) and allocation decisions captured by the allocator‑lite.

**6. Technology Kernel (Data‑Agnostic Definition)**
- Inputs: streaming sequence `x_t`, configurable encoder `E(·)`, window size `w`, stride `s`.
- Outputs per step: feature vector `f_t = {C_t,S_t,R_t,H_t}`, optional derivatives (e.g., `ΔC_t`).
- Coherence kernel: normalized agreement of signed steps in `W_t` (e.g., `C_t = |mean(sign(Δx))|` or cosine similarity between `W_t` and a low‑rank basis).
- Stability kernel: self‑similarity across adjacent windows (e.g., average cosine between `W_t` and `W_{t−k}`, k∈{1..K}).
- Rupture kernel: change‑point energy across a cut τ in `W_t` (e.g., max two‑sample statistic or cumulative sum deviation normalized by window variance).
- Entropy kernel: approximate/sample entropy of the encoded sequence in `W_t` (lower implies more regular patterns).
- Contract:
  - `process(x_t) → f_t` where `E` and kernels are parameterized but domain agnostic.
  - Deterministic given `E`, `w`, `s`; bounded compute and memory `O(w)` per step.
  - Portable across domains by swapping `E` (e.g., price deltas, sensor residuals, packet sizes).

Pseudocode:
- `b_t = E(x_t)`; `W_t = {b_{t−w+1}..b_t}`; compute C/S/R/H; emit `f_t` and quality flags.
- Decision layer consumes `f_t` via a small grammar: floors → hysteresis → cooldown → blob.

**6. Implementation Map (Reproducibility)**
- State store (Valkey):
  - Signals: `sep:signal_index:{instrument}`; Manifolds: `manifold:{instrument}:{YYYY-MM-DD}`.
  - Rolling summaries: `bt:rolling:summary:{instrument}`; History: `bt:rolling:hist:{instrument}:{param_key}`.
  - Eligibility: `opt:rolling:eligible:{instrument}`, passes/fails/misses, `opt:rolling:cooldown:{instrument}`.
  - Allocator weights: `risk:allocation_weights`; Gates blob: `opt:rolling:gates_blob`.
- Environment:
  - `EVAL_INTERVAL_SEC` (rolling cadence), `ALLOC_INTERVAL_SEC` (allocator tick), `HOTBAND_PAIRS` (universe).
  - Floors: `EVAL_MIN_TRADES`, `EVAL_CALMAR_MIN`, `EVAL_PNL_MIN`.
  - Hysteresis: `HYSTERESIS_DEFAULT`, `HYSTERESIS_PAIR_JSON` (clamped [0.00, 0.30]).
- Services:
  - Rolling Evaluator (30d strict), Allocator‑Lite (Top‑K), Backend (API + StrategyEngine), WS Hydrator.
  - Status endpoints: `GET :8100/status` (allocator), `/metrics` (backend), various `/api/*` for ops.

**6.1 Proof‑of‑Work Artifacts (Repo & Metrics)**
- Keys/values:
  - `opt:rolling:gates_blob` — single source snapshot `{ts,gates,cooldowns,bufs}` used by allocator.
  - `opt:rolling:eligible:{instrument}`, `opt:rolling:passes:{instrument}`, `opt:rolling:fails:{instrument}` — hysteresis memory.
  - `bt:rolling:summary:{instrument}` — latest strict backtest summary `{ts,params,summary,gates,eligible}`.
- Prometheus lines (backend `/metrics`):
  - `sep_rolling_gate{instrument="EUR_USD",eligible="1",buf="0.10"} 1`
  - `sep_allocator_selected{instrument="EUR_USD"} 1`
  - `sep_allocator_cooldown{instrument="USD_CHF"} 0`
- Allocator status (HTTP): `GET :8100/status` → `{"ts":...,"weights":{"EUR_USD":0.34,...},"top_k":3}`.

**6.2 Reproducibility (Commands)**
- Prime historical signals (14d): `python3 scripts/ops/prime_qfh_history.py --days 14`
- Run strict 30d backtest (single):
  - `python3 -c "from scripts.backtest_engine import run_backtest; print(run_backtest('EUR_USD', 30, overrides={'c_min':0.55,'direction_lookback':12,'dir_threshold_bps':2.0,'entry_cooldown_sec':60,'exit_cooldown_sec':60}, persist=False))"`
- Verify rolling evaluator writing blob:
  - `redis-cli GET opt:rolling:gates_blob | jq`
- Check exporter and selection:
  - `curl -s :8000/metrics | rg '^sep_rolling_gate\{|^sep_allocator_selected\{|^sep_allocator_cooldown\{'`
- Allocator staleness guard (CI): ensure `should_publish` blocks stale blobs; tests: `make test-allocator`.

**6.3 Figures (Generated from Repo Data)**
- Latest metrics per instrument (kernel): `docs/reports/figures/latest_metrics_per_instrument.png`.
- Signal counts time‑series per instrument: `docs/reports/figures/signal_counts_timeseries.png`.
- Source CSV: `docs/reports/snapshots/manifold_metrics_all.csv` (built from `output/manifolds/*/*.json`).
- Allocator snapshot JSON: `docs/reports/snapshots/snap_alloc.json`.

**7. Results Snapshot (Trading Case Study)**
- Universe: 8 major FX pairs, strict 30‑day backtests; 5–33 trades per pair with event‑time cadence vs ~0–1 under naive wall‑clock.
- Selection stability: with per‑pair hysteresis and cooldown, eligible sets persist 10–30 minutes across cycles; Top‑3 rotates only when metrics genuinely shift.
- Risk discipline: exposure remains ≤ 30% with ~10% per instrument; no manual overrides required.

Worked decision trace (realistic):
- Backtest summary (30d): `trades=18`, `calmar=0.22`, `pnl_total=+0.8%` → passes floors.
- Live kernel at t: `C=0.64`, `S=0.58`, `R=0.12`, `H=0.27`; prior eligible=0, passes=1.
- Hysteresis: with `buf=0.10`, second consecutive pass flips eligible=1; blob publishes `gates[inst]=1`.
- Allocator Top‑3 includes instrument with ~10% budget; exporter shows `sep_rolling_gate{eligible="1",buf="0.10"}`.

**7.1 Key Results (Measured from Repo Artifacts)**
- Kernel throughput (sampled): `signals.count` per instrument per day ranged from 158 (USD_JPY, 2025‑08‑31) to 1431 (EUR_USD, 2025‑09‑10) in `output/manifolds/*/*.json`.
- Manifold quality (sampled): coherence ~0.48–0.52; stability ~0.86–0.89; entropy ~0.87–0.90; rupture ~0.11–0.14 (see 5.2 for exact values).
- Allocation snapshot: Top‑K weights concentrated ~18–31% across four majors at the time of capture (`output/snap_alloc.json`).


**8. Cross‑Industry Adaptations**
- Predictive maintenance (industrial IoT): high rupture + entropy → failure risk; allocate inspections; cooldown to avoid oscillating work orders.
- Energy grids: coherence across substations → coordinated oscillations; guard and allocate reserves/load‑shedding accordingly.
- Supply chain: rupture in transit times + entropy in inventory → disruption; allocate capacity to critical lanes.
- Networks/SRE: stability/coherence of latency/throughput; rupture alerts for incidents; allocate bandwidth/mitigation.

Blueprints (encoder swaps):
- Manufacturing: `E` = residuals of temperature/pressure vs control charts; window `w` short for fault onset.
- Energy: `E` = frequency deviations and phase angles; coherence across nodes indicates oscillations.
- Supply chain: `E` = z‑scored transit times; rupture flags disruptions; stability tracks seasonality.
- Networks: `E` = packet size/flow deltas; entropy spikes on scanning/attack; coherence on coordinated floods.

**9. Limitations and Risk Considerations**
- Data quality: gaps and timestamp skew degrade coherence/stability; span checks mitigate but do not eliminate risk.
- Regime shifts: percentile‑based gates can lag true breaks; rupture helps but cannot foresee exogenous shocks.
- Slippage and exits: selection quality does not guarantee execution quality; audit MAE/MFE and exit efficiency.

Validation notes:
- Event‑time replay eliminates cooldown artifacts; strict spans guarantee monotonic counts; CI enforces exporter, blob staleness, and hysteresis transitions.
- A/B: track NAV vs selected‑pair backtest calmar; investigate exits if divergence persists.

**10. Future Work**
- Multivariate coherence across instruments; cross‑asset context.
- Robust estimators for entropy and change‑points under heavy tails.
- Automated, per‑pair hysteresis learning from realized churn and PnL.
- Theoretical bounds on false‑positive rates for gating thresholds.

Extensions in current OANDA application:
- Multiscale kernels: combine short/medium windows with a learned mixer; penalize contradictory C/S across scales.
- Learned per‑pair buffers: adapt `buf` via observed churn/NAV contribution while honoring clamps and CI.
- Spread‑aware gating: integrate live spread/latency to suppress opens during microstructure stress.
- Cross‑pair context: add multivariate coherence to avoid redundant picks; diversify Top‑K.
- Exit efficiency telemetry: compute `sep_exit_efficiency{instrument}` = realized PnL / MFE; alert when < 0.4.

**11. Figures to Include (Deck/Appendix)**
- Eligible count vs NAV (7d) with shaded cooldowns.
- Selection stability histogram: duration of continuous selection per instrument.
- Churn vs buffer (scatter): observed flips/day vs `buf` per pair.
- Blob freshness heatmap: time since last blob by instrument across day.
- Backtest vs live: calmar/pnl scatter pre/post hysteresis rollout.
 - Kernel quality over time: use `docs/reports/snapshots/manifold_metrics_samples.csv` to seed plots of C/S/H/R per instrument/date.

**Appendix A: Operational Guardrails (Summary)**
- Strict spans only; adaptive gates with floors; per‑pair hysteresis (clamped) and cooldown.
- One truth blob for allocation; allocator holds on stale.
- Risk: exposure cap 30% (~10%/instrument), live guard ≥ 0.55, ATR TP/SL.
- CI and alerts enforce behavior; see Operations Runbook for the pre‑open checklist.

**Appendix B: Kernel Interface (Sketch)**
- `class Kernel:`
  - `def __init__(self, encoder: Encoder, window: int, stride: int): ...`
  - `def step(self, x_t) -> dict:  # returns {C,S,R,H, ts}`
- `class Encoder:`
  - `def __call__(self, x_t) -> float | int | tuple:  # domain‑specific transform`
- Decision layer contract:
  - Inputs: `{C,S,R,H}`, floors, hysteresis `{prev,passes,fails}`, cooldown state.
  - Outputs: `eligible ∈ {0,1}`, updated counters, blob fields `{gates,cooldowns,bufs}`.
**5.3 New APIs (2025‑09)**

- `/api/ranking/active` — session‑aware ranking (open first) with score, flow(5m), C/S/H/R, slope, λ.
- `/api/strategy/status` — shows wired strategies and exits, plus cooldowns.
- `/api/risk/budget` — dynamic budget snapshot (balance, margin_available, target_pct, budget).
