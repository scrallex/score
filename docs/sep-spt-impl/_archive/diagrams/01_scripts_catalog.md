# Scripts & Tooling Variable Catalogue

**Scope**: everything under `scripts/` (Python helpers, ops utilities), shell entrypoints (`deploy.sh`, `bin/` wrappers), and generated CLI tools that influence runtime behaviour.

## 1. System Profile
- **Owner**: Ops / Infra
- **Runtime**: Python 3.12 (containers), Bash (host)
- **Primary responsibilities**: deployment orchestration, signal priming, backtests, data captures, Valkey manipulations.
- **External dependencies**: OANDA REST, Valkey, Docker, local filesystem (output/, config/).
- **Key data stores**: `.env.hotband`, `config/live_params.yaml`, Valkey keys (`ws:last:manifold:*`, `sep:signal_index:*`).

_Use `templates/system_profile.md` for detailed subsystem drill-downs (one per major script family)._ 

## 2. Variable Mapping Table
| Variable | Location | Type | Source | Sinks / Consumers | Notes |
| --- | --- | --- | --- | --- | --- |
| BACKFILL_ON_DEPLOY | `deploy.sh:136` | env | `.env.hotband` override; defaulted in `load_env_for_compose` | `deploy.sh:447` warmup gate; `scripts/ops/warmup_orchestrator.py --steps backfill` | Enables hot deploy candle backfill + signal priming.
| RUN_TRIALS_ON_DEPLOY | `deploy.sh:139` | env | `.env.hotband`; default 1 | `deploy.sh:564` winners stage; pipes YAML spec into `scripts/run_backtests.py` | Controls post-deploy trials + promotions.
| APPLY_WINNERS_ON_DEPLOY | `deploy.sh:140` | env | `.env.hotband`; default 1 | `deploy.sh:592` promotion block; `scripts/ops/apply_live_params.py` | Applies best configs into env & redeploys backend/websocket.
| CANDLE_COUNT | `deploy.sh:532` | env | `.env.hotband`; CLI `--count` | `deploy.sh:530` REST backfill; `scripts/ops/warmup_orchestrator.py --steps backfill` | Limits candles per instrument during warmup/backfill.
| HOTBAND_PAIRS | `deploy.sh:131` | env | `.env.hotband`; CLI `--pairs` | Warmup helpers in `deploy.sh`; `scripts/ops/warmup_orchestrator.py`, `scripts/ops/prime_qfh_history.py`, `scripts/run_backtests.py`, `scripts/report_trials.py` | Canonical instrument list shared across orchestration.
| opt:best_config:{instrument} | `scripts/report_trials.py:173` | Valkey key | `scripts/report_trials.py` | `scripts/ops/promote_winners.py:26`, deploy apply step, UI overlays | Per-instrument recommendation blob.
| ws:last:manifold:{instrument} | `scripts/ops/warmup_orchestrator.py` | Valkey key | `ws-hydrator`; `deploy.sh` warmup hydrate | Websocket server + frontend gating | TTL warm cache for manifolds.
| bt:trial:index | `scripts/run_backtests.py` | Valkey key | `scripts/run_backtests.py` | `scripts/report_trials.py:229`, analytics dashboards | Trials index (ZSET) scanned for winners.
| opt:backtests:report:latest | `scripts/report_trials.py:324` | Valkey key | `scripts/report_trials.py` | Frontend analytics (report modal), ops reviews | Cached full report JSON for UI consumption.
| output/reports/backtests/last_{days}d_report.md | `scripts/report_trials.py:309` | file | `scripts/report_trials.py` | Ops review docs, portfolio postmortems | Markdown report summarising latest trials run.
| docs/reports/snapshots/{stamp}_*.json | `scripts/ops/capture_live_snapshot.py:34` | file set | `scripts/ops/capture_live_snapshot.py` | Docs snapshots, incident reviews, allocator QA | Timestamped trading state captures (account/open trades/alloc/gates).
| output/manifolds/{instrument}/{date}.json | `scripts/ops/prime_qfh_history.py:321` | file set | `scripts/ops/prime_qfh_history.py` | Research notebooks, signal QA, manual diffing vs live manifolds | Stored manifold JSON per instrument-day for offline validation.
| config/live_params.yaml | `scripts/ops/promote_winners.py:88` | file | `scripts/ops/promote_winners.py` | `deploy.sh` apply step, `.env.hotband`, allocator-lite | Canonical promoted winners snapshot before env application. |
| manifold.metrics (C/S/H/R) | `src/app/manifold_generator_main.cpp:374` | metric | `bin/manifold_generator` | `scripts/ops/prime_qfh_history.py`, `ws_hydrator`, `backend /api/coherence/status`, allocator scoring | QFH metrics 0–1 gated by `AUTO_MIN_COHERENCE`, `GUARD_MIN_STABILITY`, `AUTO_MAX_ENTROPY`, `AUTO_MAX_RUPTURE`. |
| manifold.coeffs (sigma_eff, lambda) | `src/app/manifold_generator_main.cpp:444` | metric | `bin/manifold_generator` | `ws_hydrator`, risk dashboards | Vol proxy + rupture hazard used by pattern gate & risk clamps. |
| bt:rolling:summary:{instrument} | `scripts/rolling_backtest_evaluator.py:500` | Valkey key | `scripts/rolling_backtest_evaluator.py` | Ops dashboards, adaptive gates | Strict 30d summary + adaptive floors (min trades, calmar, pnl). |
| opt:rolling:eligible:{instrument} | `scripts/rolling_backtest_evaluator.py:505` | Valkey key | `scripts/rolling_backtest_evaluator.py` | `allocator_lite`, kill-switch monitors | Binary eligibility w/ hysteresis, cooldown, repetition gate. |
| opt:rolling:gates_blob | `scripts/rolling_backtest_evaluator.py:511` | Valkey key | `scripts/rolling_backtest_evaluator.py` | `allocator_lite`, frontend status | Consolidated `{ts,gates,cooldowns,bufs}` snapshot. |
| risk:allocation_weights | `scripts/allocator_lite.py:115` | Valkey key | `scripts/allocator_lite.py` | Trading service, frontend, ops snapshots | Top-K weight vector bounded by `RISK_ALLOC_TARGET_PCT`, `PM_MAX_PER_POS_PCT`. |

## 3. Variable Records
### `BACKFILL_ON_DEPLOY`
- **File / Symbol**: `deploy.sh:136`
- **Type**: env
- **Declared in**: `deploy.sh:120` (`load_env_for_compose`) exports default, consumed at `deploy.sh:447` inside `seed_pairs_after_deploy`
- **Purpose**: toggles the post-deploy warmup (OANDA/Valkey connectivity check + candle fetch) so the UI has history before traders log in.
- **Default / Range**: Defaults to `"1"` when unset; expects `0/1` string.
- **Read by**: `deploy.sh:447` (skip/execute `seed_pairs_after_deploy`).
- **Mutated by**: `.env.hotband`, manual operator export before invoking `deploy.sh`.
- **External Effects**: When `1`, `deploy.sh` calls `/api/candles/fetch` per `HOTBAND_PAIRS` and primes QFH signals, repopulating `md:candles:*` and `sep:signal_index:*` (documented in `docs/diagrams/11_data_flows.mmd`).
- **Related Concepts**: `HOTBAND_PAIRS`, `CANDLE_COUNT` (determine workload).
- **Open Questions**: None yet.

### `RUN_TRIALS_ON_DEPLOY`
- **File / Symbol**: `deploy.sh:139`
- **Type**: env
- **Declared in**: Defaulted in `load_env_for_compose` (`deploy.sh:120`); branch at `deploy.sh:564` gating `derive_winners_and_apply`
- **Purpose**: decides whether a deploy should trigger the backtest pipeline (`ops/prime_qfh_history.py` → `run_backtests.py` → `report_trials.py`).
- **Default / Range**: Defaults to `"1"`; `0` skips trials.
- **Read by**: `deploy.sh:564`.
- **Mutated by**: `.env.hotband`, CI overrides for faster deploy.
- **External Effects**: When `1`, re-computes `bt:trial:*` series and refreshes `opt:best_config:*` plus `opt:backtests:report:latest`; see `docs/diagrams/11_data_flows.mmd`.
- **Related Concepts**: `APPLY_WINNERS_ON_DEPLOY`, `DISENGAGE_KILL_ON_SUCCESS`.
- **Open Questions**: Should we throttle grid size when `days` large? (track in notes).

### `APPLY_WINNERS_ON_DEPLOY`
- **File / Symbol**: `deploy.sh:140`
- **Type**: env
- **Declared in**: Default exported in `load_env_for_compose`; branch at `deploy.sh:592`
- **Purpose**: determines if freshly derived winners should be promoted into `config/live_params.yaml` and mirrored into `.env.hotband`.
- **Default / Range**: Defaults to `"1"`; `0` leaves existing env untouched.
- **Read by**: `deploy.sh:592` (promotion block).
- **Mutated by**: `.env.hotband`, manual override.
- **External Effects**: When `1`, runs `scripts/ops/promote_winners.py` and `scripts/ops/apply_live_params.py`, updates `.env.hotband`, and restarts backend/websocket; flow captured in `docs/diagrams/11_data_flows.mmd` (Deploy warmup lane).
- **Related Concepts**: `RUN_TRIALS_ON_DEPLOY`, `opt:best_config:{instrument}`.
- **Open Questions**: Should backend reload happen via compose rolling update instead of `up -d`? (note tracked).

### `CANDLE_COUNT`
- **File / Symbol**: `deploy.sh:532`
- **Type**: env
- **Declared in**: Warmup helper defaults (`deploy.sh:530`) with overrides accepted by `scripts/ops/warmup_orchestrator.py`.
- **Purpose**: caps the number of candles fetched per instrument during warmup/backfill to avoid API throttling.
- **Default / Range**: Default `1500`; integer >0; typically 1-2k.
- **Read by**: `deploy.sh:530` backfill helper; `scripts/ops/warmup_orchestrator.py --steps backfill`.
- **Mutated by**: `.env.hotband`, CLI `--count` flag when running warmup orchestrator manually.
- **External Effects**: Drives the payload for `/api/candles/fetch`; influences Valkey `md:candles:{instrument}:M1` density.
- **Related Concepts**: `CANDLE_GRAN` (granularity), `HOTBAND_PAIRS`.
- **Open Questions**: None.

- **File / Symbol**: `deploy.sh:131`
- **Type**: env
- **Declared in**: Default export in `deploy.sh`; reused across orchestrator and analytics scripts (`scripts/ops/warmup_orchestrator.py`, `scripts/ops/prime_qfh_history.py:201`, `scripts/run_backtests.py`, `scripts/report_trials.py:209`).
- **Purpose**: canonical list of instruments for hotband operations (warmup, trials, manifolds).
- **Default / Range**: Defaults to `EUR_USD,USD_JPY,GBP_USD,EUR_JPY,USD_CAD,NZD_USD,AUD_USD,USD_CHF`; expects comma-separated uppercase instruments.
- **Read by**: Warmup orchestrator, backtest scripts, allocator promotion flow.
- **Mutated by**: `.env.hotband`, CLI `--pairs`.
- **External Effects**: Drives all loops that touch Valkey and OANDA; ensures consistent coverage; diagrammed in `docs/diagrams/11_data_flows.mmd`.
- **Related Concepts**: `.env.hotband` instrumentation, `opt:best_config:{instrument}` naming scheme.
- **Open Questions**: Duplicate defaults across scripts risk drift—collapse candidate logged in notes.

### `opt:best_config:{instrument}`
- **File / Symbol**: `scripts/report_trials.py:173`
- **Type**: Valkey key
- **Declared in**: Written in `_write_best_keys` (`scripts/report_trials.py:162-191`); read in `scripts/ops/promote_winners.py:26` and applied via `deploy.sh:592`.
- **Purpose**: Stores per-instrument best backtest params + summaries for subsequent promotion into live env.
- **Default / Range**: JSON payload with `params`, `env`, `summary`, `ts`; TTL defaults to 14 days.
- **Read by**: `scripts/ops/promote_winners.py`, `deploy.sh` apply step, potential UI gating.
- **Mutated by**: `scripts/report_trials.py` (latest winners).
- **External Effects**: Source of truth for `config/live_params.yaml`, drives `.env.hotband` updates; referenced in `docs/diagrams/12_services_variables.mmd` mapping.
- **Related Concepts**: `opt:live_config:changed:{instrument}` (pub/sub), `RUN_TRIALS_ON_DEPLOY`.
- **Open Questions**: Need to confirm TTL alignment with deploy cadence.

### `ws:last:manifold:{instrument}`
- **File / Symbol**: `scripts/ops/warmup_orchestrator.py`
- **Type**: Valkey key
- **Declared in**: Hydrate step within the orchestrator; also maintained by `ws-hydrator` service (noted in `docs/diagrams/10_architecture_overview.mmd`).
- **Purpose**: Cache of latest manifold snapshot per instrument, used by websocket broadcasts and UI readiness checks.
- **Default / Range**: JSON payload with metrics, coeffs, diagnostics; TTL default 300s (warmup helper ensures >=60s).
- **Read by**: Websocket server, frontend dashboards, monitoring scripts.
- **Mutated by**: `scripts/ops/warmup_orchestrator.py --steps hydrate`, `ws-hydrator`, backend `/api/coherence/status` pipeline.
- **External Effects**: Publishes to `ws:manifold` channel when hydrated; ensures UI charts show post-deploy metrics quickly; see `docs/diagrams/11_data_flows.mmd`.
- **Related Concepts**: `sep:signal_index:{instrument}`, `HOTBAND_PAIRS`.
- **Open Questions**: Align TTL with websocket hydrator interval? (tracked in notes).


### `opt:backtests:report:latest`
- **File / Symbol**: `scripts/report_trials.py:324`
- **Type**: Valkey key
- **Declared in**: `_write_best_keys` publishes report metadata inside `scripts/report_trials.py` before returning (`scripts/report_trials.py:315-324`).
- **Purpose**: stores the latest backtest report (Markdown filename, rendered content, metadata) for UI consumption and post-deploy audits.
- **Default / Range**: JSON blob with `filename`, `content`, `ts`, `days`, `instruments`; TTL defaults to 14 days.
- **Read by**: Frontend analytics panels (`opt:backtests`) and Ops reviewing deploy health.
- **Mutated by**: `scripts/report_trials.py` after each trials run.
- **External Effects**: Ensures rebuilds still surface the most recent trials summary; referenced in `docs/diagrams/11_data_flows.mmd`.
- **Related Concepts**: `output/reports/backtests/last_{days}d_report.md`, `opt:best_config:{instrument}`.
- **Open Questions**: Should TTL mirror deploy cadence or be persisted indefinitely for history?

### `output/reports/backtests/last_{days}d_report.md`
- **File / Symbol**: `scripts/report_trials.py:309`
- **Type**: file (Markdown)
- **Declared in**: `scripts/report_trials.py` writes the report to disk just before caching it in Valkey (`scripts/report_trials.py:309-313`).
- **Purpose**: human-readable summary of the latest trials window, aligning promoted parameters with underlying stats.
- **Default / Range**: Filename defaults to `output/reports/backtests/last_14d_report.md`; `--days` flag swaps the suffix.
- **Read by**: Operators during deploy QA, research notebooks referencing archived reports, documentation logs.
- **Mutated by**: `scripts/report_trials.py`; manual edits discouraged.
- **External Effects**: Serves as canonical report for portfolio postmortems; path linked from `docs/PORTFOLIO_Optimization_Log.md`.
- **Related Concepts**: `opt:backtests:report:latest`, `config/live_params.yaml`.
- **Open Questions**: Consider versioning/archiving multiple windows for historical comparison.

### `docs/reports/snapshots/{stamp}_*.json`
- **File / Symbol**: `scripts/ops/capture_live_snapshot.py:34`
- **Type**: file set
- **Declared in**: `capture` helper materialises per-metric JSON into `docs/reports/snapshots` (`scripts/ops/capture_live_snapshot.py:34-49`).
- **Purpose**: capture live trading state (OANDA account, open trades, allocator status, gates blob) for documentation and incident review.
- **Default / Range**: Filenames formatted as `<UTC stamp>_account.json`, `_open_trades.json`, `_allocation.json`, `_gates.json`.
- **Read by**: Postmortem authors, allocator QA, documentation builds embedding real snapshots.
- **Mutated by**: `scripts/ops/capture_live_snapshot.py`; optionally cron/systemd tasks.
- **External Effects**: Provides immutable evidence of live conditions; ensures docs remain grounded in actual trading states.
- **Related Concepts**: `opt:rolling:gates_blob`, `/api/oanda/*` endpoints.
- **Open Questions**: Should we compress/rotate historical snapshots to control repo size?

### `output/manifolds/{instrument}/{date}.json`
- **File / Symbol**: `scripts/ops/prime_qfh_history.py:321`
- **Type**: file set
- **Declared in**: `prime_qfh_history.py` writes manifold JSON after running the generator (`scripts/ops/prime_qfh_history.py:321-335`).
- **Purpose**: store per-instrument/day manifold outputs for offline validation and debugging mismatches with live hydrator payloads.
- **Default / Range**: Directory root defaults to `output/manifolds`; structure `<instrument>/<YYYY-MM-DD>.json`.
- **Read by**: Research notebooks diffing manifolds, QA comparing to ws:last payloads, operators verifying signal continuity.
- **Mutated by**: `scripts/ops/prime_qfh_history.py` during deploy warmup or manual runs.
- **External Effects**: Enables reproducibility for trials and hydrator debugging; ensures warmup outputs persist beyond container lifecycle.
- **Related Concepts**: `ws:last:manifold:{instrument}`, `sep:signal_index:{instrument}`.
- **Open Questions**: Retention/rotation policy for historical manifolds still TBD.

### `config/live_params.yaml`
- **File / Symbol**: `scripts/ops/promote_winners.py:88`
- **Type**: file
- **Declared in**: `scripts/ops/promote_winners.py` writes promoted winners to YAML before deployment apply step (`scripts/ops/promote_winners.py:88-94`).
- **Purpose**: intermediate snapshot of per-instrument winners destined for `.env.hotband` via `apply_live_params.py`.
- **Default / Range**: Sorted YAML keyed by instrument; includes params and summary metrics.
- **Read by**: `deploy.sh` apply step, `scripts/ops/apply_live_params.py`, operators reviewing winners before promotion.
- **Mutated by**: `scripts/ops/promote_winners.py`; manual edits by ops prior to apply (rare).
- **External Effects**: Drives environment updates for allocator thresholds; ensures repeatable promotions.
- **Related Concepts**: `opt:best_config:{instrument}`, `.env.hotband`.
- **Open Questions**: Should we commit snapshots for audit trail or keep local-only?
### `manifold.metrics (coherence, stability, entropy, rupture)`
- **File / Symbol**: `src/app/manifold_generator_main.cpp:374`
- **Type**: metric
- **Declared in**: `manifold_generator` emits these per-snapshot fields when building manifolds/signals.
- **Purpose**: quantifies directional agreement (C), persistence (S), randomness (H), and structural breaks (R) for each instrument window.
- **Default / Range**: Float `0.0–1.0`. Live thresholds sourced from `.env.hotband` (`AUTO_MIN_COHERENCE=0.55`, `GUARD_MIN_STABILITY=0.40`, `AUTO_MAX_ENTROPY=0.70`, `AUTO_MAX_RUPTURE=0.60`).
- **Read by**: `scripts/ops/prime_qfh_history.py`, `scripts/ws_hydrator.py`, backend `/api/coherence/status`, `allocator_lite`, rolling evaluator.
- **Mutated by**: `bin/manifold_generator` (native) and `scripts/ws_hydrator.py` (REST mirror).
- **External Effects**: Drives gating, allocator scoring, UI diagnostics; forms the basis for pattern repetition detection via `ws:manifold:hist:{instrument}`.
- **Related Concepts**: `HOTBAND_PROCESSING_INTERVAL`, `STRANDS_C_MIN`, `AUTO_*` guard envs.
- **Open Questions**: Confirm whether entropy upper bound should adapt intraday vs static `AUTO_MAX_ENTROPY`.

### `manifold.coeffs (sigma_eff, lambda)`
- **File / Symbol**: `src/app/manifold_generator_main.cpp:444`
- **Type**: metric
- **Declared in**: `manifold_generator` post-processing step computing volatility proxy and hazard probability.
- **Purpose**: `sigma_eff` captures effective log-return volatility; `lambda` estimates 1-minute rupture hazard, feeding repetition/band risk checks.
- **Default / Range**: `sigma_eff` ≥ 0 (float); `lambda` clamped `0–1`. No direct env overrides; downstream heuristics clamp hazard contributions.
- **Read by**: `scripts/ws_hydrator.py`, `ws:last:manifold` consumers, risk dashboards, cooling logic in wake of ruptures.
- **Mutated by**: `bin/manifold_generator`; preserved by hydrator when republishing manifolds.
- **External Effects**: Higher `lambda` increases likelihood of repetition gate blocking allocations; `sigma_eff` informs risk overlays (planned).
- **Related Concepts**: `opt:rolling:cooldown:{instrument}`, pattern repetition guard, `AUTO_MAX_HAZARD`.
- **Open Questions**: Should `lambda` feed allocator penalties directly or remain observational?

### `bt:rolling:summary:{instrument}`
- **File / Symbol**: `scripts/rolling_backtest_evaluator.py:500`
- **Type**: Valkey key (JSON)
- **Declared in**: Rolling evaluator writes summary after each strict 30d backtest pass.
- **Purpose**: Persist per-instrument stats (`trades`, `win_rate`, `pnl_total`, `max_drawdown`, `calmar`) alongside adaptive floors (`min_trades`, `calmar_min`, `pnl_min`).
- **Default / Range**: Updated every `EVAL_INTERVAL_SEC` (default 900s); floors derived from percentiles with env floors (`EVAL_MIN_TRADES`≈8, `EVAL_CALMAR_MIN`≈0.10, `EVAL_PNL_MIN`≈0.0). TTL 1h.
- **Read by**: Ops dashboards, historical auditing, potential Prometheus exporters.
- **Mutated by**: `scripts/rolling_backtest_evaluator.py`.
- **External Effects**: Supplies inputs to hysteresis + cooldown gating and informs manual tuning.
- **Related Concepts**: `bt:rolling:hist:{instrument}:{param_key}`, `opt:rolling:eligible:{instrument}`.
- **Open Questions**: Add Prometheus exporter for calmar/PNL trends?

### `opt:rolling:eligible:{instrument}`
- **File / Symbol**: `scripts/rolling_backtest_evaluator.py:505`
- **Type**: Valkey key (string 0/1)
- **Declared in**: Rolling evaluator after hysteresis evaluation.
- **Purpose**: Binary eligibility flag consumed by allocator-lite and safety tooling. Requires two consecutive passes/fails and respects cooldown + pattern repetition guard.
- **Default / Range**: `"0"` or `"1"`; consecutive passes/fails threshold `REQ_CONSEC=2`; cooldown arms after 4 misses (`SESSION_PATTERN_MIN_OCCURRENCES` default 2).
- **Read by**: `scripts/allocator_lite.py`, backend metrics exporter, ops monitors.
- **Mutated by**: `scripts/rolling_backtest_evaluator.py`; session gating forces `0` near market close.
- **External Effects**: Directly gates allocations; also mirrored into `opt:rolling:gates_blob`.
- **Related Concepts**: `HYSTERESIS_DEFAULT`, `HYSTERESIS_PAIR_JSON`, `SESSION_TRADING_ENABLED`, `SESSION_EXIT_MINUTES`.
- **Open Questions**: Should pattern repetition threshold increase during high-vol regimes?

### `opt:rolling:gates_blob`
- **File / Symbol**: `scripts/rolling_backtest_evaluator.py:511`
- **Type**: Valkey key (JSON)
- **Declared in**: Rolling evaluator consolidates eligibility + cooldown + buffers each cycle.
- **Purpose**: Single source-of-truth for allocator decisions `{ ts, gates, cooldowns, bufs }`.
- **Default / Range**: Millisecond timestamp; TTL implied by refresh cadence (≤ evaluation interval). Buffers clamped [0.00, 0.30].
- **Read by**: `scripts/allocator_lite.py`, frontend status tiles, snapshot tooling, kill-switch checks.
- **Mutated by**: `scripts/rolling_backtest_evaluator.py`; overwritten atomically per evaluation loop.
- **External Effects**: Missing/stale blob triggers allocator hold; used to audit hysteresis and pattern gating.
- **Related Concepts**: `opt:rolling:eligible:{instrument}`, `opt:rolling:cooldown:{instrument}`, `risk:allocation_weights`.
- **Open Questions**: Add schema validation + max-age alarm in allocator?

### `risk:allocation_weights`
- **File / Symbol**: `scripts/allocator_lite.py:115`
- **Type**: Valkey key (JSON)
- **Declared in**: Allocator-lite publishes weights after scoring eligible instruments.
- **Purpose**: Expose Top-K allocation percentages computed from metric-weighted score and risk caps.
- **Default / Range**: Weights sum ≤ `RISK_ALLOC_TARGET_PCT` (default 0.30); per-position cap `PM_MAX_PER_POS_PCT` (0.10); refresh every `ALLOC_INTERVAL_SEC` (600s).
- **Read by**: Trading service (order sizing), frontend dashboards, snapshot tooling, ops runbook.
- **Mutated by**: `scripts/allocator_lite.py`; holds previous weights if gates blob stale or fewer than Top-K eligible.
- **External Effects**: Direct trading exposure; downstream risk monitors compare to account margin + live positions.
- **Related Concepts**: `ALLOC_TOP_K`, `ALLOC_W_C`, `ALLOC_W_STAB`, `ALLOC_W_RUP`, `ALLOC_W_ENT`, `opt:rolling:gates_blob`, `opt:allocator:status`.
- **Open Questions**: Evaluate dynamic weighting vs static `ALLOC_W_*` for different regimes.




## 4. Audit Checklist
- [x] Catalogue environment variables consumed by `deploy.sh` and `scripts/shared_utils/config_loader`.
- [x] Document CLI arguments for `scripts/ops/warmup_orchestrator.py` (backfill/prime/hydrate) and `scripts/ops/prime_qfh_history.py`.
- [x] Track Valkey key namespaces read/written by the scripts.
- [x] Record generated artefacts (reports, snapshots) and who uses them.

## 5. Notes & TODOs
Keep scratch notes in `notes/scripts/` as the audit proceeds. Current warmup findings logged in `notes/scripts/deploy_warmup.md`.
