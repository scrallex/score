# UI Data & Diagnostics Worksheet

## Deploy Warmup Pipeline
- `deploy.sh` now preserves `valkey`, prunes other services, and rebuilds containers cleanly.
- Post-deploy warmup sequence:
  1. `scripts/ops/warmup_orchestrator.py --steps backfill` pulls fresh candles for all eight portfolio pairs via `/api/candles/fetch`.
  2. `scripts/ops/warmup_orchestrator.py --steps prime --store-manifold-to-valkey` populates strict-span manifolds and signal indices.
  3. `scripts/ops/warmup_orchestrator.py --steps hydrate --synthesize-signals` seeds `ws:last:manifold:*`, publishes to `ws:manifold`, and inserts minimal `sep:signal:*` snapshots.
- WebSocket service retries Valkey connection until ready (`WS_VALKEY_RETRY_*` knobs). Verified via `docker logs sep-websocket`.
- Diagnostics panel requirements: `/api/coherence/status` returns non-zero data for EUR_USD, USD_JPY, GBP_USD, EUR_JPY, USD_CAD, NZD_USD, AUD_USD, USD_CHF immediately after deploy.

## Diagnostics Panel Checklist
- [x] `ws:last:manifold:*` populated with current timestamps (TTL 300 s).
- [x] `/status` endpoint on `sep-ws-hydrator` reporting `count=8` events each cycle.
- [x] WebSocket pub/sub stream active (manifold updates forwarded every 10 s).
- [ ] Add automated post-deploy validator that fails if any instrument lacks coherence/stability/entropy data.

## Ranking Panels Alignment
- Active ranking API (`/api/ranking/active`) provides canonical ordering, session metadata, and metrics (score, slope, lambda).
- Progress:
  - [x] Extracted `useActiveRanking` hook (shared poller, 5 s cadence).
  - [x] `RankedInstruments` and `MultiInstrumentRankedBoard` now consume the hook so ordering, sessions, and coherence badges match.
  - [ ] Surface allocation weights/net units in the ranked table (low priority) or remove if redundant.

## Chart Component Audit
- Candles: `/api/candles` fallback works; websocket stream replenishes market data arrays.
- Strands & torches overlays: aggregator feed removed; update UI to hide or repurpose overlays until a new data source is defined.
- Slices: legacy slice endpoints/UI have been fully removed from both frontend and backend.

## Action Items
1. Add automated `verify_post_deploy` script to hit `/api/coherence/status`, `/api/ranking/active`, `/api/oanda/pricing`, `/api/risk/allocation-status` for all portfolio pairs.
2. Chart overlays follow-up:
   - [x] Disable or clearly label Slice/Similarity controls until APIs exist.
   - [ ] Enumerate required backend routes & schemas for slice features in a separate ADR.
3. Extend documentation with real-time checklist for operators (include expected metric ranges, gating states, kill-switch position).
