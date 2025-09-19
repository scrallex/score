# 08. System Audit and Completion Roadmap

_Archived 2025-09-16: Final motifs-to-quantum pivot is complete. This document is preserved for historical traceability only._


**Date**: September 12, 2025  
**Status**: Realignment completed — Motifs/GPU removed; quantum metrics live (updated 2025‑09‑13)  
**Purpose**: Comprehensive audit of all system components and definitive roadmap to completion (kept for traceability)

## Executive Summary

The SEP trading system has deviated from its original quantum pattern-based design into a complex, failing motifs approach that relies on historical pattern matching instead of real-time quantum analysis. This audit identifies all critical flaws and provides a clear path to system completion.

## Part 1: Critical System Flaws

### 1.1 Core Trading Logic Issues

#### CRITICAL: Broken Motifs Approach

- **Current State**: System searches historical motifs database for patterns
- **Problem**: Backward-looking analysis, statistical overfitting, no real-time quantum metrics
- **Impact**: Consistent trading losses
- **Fix Applied**: Updated `margin_aware_motif.py` to use real-time quantum analysis
- **Remaining Work**: Remove all motifs infrastructure

#### CRITICAL: Misaligned Strategy Execution

- **Current State**: Complex scoring system based on backtests + roots + motif quality
- **Problem**: Over-engineered, not using actual market quantum state
- **Impact**: Poor entry/exit decisions
- **Required**: Simplify to pure quantum metrics (coherence, stability, rupture, entropy)

### 1.2 Data Flow and Storage Issues

#### Unnecessary Complexity

- **Current State**:
- **Problem**: These services add no value with corrected quantum approach
- **Action**: Disable/remove these services

#### Valkey Storage Bloat

- **Current Keys**:
  - `strand:motifs:*` - Historical motif patterns (DELETE)
  - `opt:*` - Optimizer results (KEEP for portfolio weights only)
  - `sep:signal:*` - Quantum signals (KEEP - this is correct)
  - `manifold:*` - Daily manifolds (KEEP - quantum metrics)

### 1.3 Configuration Issues

#### Environment Variables Misalignment

```bash
# INCORRECT (motifs-based)
MOTIF_GATE_ENABLED=1
MOTIF_SAMPLES_MIN=24
MOTIF_MIN_TREND_BPS=6.0
MOTIF_P_MIN=0.55

# CORRECT (quantum-based)
GUARD_MIN_COHERENCE=0.65
GUARD_MAX_RUPTURE=0.6
GUARD_MIN_STABILITY=0.4
AUTO_MAX_ENTROPY=0.7
```

#### Docker Services Configuration

- **Unnecessary Services Running**:
  - sep-motifs-indexer
  - sep-motif-analyzer
  - sep-optimizer-batch (questionable value)

### 1.4 Documentation Inconsistencies

All documentation still describes motifs-centered approach:

- `01_System_Concepts.md` - Needs complete rewrite
- `docs/_archive/05_Motif_Intelligence.md` - archived for history
- `docs/_archive/07_Alpha_Generation.md` - archived for history

## Part 2: System Architecture (Corrected)

### 2.1 Correct Data Flow

```
Market Data (OANDA)
    ↓
Quantum Analysis (C++ QFH)
    ↓
Real-time Metrics (coherence, stability, rupture, entropy)
    ↓
Trading Decision (quantum thresholds)
    ↓
Position Management (margin-aware)
    ↓
Order Execution (OANDA)
```

### 2.2 Required Services Only (current compose)

```yaml
# Essential Services
sep-backend          # Trading logic + API + quantum metrics (via libquantum_metrics.so)
sep-websocket        # Real-time updates
sep-allocator-lite   # Publishes Top‑K portfolio weights
sep-ws-hydrator      # Mirrors minimal manifold payloads for the UI
sep-candle-fetcher   # REST-based market data priming
sep-trade-monitor    # Position/open-trade snapshots for ops
sep-frontend         # UI (nginx)

# Removed
# sep-motifs-indexer  # Historical patterns — removed
# sep-motif-analyzer  # Motif alpha — removed
# sep-optimizer-batch # Obsolete with allocator-lite — removed
```

### 2.3 Simplified Trading Logic

```python
# Correct approach (already implemented in updated margin_aware_motif.py)
1. Get real-time quantum metrics from PathMetrics
2. Check coherence > 0.65, rupture < 0.6, stability > 0.4
3. Determine direction from price momentum + quantum state
4. Apply margin management and risk controls
5. Execute trade
```

## Part 3: Completion Roadmap

### Phase 1: Remove Broken Components (Day 1)

- [ ] Stop motifs-indexer service
- [ ] Stop motif-analyzer service
- [ ] Clean Valkey: DELETE all `strand:motifs:*` keys
- [x] Update docker-compose.hotband.yml to remove motif services (no services remaining)
- [x] Update deploy.sh to skip motif services (delegates to warmup orchestrator)

### Phase 2: Simplify Configuration (Day 1-2)

- [x] Update .env.hotband to remove MOTIF_* variables
- [ ] Set correct quantum thresholds
- [ ] Simplify portfolio allocation (consider equal weights)
- [ ] Review if optimizer-batch is needed at all

### Phase 3: Fix Quantum Pipeline (Day 2-3)

- [ ] Verify C++ quantum analysis is working correctly
- [ ] Ensure PathMetrics contain all quantum fields
- [ ] Validate signal generation frequency
- [ ] Check manifold generation and storage

### Phase 4: Update Documentation (Day 3-4)

- [ ] Rewrite 01_System_Concepts.md for quantum approach
- [x] Archive 05_Motif_Intelligence.md (moved to docs/_archive)
- [ ] Update 02_Operations_Runbook.md
- [ ] Create new 09_Quantum_Trading_Guide.md

### Phase 5: Testing and Validation (Day 4-5)

- [ ] Run system with single instrument (EUR_USD)
- [ ] Verify quantum metrics are updating
- [ ] Check trading decisions match quantum state
- [ ] Monitor performance for 24 hours
- [ ] Gradually add more instruments

### Phase 6: Performance Optimization (Day 5-7)

- [ ] Tune quantum thresholds based on results
- [ ] Optimize position sizing
- [ ] Implement proper stop-loss/take-profit
- [ ] Add trade journaling for analysis

## Part 4: Critical Code Changes Required

### 4.1 Portfolio Manager Simplification

```python
# scripts/trading/portfolio_manager.py
# REMOVE: Complex scoring system
# ADD: Simple equal weights or market-cap weights
def _get_target_weights(self):
    instruments = self._get_active_instruments()
    # Simple equal weight
    weight = 1.0 / len(instruments)
    return {inst: weight for inst in instruments}
```

### 4.2 Remove Motif Dependencies

```python
# scripts/trading_service.py
# REMOVE: All motif-related imports and logic
# REMOVE: from scripts.shared_utils.motif_scoring import ...
```

### 4.3 Simplified Strategy Configuration

```python
# scripts/trading/engine.py
# Use only quantum-based strategies
strategies = [
    MarginAwareMotifStrategy(  # Now uses quantum, not motifs
        gate=coherence_gate,
        trading_service=self,
        coherence_threshold=0.65,
        max_rupture=0.6,
        min_stability=0.4
    )
]
```

## Part 5: Deployment Configuration

### 5.1 Updated docker-compose.hotband.yml

```yaml
services:
  # KEEP THESE
  backend:
    # ... existing config

  websocket:
    # ... existing config

  frontend:
    # ... existing config

  hotband-processor:
    # ... existing config

  diagnostics-hydrator:
    # ... existing config

  # REMOVE THESE
  # motifs-indexer: DELETE
  # motif-analyzer: DELETE
```

### 5.2 Environment Variables (Corrected)

```bash
# Quantum Thresholds
GUARD_MIN_COHERENCE=0.65
GUARD_MAX_RUPTURE=0.6
GUARD_MIN_STABILITY=0.4
AUTO_MAX_ENTROPY=0.7

# Risk Management
RISK_ALLOC_TARGET_PCT=0.30   # Portfolio budget = marginAvailable × pct
MARGIN_OVERLOAD=0.80
KILL_SWITCH=0  # Set to 1 initially

# Position Sizing
AUTO_UNITS=8000
# Exposure sizing uses a fixed internal inverse‑leverage proxy (~0.02) inside the codebase;
# no RISK_EXPOSURE_SCALE environment variable is used anymore.

# Remove ALL MOTIF_* variables
```

## Part 6: Validation Checklist

### Pre-Launch

- [ ] All motif services stopped
- [ ] Valkey cleaned of motif keys
- [ ] Configuration updated
- [ ] Documentation updated
- [ ] Code dependencies removed

### Post-Launch Monitoring

- [ ] Quantum metrics updating (check /api/coherence/status)
- [ ] Trading decisions based on quantum state
- [ ] No references to motifs in logs
- [ ] Performance improving vs baseline
- [ ] Margin utilization stable

## Part 7: Success Metrics

### Week 1 Goals

- Zero dependency on historical motifs
- All trades based on real-time quantum metrics
- Positive P&L trajectory
- System stability (no crashes)

### Month 1 Goals

- Consistent profitability
- Sharpe ratio > 1.0
- Maximum drawdown < 10%
- All 8 instruments trading successfully

## Conclusion

The system's core issue is clear: it abandoned its original quantum pattern analysis for a complex historical pattern matching system that doesn't work. The fix is straightforward:

1. **Remove** all motifs infrastructure
2. **Simplify** to real-time quantum analysis
3. **Focus** on coherence, stability, rupture metrics
4. **Execute** based on current market quantum state

This is not a refactor - it's a return to the original, correct design. The quantum analysis infrastructure is already built and working. We just need to remove the layers of complexity that were added on top.

**Estimated Time to Completion**: 5-7 days
**Confidence Level**: High - the fix is already partially implemented
**Risk Level**: Low - we're simplifying, not adding complexity
## Finalization (Sep 2025)

- Internal Valkey: Removed external/managed Valkey; added `valkey` service to compose and standardized on `VALKEY_URL` everywhere.
- One‑button deploy: `deploy.sh` now brings up services, primes history (configurable), runs trials, writes winners to Valkey, applies env knobs into `.env.hotband`, and redeploys backend+websocket.
- Docs trimmed: Consolidated ops steps into 02_Operations_Runbook; archived motif/optimizer notes.
- Readiness scripts: Added `scripts/tools/check_readiness.py` (backend health, per‑pair coherence, ws keys, winner keys).
- Safety defaults: Kill‑switch ON by default; opt‑in `DISENGAGE_KILL_ON_SUCCESS=1` to auto‑disable after winners apply.

Open items (tracked)
- Expand trials grid and window for stronger winner confidence (current defaults use a compact grid).
- Optional: nightly job to regenerate winners and reflect knobs when performance improves.
