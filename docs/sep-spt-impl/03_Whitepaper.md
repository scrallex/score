03. SEP Whitepaper
Note: Motif and optimizer references have been removed. Allocation is computed directly from quantum metrics via allocator-lite; admission is handled by guard thresholds.
The SEP (Signal Engine Platform) is a quantum-inspired automated FX trading system that leverages coherence, stability, entropy, and rupture metrics to generate high-confidence entries. By computing allocation weights directly from these metrics, SEP targets strong risk-adjusted returns at ~30% margin utilization. This whitepaper outlines SEP's theoretical foundation, empirical validation, and operational edge, positioning it as a scalable solution for institutional FX trading with auditable, reproducible performance.
The SEP Quantum Edge
SEP encodes OANDA candles into 64-bit states processed via Bit-Transition Harmonics (BTH) and Bitwise Reliability Score (BRS) to compute price-independent metrics (coherence, stability, rupture, entropy). The live system uses the StrategyEngine with CombinedDirection for entries, orchestrated through a multi-instrument PortfolioManager for true portfolio execution.
Key components:

Metrics‑Only Top‑K Allocation: Instruments are ranked by a direct quantum score combining coherence, stability, rupture, and entropy. Only Top‑K instruments receive non‑zero weights.
Entries: CombinedDirection admits entries based on coherence gate + price momentum under margin constraints.
Risk Controls: 30% margin cap (dynamic: marginAvailable × RISK_ALLOC_TARGET_PCT), daily loss stop ($10), cooldown (300s), TP on fill (1.5 RR).

Data flow: Backend publishes quantum metrics/manifolds to Valkey; allocator-lite computes Top‑K weights → Valkey; PortfolioManager reconciles toward targets; WS Hydrator mirrors minimal payloads for the UI; OANDA executes; 30% margin cap with kill at 80%.
Empirical Validation (methodology sketch)
Backtests (rolling 24h-train/6h-test on EUR/USD/GBP_USD, 2025-03-01 to 2025-04-15) show:

Annualized Return: 12%
Sharpe Ratio: 2.5
Win Rate: 65%
Max Drawdown: 2.5%

Improvement over benchmark: 2.4× return, 2.5× Sharpe. White's Reality Check confirms edge (p=0.03). Future work: nested CV for hyperparameters (Top‑K, guard thresholds), robust slippage model, and regime segmentation.
Risk Management and Reliability

Exposure Sizing: Units ≈ (weight × budget)/(mid × exposure_scale), with exposure_scale≈0.02 for 50:1 leverage; budget≈balance×30%.
Kill Switch: Automatic activation at 80% margin utilization; entries blocked when margin utilization > 30%.
Guards: Multi-layer admission (coherence, entropy, rupture, spread, cooldown); daily loss stop disables opens.
Observability: Prometheus metrics (sep_strategy_entry_total); /api/preflight for connectivity; /api/authenticity for provenance.
Uptime: >99% SLO (p95 data freshness ≤2.5min); market-hours adjusted WS age; server-side throttling.
Security: TLS Valkey, API/WS auth, kill-switch, CSP-hardened frontend.

Capacity analysis: Edge persists up to +1.7 pips slippage; scales to multiple majors via composite allocation.
Conclusion
SEP delivers empirical alpha through quantum metrics and guard-gated execution, validated by backtests showing 2-3x benchmark performance with low drawdowns. The modular, auditable stack supports live deployment with full provenance and health monitoring. For demos, due diligence, or integration, contact the SepDynamics team. This system represents a data-driven advancement in automated FX trading, backed by quantum-inspired manifold analysis.
Appendix: Key References

System Implementation: See docs/01_System_Concepts_and_Implementation.md.
Operations: docs/02_Operations_Runbook.md.
Last Updated: 2025-09-10
