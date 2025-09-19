%% Mermaid: Core data flows and keyspaces
flowchart TD
  subgraph DeployWarmup [Deploy Warmup]
    Deploy["deploy.sh"]
    Backfill["backfill_candles_post_deploy
(+ scripts/ops/backfill_live_candles.py)"]
    Prime["prime_signals_post_deploy
scripts/ops/prime_qfh_history.py"]
    HydrateWarm["hydrate_manifolds_post_deploy
scripts/ops/hydrate_manifolds_once.py"]
    Trials["derive_winners_and_apply
run_backtests.py + report_trials.py"]
    Promote["promote_winners.py"]
    ApplyEnv["apply_live_params.py"]
  end

  subgraph LiveLoop [Live Evaluation Loop]
    CandleSvc["candle-fetcher
OANDA M1 stream"]
    ManifoldBin["manifold_generator
(bin)"]
    Hydrator["ws-hydrator
REST mirror"]
    RollingEval["rolling-evaluator
adaptive floors"]
    AllocLite["allocator-lite
Top-K weights"]
    TradeSvc["trading_service
orders / guards"]
  end

  subgraph BackendAPI["sep-backend REST"]
    Candles[/api/candles/fetch/]
    Coherence[/api/coherence/status/]
    Kill[/api/kill-switch/]
  end

  subgraph Store[Valkey]
    MDCAND["md:candles:{instrument}:M1"]
    SIGIDX["sep:signal_index:{instrument}"]
    SIG["sep:signal:{instrument}:{ts_ns}"]
    LASTMF["ws:last:manifold:{instrument}"]
    MANCHAN[["ws:manifold (channel)"]]
    MANHIST["ws:manifold:hist:{instrument}"]
    SUMMARY["bt:rolling:summary:{instrument}"]
    ELIG["opt:rolling:eligible:{instrument}"]
    GATES["opt:rolling:gates_blob"]
    WEIGHTS["risk:allocation_weights"]
    TRIALIDX["bt:trial:index"]
    TRIALRUN["bt:trial:run:{run_id}"]
    TRADES["bt:trial:trades:{run_id}"]
    BEST["opt:best_config:{instrument}"]
    BESTMETA["opt:best_config_meta:{instrument}"]
    GLOBAL["opt:best_config:GLOBAL"]
    REPORT["opt:backtests:report:latest"]
  end

  subgraph Files
    LIVEYAML["config/live_params.yaml"]
    ENVFILE[".env.hotband"]
  end

  OANDA["OANDA REST"]
  WSsvc["sep-websocket"]

  %% Warmup orchestration
  Deploy --> Backfill
  Deploy --> Prime
  Deploy --> HydrateWarm
  Deploy --> Trials
  Trials --> Promote
  Promote --> ApplyEnv

  %% Warmup API interactions
  Backfill --> Candles
  Candles --> MDCAND

  Prime --> SIGIDX
  Prime --> SIG
  Prime --> MDCAND
  OANDA --> Prime

  HydrateWarm --> Coherence
  HydrateWarm --> LASTMF
  HydrateWarm -.publish.-> MANCHAN
  HydrateWarm -.optional synth.-> SIGIDX

  Trials --> TRIALRUN
  Trials --> TRIALIDX
  Trials --> TRADES
  Trials --> BEST
  Trials --> BESTMETA
  Trials --> GLOBAL
  Trials --> REPORT

  BEST --> Promote
  Promote --> LIVEYAML
  ApplyEnv --> ENVFILE
  LIVEYAML --> ApplyEnv
  ENVFILE --> Deploy

  %% Live loop
  OANDA --> CandleSvc
  CandleSvc --> MDCAND
  MDCAND --> ManifoldBin
  ManifoldBin --> SIG
  ManifoldBin --> SIGIDX
  ManifoldBin --> LASTMF
  Hydrator --> Coherence
  Hydrator --> LASTMF
  Hydrator -.publish.-> MANCHAN
  LASTMF --> MANHIST
  MANHIST --> RollingEval
  RollingEval --> SUMMARY
  SUMMARY --> RollingEval
  RollingEval --> ELIG
  RollingEval --> GATES
  ELIG --> GATES
  GATES --> AllocLite
  AllocLite --> Coherence
  AllocLite --> WEIGHTS
  WEIGHTS --> TradeSvc
  TradeSvc --> Kill
  TradeSvc --> OANDA

  MANCHAN --> WSsvc
  LASTMF --> WSsvc

  %% Kill switch + snapshots
  Deploy --> Kill

  classDef key fill:#5c7cfa,stroke:#364fc7,color:#fff;
  classDef svc fill:#adb5bd,stroke:#495057,color:#212529;
  class MDCAND,SIGIDX,SIG,LASTMF,MANCHAN,MANHIST,SUMMARY,ELIG,GATES,WEIGHTS,TRIALIDX,TRIALRUN,TRADES,BEST,BESTMETA,GLOBAL,REPORT key;
  class WSsvc,CandleSvc,ManifoldBin,Hydrator,RollingEval,AllocLite,TradeSvc svc;
