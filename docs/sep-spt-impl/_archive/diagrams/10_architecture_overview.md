%% Mermaid: High-level architecture overview
flowchart LR
  %% External
  OANDA["OANDA REST / Pricing / Orders"]

  %% Data store
  VALKEY(("Valkey\nredis://sep-valkey:6379/0"))
  ENV[".env.hotband\nlive config snapshot"]

  %% Backend services
  subgraph Backend
    API["sep-backend\nREST API (/api/**)"]
    WS["sep-websocket\nWS server (:8001)"]
    HYD["ws-hydrator\ncoherence→ws:last:manifold + ws:manifold"]
    EVAL["rolling-evaluator\npublishes opt:rolling:gates_blob"]
    ALOC["allocator-lite\nweights → risk:allocation_weights"]
    CND["candle-fetcher\nseed M1/M5"]
    MON["trade-monitor"]
    STDY["study-collector"]
  end

  %% Frontend
  FRONT["Frontend (Nginx)\nReact/Vite"]

  %% Ops
  subgraph Ops
    DEPLOY["deploy.sh\noperator entry"]
    WARM["Warmup helpers\n(backfill / prime / hydrate)"]
    TRIALS["Trials pipeline\nops/prime_qfh_history → run_backtests → report_trials"]
    APPLY["Promotion step\npromote_winners + apply_live_params"]
  end

  %% Edges
  OANDA --> API
  API <--> VALKEY
  CND --> API

  HYD --> VALKEY
  VALKEY -.pub/sub.-> WS
  WS <--> FRONT
  FRONT <--> API

  EVAL <--> VALKEY
  ALOC <--> VALKEY

  DEPLOY --> WARM
  WARM --> API
  WARM --> VALKEY
  DEPLOY --> TRIALS
  TRIALS --> VALKEY
  TRIALS --> APPLY
  VALKEY --> APPLY
  APPLY --> ENV
  ENV --> API
  ENV --> WS
  ENV --> ALOC
  ENV --> FRONT

  %% Key channels/keys (labels)
  classDef key fill:#0b7285,stroke:#0b7285,color:#fff,opacity:0.9;
  VALKEY:::key
  ENV:::key
