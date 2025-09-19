%% Mermaid: Services ↔ critical variables
classDiagram
  class DeploySh {
    +BACKFILL_ON_DEPLOY : bool
    +RUN_TRIALS_ON_DEPLOY : bool
    +APPLY_WINNERS_ON_DEPLOY : bool
    +DISENGAGE_KILL_ON_SUCCESS : bool
    +SEP_SKIP_PROCESSOR : bool
    orchestrates warmup + trials
  }

  class WarmupHelpers {
    +HOTBAND_PAIRS : list
    +CANDLE_GRAN : str
    +CANDLE_COUNT : int
    calls /api/candles/fetch
    writes md:candles:{instrument}:M1
    writes sep:signal_index:{instrument}
    writes ws:last:manifold:{instrument}
  }

  class TrialsPipeline {
    +HOTBAND_PAIRS : list
    +--days : int
    +--c-min : list[float]
    +--lookback : list[int]
    +--thr-bps : list[float]
    writes bt:trial:run:{run_id}
    writes bt:trial:index
    writes opt:best_config:{instrument}
    writes opt:best_config_meta:{instrument}
  }

  class PromotionStep {
    reads opt:best_config:{instrument}
    writes config/live_params.yaml
    updates .env.hotband (GUARD_MIN_COHERENCE, AUTO_MIN_COHERENCE, AUTO_DIRECTION_LOOKBACK, AUTO_DIR_THRESHOLD_BPS)
  }

  class BackendAPI {
    +API_PORT : int
    +VALKEY_URL : string
    +BACKEND_API_URL : url
    exposes /api/candles/fetch, /api/coherence/status
  }

  class WebsocketSvc {
    +WS_PORT : int
    consumes ws:last:manifold:{instrument}
  }

  class AllocatorLite {
    +AUTO_MIN_COHERENCE : float
    +AUTO_DIRECTION_LOOKBACK : int
    +AUTO_DIR_THRESHOLD_BPS : float
    reads opt:rolling:gates_blob
    writes risk:allocation_weights
  }

  DeploySh --> WarmupHelpers : runs warmup
  DeploySh --> TrialsPipeline : triggers
  DeploySh --> BackendAPI : compose up / kill-switch
  DeploySh --> WebsocketSvc : compose up
  TrialsPipeline --> PromotionStep : winners snapshot
  WarmupHelpers --> BackendAPI : REST calls
  BackendAPI --> WarmupHelpers : returns candles/coherence
  WarmupHelpers --> WebsocketSvc : hydrates ws:last:manifold
  PromotionStep --> AllocatorLite : updates env thresholds
  PromotionStep --> BackendAPI : env refreshed
  PromotionStep --> WebsocketSvc : env refreshed
