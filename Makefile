.PHONY: scorecard plots lead twins onset all clean clean-demo pack stream sweep permutation report
.PHONY: bench-seen
.PHONY: demo-payload demo-up demo-down
.PHONY: planbench-all planbench-scale codetrace-report
.PHONY: semantic-guardrail-demo final-report

PLANBENCH_COUNT ?= 300
PLANBENCH_TARGETS ?= logistics blocksworld mystery_bw
PLANBENCH_WINDOW_BYTES ?= 256
PLANBENCH_STRIDE ?= 128
PERMUTATION_ITERS ?= 20000
PLANBENCH_ENRICH_BASE ?= output/planbench_by_domain/logistics/gold_state.json
PLANBENCH_EXTRA_TWINS ?=
PLANBENCH_ENRICH_NOTE ?= logistics-twin-enrichment
PLANBENCH_MATCH_SIGNATURE ?= 0
PLANBENCH_VALIDATOR ?= external/VAL/build/bin/Validate

PLANBENCH_SCALE_ROOT ?= data/planbench_scale500
PLANBENCH_SCALE_OUTPUT ?= output/planbench_scale500
PLANBENCH_SCALE_TARGETS ?= logistics blocksworld mystery_bw
PLANBENCH_SCALE_COUNT ?= 500
PLANBENCH_SCALE_WINDOW_BYTES ?= $(PLANBENCH_WINDOW_BYTES)
PLANBENCH_SCALE_STRIDE ?= $(PLANBENCH_STRIDE)
PLANBENCH_SCALE_ITERATIONS ?= $(PERMUTATION_ITERS)
PLANBENCH_SCALE_WINDOW ?= 0.005

comma := ,
empty :=
space := $(empty) $(empty)

scorecard:
	python scripts/make_scorecard.py

plots:
	python -m sep_text_manifold.cli_plots --state analysis/mms_0000_state.json --out-dir docs/plots

lead:
	python -m sep_text_manifold.cli_lead --state analysis/mms_0000_state.json \
	  --onset 2017-09-08T00:40:00 --output analysis/mms_0000_leadtime.json \
	  --plot docs/plots/mms_0000_lead.png

stats:
	python scripts/lead_permutation.py analysis/mms_0000_state.json analysis/router_config_0000.json \
	  --start 2017-09-08T00:00:00 --stop 2017-09-08T01:00:00 --onset 2017-09-08T00:40:00 \
	  --iterations 1000 --seed 42 > analysis/mms_0000_lead_perm.json
	python scripts/ann_bootstrap.py analysis/mms_twins_0000_to_0913.json --min-windows 50 \
	  --iterations 2000 --seed 42 > analysis/mms_0000_ann_boot.json

onset:
	python scripts/run_onset_sweep.py --state analysis/mms_0000_state.json \
	  --times 00:20 00:30 00:35 00:40 00:45 00:50

twins:
	python scripts/twin_diagnostics.py analysis/mms_twins_0000_to_0913.json > analysis/mms_twins_0000_diagnostics.json
	python scripts/twin_diagnostics.py analysis/mms_twins_0100_to_0913.json > analysis/mms_twins_0100_diagnostics.json

all: scorecard plots lead twins onset stats

demo-payload:
	PYTHONPATH=src python demo/standalone.py --no-pretty

demo-up:
	docker compose -f docker-compose.demo.yml up --build -d

demo-down:
	docker compose -f docker-compose.demo.yml down

planbench-all:
	.venv/bin/python scripts/generate_planbench_dataset.py --count $(PLANBENCH_COUNT)
	.venv/bin/python scripts/planbench_to_stm.py \
	  --input-root data/planbench_public \
	  --domains $(subst $(space),$(comma),$(PLANBENCH_TARGETS)) \
	  --output output/planbench_public \
	  --window-bytes $(PLANBENCH_WINDOW_BYTES) \
	  --stride $(PLANBENCH_STRIDE) \
	  --path-threshold 0.10 --signal-threshold 0.10 \
	  --twin-distance 0.40 --twin-top-k 3 --verbose
	for dom in $(PLANBENCH_TARGETS); do \
	  label=$$(echo $$dom | tr '[:lower:]' '[:upper:]'); \
	  enrich_args=""; \
	  if [ "$$dom" = "blocksworld" ] || [ "$$dom" = "mystery_bw" ]; then \
	    enrich_args="--enrich-note \"$(PLANBENCH_ENRICH_NOTE)\""; \
	    for extra in $(PLANBENCH_ENRICH_BASE) $(PLANBENCH_EXTRA_TWINS); do \
	      if [ -n "$$extra" ]; then \
	        enrich_args="$$enrich_args --enrich-from $$extra"; \
	      fi; \
	    done; \
	  fi; \
	  match_args=""; \
	  if [ "$(PLANBENCH_MATCH_SIGNATURE)" = "1" ] && ( [ "$$dom" = "blocksworld" ] || [ "$$dom" = "mystery_bw" ] ); then \
	    match_args="--match-signature"; \
	  fi; \
	  .venv/bin/python scripts/planbench_to_stm.py \
	    --input-root data/planbench_public \
	    --domains $$dom \
	    --output output/planbench_by_domain/$$dom \
	    --window-bytes $(PLANBENCH_WINDOW_BYTES) \
	    --stride $(PLANBENCH_STRIDE) \
	    --path-threshold 0.10 --signal-threshold 0.10 \
	    --twin-distance 0.40 --twin-top-k 3 --verbose $$match_args $$enrich_args; \
	  calibrate_args="--target-low 0.05 --target-high 0.07 --output analysis/router_config_$${dom}_invalid_5pct.json"; \
	  if [ "$$dom" = "logistics" ]; then \
	    calibrate_args="$$calibrate_args --domain-root output/planbench_by_domain/$$dom --dynamic-target 0.025 --dynamic-window 0.005 --pvalue-threshold 0.05 --pvalue-metric min"; \
	  fi; \
	  .venv/bin/python scripts/calibrate_router.py output/planbench_by_domain/$$dom/invalid_state.json $$calibrate_args; \
	  .venv/bin/python scripts/run_permutation_guardrail.py \
	    output/planbench_by_domain/$$dom \
	    analysis/router_config_$${dom}_invalid_5pct.json \
	    --iterations $(PERMUTATION_ITERS) \
	    --output docs/tests/permutation_$${dom}_5pct.json; \
	  .venv/bin/python scripts/guardrail_sweep.py \
	    output/planbench_by_domain/$$dom/invalid_state.json \
	    output/planbench_by_domain/$$dom \
	    --prefix $${dom}_invalid \
	    --label PlanBench-$$label \
	    --appendix docs/note/appendix_guardrail_sweep.csv; \
	done
	.venv/bin/python scripts/calibrate_router.py output/planbench_public/invalid_state.json \
	  --target-low 0.05 --target-high 0.07 \
	  --output analysis/router_config_invalid_5pct.json
	.venv/bin/python scripts/run_permutation_guardrail.py \
	  output/planbench_public \
	  analysis/router_config_invalid_5pct.json \
	  --iterations $(PERMUTATION_ITERS) \
	  --output docs/tests/permutation_planbench_invalid_5pct.json
	.venv/bin/python scripts/guardrail_sweep.py \
	  output/planbench_public/invalid_state.json \
	  output/planbench_public \
	  --prefix planbench_invalid \
	  --label PlanBench-Aggregate \
	  --appendix docs/note/appendix_guardrail_sweep.csv

planbench-scale:
	.venv/bin/python scripts/generate_planbench_dataset.py --root $(PLANBENCH_SCALE_ROOT) --count $(PLANBENCH_SCALE_COUNT)
	.venv/bin/python scripts/inject_plan_corruption.py --root $(PLANBENCH_SCALE_ROOT) \
	  --domains $(subst $(space),$(comma),$(PLANBENCH_SCALE_TARGETS)) --validator $(PLANBENCH_VALIDATOR)
	.venv/bin/python scripts/val_to_trace.py --root $(PLANBENCH_SCALE_ROOT) \
	  --domains $(subst $(space),$(comma),$(PLANBENCH_SCALE_TARGETS)) --validator $(PLANBENCH_VALIDATOR)
	for dom in $(PLANBENCH_SCALE_TARGETS); do \
	  label=$$(python -c 'import sys; parts=sys.argv[1].split("_");\
	def fmt(token):\
	    return token.upper() if len(token) <= 2 else token.capitalize();\
	print("-".join(fmt(p) for p in parts))' "$$dom"); \
	  .venv/bin/python scripts/planbench_to_stm.py \
	    --input-root $(PLANBENCH_SCALE_ROOT) \
	    --domains $$dom \
	    --output $(PLANBENCH_SCALE_OUTPUT)/$$dom \
	    --window-bytes $(PLANBENCH_SCALE_WINDOW_BYTES) \
	    --stride $(PLANBENCH_SCALE_STRIDE) \
	    --path-threshold 0.10 --signal-threshold 0.10 \
	    --twin-distance 0.40 --twin-top-k 3 --verbose; \
	  .venv/bin/python scripts/guardrail_sweep.py \
	    $(PLANBENCH_SCALE_OUTPUT)/$$dom/invalid_state.json \
	    $(PLANBENCH_SCALE_OUTPUT)/$$dom \
	    --prefix $${dom}_scale$(PLANBENCH_SCALE_COUNT)_invalid \
	    --label PlanBench-$$label-$(PLANBENCH_SCALE_COUNT) \
	    --iteration $(PLANBENCH_SCALE_ITERATIONS) \
	    --window $(PLANBENCH_SCALE_WINDOW) \
	    --summary-json analysis/guardrail_sweep_$${dom}_scale$(PLANBENCH_SCALE_COUNT)_invalid.summary.json; \
	done

semantic-guardrail-demo:
	@echo "[semantic-guardrail] Preparing truth-pack manifold"
	@if [ ! -f analysis/truth_packs/docs_demo/manifest.json ]; then \
	  PYTHONPATH=src .venv/bin/python scripts/reality_filter_pack.py docs \
	    --output-root analysis/truth_packs/docs_demo \
	    --extensions md txt json yaml \
	    --drop-numeric \
	    --min-token-len 3 \
	    --min-occurrences 1 \
	    --semantic-min-occ 2 \
	    --seeds risk resilience volatility anomaly "predictive maintenance"; \
	fi
	@echo "[semantic-guardrail] Generating reality-filter stream"
	@PYTHONPATH=src .venv/bin/python scripts/reality_filter_stream.py \
	  --manifest analysis/truth_packs/docs_demo/manifest.json \
	  --spans demo/truth_pack/sample_spans.json \
	  --seeds risk resilience volatility anomaly "predictive maintenance" \
	  --semantic-threshold 0.25 \
	  --structural-threshold 0.46 \
	  --r-min 2 \
	  --hazard-max 0.55 \
	  --sigma-min 0.28 \
	  --repair \
	  --output results/semantic_guardrail_stream.jsonl \
	  --metrics-output results/semantic_guardrail_metrics.json
	@echo "[semantic-guardrail] Launching dashboard"
	@PYTHONPATH=src .venv/bin/python scripts/demos/semantic_guardrail_dashboard.py \
	  --stream results/semantic_guardrail_stream.jsonl \
	  --background analysis/truth_packs/docs_demo/semantic_scatter.png \
	  --states analysis/truth_packs/docs_demo/manifold_state.json \
	  --seeds risk resilience volatility anomaly "predictive maintenance"

PACK ?= docs_demo
PACK_PATH ?= analysis/truth_packs/$(PACK)
PACK_SRC ?= docs
PACK_ARGS ?=
SPANS ?= demo/truth_pack/sample_spans.json
SEEDS ?= risk resilience volatility anomaly "predictive maintenance"

pack:
	PYTHONPATH=src .venv/bin/python scripts/reality_filter_pack.py $(PACK_SRC) \
	  --output-root $(PACK_PATH) \
	  --extensions md txt json yaml \
	  --drop-numeric --min-token-len 3 --min-occurrences 1 --semantic-min-occ 2 \
	  --seeds $(SEEDS) $(PACK_ARGS)

stream:
	PYTHONPATH=src .venv/bin/python scripts/reality_filter_stream.py \
	  --manifest $(PACK_PATH)/manifest.json \
	  --spans $(SPANS) \
	  --seeds $(SEEDS) \
	  --semantic-threshold 0.25 \
	  --structural-threshold 0.46 \
	  --r-min 2 \
	  --hazard-max 0.25 \
	  --sigma-min 0.28 \
	  --repair \
	  --output results/$(PACK)_stream.jsonl \
	  --metrics-output results/$(PACK)_metrics.json

sweep:
	PYTHONPATH=src .venv/bin/python scripts/reality_filter_sweep.py \
	  --manifest $(PACK_PATH)/manifest.json \
	  --spans $(SPANS) \
	  --output results/sweeps/$(PACK).csv

permutation:
	PYTHONPATH=src .venv/bin/python scripts/reality_filter_permutation.py \
	  --manifest $(PACK_PATH)/manifest.json \
	  --spans $(SPANS) \
	  --output results/permutation/$(PACK).json

report:
	PYTHONPATH=src .venv/bin/python scripts/reality_filter_report.py \
	  --packs $(PACK)

bench-seen:
	PYTHONPATH=src .venv/bin/python scripts/benchmark_seen.py --manifest $(PACK_PATH)/manifest.json --requests 4000 --concurrency 600 --hash-embeddings | tee results/bench_seen_latest.txt

codetrace-report:
	PYTHONPATH=src .venv/bin/python demo/coding/run_comparison.py

final-report:
	@PYTHONPATH=src .venv/bin/python scripts/analysis/summarize_semantic_guardrail.py
	@if [ -f results/semantic_bridge_combined.png ]; then \
	  mkdir -p docs/whitepaper/figures; \
	  cp results/semantic_bridge_combined.png docs/whitepaper/figures/semantic_bridge_combined.png; \
	fi
	@latexmk -pdf -quiet -f -g -output-directory=docs/whitepaper docs/whitepaper/Semantic_Guardrail_Whitepaper.tex

clean:
	rm -f analysis/semantic_demo_state.json analysis/mms_state.json
	rm -f results/semantic_bridge_docs.json results/semantic_bridge_mms.json
	rm -f results/semantic_bridge_scatter.png results/semantic_bridge_mms_scatter.png
	rm -f results/semantic_bridge_combined.png results/semantic_guardrail_stream.jsonl results/semantic_guardrail_metrics.json
	rm -f docs/whitepaper/figures/semantic_bridge_combined.png
	rm -rf analysis/truth_packs/docs_demo
	 rm -f results/*_metrics.json results/*_stream.jsonl
	 rm -rf results/sweeps results/permutation results/report

clean-demo: clean
