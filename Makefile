.PHONY: scorecard plots lead twins onset all
.PHONY: demo-payload demo-up demo-down
.PHONY: planbench-all planbench-scale codetrace-report
.PHONY: semantic-guardrail-demo

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
	@echo "[semantic-guardrail] Preparing documentation manifold"
	@if [ ! -f analysis/semantic_demo_state.json ]; then \
	  PYTHONPATH=/score .venv/bin/stm ingest docs --extensions md txt json yaml \
	    --output analysis/semantic_demo_state.json --store-signals --min-token-len 3 --drop-numeric; \
	fi
	@if [ -d data/mms ] && [ ! -f analysis/mms_state.json ]; then \
	  PYTHONPATH=/score .venv/bin/stm ingest data/mms --output analysis/mms_state.json --store-signals; \
	fi
	@echo "[semantic-guardrail] Building semantic projections"
	@PYTHONPATH=src .venv/bin/python scripts/semantic_bridge_demo.py \
	  analysis/semantic_demo_state.json \
	  --seeds risk resilience volatility anomaly "predictive maintenance" \
	  --top-k 15 --min-occurrences 3 --embedding-method transformer \
	  --output results/semantic_bridge_docs.json
	@PYTHONPATH=src .venv/bin/python scripts/semantic_bridge_plot.py \
	  analysis/semantic_demo_state.json \
	  --seeds risk resilience volatility anomaly "predictive maintenance" \
	  --embedding-method transformer \
	  --output results/semantic_bridge_scatter.png
	@if [ -f analysis/mms_state.json ]; then \
	  PYTHONPATH=src .venv/bin/python scripts/semantic_bridge_demo.py \
	    analysis/mms_state.json \
	    --seeds risk resilience volatility anomaly "predictive maintenance" \
	    --top-k 15 --min-occurrences 1 --embedding-method transformer \
	    --output results/semantic_bridge_mms.json; \
	  PYTHONPATH=src .venv/bin/python scripts/semantic_bridge_plot.py \
	    analysis/mms_state.json \
	    --seeds risk resilience volatility anomaly "predictive maintenance" \
	    --embedding-method transformer \
	    --output results/semantic_bridge_mms_scatter.png; \
	fi
	@python - <<-'PY'
	from pathlib import Path
	from PIL import Image
	left = Path('results/semantic_bridge_scatter.png')
	right = Path('results/semantic_bridge_mms_scatter.png')
	if left.exists() and right.exists():
	    canvas_path = Path('results/semantic_bridge_combined.png')
	    left_img = Image.open(left)
	    right_img = Image.open(right)
	    canvas = Image.new('RGB', (left_img.width + right_img.width, max(left_img.height, right_img.height)), 'white')
	    canvas.paste(left_img, (0, 0))
	    canvas.paste(right_img, (left_img.width, 0))
	    canvas.save(canvas_path)
	PY
	@mkdir -p docs/whitepaper/figures
	@if [ -f results/semantic_bridge_combined.png ]; then \
	  cp results/semantic_bridge_combined.png docs/whitepaper/figures/semantic_bridge_combined.png; \
	fi
	@echo "[semantic-guardrail] Generating stream"
	@PYTHONPATH=src .venv/bin/python scripts/semantic_guardrail_stream.py \
	  --seeds risk resilience volatility anomaly "predictive maintenance" \
	  --samples 6
	@echo "[semantic-guardrail] Launching dashboard"
	@PYTHONPATH=src .venv/bin/python scripts/demos/semantic_guardrail_dashboard.py \
	  --stream results/semantic_guardrail_stream.jsonl \
	  --background results/semantic_bridge_combined.png \
	  --states analysis/semantic_demo_state.json analysis/mms_state.json

codetrace-report:
	PYTHONPATH=src .venv/bin/python demo/coding/run_comparison.py
