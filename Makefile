.PHONY: scorecard plots lead twins onset all
.PHONY: demo-payload demo-up demo-down

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
