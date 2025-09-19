.PHONY: scorecard plots lead twins onset all

scorecard:
	python scripts/make_scorecard.py

plots:
	python -m sep_text_manifold.cli_plots --state analysis/mms_0000_state.json --out-dir docs/plots

lead:
	python -m sep_text_manifold.cli_lead --state analysis/mms_0000_state.json \
	  --onset 2017-09-08T00:40:00 --output analysis/mms_0000_leadtime.json \
	  --plot docs/plots/mms_0000_lead.png

onset:
	python - <<'PY'
import csv
import json
import subprocess
from pathlib import Path

state = "analysis/mms_0000_state.json"
pairs = ["00:20", "00:30", "00:35", "00:40", "00:45", "00:50"]
rows = []
for time_str in pairs:
    onset = f"2017-09-08T{time_str}:00"
    suffix = time_str.replace(":", "")
    out_json = Path(f"analysis/mms_0000_lead_{suffix}.json")
    out_plot = Path(f"docs/plots/mms_0000_lead_{suffix}.png")
    cmd = [
        "python", "-m", "sep_text_manifold.cli_lead",
        "--state", state,
        "--onset", onset,
        "--output", str(out_json),
        "--plot", str(out_plot),
    ]
    subprocess.run(cmd, check=True)
    payload = json.loads(out_json.read_text())
    last_density = payload["bins"][-1]["density"] if payload.get("bins") else 0.0
    rows.append({
        "onset": onset,
        "fg_density_last_bin": last_density,
        "monotonic_increase": payload.get("monotonic_increase"),
    })

scorecard = Path("docs/note/tab4a_midnight_onset_sweep.csv")
scorecard.parent.mkdir(parents=True, exist_ok=True)
with scorecard.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=["onset", "fg_density_last_bin", "monotonic_increase"])
    writer.writeheader()
    writer.writerows(rows)
print(f"Wrote {scorecard}")
PY

twins:
	python scripts/twin_diagnostics.py analysis/mms_twins_0000_to_0913.json > analysis/mms_twins_0000_diagnostics.json
	python scripts/twin_diagnostics.py analysis/mms_twins_0100_to_0913.json > analysis/mms_twins_0100_diagnostics.json

all: scorecard plots lead twins onset
