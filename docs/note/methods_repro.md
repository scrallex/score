# MMS Structural Precursors – Reproducibility Cheat Sheet

Environment reference:

- OS: Ubuntu 22.04 LTS (64-bit)
- Python: `python --version` → 3.11.x (match your env)
- Dependencies: `pip install -r requirements.txt` (`pip freeze > docs/note/requirements.txt` to snapshot)

All commands assume repository root (`/score`) with STM CLI available in the active virtualenv.

1. **Bit-slice preparation**
   ```bash
   python scripts/slice_csv_interval.py nasa/mms/csv/2017-09-08/mms1_fgm_bits.csv \
     2017-09-08T00:00 2017-09-08T01:00 nasa/mms/csv_zoom/2017-09-08_0000-0100/mms1_fgm_bits.csv
   ```

2. **Ingest & guardrail calibration**
   ```bash
   export STM_NATIVE=1
   stm ingest nasa/mms/csv_zoom/2017-09-08_0000-0100 \
     --window-bytes 1024 --stride 512 --min-token-len 2 --drop-numeric --min-occ 1 \
     --cap-tokens-per-win 80 --graph-min-pmi 0.35 --theme-min-size 3 \
     --output analysis/mms_0000_state.json --store-signals --store-profiles \
     --verbose --log-file analysis/mms_0000.log
   python scripts/calibrate_router.py analysis/mms_0000_state.json
   ```

3. **Structural enforcement & artefacts**
   ```bash
   python scripts/enforce_structural.py strings   analysis/mms_0000_state.json analysis/mms_0000_top_structural.txt
   python scripts/enforce_structural.py proposals analysis/mms_0000_proposals_raw.json analysis/mms_0000_proposals_struct.json
   python scripts/enforce_structural.py twins     analysis/mms_twins_0000_to_0913_raw.json analysis/mms_twins_0000_to_0913.json
   python scripts/twin_diagnostics.py             analysis/mms_twins_0000_to_0913.json > analysis/mms_twins_0000_diagnostics.json
   ```

4. **Plots & lead-time**
   ```bash
   python -m sep_text_manifold.cli_plots --state analysis/mms_0000_state.json --out-dir docs/plots
   python -m sep_text_manifold.cli_lead  --state analysis/mms_0000_state.json --onset 2017-09-08T00:40:00 \
     --output analysis/mms_0000_leadtime.json --plot docs/plots/mms_0000_lead.png
   ```

5. **Scorecard + tables**
   ```bash
   python scripts/make_scorecard.py
   python - <<'PY'
import json, csv
from pathlib import Path
root = Path('analysis')
p = json.loads((root / 'mms_0000_proposals_struct.json').read_text())['proposals']
out = Path('docs/note/tab2_midnight_proposals.csv')
with out.open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['string','coherence','entropy','stability','ann_distance','connector','score','occurrences'])
    writer.writeheader()
    for entry in p:
        m = entry.get('metrics', {})
        d = entry.get('diagnostics', {})
        writer.writerow({
            'string': entry['string'],
            'coherence': m.get('coherence'),
            'entropy': m.get('entropy'),
            'stability': m.get('stability'),
            'ann_distance': d.get('distance'),
            'connector': entry.get('connector'),
            'score': entry.get('score'),
            'occurrences': entry.get('occurrences'),
        })
PY
   ```

Repeat the same sequence for 22:30–23:30 (`_2230-2330`), 23:00–00:00 (`_2300-0000`), 01:00–02:00 (`_0100-0200`), and quiet baseline (`2017-09-10_0000-0100`). Run `make all` afterwards to refresh plots, lead-time summaries, twin diagnostics, and the scorecard.
