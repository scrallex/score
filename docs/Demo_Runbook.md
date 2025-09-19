# Demo Runbook (10-minute session)

1. **Spin up streaming runtime**
   ```bash
   stm stream --config configs/mms_stream.yaml &
   sleep 1
   curl -s localhost:8000/stm/health | jq
   ```

2. **Show foreground clustering**
   ```bash
   curl -s localhost:8000/stm/seen \
     -H 'content-type: application/json' \
     -d '{"trigger":"mms1_fgm_b_gse_srvy_l2_x__RANGEEXP","top":25}' | jq
   ```
   - Point out contiguous foreground windows before onset.

3. **Open evidence**
   - Figure: `docs/note/fig2_mms_0000_zoom.png` (Bx/Bz overlays + heat strip).
   - Scorecard row (midnight vs quiet) from `docs/note/tab1_scorecard.csv`.
   - Twins diagnostics: `analysis/mms_twins_0000_diagnostics.json` (50 windows, ANN ≈ 2e-3).
   - Lead-time table/plot: `docs/note/tab4_leadtime_bins_midnight.csv`, `docs/plots/mms_0000_lead.png`.

4. **Autolabel + lead-time refresh**
   ```bash
   stm onsets --state analysis/mms_0000_state.json \
     --config analysis/router_config_0000.json \
     --start 2017-09-08T00:00:00 --stop 2017-09-08T01:00:00 \
     --lead-output analysis/mms_0000_lead_autolabel.json \
     --lead-plot docs/plots/mms_0000_lead_autolabel.png
   ```
   - Show that auto labels reproduce the rising trend.

5. **Adapter example (optional)**
   ```bash
   stm ingest --adapter nasa_themis THEMIS_raw.csv \
     --output analysis/themis_state.json --log-file analysis/themis.log
   python scripts/calibrate_router.py analysis/themis_state.json
   stm report --state analysis/themis_state.json --onset <themis_onset_utc>
   ```

6. **Close with scorecard + next steps**
   - Reiterate twins, lead-time gain, guardrailed foreground, and portability to other datasets.
