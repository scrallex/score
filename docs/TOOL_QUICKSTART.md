# STM Tool Quickstart

## Install

```bash
pip install -e .[native,index]
```

## Generate plots & tables for MMS midnight slice

```bash
stm plots --state analysis/mms_0000_state.json --out-dir docs/plots
stm lead --state analysis/mms_0000_state.json --onset 2017-09-08T00:40:00
make scorecard
```

## Run streaming router (placeholder)

```bash
stm stream --config configs/mms_stream.yaml
```

## Auto-label onsets and compute lead-time (guardrail rules)

```bash
stm onsets --state analysis/mms_0000_state.json \
  --config analysis/router_config_0000.json \
  --start 2017-09-08T00:00:00 --stop 2017-09-08T01:00:00 \
  --lead-output analysis/mms_0000_lead_autolabel.json \
  --lead-plot docs/plots/mms_0000_lead_autolabel.png
```
