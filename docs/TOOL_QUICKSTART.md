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
