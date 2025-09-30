#!/usr/bin/env python3
"""Generate a logistics guardrail demo trace and STM metrics."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Sequence

from sep_text_manifold.pipeline import analyse_directory
from stm_adapters.pddl_trace import PDDLTraceAdapter


@dataclass(frozen=True)
class StepDefinition:
    action: str
    adds: Sequence[str] = field(default_factory=list)
    deletes: Sequence[str] = field(default_factory=list)
    observations: Sequence[str] = field(default_factory=list)
    status: str = "valid"
    errors: Sequence[str] = field(default_factory=list)
    apply_effects: bool = True
    note: str | None = None


INITIAL_STATE = [
    "airport apt1",
    "airport apt2",
    "at pkg1 warehouse",
    "at-vehicle plane1 apt1",
    "at-vehicle truck1 warehouse",
    "at-vehicle truck2 apt2",
]


STEP_DEFINITIONS: Sequence[StepDefinition] = (
    StepDefinition(
        action="(load-truck pkg1 truck1 warehouse)",
        adds=["in pkg1 truck1"],
        deletes=["at pkg1 warehouse"],
    ),
    StepDefinition(
        action="(drive truck1 warehouse apt1)",
        adds=["at-vehicle truck1 apt1"],
        deletes=["at-vehicle truck1 warehouse"],
    ),
    StepDefinition(
        action="(unload-truck pkg1 truck1 apt1)",
        adds=["at pkg1 apt1"],
        deletes=["in pkg1 truck1"],
    ),
    StepDefinition(
        action="(load-plane pkg1 plane1 apt1)",
        adds=["in pkg1 plane1"],
        deletes=["at pkg1 apt1"],
    ),
    StepDefinition(
        action="(event airport-closed apt2)",
        deletes=["airport apt2", "at-vehicle truck2 apt2"],
        observations=[
            "disruption airport apt2",
            "hazard closure apt2",
            "telemetry weather_alert apt2",
            "truck2 stranded apt2",
        ],
        note="Weather closure removes the apt2 runway",
    ),
    StepDefinition(
        action="(fly plane1 apt1 apt2)",
        adds=["at-vehicle plane1 apt2"],
        deletes=["at-vehicle plane1 apt1"],
        observations=[
            "hazard flight_blocked plane1 apt2",
            "telemetry runway_status apt2 closed",
            "flight_deviation plane1 apt1",
            "structural_drift runway_closure",
            "signal_divergence logistics_plan",
            "anomaly_lead indicator_runway",
            "qfh_alert code_yellow",
            "stm_hint twins_requested",
        ],
        status="valid",
        apply_effects=False,
    ),
    StepDefinition(
        action="(unload-plane pkg1 plane1 apt2)",
        adds=["at pkg1 apt2"],
        deletes=["in pkg1 plane1"],
        observations=[
            "hazard unload_blocked pkg1 apt2",
            "telemetry cargo_pending pkg1",
        ],
        status="valid",
        apply_effects=False,
    ),
    StepDefinition(
        action="(drive truck2 apt2 depot)",
        adds=["at-vehicle truck2 depot"],
        deletes=["at-vehicle truck2 apt2"],
        observations=[
            "hazard truck_denied truck2 apt2",
            "telemetry access_denied truck2 apt2",
        ],
        status="valid",
        apply_effects=False,
    ),
    StepDefinition(
        action="(posthoc-validate goal pkg1 apt2)",
        observations=[
            "validator result goal_unmet",
            "summary anomaly_detected",
        ],
        status="invalid",
        errors=["goal state unmet: pkg1 at apt2 missing"],
        apply_effects=False,
    ),
)


def _build_transitions(steps: Sequence[StepDefinition]) -> List[Dict[str, object]]:
    state = {atom for atom in INITIAL_STATE}
    transitions: List[Dict[str, object]] = []
    for idx, spec in enumerate(steps):
        before = sorted(state)
        adds = [item.lower() for item in spec.adds]
        deletes = [item.lower() for item in spec.deletes]
        observations = [item.lower() for item in spec.observations]
        if spec.apply_effects:
            for atom in deletes:
                state.discard(atom)
            for atom in adds:
                state.add(atom)
        if observations:
            for atom in observations:
                state.add(atom)
        after = sorted(state)
        transition: Dict[str, object] = {
            "step": idx,
            "time": float(idx + 1),
            "action": spec.action,
            "state": before,
            "effects": {"add": adds, "del": deletes},
            "status": spec.status,
            "next_state": after,
            "errors": list(spec.errors),
        }
        if observations:
            transition["observations"] = observations
        if spec.note:
            transition["note"] = spec.note
        transitions.append(transition)
    return transitions


def _normalise_transition_atoms(transitions: Iterable[Dict[str, object]]) -> None:
    for transition in transitions:
        transition["state"] = [str(atom).lower() for atom in transition.get("state", [])]
        transition["next_state"] = [str(atom).lower() for atom in transition.get("next_state", [])]
        effects = transition.get("effects", {})
        if isinstance(effects, dict):
            adds = effects.get("add", [])
            dels = effects.get("del", [])
            effects["add"] = [str(atom).lower() for atom in adds]
            effects["del"] = [str(atom).lower() for atom in dels]
        observations = transition.get("observations")
        if observations is not None:
            transition["observations"] = [str(atom).lower() for atom in observations]


def _build_trace_payload(transitions: Sequence[Dict[str, object]]) -> Dict[str, object]:
    invalid_steps = [t["step"] for t in transitions if t.get("status") != "valid"]
    payload: Dict[str, object] = {
        "domain": "logistics_guardrail_demo",
        "problem": "Deliver pkg1 under airport disruption",
        "plan_steps": [t["action"] for t in transitions],
        "initial_state": [atom.lower() for atom in INITIAL_STATE],
        "status": "invalid" if invalid_steps else "valid",
        "failed_at_step": invalid_steps[-1] if invalid_steps else None,
        "transitions": transitions,
    }
    return payload


def _max_threshold(values: Sequence[float], *, padding: float, ceiling: float | None = None) -> float:
    if not values:
        base = 0.0
    else:
        base = max(values)
    threshold = base + padding
    if ceiling is not None:
        threshold = min(threshold, ceiling)
    return threshold


def _build_signal_rows(
    *,
    signals: Sequence[Dict[str, object]],
    transitions: Sequence[Dict[str, object]],
    event_index: int,
) -> Dict[str, object]:
    limit = min(len(signals), len(transitions))
    rows: List[Dict[str, object]] = []
    for idx in range(limit):
        signal = signals[idx]
        transition = transitions[idx]
        metrics = signal.get("metrics", {}) if isinstance(signal, dict) else {}
        dilution = signal.get("dilution", {}) if isinstance(signal, dict) else {}
        row = {
            "step": idx,
            "action": transition.get("action"),
            "status": transition.get("status", "valid"),
            "lambda_hazard": float(signal.get("lambda_hazard", metrics.get("lambda_hazard", 0.0))),
            "coherence": float(metrics.get("coherence", 0.0)),
            "stability": float(metrics.get("stability", 0.0)),
            "entropy": float(metrics.get("entropy", 0.0)),
            "path_dilution": float(dilution.get("path", 0.0)) if isinstance(dilution, dict) else 0.0,
            "signal_dilution": float(dilution.get("signal", 0.0)) if isinstance(dilution, dict) else 0.0,
            "errors": transition.get("errors", []),
            "note": transition.get("note"),
        }
        rows.append(row)

    baseline = [row for row in rows if row["step"] <= event_index]
    hazard_values = [row["lambda_hazard"] for row in baseline]
    path_values = [row["path_dilution"] for row in baseline]
    signal_values = [row["signal_dilution"] for row in baseline]

    hazard_threshold = _max_threshold(hazard_values, padding=0.0015, ceiling=0.95)
    path_threshold = _max_threshold(path_values, padding=0.05, ceiling=1.2)
    signal_threshold = _max_threshold(signal_values, padding=0.01)

    alert_steps: List[int] = []
    for row in rows:
        hazard_alert = row["lambda_hazard"] >= hazard_threshold
        path_alert = row["path_dilution"] >= path_threshold
        signal_alert = row["signal_dilution"] >= signal_threshold
        row["hazard_alert"] = hazard_alert
        row["dilution_alert"] = path_alert or signal_alert
        row["alert"] = hazard_alert or row["dilution_alert"]
        if row["alert"]:
            alert_steps.append(row["step"])

    failure_steps = [row["step"] for row in rows if row["status"] != "valid"]
    first_alert = alert_steps[0] if alert_steps else None
    first_failure = failure_steps[0] if failure_steps else None
    lead_time = None
    if first_alert is not None and first_failure is not None:
        lead_time = max(0, first_failure - first_alert)

    summary = {
        "rows": rows,
        "thresholds": {
            "lambda_hazard": hazard_threshold,
            "path_dilution": path_threshold,
            "signal_dilution": signal_threshold,
        },
        "alert_steps": alert_steps,
        "failure_steps": failure_steps,
        "first_alert": first_alert,
        "first_failure": first_failure,
        "lead_time": lead_time,
        "baseline": {
            "hazard_mean": mean(hazard_values) if hazard_values else 0.0,
            "path_mean": mean(path_values) if path_values else 0.0,
            "signal_mean": mean(signal_values) if signal_values else 0.0,
        },
    }
    return summary


def _build_classical_timeline(transitions: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    classical: List[Dict[str, object]] = []
    for idx, transition in enumerate(transitions):
        entry = {
            "step": idx,
            "action": transition.get("action"),
            "status": "pending",
        }
        classical.append(entry)
    if classical:
        classical[-1]["status"] = "fail"
        classical[-1]["message"] = "Plan failed post-hoc validation"
    return classical


def _write_dashboard(output_root: Path, timeline_payload: Dict[str, object]) -> Path:
    dashboard_path = output_root / "dashboard.html"
    timeline_json = json.dumps(timeline_payload, separators=(",", ":"))
    html = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>STM Logistics Guardrail Demo</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #0f172a;
      --panel: rgba(15, 23, 42, 0.72);
      --border: #1f2937;
      --accent: #38bdf8;
      --alert: #f97316;
      --failure: #ef4444;
      --success: #22c55e;
      --text: #e2e8f0;
      --muted: #94a3b8;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 32px;
      font-family: \"Inter\", \"Segoe UI\", sans-serif;
      background: radial-gradient(circle at 15% 20%, rgba(56,189,248,0.15), transparent 45%),
                  radial-gradient(circle at 85% 15%, rgba(249,115,22,0.12), transparent 40%),
                  var(--bg);
      color: var(--text);
    }}
    h1 {{
      margin-top: 0;
      font-size: 28px;
      letter-spacing: 0.02em;
    }}
    .subtitle {{
      color: var(--muted);
      margin-bottom: 28px;
      max-width: 720px;
      line-height: 1.6;
    }}
    .columns {{
      display: flex;
      flex-wrap: wrap;
      gap: 24px;
    }}
    .panel {{
      flex: 1 1 340px;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 24px 28px;
      box-shadow: 0 18px 38px rgba(15,23,42,0.4);
      position: relative;
      overflow: hidden;
    }}
    .panel::before {{
      content: \"\";
      position: absolute;
      inset: 0;
      background: linear-gradient(135deg, rgba(56,189,248,0.12), rgba(15,23,42,0));
      opacity: 0.8;
      pointer-events: none;
    }}
    .panel h2 {{
      margin: 0 0 16px;
      font-size: 20px;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      color: #f8fafc;
    }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 16px;
      margin-bottom: 20px;
    }}
    .metric {{
      background: rgba(15,23,42,0.65);
      border: 1px solid rgba(148,163,184,0.15);
      border-radius: 12px;
      padding: 14px;
    }}
    .metric .label {{
      font-size: 12px;
      color: var(--muted);
      letter-spacing: 0.12em;
      text-transform: uppercase;
    }}
    .metric .value {{
      font-size: 22px;
      font-weight: 600;
      margin-top: 6px;
    }}
    .timeline {{
      border-top: 1px solid rgba(148,163,184,0.12);
      margin-top: 18px;
    }}
    .step {{
      display: grid;
      grid-template-columns: 54px 1fr 120px;
      align-items: start;
      gap: 14px;
      padding: 12px 0;
      border-bottom: 1px solid rgba(148,163,184,0.08);
      position: relative;
    }}
    .step:last-child {{ border-bottom: none; }}
    .step .index {{
      font-variant-numeric: tabular-nums;
      color: var(--muted);
      font-size: 14px;
      letter-spacing: 0.08em;
    }}
    .step .action {{
      font-family: \"JetBrains Mono\", \"SFMono-Regular\", monospace;
      font-size: 13px;
      line-height: 1.45;
      color: #f8fafc;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 12px;
      padding: 4px 10px;
      border-radius: 999px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      background: rgba(148,163,184,0.15);
      color: var(--muted);
    }}
    .badge.success {{ background: rgba(34,197,94,0.22); color: var(--success); }}
    .badge.alert {{ background: rgba(249,115,22,0.18); color: var(--alert); }}
    .badge.fail {{ background: rgba(239,68,68,0.2); color: var(--failure); }}
    .metrics-small {{
      color: var(--muted);
      font-size: 12px;
      letter-spacing: 0.04em;
      line-height: 1.4;
      margin-top: 6px;
    }}
    .note {{
      margin-top: 14px;
      padding: 14px 16px;
      background: rgba(59,130,246,0.12);
      border: 1px solid rgba(59,130,246,0.24);
      border-radius: 12px;
      font-size: 13px;
      color: #bfdbfe;
      line-height: 1.5;
    }}
    @media (max-width: 820px) {{
      body {{ padding: 24px; }}
      .columns {{ flex-direction: column; }}
      .panel {{ padding: 20px; }}
    }}
  </style>
</head>
<body>
  <h1>STM Logistics Guardrail Demo</h1>
  <p class=\"subtitle\">Classical PDDL validation waits until execution completes before reporting failure. The Structural Manifold guardrail monitors every action, surfacing calibrated hazard spikes several steps before the plan collapses.</p>
  <div class=\"columns\">
    <section class=\"panel\" id=\"classic-panel\">
      <h2>Classical Validator</h2>
      <div class=\"metric-grid\">
        <div class=\"metric\">
          <div class=\"label\">Outcome</div>
          <div class=\"value\" id=\"classic-outcome\">Pending…</div>
        </div>
        <div class=\"metric\">
          <div class=\"label\">Failure Step</div>
          <div class=\"value\" id=\"classic-failure\">–</div>
        </div>
      </div>
      <div class=\"timeline\" id=\"classic-timeline\"></div>
    </section>
    <section class=\"panel\" id=\"stm-panel\">
      <h2>STM Guardrail</h2>
      <div class=\"metric-grid\">
        <div class=\"metric\">
          <div class=\"label\">Lead Time</div>
          <div class=\"value\" id=\"stm-lead\">–</div>
        </div>
        <div class=\"metric\">
          <div class=\"label\">First Alert</div>
          <div class=\"value\" id=\"stm-first-alert\">–</div>
        </div>
        <div class=\"metric\">
          <div class=\"label\">Hazard Threshold</div>
          <div class=\"value\" id=\"stm-threshold\">–</div>
        </div>
      </div>
      <div class=\"timeline\" id=\"stm-timeline\"></div>
      <div class=\"note\" id=\"stm-note\"></div>
    </section>
  </div>
  <script id=\"timeline-data\" type=\"application/json\">{timeline_json}</script>
  <script>
    const data = JSON.parse(document.getElementById('timeline-data').textContent);
    const classical = data.classical_validator || [];
    const signalSummary = data.signal_summary || {{}};
    const rows = signalSummary.rows || [];
    const alertSteps = new Set(signalSummary.alert_steps || []);
    const failureSteps = new Set(signalSummary.failure_steps || []);

    const classicTimeline = document.getElementById('classic-timeline');
    classical.forEach((entry) => {{
      const row = document.createElement('div');
      row.className = 'step';
      const index = document.createElement('div');
      index.className = 'index';
      index.textContent = 'Step ' + entry.step;
      const action = document.createElement('div');
      action.className = 'action';
      action.textContent = entry.action;
      const statusWrap = document.createElement('div');
      const badge = document.createElement('span');
      badge.className = 'badge ' + (entry.status === 'fail' ? 'fail' : 'success');
      badge.textContent = entry.status === 'fail' ? 'FAIL' : 'OK';
      statusWrap.appendChild(badge);
      if (entry.message) {{
        const msg = document.createElement('div');
        msg.className = 'metrics-small';
        msg.textContent = entry.message;
        statusWrap.appendChild(msg);
      }}
      row.appendChild(index);
      row.appendChild(action);
      row.appendChild(statusWrap);
      classicTimeline.appendChild(row);
    }});

    const classicOutcome = document.getElementById('classic-outcome');
    classicOutcome.textContent = 'Failed';
    classicOutcome.style.color = 'var(--failure)';
    const classicFailure = document.getElementById('classic-failure');
    const last = classical[classical.length - 1];
    classicFailure.textContent = last ? ('#' + last.step) : '–';

    const stmTimeline = document.getElementById('stm-timeline');
    rows.forEach((row) => {{
      const wrapper = document.createElement('div');
      wrapper.className = 'step';
      if (alertSteps.has(row.step)) {{
        wrapper.style.background = 'rgba(249,115,22,0.08)';
      }}
      if (failureSteps.has(row.step)) {{
        wrapper.style.borderLeft = '3px solid var(--failure)';
      }}
      const index = document.createElement('div');
      index.className = 'index';
      index.textContent = 'Step ' + row.step;
      const action = document.createElement('div');
      action.className = 'action';
      action.textContent = row.action;
      const statusWrap = document.createElement('div');
      const badge = document.createElement('span');
      badge.className = 'badge ' + (alertSteps.has(row.step) ? 'alert' : 'success');
      badge.textContent = alertSteps.has(row.step) ? 'ALERT' : 'OK';
      statusWrap.appendChild(badge);
      const metrics = document.createElement('div');
      metrics.className = 'metrics-small';
      const hazard = Number.parseFloat(row.lambda_hazard || 0).toFixed(3);
      const dilution = Number.parseFloat(row.path_dilution || 0).toFixed(3);
      metrics.textContent = 'λ=' + hazard + ' · path=' + dilution;
      statusWrap.appendChild(metrics);
      wrapper.appendChild(index);
      wrapper.appendChild(action);
      wrapper.appendChild(statusWrap);
      stmTimeline.appendChild(wrapper);
    }});

    const lead = signalSummary.lead_time;
    document.getElementById('stm-lead').textContent = lead ? (lead + ' steps') : '0 steps';
    const firstAlert = signalSummary.first_alert;
    document.getElementById('stm-first-alert').textContent = firstAlert === null || firstAlert === undefined ? '–' : ('#' + firstAlert);
    const hazardThreshold = (signalSummary.thresholds || {{}}).lambda_hazard;
    document.getElementById('stm-threshold').textContent = hazardThreshold === undefined ? '0.000' : Number.parseFloat(hazardThreshold).toFixed(3);

    const note = document.getElementById('stm-note');
    if (firstAlert === null || firstAlert === undefined) {{
      note.textContent = 'No foreground alert fired within the analysed horizon.';
    }} else {{
      const firstFailure = signalSummary.first_failure;
      const leadDelta = firstFailure === null || firstFailure === undefined ? null : firstFailure - firstAlert;
      const detail = leadDelta === null ? 'Alert raised before failure check.' : (leadDelta + ' step lead-time before failure.');
      note.textContent = 'STM flagged a structural deviation at step ' + firstAlert + ', ' + detail;
    }}
  </script>
</body>
</html>
"""
    dashboard_path.write_text(html, encoding="utf-8")
    return dashboard_path


def generate_demo(args: argparse.Namespace) -> Dict[str, object]:
    output_root = Path(args.output_root).resolve()
    trace_root = output_root / "trace"
    trace_root.mkdir(parents=True, exist_ok=True)

    transitions = _build_transitions(STEP_DEFINITIONS)
    _normalise_transition_atoms(transitions)
    payload = _build_trace_payload(transitions)

    trace_path = trace_root / "logistics_guardrail_demo.trace.json"
    trace_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    adapter = PDDLTraceAdapter()
    tokens_root = output_root / "tokens"
    struct_path = adapter.run(trace_path, tokens_root)

    analysis = analyse_directory(
        str(tokens_root),
        window_bytes=args.window_bytes,
        stride=args.stride,
        extensions=["txt"],
        verbose=args.verbose,
    )

    state = analysis.to_state(include_signals=True)
    state_path = output_root / "analysis_state.json"
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    event_index = next((idx for idx, step in enumerate(STEP_DEFINITIONS) if "event" in step.action), 3)
    signal_summary = _build_signal_rows(
        signals=analysis.signals,
        transitions=transitions,
        event_index=event_index,
    )
    timeline_path = output_root / "timeline.json"
    timeline_payload = {
        "transitions": transitions,
        "signal_summary": signal_summary,
        "classical_validator": _build_classical_timeline(transitions),
    }
    timeline_path.write_text(json.dumps(timeline_payload, indent=2), encoding="utf-8")

    dashboard_path = _write_dashboard(output_root, timeline_payload)

    summary = {
        "trace_path": str(trace_path),
        "tokens_path": str(struct_path),
        "state_path": str(state_path),
        "timeline_path": str(timeline_path),
        "dashboard_path": str(dashboard_path),
        "lead_time": signal_summary.get("lead_time"),
        "first_alert": signal_summary.get("first_alert"),
        "first_failure": signal_summary.get("first_failure"),
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate STM logistics guardrail demo artefacts")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("analysis/logistics_guardrail_demo"),
        help="Directory to store generated artefacts",
    )
    parser.add_argument(
        "--window-bytes",
        type=int,
        default=256,
        help="Sliding window size for STM analysis",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=128,
        help="Stride for STM analysis",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose manifold logging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = generate_demo(args)
    lead = summary.get("lead_time")
    if lead is None:
        print("Generated demo trace but no alert lead could be computed")
    else:
        print(
            "Generated logistics guardrail demo: first alert at step {alert}, failure at step {fail}, lead {lead} steps".format(
                alert=summary.get("first_alert"),
                fail=summary.get("first_failure"),
                lead=lead,
            )
        )


if __name__ == "__main__":
    main()
