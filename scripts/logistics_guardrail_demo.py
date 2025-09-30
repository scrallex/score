#!/usr/bin/env python3
"""Generate a logistics guardrail demo trace and STM metrics."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from sep_text_manifold import analyse_directory, summarise_guardrail, suggest_twin_action, TwinSuggestion
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



def _normalise_signature(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _filtered_keywords(tokens: Sequence[str], limit: int) -> List[str]:
    words: List[str] = []
    for token in tokens:
        if not token:
            continue
        if "__" in token or token.startswith("action"):
            continue
        cleaned = token.replace("_", " ").strip()
        if not cleaned:
            continue
        if cleaned in words:
            continue
        words.append(cleaned)
        if len(words) >= max(20, limit * 4):
            break
    if not words:
        return []
    words.sort(key=lambda item: (-len(item), item.lower()))
    return words[:limit]


def _headline_from_keywords(keywords: Sequence[str]) -> str:
    if not keywords:
        return "Nearest precedent available for review."
    if len(keywords) == 1:
        return f"Stabilise around {keywords[0]}."
    if len(keywords) == 2:
        return f"Bridge {keywords[0]} via {keywords[1]}."
    joined = " -> ".join(keywords[:3])
    return f"Precedent emphasises {joined}."


def _serialise_twin(suggestion: TwinSuggestion, *, keyword_limit: int) -> Dict[str, Any]:
    keywords = _filtered_keywords(suggestion.tokens, keyword_limit)
    metrics = {key: float(value) for key, value in suggestion.metrics.items()}
    payload = {
        "window_id": suggestion.window_id,
        "signature": _normalise_signature(suggestion.signature),
        "distance": float(suggestion.distance),
        "metrics": metrics,
        "keywords": keywords,
        "headline": _headline_from_keywords(keywords),
    }
    return payload


def _recommend_twins(
    *,
    alert_row: Dict[str, Any],
    twin_state: Dict[str, Any],
    top_k: int,
    max_distance: float,
    match_signature: bool,
    keyword_limit: int,
) -> List[Dict[str, Any]]:
    invalid_action: Dict[str, Any] = {
        "metrics": {
            "coherence": float(alert_row.get("coherence", 0.0)),
            "stability": float(alert_row.get("stability", 0.0)),
            "entropy": float(alert_row.get("entropy", 0.0)),
            "rupture": float(alert_row.get("lambda_hazard", 0.0)),
            "lambda_hazard": float(alert_row.get("lambda_hazard", 0.0)),
        }
    }
    window_id = alert_row.get("window_id")
    if window_id is not None:
        invalid_action["window_id"] = window_id
    signature = _normalise_signature(alert_row.get("signature"))
    if signature is not None:
        invalid_action["signature"] = signature

    suggestions = suggest_twin_action(
        invalid_action,
        twin_state,
        top_k=top_k,
        max_distance=max_distance,
        match_signature=match_signature,
        top_tokens=max(12, keyword_limit * 4),
    )
    return [
        _serialise_twin(suggestion, keyword_limit=keyword_limit)
        for suggestion in suggestions
    ]


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
      --path: #c084fc;
      --alert: #f97316;
      --event: rgba(56, 189, 248, 0.65);
      --failure: #ef4444;
      --failure-band: rgba(239, 68, 68, 0.12);
      --event-band: rgba(56, 189, 248, 0.08);
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
    .panel h3 {{
      margin: 24px 0 12px;
      font-size: 13px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: rgba(226,232,240,0.82);
    }}
    .chart-card {{
      margin-top: 18px;
      background: rgba(15,23,42,0.55);
      border: 1px solid rgba(148,163,184,0.12);
      border-radius: 12px;
      padding: 16px 18px;
    }}
    .chart-card canvas {{
      width: 100%;
      height: 220px;
      display: block;
    }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 16px;
      margin-top: 12px;
      font-size: 12px;
      color: var(--muted);
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    .legend-item {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }}
    .legend-item::before {{
      content: '';
      width: 12px;
      height: 12px;
      border-radius: 50%;
      background: currentColor;
      opacity: 0.88;
    }}
    .legend-item.hazard {{ color: var(--accent); }}
    .legend-item.path {{ color: var(--path); }}
    .legend-item.threshold {{ color: rgba(148,163,184,0.82); }}
    .legend-item.event {{ color: var(--event); }}
    .legend-item.failure {{ color: var(--failure); }}
    .classic-note {{
      margin-top: 14px;
      font-size: 13px;
      color: var(--muted);
      line-height: 1.5;
    }}
    .twin-card {{
      margin-top: 18px;
      padding: 18px;
      background: rgba(20,31,53,0.62);
      border: 1px solid rgba(56,189,248,0.18);
      border-radius: 12px;
    }}
    .twin-card p {{
      margin: 0 0 12px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }}
    .twin-entry {{
      margin-bottom: 14px;
      padding-bottom: 12px;
      border-bottom: 1px solid rgba(148,163,184,0.12);
    }}
    .twin-entry:last-child {{
      margin-bottom: 0;
      padding-bottom: 0;
      border-bottom: none;
    }}
    .twin-entry .headline {{
      font-size: 13px;
      font-weight: 600;
      color: var(--text);
      letter-spacing: 0.04em;
    }}
    .twin-entry .meta {{
      font-size: 12px;
      color: var(--muted);
      margin-top: 4px;
      letter-spacing: 0.06em;
    }}
    .twin-entry .keywords {{
      margin-top: 6px;
      font-size: 12px;
      color: rgba(148,163,184,0.85);
      letter-spacing: 0.04em;
    }}
    .twin-empty {{
      color: rgba(148,163,184,0.75);
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
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
      <div class=\"classic-note\" id=\"classic-note\"></div>
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
      <div class=\"chart-card\">
        <canvas id=\"metric-chart\"></canvas>
        <div class=\"legend\">
          <span class=\"legend-item hazard\">λ hazard</span>
          <span class=\"legend-item path\">Path dilution</span>
          <span class=\"legend-item threshold\">λ threshold</span>
          <span class=\"legend-item event\">Disruption</span>
          <span class=\"legend-item failure\">Failure check</span>
        </div>
      </div>
      <div class=\"timeline\" id=\"stm-timeline\"></div>
      <div class=\"note\" id=\"stm-note\"></div>
      <div class=\"twin-card\">
        <h3>Recovery Recommendation</h3>
        <p id=\"twin-summary\"></p>
        <div id=\"twin-recovery\"></div>
      </div>
    </section>
  </div>
  <script id="timeline-data" type="application/json">{timeline_json}</script>
  <script>
    const data = JSON.parse(document.getElementById('timeline-data').textContent);
    const classical = data.classical_validator || [];
    const signalSummary = data.signal_summary || {{}};
    const rows = signalSummary.rows || [];
    const alertSteps = new Set(signalSummary.alert_steps || []);
    const failureSteps = new Set(signalSummary.failure_steps || []);
    const eventStep = data.event_step;
    const twin = data.twin || {{}};
    const twinSuggestions = twin.suggestions || [];
    const firstAlert = signalSummary.first_alert;
    const firstFailure = signalSummary.first_failure;
    const lead = signalSummary.lead_time;
    const hazardThreshold = (signalSummary.thresholds || {{}}).lambda_hazard;
    const classicNote = document.getElementById('classic-note');

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
    if (classicNote) {{
      if (!classical.length) {{
        classicNote.textContent = 'No classical validation data was available for this trace.';
      }} else if (firstFailure === null || firstFailure === undefined) {{
        classicNote.textContent = 'Classical validator did not report a terminal failure within the current horizon.';
      }} else if (firstAlert === null || firstAlert === undefined) {{
        classicNote.textContent = 'Validator reports the first failure at step ' + firstFailure + ' with no prior hazard telemetry.';
      }} else {{
        const lag = Math.max(0, firstFailure - firstAlert);
        const leadDescriptor = lag === 0 ? 'simultaneously with STM alerting.' : (lag + ' step lag behind STM.');
        classicNote.textContent = 'Validator flags the run at step ' + firstFailure + ', ' + leadDescriptor;
      }}
    }}

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

    document.getElementById('stm-lead').textContent = lead ? (lead + ' steps') : '0 steps';
    document.getElementById('stm-first-alert').textContent = firstAlert === null || firstAlert === undefined ? '–' : ('#' + firstAlert);
    document.getElementById('stm-threshold').textContent = hazardThreshold === undefined ? '0.000' : Number.parseFloat(hazardThreshold).toFixed(3);

    const note = document.getElementById('stm-note');
    if (firstAlert === null || firstAlert === undefined) {{
      note.textContent = 'No foreground alert fired within the analysed horizon.';
    }} else {{
      const leadDelta = firstFailure === null || firstFailure === undefined ? null : firstFailure - firstAlert;
      const detail = leadDelta === null ? 'Alert raised before failure check.' : (leadDelta + ' step lead-time before failure.');
      note.textContent = 'STM flagged a structural deviation at step ' + firstAlert + ', ' + detail;
    }}

    const chartCanvas = document.getElementById('metric-chart');
    function renderChart() {{
      if (!chartCanvas || !rows.length) {{
        return;
      }}
      const rootStyles = getComputedStyle(document.documentElement);
      const accent = rootStyles.getPropertyValue('--accent').trim() || '#38bdf8';
      const alertColor = rootStyles.getPropertyValue('--alert').trim() || '#f97316';
      const pathColor = rootStyles.getPropertyValue('--path').trim() || '#c084fc';
      const guideColor = 'rgba(148,163,184,0.18)';
      const textColor = rootStyles.getPropertyValue('--muted').trim() || '#94a3b8';
      const eventBand = rootStyles.getPropertyValue('--event-band').trim() || 'rgba(56,189,248,0.08)';
      const failureBand = rootStyles.getPropertyValue('--failure-band').trim() || 'rgba(239,68,68,0.12)';
      const dpr = window.devicePixelRatio || 1;
      const rect = chartCanvas.getBoundingClientRect();
      const width = (rect.width || chartCanvas.clientWidth || 640);
      const height = 220;
      chartCanvas.width = width * dpr;
      chartCanvas.height = height * dpr;
      const ctx = chartCanvas.getContext('2d');
      if (!ctx) {{
        return;
      }}
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, width * dpr, height * dpr);
      ctx.scale(dpr, dpr);

      const padding = {{ left: 48, right: 20, top: 16, bottom: 28 }};
      const chartWidth = width - padding.left - padding.right;
      const chartHeight = height - padding.top - padding.bottom;
      if (chartWidth <= 0 || chartHeight <= 0) {{
        return;
      }}

      const lambdaValues = rows.map((row) => Number(row.lambda_hazard || 0));
      const pathValues = rows.map((row) => Number(row.path_dilution || 0));
      const seriesValues = lambdaValues.concat(pathValues);
      if (typeof hazardThreshold === 'number') {{
        seriesValues.push(Number(hazardThreshold));
      }}
      seriesValues.push(0);
      const maxVal = Math.max(...seriesValues);
      const minVal = Math.min(...seriesValues);
      const span = maxVal - minVal || 1;
      const xStep = rows.length <= 1 ? 0 : chartWidth / (rows.length - 1);
      const xFor = (idx) => padding.left + idx * xStep;
      const yFor = (value) => padding.top + chartHeight - ((value - minVal) / span) * chartHeight;

      // grid lines
      ctx.strokeStyle = guideColor;
      ctx.lineWidth = 1;
      ctx.font = '12px "Inter", "Segoe UI", sans-serif';
      ctx.fillStyle = textColor;
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      const gridLevels = 4;
      for (let i = 0; i <= gridLevels; i += 1) {{
        const value = minVal + (span / gridLevels) * i;
        const y = yFor(value);
        ctx.beginPath();
        ctx.moveTo(padding.left, y);
        ctx.lineTo(width - padding.right, y);
        ctx.stroke();
        ctx.fillText(value.toFixed(2), padding.left - 8, y);
      }}

      if (typeof hazardThreshold === 'number' && !Number.isNaN(hazardThreshold)) {{
        const yHaz = yFor(hazardThreshold);
        ctx.setLineDash([6, 6]);
        ctx.strokeStyle = 'rgba(148,163,184,0.55)';
        ctx.beginPath();
        ctx.moveTo(padding.left, yHaz);
        ctx.lineTo(width - padding.right, yHaz);
        ctx.stroke();
        ctx.setLineDash([]);
      }}

      if (typeof eventStep === 'number' && eventStep >= 0 && eventStep < rows.length) {{
        const x = xFor(eventStep);
        ctx.fillStyle = eventBand;
        ctx.fillRect(x - Math.max(2, xStep * 0.25), padding.top, Math.max(4, xStep * 0.5), chartHeight);
      }}

      if (typeof firstFailure === 'number' && firstFailure >= 0 && firstFailure < rows.length) {{
        const x = xFor(firstFailure);
        ctx.fillStyle = failureBand;
        ctx.fillRect(x - Math.max(2, xStep * 0.25), padding.top, Math.max(4, xStep * 0.5), chartHeight);
      }}

      ctx.strokeStyle = accent;
      ctx.lineWidth = 2;
      ctx.beginPath();
      lambdaValues.forEach((value, idx) => {{
        const x = xFor(idx);
        const y = yFor(value);
        if (idx === 0) {{
          ctx.moveTo(x, y);
        }} else {{
          ctx.lineTo(x, y);
        }}
      }});
      ctx.stroke();

      ctx.strokeStyle = pathColor;
      ctx.lineWidth = 2;
      ctx.beginPath();
      pathValues.forEach((value, idx) => {{
        const x = xFor(idx);
        const y = yFor(value);
        if (idx === 0) {{
          ctx.moveTo(x, y);
        }} else {{
          ctx.lineTo(x, y);
        }}
      }});
      ctx.stroke();

      lambdaValues.forEach((value, idx) => {{
        if (!alertSteps.has(rows[idx].step)) {{
          return;
        }}
        const x = xFor(idx);
        const y = yFor(value);
        ctx.fillStyle = alertColor;
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fill();
      }});
    }}

    renderChart();
    window.addEventListener('resize', renderChart);

    const twinSummary = document.getElementById('twin-summary');
    const twinContainer = document.getElementById('twin-recovery');
    if (twinContainer) {{
      twinContainer.innerHTML = '';
      if (!twinSuggestions.length) {{
        if (twinSummary) {{
          twinSummary.textContent = 'No cached twin corpus supplied for this run.';
        }}
        const empty = document.createElement('div');
        empty.className = 'twin-empty';
        empty.textContent = firstAlert === null || firstAlert === undefined ? 'Awaiting hazard before twin lookup.' : 'No twin suggestions available for the alert window.';
        twinContainer.appendChild(empty);
      }} else {{
        if (twinSummary) {{
          const source = twin.source || 'cached corpus';
          const distance = twinSuggestions[0].distance;
          const message = typeof distance === 'number' ? `Nearest precedent from ${{source}} (distance ${{distance.toFixed(3)}}).` : `Nearest precedent from ${{source}}.`;
          twinSummary.textContent = message;
        }}
        twinSuggestions.forEach((entry) => {{
          const wrapper = document.createElement('div');
          wrapper.className = 'twin-entry';
          const headline = document.createElement('div');
          headline.className = 'headline';
          headline.textContent = entry.headline || 'Twin suggestion';
          wrapper.appendChild(headline);
          const meta = document.createElement('div');
          meta.className = 'meta';
          const metrics = entry.metrics || {{}};
          const lambdaValue = metrics.lambda_hazard === undefined ? '0.000' : Number(metrics.lambda_hazard).toFixed(3);
          const distance = entry.distance === undefined ? 'N/A' : Number(entry.distance).toFixed(3);
          const signatureText = entry.signature ? ' · ' + entry.signature : '';
          meta.textContent = 'lambda=' + lambdaValue + ' · dist=' + distance + signatureText;
          wrapper.appendChild(meta);
          if (Array.isArray(entry.keywords) && entry.keywords.length) {{
            const keywords = document.createElement('div');
            keywords.className = 'keywords';
            keywords.textContent = 'Keywords: ' + entry.keywords.join(' · ');
            wrapper.appendChild(keywords);
          }}
          twinContainer.appendChild(wrapper);
        }});
      }}
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
    signal_summary = summarise_guardrail(
        signals=analysis.signals,
        transitions=transitions,
        event_index=event_index,
    )

    twin_payload: Dict[str, Any] = {"suggestions": []}
    twin_state_path = getattr(args, "twin_state", None)
    if twin_state_path:
        twin_path = Path(twin_state_path)
        if twin_path.exists():
            twin_state_data = json.loads(twin_path.read_text(encoding="utf-8"))
            first_alert = signal_summary.get("first_alert")
            rows = signal_summary.get("rows", [])
            if isinstance(first_alert, int) and 0 <= first_alert < len(rows):
                alert_row = rows[first_alert]
                suggestions = _recommend_twins(
                    alert_row=alert_row,
                    twin_state=twin_state_data,
                    top_k=max(1, int(getattr(args, "twin_top_k", 3))),
                    max_distance=float(getattr(args, "twin_max_distance", 0.2)),
                    match_signature=bool(getattr(args, "twin_match_signature", False)),
                    keyword_limit=max(2, int(getattr(args, "twin_keyword_limit", 4))),
                )
                twin_payload = {
                    "suggestions": suggestions,
                    "source": twin_path.name,
                    "source_path": str(twin_path),
                }
            else:
                twin_payload = {
                    "suggestions": [],
                    "source": twin_path.name,
                    "source_path": str(twin_path),
                    "message": "No alert fired within the analysed horizon.",
                }
                if getattr(args, "verbose", False):
                    print("[twin] alert window not reached; skipping twin lookup", flush=True)
        else:
            twin_payload = {
                "suggestions": [],
                "source": str(twin_path),
                "missing": True,
            }
            if getattr(args, "verbose", False):
                print(f"[twin] twin state not found: {twin_path}", flush=True)
    timeline_path = output_root / "timeline.json"
    timeline_payload = {
        "transitions": transitions,
        "signal_summary": signal_summary,
        "classical_validator": _build_classical_timeline(transitions),
        "event_step": event_index,
        "twin": twin_payload,
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
        "twin_suggestions": twin_payload.get("suggestions", []),
        "twin_source": twin_payload.get("source_path", twin_payload.get("source")),
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
    parser.add_argument(
        "--twin-state",
        type=Path,
        default=None,
        help="Optional STM state JSON to query for recovery twins",
    )
    parser.add_argument(
        "--twin-top-k",
        type=int,
        default=3,
        help="Maximum number of twin suggestions to surface",
    )
    parser.add_argument(
        "--twin-max-distance",
        type=float,
        default=0.2,
        help="Euclidean distance ceiling applied when matching twins",
    )
    parser.add_argument(
        "--twin-match-signature",
        action="store_true",
        help="Require matching STM signatures for twin candidates",
    )
    parser.add_argument(
        "--twin-keyword-limit",
        type=int,
        default=4,
        help="Number of keywords to display per twin suggestion",
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
