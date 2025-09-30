
#!/usr/bin/env python3
"""Generate whitepaper figures for the logistics guardrail demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt

ACCENT = '#38bdf8'
ALERT = '#f97316'
FAIL = '#ef4444'
SUCCESS = '#22c55e'
PATH_COLOR = '#c084fc'
GRID_COLOR = '#1f2937'


def load_timeline(path: Path) -> dict:
    data = json.loads(path.read_text(encoding='utf-8'))
    if 'signal_summary' not in data:
        raise ValueError('timeline JSON missing signal_summary section')
    return data


def _step_colors(statuses: Sequence[str], *, alert_steps: set[int] | None = None) -> list[str]:
    colors = []
    for idx, status in enumerate(statuses):
        if alert_steps and idx in alert_steps:
            colors.append(ALERT)
        elif status == 'fail':
            colors.append(FAIL)
        else:
            colors.append(SUCCESS)
    return colors


def _render_dashboard(timeline: dict, output: Path) -> None:
    classical = timeline.get('classical_validator', [])
    signal_summary = timeline['signal_summary']
    rows = signal_summary.get('rows', [])
    alert_steps = set(signal_summary.get('alert_steps', []))
    lead_time = signal_summary.get('lead_time')
    first_alert = signal_summary.get('first_alert')
    first_failure = signal_summary.get('first_failure')
    threshold = signal_summary.get('thresholds', {}).get('lambda_hazard')

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    fig.suptitle('Logistics Guardrail — Classical vs STM', fontsize=16, weight='bold')

    # Classical panel
    ax_left = axes[0]
    statuses = [entry.get('status', 'pending') for entry in classical]
    colors = _step_colors(statuses)
    steps = list(range(len(classical)))
    ax_left.barh(steps, [1] * len(steps), color=colors, edgecolor='none')
    ax_left.set_title('Classical Validator', fontsize=13)
    ax_left.set_xlim(0, 1)
    ax_left.set_xticks([])
    ax_left.set_yticks(steps)
    ax_left.set_yticklabels([f'Step {step}' for step in steps])
    ax_left.invert_yaxis()
    ax_left.grid(axis='y', color='0.2', alpha=0.3, linestyle='--')
    if classical:
        failure = classical[-1]
        if failure.get('status') == 'fail':
            message = failure.get('message', 'Plan failed post-hoc validation')
            ax_left.text(0.02, len(steps) - 0.5, message, color=FAIL, fontsize=10, ha='left', va='center')

    ax_left.text(0.02, -0.8, 'Verdict emitted after execution completes', fontsize=10, color='0.6')

    # STM panel
    ax_right = axes[1]
    guardrail_status = ['alert' if row.get('alert') else 'ok' for row in rows]
    guardrail_colors = _step_colors(guardrail_status, alert_steps=alert_steps)
    steps_guardrail = list(range(len(rows)))
    ax_right.barh(steps_guardrail, [1] * len(steps_guardrail), color=guardrail_colors, edgecolor='none')
    ax_right.set_title('STM Guardrail', fontsize=13)
    ax_right.set_xlim(0, 1)
    ax_right.set_xticks([])
    ax_right.set_yticks(steps_guardrail)
    ax_right.set_yticklabels([f'Step {step}' for step in steps_guardrail])
    ax_right.invert_yaxis()
    ax_right.grid(axis='y', color='0.2', alpha=0.3, linestyle='--')

    for idx, row in enumerate(rows):
        hazard = float(row.get('lambda_hazard', 0.0))
        label = f"lambda={hazard:.3f}"
        ax_right.text(0.02, idx, label, va='center', ha='left', fontsize=9, color='0.75')

    summary_lines = []
    if first_alert is not None:
        summary_lines.append(f'Alert at step {first_alert}')
    if first_failure is not None:
        summary_lines.append(f'Failure at step {first_failure}')
    if lead_time is not None:
        summary_lines.append(f'Lead time {lead_time} steps')
    if threshold is not None:
        summary_lines.append(f'Hazard threshold {threshold:.3f}')
    summary_text = '\n'.join(summary_lines)
    ax_right.text(0.02, -0.8, summary_text, fontsize=10, color='0.7', ha='left', va='top')

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches='tight')
    plt.close(fig)


def _render_hazard_chart(timeline: dict, output: Path) -> None:
    signal_summary = timeline['signal_summary']
    rows = signal_summary.get('rows', [])
    event_step = timeline.get('event_step')
    alert_steps = set(signal_summary.get('alert_steps', []))
    first_failure = signal_summary.get('first_failure')
    threshold = signal_summary.get('thresholds', {}).get('lambda_hazard')

    steps = [row.get('step', idx) for idx, row in enumerate(rows)]
    hazard = [float(row.get('lambda_hazard', 0.0)) for row in rows]
    path = [float(row.get('path_dilution', 0.0)) for row in rows]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.set_title('Hazard lambda and Path Dilution Over Time', fontsize=14, weight='bold')
    ax.plot(steps, hazard, color=ACCENT, linewidth=2.4, label='lambda hazard')
    ax.plot(steps, path, color=PATH_COLOR, linewidth=1.8, linestyle='--', label='Path dilution')

    if threshold is not None:
        ax.axhline(threshold, color='#94a3b8', linestyle=':', linewidth=1.4, label=f'Threshold {threshold:.3f}')
    if isinstance(event_step, int):
        ax.axvline(event_step, color=ACCENT, linestyle=':', alpha=0.4, linewidth=1)
        ax.text(event_step, ax.get_ylim()[1], ' Disruption', color=ACCENT, fontsize=9, va='top')
    if isinstance(first_failure, int):
        ax.axvline(first_failure, color=FAIL, linestyle='--', alpha=0.5, linewidth=1)
        ax.text(first_failure, ax.get_ylim()[1], ' Failure', color=FAIL, fontsize=9, va='top')

    for idx, value in enumerate(hazard):
        if idx in alert_steps:
            ax.scatter(steps[idx], value, color=ALERT, s=60, zorder=5, label='Alert' if idx == min(alert_steps) else '')

    ax.set_xlabel('Plan step')
    ax.set_ylabel('Metric value')
    ax.grid(alpha=0.2)
    ax.legend(loc='upper left', frameon=False)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches='tight')
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot logistics guardrail figures for documentation')
    parser.add_argument('--timeline', type=Path, required=True, help='Path to timeline.json generated by the demo')
    parser.add_argument('--output-dir', type=Path, required=True, help='Directory to write figure PNGs')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timeline = load_timeline(args.timeline)
    dashboard_path = args.output_dir / 'logistics_guardrail_dashboard.png'
    hazard_path = args.output_dir / 'logistics_guardrail_hazard.png'
    _render_dashboard(timeline, dashboard_path)
    _render_hazard_chart(timeline, hazard_path)
    print(json.dumps({
        'dashboard': str(dashboard_path),
        'hazard': str(hazard_path)
    }, indent=2))


if __name__ == '__main__':
    main()
