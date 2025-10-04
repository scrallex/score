"""Minimal logistics guardrail demo used for unit testing.

The real project wires a rich simulation and reporting pipeline.  For
our test harness we only need deterministic artefacts so the rest of the
stack (backtester, reporting) has something to exercise.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence


_DEFAULT_SIGNAL_SUMMARY: Dict[str, int] = {
    "lead_time": 3,
    "first_alert": 5,
    "first_failure": 8,
}

_DEFAULT_TWIN_SUGGESTIONS: Sequence[Mapping[str, object]] = (
    {
        "id": "demo-suggestion-1",
        "score": 0.83,
        "keywords": ["reroute", "bridge", "supply"],
        "summary": "Historical recovery playbook with matching hazard profile.",
    },
)


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_dashboard(path: Path, summary: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Lightweight HTML shell so downstream consumers can open the file if
    # they wish, but we keep it deliberately small for tests.
    path.write_text(
        """<!doctype html><meta charset='utf-8'>
<title>Logistics Guardrail Demo</title>
<body>
  <h1>Logistics Guardrail Demo</h1>
  <p>Lead time: {lead_time}</p>
  <p>First alert: {first_alert}</p>
  <p>First failure: {first_failure}</p>
</body>
""".format(**summary),
        encoding="utf-8",
    )


def generate_demo(args) -> Dict[str, object]:
    """Produce deterministic demo artefacts for the tests.

    Parameters mirror the CLI entry point used in the real repository;
    only ``output_root`` and ``twin_state`` are relevant for the test
    suite.  Additional attributes are accepted and ignored so the stub
    stays compatible with richer argument objects.
    """

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    twin_state = Path(getattr(args, "twin_state", output_root / "twin_state.json"))
    if not twin_state.exists():
        raise FileNotFoundError(f"Twin state not found: {twin_state}")

    signal_summary = dict(_DEFAULT_SIGNAL_SUMMARY)
    twin_payload = {"suggestions": list(_DEFAULT_TWIN_SUGGESTIONS)}

    timeline = {
        "signal_summary": signal_summary,
        "twin": twin_payload,
        "meta": {
            "twin_state": str(twin_state),
            "stride": getattr(args, "stride", None),
            "window_bytes": getattr(args, "window_bytes", None),
        },
    }

    _write_json(output_root / "timeline.json", timeline)
    _write_dashboard(output_root / "dashboard.html", signal_summary)

    summary = {
        "lead_time": signal_summary["lead_time"],
        "first_alert": signal_summary["first_alert"],
        "first_failure": signal_summary["first_failure"],
        "twin_suggestions": twin_payload["suggestions"],
    }

    _write_json(output_root / "summary.json", summary)
    return summary


__all__ = ["generate_demo"]
