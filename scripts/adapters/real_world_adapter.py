"""Adapters that convert real-world trace sources into STM-friendly artefacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import yaml

from scripts.features import CausalFeatureExtractor
from scripts.features.causal_features import _difference_score, _flatten_state


@dataclass
class NormalisedEvent:
    """Unified representation for planner events."""

    step: int
    action: str
    state: Mapping[str, Any]
    failure: bool = False
    metadata: Mapping[str, Any] | None = None


class RealWorldAdapter:
    """Build STM states from domain-specific telemetry."""

    def __init__(self, *, feature_extractor: Optional[CausalFeatureExtractor] = None) -> None:
        self.feature_extractor = feature_extractor or CausalFeatureExtractor()

    # ------------------------------------------------------------------
    # ROS motion planning
    # ------------------------------------------------------------------
    def from_ros_bag(self, bag_file: Path) -> Dict[str, Any]:
        payload = self._load_serialised(bag_file)
        events = self._extract_events(payload, action_key="action", state_key="state")
        return self._build_state(events, domain="ros", source=str(bag_file))

    # ------------------------------------------------------------------
    # Kubernetes orchestration
    # ------------------------------------------------------------------
    def from_kubernetes_logs(self, log_dir: Path) -> Dict[str, Any]:
        events: List[NormalisedEvent] = []
        step = 0
        for path in sorted(log_dir.glob("*.json")):
            data = self._load_serialised(path)
            for entry in self._coerce_iterable(data):
                action = str(entry.get("reason") or entry.get("action") or "unknown")
                state = entry.get("state") or entry
                failure = bool(entry.get("type", "").lower() in {"warning", "error"})
                events.append(
                    NormalisedEvent(
                        step=step,
                        action=action,
                        state=state,
                        failure=failure,
                        metadata={"node": entry.get("involvedObject", {}).get("name")},
                    )
                )
                step += 1
        return self._build_state(events, domain="kubernetes", source=str(log_dir))

    # ------------------------------------------------------------------
    # GitHub Actions workflows
    # ------------------------------------------------------------------
    def from_github_actions(self, workflow_runs: Path) -> Dict[str, Any]:
        data = self._load_serialised(workflow_runs)
        events = []
        step = 0
        for run in self._coerce_iterable(data.get("workflow_runs") if isinstance(data, Mapping) else data):
            steps = run.get("steps") or []
            for step_payload in self._coerce_iterable(steps):
                action = f"{run.get('name', 'run')}::{step_payload.get('name', 'step')}"
                state = {
                    "conclusion": step_payload.get("conclusion"),
                    "status": step_payload.get("status"),
                    "duration_ms": step_payload.get("duration_ms"),
                }
                failure = step_payload.get("conclusion") not in {None, "success"}
                events.append(
                    NormalisedEvent(
                        step=step,
                        action=action,
                        state=state,
                        failure=failure,
                        metadata={"run_id": run.get("id"), "url": run.get("html_url")},
                    )
                )
                step += 1
        return self._build_state(events, domain="github_actions", source=str(workflow_runs))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_state(
        self,
        events: Sequence[NormalisedEvent],
        *,
        domain: str,
        source: str,
    ) -> Dict[str, Any]:
        signals: List[Dict[str, Any]] = []
        history: List[Dict[str, Any]] = []
        failure_index: Optional[int] = None
        for event in events:
            metrics = self._compute_metrics(event, history)
            window: Dict[str, Any] = {
                "id": event.step,
                "window_start": event.step,
                "window_end": event.step + 1,
                "index": event.step,
                "metrics": metrics,
                "dilution": {
                    "path": metrics["resource_commitment_ratio"],
                    "signal": metrics["pattern_break_score"],
                },
                "action": event.action,
                "state": event.state,
                "failure": event.failure,
                "metadata": event.metadata,
            }
            features = self.feature_extractor.extract(window, history=history)
            window.setdefault("features", {})["causal"] = features
            window["metrics"].update({
                "coherence": metrics["coherence"],
                "entropy": metrics["entropy"],
                "stability": metrics["stability"],
            })
            signals.append(window)
            history.append(window)
            if event.failure and failure_index is None:
                failure_index = event.step

        return {
            "metadata": {
                "domain": domain,
                "source": source,
                "event_count": len(events),
            },
            "signals": signals,
            "failure_index": failure_index,
        }

    def _compute_metrics(
        self,
        event: NormalisedEvent,
        history: Sequence[Mapping[str, Any]],
    ) -> Dict[str, float]:
        history_states = [prev.get("state") for prev in history]
        coherence = self._coherence(event.state, history_states)
        entropy = self._entropy(event.action, history)
        stability = self._stability(event.failure, history)
        resource_commitment = self._resource_commitment_ratio(event.state)
        pattern_break = self._pattern_break(entropy, history)

        return {
            "coherence": coherence,
            "entropy": entropy,
            "stability": stability,
            "resource_commitment_ratio": resource_commitment,
            "pattern_break_score": pattern_break,
        }

    def _coherence(self, state: Mapping[str, Any], history_states: Sequence[Mapping[str, Any]]) -> float:
        if not history_states:
            return 0.5
        prev_flat = _flatten_state(history_states[-1])
        curr_flat = _flatten_state(state)
        diff = _difference_score(curr_flat, prev_flat)
        return max(0.0, 1.0 - diff)

    def _entropy(self, action: str, history: Sequence[Mapping[str, Any]]) -> float:
        seen = {window.get("action") for window in history if window.get("action") is not None}
        seen.add(action)
        unique = len(seen)
        total = len(history) + 1
        return min(1.0, unique / max(total, 1))

    def _stability(self, failure: bool, history: Sequence[Mapping[str, Any]]) -> float:
        failure_count = sum(1 for window in history if window.get("failure")) + (1 if failure else 0)
        total = len(history) + 1
        return max(0.0, 1.0 - failure_count / max(total, 1))

    def _resource_commitment_ratio(self, state: Mapping[str, Any]) -> float:
        flat = _flatten_state(state)
        if not flat:
            return 0.0
        locked_keys = [key for key in flat if "lock" in key.lower() or "allocated" in key.lower()]
        if not locked_keys:
            return min(1.0, sum(abs(value) for value in flat.values()) / (len(flat) * 10.0))
        locked = sum(abs(float(flat[key])) for key in locked_keys)
        total = sum(abs(value) for value in flat.values()) or 1.0
        return min(1.0, locked / total)

    def _pattern_break(self, entropy: float, history: Sequence[Mapping[str, Any]]) -> float:
        entropy_history = [float(window.get("metrics", {}).get("entropy", 0.0)) for window in history]
        entropy_history.append(entropy)
        if len(entropy_history) < 2:
            return entropy
        deltas = [abs(entropy_history[idx] - entropy_history[idx - 1]) for idx in range(1, len(entropy_history))]
        magnitude = sum(deltas) / len(deltas)
        return min(1.0, max(entropy, magnitude))

    @staticmethod
    def _load_serialised(path: Path) -> Any:
        suffix = path.suffix.lower()
        if suffix in {".json", ".ndjson", ".jsonl"}:
            text = path.read_text(encoding="utf-8")
            if suffix in {".ndjson", ".jsonl"}:
                return [json.loads(line) for line in text.splitlines() if line.strip()]
            return json.loads(text)
        if suffix in {".yaml", ".yml"}:
            return yaml.safe_load(path.read_text(encoding="utf-8"))
        raise ValueError(f"Unsupported file type: {path}")

    @staticmethod
    def _extract_events(
        payload: Any,
        *,
        action_key: str,
        state_key: str,
    ) -> List[NormalisedEvent]:
        events: List[NormalisedEvent] = []
        step = 0
        for entry in RealWorldAdapter._coerce_iterable(payload):
            action = str(_dig(entry, action_key) or "unknown_action")
            state = _dig(entry, state_key) or entry
            failure = bool(entry.get("failure") or entry.get("status") in {"FAILED", "ERROR"})
            events.append(NormalisedEvent(step=step, action=action, state=state, failure=failure, metadata=entry))
            step += 1
        return events

    @staticmethod
    def _coerce_iterable(payload: Any) -> Iterable[Mapping[str, Any]]:
        if payload is None:
            return []
        if isinstance(payload, Mapping):
            if "events" in payload:
                return RealWorldAdapter._coerce_iterable(payload["events"])
            return [payload]
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, Mapping)]
        return []


def _dig(mapping: Mapping[str, Any], dotted: str) -> Optional[Any]:
    parts = dotted.split(".")
    current: Any = mapping
    for part in parts:
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current
