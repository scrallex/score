"""Reference agent loop that wires STM calls into a codebot run."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional

try:  # requests keeps the example ergonomic; fall back to urllib if absent.
    import requests
except ModuleNotFoundError:  # pragma: no cover
    requests = None  # type: ignore[assignment]

from stm_adapters.code_trace_adapter import CodeTokenizer, TokenBundle


@dataclass
class AgentStep:
    """Minimal representation of a code-agent step."""

    id: str
    action: str
    metadata: Mapping[str, object]
    timestamp: Optional[str] = None

    def as_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "id": self.id,
            "action": self.action,
            "metadata": dict(self.metadata),
        }
        if self.timestamp:
            payload["timestamp"] = self.timestamp
        return payload


class StreamingAgent:
    """Protocol-like base class for adapters that yield agent telemetry."""

    def run(self, task: Mapping[str, object]) -> Iterator[AgentStep]:  # pragma: no cover - example hook
        raise NotImplementedError

    def apply_patch(self, patch: str, context: Mapping[str, object]) -> None:  # pragma: no cover - example hook
        raise NotImplementedError

    def receive_warning(self, warning: Mapping[str, object]) -> None:  # pragma: no cover - example hook
        raise NotImplementedError


@dataclass
class STMClient:
    """Lightweight HTTP client for the STM coprocessor endpoints."""

    base_url: str
    timeout: float = 10.0
    session: Optional["requests.Session"] = None

    def _post(self, path: str, payload: Mapping[str, object]) -> Mapping[str, object]:
        if requests is None:  # pragma: no cover - guard for missing dependency
            raise RuntimeError("requests is required for STMClient")
        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"
        sess = self.session or requests
        response = sess.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def dilution(self, tokens: Iterable[str]) -> Mapping[str, object]:
        return self._post("stm/dilution", {"tokens": list(tokens)})

    def seen(self, window: Iterable[str]) -> Mapping[str, object]:
        return self._post("stm/seen", {"window": list(window)})

    def propose(self, seeds: Iterable[str]) -> Mapping[str, object]:
        return self._post("stm/propose", {"seeds": list(seeds)})


@dataclass
class ReferenceAgentLoop:
    """Reference integration pattern for running an agent with STM support."""

    agent: StreamingAgent
    stm: STMClient
    tokenizer: CodeTokenizer = field(default_factory=CodeTokenizer)
    guardrail_target: float = 0.12
    twin_accept_threshold: float = 0.6

    def run(self, task: Mapping[str, object]) -> Dict[str, object]:
        trace: List[Dict[str, object]] = []
        structural_windows: List[str] = []
        semantic_windows: List[str] = []
        foreground_hits = 0

        for step in self.agent.run(task):
            trace.append(step.as_dict())
            bundle = self.tokenizer.encode(step.as_dict())
            structural_windows.append(" ".join(bundle.structural))
            semantic_windows.append(" ".join(bundle.semantic))

            feedback = self._stm_feedback(bundle)
            if feedback.get("foreground"):
                foreground_hits += 1
                self._handle_foreground(step, feedback)
            elif feedback.get("warnings"):
                self.agent.receive_warning(feedback["warnings"])

        return {
            "trace": trace,
            "structural_windows": structural_windows,
            "semantic_windows": semantic_windows,
            "foreground_ratio": foreground_hits / max(len(trace), 1),
        }

    # Internal helpers ------------------------------------------------------------

    def _stm_feedback(self, bundle: TokenBundle) -> Mapping[str, object]:
        if not bundle.structural:
            return {}

        dilution = self.stm.dilution(bundle.structural)
        seen = self.stm.seen(bundle.structural)

        decisive = bool(dilution.get("decisive"))
        foreground = bool(seen.get("foreground"))
        feedback: Dict[str, object] = {
            "decisive": decisive,
            "foreground": foreground,
            "warnings": [],
        }
        if decisive and dilution.get("guardrail", 1.0) > self.guardrail_target:
            feedback["warnings"].append({
                "type": "guardrail",
                "message": "STM coverage drifting above guardrail target",
                "payload": dilution,
            })
        if foreground:
            tokens = seen.get("tokens", [])
            if isinstance(tokens, list) and tokens:
                proposals = self.stm.propose(tokens[:8])
                feedback["proposals"] = proposals
        return feedback

    def _handle_foreground(self, step: AgentStep, feedback: Mapping[str, object]) -> None:
        proposals = feedback.get("proposals")
        if not isinstance(proposals, Mapping):
            return
        matches = proposals.get("results") or []
        if not isinstance(matches, Iterable):
            return
        best_patch = None
        best_score = -1.0
        for result in matches:
            if not isinstance(result, Mapping):
                continue
            patch = result.get("patch") or result.get("snippet")
            score = float(result.get("patternability", 0.0))
            if patch and score > best_score:
                best_patch = patch
                best_score = score
        if best_patch and best_score >= self.twin_accept_threshold:
            self.agent.apply_patch(str(best_patch), {"step_id": step.id, "score": best_score})
        elif proposals:
            self.agent.receive_warning(
                {
                    "type": "twin",
                    "message": "STM twin available but below acceptance threshold",
                    "payload": proposals,
                }
            )


def persist_trace(structural_windows: List[str], semantic_windows: List[str], output_path: str) -> None:
    """Persist windows to disk using the adapter output convention."""

    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    base = path.stem
    structural_path = path.parent / f"{base}_struct.txt"
    semantic_path = path.parent / f"{base}_semantic.txt"

    structural_path.write_text("\n".join(structural_windows), encoding="utf-8")
    semantic_path.write_text("\n".join(semantic_windows), encoding="utf-8")

    manifest = {
        "source": base,
        "structural": structural_path.name,
        "semantic": semantic_path.name,
        "windows": len(structural_windows),
    }
    (path.parent / f"{base}_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
