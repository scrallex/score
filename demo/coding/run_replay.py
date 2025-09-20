#!/usr/bin/env python3
"""Replay a recorded trace through the reference STM agent loop."""

from __future__ import annotations

import argparse
import json
import os
import importlib.util
import sys
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


code_adapter_module = _load_module(
    "stm_adapters.code_trace_adapter", SRC_ROOT / "stm_adapters" / "code_trace_adapter.py"
)
reference_loop_module = _load_module(
    "stm_demo.reference_agent_loop", SRC_ROOT / "stm_demo" / "reference_agent_loop.py"
)

CodeTraceAdapter = getattr(code_adapter_module, "CodeTraceAdapter")
AgentStep = getattr(reference_loop_module, "AgentStep")
ReferenceAgentLoop = getattr(reference_loop_module, "ReferenceAgentLoop")
STMClient = getattr(reference_loop_module, "STMClient")
StreamingAgent = getattr(reference_loop_module, "StreamingAgent")
persist_trace = getattr(reference_loop_module, "persist_trace")

TASK_ROOT = Path(__file__).resolve().parent / "tasks"
OUTPUT_ROOT = Path(__file__).resolve().parent / "output"


class RecordedAgent(StreamingAgent):
    """Streams recorded steps back into the agent loop."""

    def __init__(self, steps: List[Mapping[str, object]]) -> None:
        self._steps = steps
        self.applied_patches: List[Mapping[str, object]] = []
        self.warnings: List[Mapping[str, object]] = []

    def run(self, task: Mapping[str, object]) -> Iterator[AgentStep]:
        for payload in self._steps:
            metadata = payload.get("metadata")
            if not isinstance(metadata, Mapping):
                metadata = {}
            yield AgentStep(
                id=str(payload.get("id", "")),
                action=str(payload.get("action", "")),
                metadata=metadata,
                timestamp=payload.get("timestamp") if isinstance(payload.get("timestamp"), str) else None,
            )

    def apply_patch(self, patch: str, context: Mapping[str, object]) -> None:
        self.applied_patches.append({"patch": patch, "context": dict(context)})

    def receive_warning(self, warning: Mapping[str, object]) -> None:
        self.warnings.append(dict(warning))


class NullSTMClient:
    """Fallback STM client that yields no foreground decisions."""

    def dilution(self, tokens: Iterable[str]) -> Mapping[str, object]:
        return {"decisive": False, "guardrail": 0.0, "tokens": list(tokens)}

    def seen(self, window: Iterable[str]) -> Mapping[str, object]:
        return {"foreground": False, "tokens": list(window)}

    def propose(self, seeds: Iterable[str]) -> Mapping[str, object]:
        return {"results": [], "seeds": list(seeds)}


def load_trace(task: str, variant: str) -> List[Mapping[str, object]]:
    path = TASK_ROOT / task / f"{variant}.jsonl"
    text = path.read_text(encoding="utf-8").strip()
    steps: List[Mapping[str, object]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        if isinstance(data, Mapping):
            steps.append(data)
    return steps


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a trace with the reference STM loop")
    parser.add_argument("--task", required=True, help="Task directory name")
    parser.add_argument("--variant", choices=("baseline", "stm"), default="stm")
    parser.add_argument("--persist", action="store_true", help="Persist STM-formatted windows under output/")
    args = parser.parse_args()

    steps = load_trace(args.task, args.variant)
    agent = RecordedAgent(steps)

    base_url = os.environ.get("STM_BASE_URL")
    if base_url:
        stm_client = STMClient(base_url=base_url)
    else:
        stm_client = NullSTMClient()

    loop = ReferenceAgentLoop(agent=agent, stm=stm_client)
    result = loop.run({"task": args.task, "variant": args.variant})

    print(json.dumps({
        "task": args.task,
        "variant": args.variant,
        "steps": result["trace"],
        "foreground_ratio": result["foreground_ratio"],
        "warnings": agent.warnings,
        "applied_patches": agent.applied_patches,
    }, indent=2))

    if args.persist:
        windows_dir = OUTPUT_ROOT / args.task / args.variant
        windows_dir.mkdir(parents=True, exist_ok=True)
        persist_trace(result["structural_windows"], result["semantic_windows"], str(windows_dir / "replay"))

        adapter = CodeTraceAdapter()
        trace_path = TASK_ROOT / args.task / f"{args.variant}.jsonl"
        adapter.run(trace_path, windows_dir / "adapter")


if __name__ == "__main__":
    main()
