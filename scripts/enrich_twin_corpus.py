#!/usr/bin/env python3
"""Merge additional STM states into a base twin corpus."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple


def load_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"State file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _next_id(signals: Iterable[Mapping[str, Any]]) -> int:
    max_id = -1
    for sig in signals:
        try:
            sig_id = int(sig.get("id", sig.get("index", -1)))
        except (TypeError, ValueError):
            continue
        if sig_id > max_id:
            max_id = sig_id
    return max_id + 1


def merge_state(
    base_state: Dict[str, Any],
    extra_state: Dict[str, Any],
    *,
    source: Path,
    note: str | None = None,
) -> Dict[str, Any]:
    base_signals: List[Dict[str, Any]] = base_state.setdefault("signals", [])  # type: ignore[assignment]
    next_id = _next_id(base_signals)
    id_map: Dict[int, int] = {}
    added_signals = 0

    for sig in extra_state.get("signals", []):
        sig_copy = copy.deepcopy(sig)
        try:
            previous_id = int(sig_copy.get("id", sig_copy.get("index", next_id)))
        except (TypeError, ValueError):
            continue
        mapped_id = next_id
        next_id += 1
        sig_copy["id"] = mapped_id
        sig_copy["index"] = mapped_id
        base_signals.append(sig_copy)
        id_map[previous_id] = mapped_id
        added_signals += 1

    base_strings: Dict[str, Dict[str, Any]] = base_state.setdefault("string_scores", {})  # type: ignore[assignment]
    new_strings = 0
    updated_strings = 0

    for token, payload in extra_state.get("string_scores", {}).items():
        payload_copy = copy.deepcopy(payload)
        mapped_ids: List[int] = []
        for wid in payload_copy.get("window_ids", []):
            try:
                mapped = id_map[int(wid)]
            except (KeyError, TypeError, ValueError):
                continue
            mapped_ids.append(mapped)
        if not mapped_ids:
            continue
        payload_copy["window_ids"] = sorted(set(mapped_ids))
        payload_copy["occurrences"] = int(payload_copy.get("occurrences", len(mapped_ids)))
        occ = payload_copy["occurrences"]

        base_payload = base_strings.get(token)
        if base_payload is None:
            base_strings[token] = payload_copy
            new_strings += 1
            continue

        updated_strings += 1
        base_occ = int(base_payload.get("occurrences", 0))
        total_occ = base_occ + occ if base_occ + occ > 0 else 1
        base_payload["occurrences"] = base_occ + occ

        existing_ids = set(int(wid) for wid in base_payload.get("window_ids", []))
        existing_ids.update(mapped_ids)
        base_payload["window_ids"] = sorted(existing_ids)

        def blend_metric(key: str, new_val: Any) -> None:
            if not isinstance(new_val, (int, float)):
                return
            current = base_payload.get(key)
            if isinstance(current, (int, float)):
                base_payload[key] = (current * base_occ + float(new_val) * occ) / total_occ
            else:
                base_payload[key] = float(new_val)

        for metric_key in ("coherence", "stability", "entropy", "rupture", "connector", "patternability"):
            if metric_key in payload_copy:
                blend_metric(metric_key, payload_copy[metric_key])

        extra_metrics = payload_copy.get("metrics")
        if isinstance(extra_metrics, dict):
            base_metrics = base_payload.setdefault("metrics", {})
            for key, value in extra_metrics.items():
                if isinstance(value, (int, float)):
                    current = base_metrics.get(key)
                    if isinstance(current, (int, float)):
                        base_metrics[key] = (current * base_occ + float(value) * occ) / total_occ
                    else:
                        base_metrics[key] = float(value)
                else:
                    base_metrics[key] = value

        extra_graph = payload_copy.get("graph_metrics")
        if isinstance(extra_graph, dict):
            base_payload.setdefault("graph_metrics", extra_graph)

    file_index = {entry.get("file_id"): entry for entry in base_state.get("files", [])}
    for record in extra_state.get("files", []):
        file_id = record.get("file_id")
        if file_id and file_id not in file_index:
            base_state.setdefault("files", []).append(record)
            file_index[file_id] = record

    summary = base_state.setdefault("summary", {})
    summary["window_count"] = len(base_signals)
    summary["string_count"] = len(base_strings)
    summary["token_count"] = sum(int(payload.get("occurrences", 0)) for payload in base_strings.values())
    base_state["token_count"] = summary["token_count"]
    base_state["corpus_size_bytes"] = base_state.get("corpus_size_bytes", 0) + extra_state.get("corpus_size_bytes", 0)
    base_state["enriched_sources"] = sorted({
        *base_state.get("enriched_sources", []),
        str(source),
    })

    enrichment_log = base_state.setdefault("enrichment_log", [])  # type: ignore[assignment]
    enrichment_entry = {
        "source": str(source),
        "added_signals": added_signals,
        "new_strings": new_strings,
        "updated_strings": updated_strings,
    }
    if note:
        enrichment_entry["note"] = note
    enrichment_log.append(enrichment_entry)

    return enrichment_entry


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich an STM twin corpus with additional states")
    parser.add_argument("base", type=Path, help="Base gold_state JSON path to update in-place")
    parser.add_argument(
        "--extra",
        type=Path,
        action="append",
        default=[],
        help="Additional STM state JSON files to merge (can be repeated)",
    )
    parser.add_argument(
        "--note",
        type=str,
        default=None,
        help="Optional annotation stored in the enrichment log",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing outputs")
    args = parser.parse_args()

    if not args.extra:
        parser.error("At least one --extra state must be provided")

    base_path = args.base
    original_text = base_path.read_text(encoding="utf-8")
    base_state = json.loads(original_text)

    summary: List[Dict[str, Any]] = []
    for extra_path in args.extra:
        extra_state = load_state(extra_path)
        summary.append(merge_state(base_state, extra_state, source=extra_path, note=args.note))

    if args.dry_run:
        print(json.dumps({"preview": summary, "signals": len(base_state.get("signals", []))}, indent=2))
        return

    backup_path = base_path.with_suffix(".pre_enrich.json")
    if not backup_path.exists():
        backup_path.write_text(original_text, encoding="utf-8")

    base_path.write_text(json.dumps(base_state, indent=2), encoding="utf-8")
    print(json.dumps({"updated": str(base_path), "backup": str(backup_path), "enrichment": summary}, indent=2))


if __name__ == "__main__":
    main()
