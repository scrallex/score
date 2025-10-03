#!/usr/bin/env python3
"""Visual dashboard for the Semantic Guardrail demo.

The dashboard replays a pre-generated event stream (JSONL) and contrasts three
perspectives:

1. Naïve semantic alerts (LLM-style keyword/embedding search)
2. Naïve structural alerts (pure QFH/STM structural scoring)
3. The hybrid guardrail that requires both semantic relevance and structural
   stability (upper-right quadrant of the semantic/structural scatter plot).

The right-hand panel renders the structural/semantic plane, optionally with the
static combined scatter background, and animates incoming events.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from sep_text_manifold.semantic import EmbeddingConfig, SemanticEmbedder, seed_similarity

PANEL_MAX_LOG = 6


def load_events(path: Path) -> List[Dict[str, object]]:
    events: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not events:
        raise ValueError(f"No events found in {path}")
    return events


def load_state_points(
    states: Sequence[Path],
    seeds: Sequence[str],
    *,
    max_points: int = 4000,
) -> Dict[str, np.ndarray]:
    embedder = SemanticEmbedder(EmbeddingConfig(method="transformer"))
    xs: List[float] = []
    ys: List[float] = []
    coherences: List[float] = []
    for state_path in states:
        if not state_path.exists():
            continue
        data = json.loads(state_path.read_text())
        string_scores: Dict[str, Dict[str, object]] = data.get("string_scores", {})
        if not string_scores:
            continue
        strings = list(string_scores.keys())
        semantics = seed_similarity(strings, embedder=embedder, seeds=seeds)
        for idx, string in enumerate(strings):
            entry = string_scores[string]
            metrics = entry.get("metrics", {})
            xs.append(float(entry.get("patternability", metrics.get("coherence", 0.0))))
            ys.append(float(semantics[idx]))
            coherences.append(float(entry.get("coherence", metrics.get("coherence", 0.0))))
    if not xs:
        raise ValueError("No string scores available to seed scatter plot")
    xs_np = np.asarray(xs)
    ys_np = np.asarray(ys)
    coh_np = np.asarray(coherences)
    if len(xs_np) > max_points:
        idx = np.random.choice(len(xs_np), size=max_points, replace=False)
        xs_np = xs_np[idx]
        ys_np = ys_np[idx]
        coh_np = coh_np[idx]
    return {"x": xs_np, "y": ys_np, "coherence": coh_np}


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", type=Path, default=Path("results/semantic_guardrail_stream.jsonl"))
    parser.add_argument(
        "--states",
        nargs="*",
        type=Path,
        default=[
            Path("analysis/semantic_demo_state.json"),
            Path("analysis/mms_state.json"),
        ],
        help="State files used to seed the background scatter",
    )
    parser.add_argument("--background", type=Path, default=Path("results/semantic_bridge_combined.png"))
    parser.add_argument("--seeds", nargs="+", default=["risk", "resilience", "volatility", "anomaly", "predictive maintenance"])
    parser.add_argument("--interval", type=int, default=1000, help="Animation interval in milliseconds")
    parser.add_argument("--pause", type=float, default=0.0, help="Optional initial pause before playback (seconds)")
    args = parser.parse_args(argv)

    events = load_events(args.stream)
    if args.pause > 0:
        time.sleep(args.pause)

    state_points = load_state_points(args.states, args.seeds)
    x_background = state_points["x"]
    y_background = state_points["y"]
    c_background = state_points["coherence"]

    x_min = float(x_background.min()) - 0.002
    x_max = float(x_background.max()) + 0.002
    y_min = max(0.0, float(y_background.min()) - 0.02)
    y_max = min(1.0, float(y_background.max()) + 0.02)

    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(2, 2, width_ratios=(1, 2), height_ratios=(1, 1), hspace=0.25)
    ax_sem = fig.add_subplot(gs[0, 0])
    ax_struct = fig.add_subplot(gs[1, 0])
    ax_scatter = fig.add_subplot(gs[:, 1])

    ax_sem.set_title("Naïve semantic guardrail")
    ax_struct.set_title("Naïve structural guardrail")
    for ax in (ax_sem, ax_struct):
        ax.axis("off")

    ax_scatter.set_title("Hybrid guardrail: structural vs semantic alignment")
    ax_scatter.set_xlabel("Structural patternability")
    ax_scatter.set_ylabel("Semantic similarity to seeds")
    ax_scatter.set_xlim(x_min, x_max)
    ax_scatter.set_ylim(y_min, y_max)

    if args.background.exists():
        try:
            img = plt.imread(args.background)
            ax_scatter.imshow(img, extent=(x_min, x_max, y_min, y_max), aspect="auto", alpha=0.3, zorder=0)
        except Exception:  # pragma: no cover - defensive
            pass

    base = ax_scatter.scatter(
        x_background,
        y_background,
        c=c_background,
        s=15,
        cmap="viridis",
        alpha=0.25,
        zorder=1,
        edgecolors="none",
    )
    plt.colorbar(base, ax=ax_scatter, label="Mean coherence", fraction=0.046, pad=0.04)

    scatter = ax_scatter.scatter([], [], s=[], c=[], cmap="coolwarm", vmin=0.0, vmax=1.0, edgecolors="k", linewidths=0.5, zorder=3)

    log_sem = deque(maxlen=PANEL_MAX_LOG)
    log_struct = deque(maxlen=PANEL_MAX_LOG)
    sem_text = ax_sem.text(0.02, 0.98, "", va="top", ha="left", fontsize=10, family="monospace")
    struct_text = ax_struct.text(0.02, 0.98, "", va="top", ha="left", fontsize=10, family="monospace")

    alert_text = ax_scatter.text(
        0.5,
        1.02,
        "",
        transform=ax_scatter.transAxes,
        ha="center",
        va="bottom",
        fontsize=12,
        weight="bold",
        color="firebrick",
    )

    offsets: List[List[float]] = []
    colors: List[float] = []
    sizes: List[float] = []

    sem_alerts = 0
    struct_alerts = 0
    hybrid_alerts = 0
    blocked = 0
    repairs = 0
    citations = 0
    total_latency = 0.0

    processed_events: List[Dict[str, object]] = []

    metrics_text = fig.text(
        0.01,
        0.02,
        "",
        transform=fig.transFigure,
        ha="left",
        va="bottom",
        fontsize=10,
        family="monospace",
        color="black",
    )

    cluster_labels = {
        "semantic_only": "Semantic only",
        "structural_only": "Structural only",
        "neutral": "Neutral",
        "hybrid_alert": "Hybrid alert",
        "both_disagree": "Semantic + structural",
    }

    def format_log(lines: Iterable[str], header: str, count: int) -> str:
        body = "\n".join(lines) if lines else "(no alerts)"
        return f"{header} [{count}]\n{body}"

    def update(frame: int):
        nonlocal sem_alerts, struct_alerts, hybrid_alerts
        nonlocal sem_alerts, struct_alerts, hybrid_alerts, blocked, repairs, citations, total_latency
        if frame >= len(events):
            return scatter,
        event = events[frame]
        offsets.append([event["patternability"], event["semantic_similarity"]])
        colors.append(float(event.get("coherence", 0.0)))
        size = 80 if event.get("hybrid_guardrail_alert") else 40
        sizes.append(size)
        scatter.set_offsets(np.asarray(offsets))
        scatter.set_array(np.asarray(colors))
        scatter.set_sizes(np.asarray(sizes))

        label = cluster_labels.get(event.get("cluster"), event.get("cluster", ""))
        name = event.get("event") or event.get("span") or "<span>"
        annotation = f"step {event['step']:02d} | {name} ({label})"
        if event.get("question"):
            annotation += f"\nQ: {event['question']}"

        processed_events.append(event)

        if event.get("naive_semantic_alert"):
            sem_alerts += 1
            log_sem.appendleft(annotation)
        if event.get("naive_structural_alert"):
            struct_alerts += 1
            log_struct.appendleft(annotation)
        sem_text.set_text(format_log(log_sem, "Semantic alerts", sem_alerts))
        struct_text.set_text(format_log(log_struct, "Structural alerts", struct_alerts))

        if event.get("hybrid_guardrail_alert"):
            hybrid_alerts += 1
            citations += 1 if event.get("twins") else 0
            message = (
                f"High-confidence alert #{hybrid_alerts}: {name}\n"
                f"pattern={event['patternability']:.3f} | semantic={event['semantic_similarity']:.3f}"
                f" | hazard={event.get('hazard', 0.0):.3f}"
            )
            twins = event.get("twins") or []
            if twins:
                primary = twins[0]
                message += f"\nCitation: {primary['string']} (occ={primary['occurrences']})"
            alert_text.set_color("darkgreen")
            alert_text.set_text(message)
        else:
            blocked += 1
            if event.get("repair_applied") and event.get("repair_suggestion"):
                repairs += 1
                suggestion = event["repair_suggestion"]
                message = (
                    f"Blocked line: {name}\n"
                    f"Auto-repaired with twin '{suggestion['string']}' (occ={suggestion['occurrences']})"
                )
            else:
                message = f"Blocked line: {name}"
            message += f"\npattern={event['patternability']:.3f} | semantic={event['semantic_similarity']:.3f}"
            message += f" | hazard={event.get('hazard', 0.0):.3f}"
            reasons = []
            if not event.get("repeat_ok", True):
                reasons.append("repeat<r_min")
            if not event.get("hazard_ok", True):
                reasons.append("hazard>lambda_max")
            if not event.get("semantic_ok", True):
                reasons.append("semantic")
            if reasons:
                message += "\nReasons: " + ", ".join(reasons)
            alert_text.set_color("firebrick")
            alert_text.set_text(message)

        total_latency += float(event.get("latency_ms", 0.0))

        total = len(processed_events)
        approved = hybrid_alerts
        blocked_total = blocked
        hall_rate = blocked_total / total if total else 0.0
        repair_yield = repairs / blocked_total if blocked_total else 0.0
        citation_coverage = citations / approved if approved else 0.0
        latency_mean = total_latency / total if total else 0.0

        metrics_text.set_text(
            "Hallucination rate: {hall:.1%}\n"
            "Repair yield: {repair:.1%}\n"
            "Citation coverage: {cite:.1%}\n"
            "Latency overhead: {latency:.1f} ms".format(
                hall=hall_rate,
                repair=repair_yield,
                cite=citation_coverage,
                latency=latency_mean,
            )
        )

        return scatter, sem_text, struct_text, alert_text, metrics_text

    ani = FuncAnimation(fig, update, frames=len(events) + 10, interval=args.interval, blit=False, repeat=False)
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
