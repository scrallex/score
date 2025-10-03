#!/usr/bin/env python3
"""Generate a span-level reality-filter stream from a truth-pack manifest.

For each candidate span we compute semantic similarity, structural repetition,
hazard, and twin suggestions using the prepared manifold.
The output JSONL matches the schema consumed by the demo dashboard
(naïve semantic & structural alerts plus hybrid admission decisions).
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from sep_text_manifold.semantic import EmbeddingConfig, SemanticEmbedder


@dataclass
class Twin:
    string: str
    occurrences: int
    patternability: float
    semantic_similarity: float
    hazard: float


class TruthPackEngine:
    def __init__(
        self,
        manifest_path: Path,
        *,
        seeds: Sequence[str],
        embedding_method: str,
        model_name: str,
        hash_dims: int,
        embedding_min_occ: int,
    ) -> None:
        manifest = json.loads(manifest_path.read_text())
        self.manifest = manifest
        state_path = Path(manifest["state_path"])
        if not state_path.exists():
            raise FileNotFoundError(f"State file not found: {state_path}")
        self.pack_name = manifest.get("name", state_path.stem)
        self.state = json.loads(state_path.read_text())
        self.strings: Dict[str, Dict[str, object]] = self.state.get("string_scores", {})  # type: ignore[assignment]
        self.signals: Dict[int, Dict[str, object]] = {
            int(sig["id"]): sig for sig in self.state.get("signals", [])  # type: ignore[list-item]
        }

        self.embedder = SemanticEmbedder(
            EmbeddingConfig(method=embedding_method, model_name=model_name, dims=hash_dims)
        )
        self.seeds = list(seeds)
        self.seed_vector = self._build_seed_vector()
        self.embedding_strings: List[str] = []
        self.embedding_matrix: Optional[np.ndarray] = None
        self._build_string_embeddings(min_occ=embedding_min_occ)
        self.hazard_cache: Dict[str, float] = {}

    def _build_seed_vector(self) -> Optional[np.ndarray]:
        if not self.seeds:
            return None
        seed_vecs = self.embedder.encode(self.seeds)
        centroid = seed_vecs.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm == 0.0:
            return np.zeros_like(centroid)
        return centroid / norm

    def _build_string_embeddings(self, min_occ: int) -> None:
        candidates = [s for s, data in self.strings.items() if data.get("occurrences", 0) >= min_occ]
        if not candidates:
            self.embedding_strings = []
            self.embedding_matrix = None
            return
        self.embedding_strings = candidates
        self.embedding_matrix = self.embedder.encode(candidates)

    def semantic_similarity(self, vector: np.ndarray) -> float:
        if self.seed_vector is None:
            return 0.0
        return float(np.clip(np.dot(vector, self.seed_vector), -1.0, 1.0))

    def compute_hazard(self, string: str, data: Optional[Dict[str, object]] = None) -> float:
        if string in self.hazard_cache:
            return self.hazard_cache[string]
        entry = data or self.strings.get(string)
        hazards: List[float] = []
        if entry:
            for wid in entry.get("window_ids", []):  # type: ignore[list-item]
                sig = self.signals.get(int(wid))
                if sig:
                    metrics = sig.get("metrics", {})
                    hazard = float(
                        sig.get(
                            "lambda_hazard",
                            metrics.get("lambda_hazard", metrics.get("rupture", 1.0)),
                        )
                    )
                    hazards.append(hazard)
        if not hazards and entry:
            metrics = entry.get("metrics", {})
            hazards.append(float(metrics.get("lambda_hazard", metrics.get("rupture", 1.0))))
        hazard_value = float(np.mean(hazards)) if hazards else 1.0
        self.hazard_cache[string] = hazard_value
        return hazard_value

    def top_twins(self, vector: np.ndarray, *, exclude: Optional[str] = None, limit: int = 3) -> List[Twin]:
        results: List[Twin] = []
        seen: set[str] = set()

        if exclude and exclude in self.strings:
            data = self.strings[exclude]
            results.append(
                Twin(
                    string=exclude,
                    occurrences=int(data.get("occurrences", 0)),
                    patternability=float(data.get("patternability", 0.0)),
                    semantic_similarity=1.0,
                    hazard=self.compute_hazard(exclude, data),
                )
            )
            seen.add(exclude)

        if self.embedding_matrix is None:
            return results

        scores = self.embedding_matrix @ vector
        order = np.argsort(scores)[::-1]
        for idx in order:
            name = self.embedding_strings[idx]
            if exclude and name == exclude:
                continue
            if name in seen:
                continue
            data = self.strings.get(name)
            if not data:
                continue
            results.append(
                Twin(
                    string=name,
                    occurrences=int(data.get("occurrences", 0)),
                    patternability=float(data.get("patternability", 0.0)),
                    semantic_similarity=float(np.clip(scores[idx], -1.0, 1.0)),
                    hazard=self.compute_hazard(name, data),
                )
            )
            seen.add(name)
            if len(results) >= limit:
                break
        return results

    def metrics_for_string(self, string: str, data: Optional[Dict[str, object]] = None) -> Tuple[float, float, float, float, float]:
        entry = data or self.strings.get(string)
        if entry:
            metrics = entry.get("metrics", {})
            return (
                float(entry.get("patternability", 0.0)),
                float(metrics.get("coherence", 0.0)),
                float(metrics.get("stability", 0.0)),
                float(metrics.get("entropy", 1.0)),
                float(metrics.get("rupture", metrics.get("rupture", 0.0))),
            )
        return 0.0, 0.0, 0.0, 1.0, 0.0

    def evaluate_span(
        self,
        span: str,
        *,
        context: Optional[str],
        semantic_threshold: float,
        structural_threshold: float,
        r_min: int,
        hazard_max: float,
        sigma_min: float,
        rng: random.Random,
    ) -> Dict[str, object]:
        vector = self.embedder.encode([span])[0]
        sem_score = self.semantic_similarity(vector)
        data = self.strings.get(span)
        occurrences = int(data.get("occurrences", 0)) if data else 0
        hazard = self.compute_hazard(span, data)
        pattern, coherence, stability, entropy, rupture = self.metrics_for_string(span, data)

        twins = self.top_twins(vector, exclude=span, limit=3)
        fallback = twins[0] if twins else None
        if occurrences == 0 and fallback is not None:
            pattern = fallback.patternability
            hazard = fallback.hazard
            base_metrics = self.strings.get(fallback.string, {})
            metrics = base_metrics.get("metrics", {}) if isinstance(base_metrics, dict) else {}
            coherence = float(metrics.get("coherence", coherence))
            stability = float(metrics.get("stability", stability))
            entropy = float(metrics.get("entropy", entropy))
            rupture = float(metrics.get("rupture", rupture))

        naive_semantic = sem_score >= semantic_threshold
        naive_structural = pattern >= structural_threshold
        repeat_ok = occurrences >= r_min
        hazard_ok = hazard <= hazard_max
        semantic_ok = sem_score >= sigma_min
        hybrid = repeat_ok and hazard_ok and semantic_ok

        if hybrid:
            cluster = "hybrid_alert"
        elif naive_semantic and not naive_structural:
            cluster = "semantic_only"
        elif naive_structural and not naive_semantic:
            cluster = "structural_only"
        elif naive_semantic and naive_structural:
            cluster = "both_disagree"
        else:
            cluster = "neutral"

        repair_suggestion = None
        repair_applied = False
        if not hybrid and fallback is not None:
            repair_suggestion = {
                "string": fallback.string,
                "occurrences": fallback.occurrences,
                "patternability": round(fallback.patternability, 6),
                "semantic_similarity": round(fallback.semantic_similarity, 6),
                "hazard": round(fallback.hazard, 6),
            }
            repair_applied = True

        latency_ms = round(rng.uniform(45.0, 95.0), 2)

        return {
            "span": span,
            "event": span,
            "context": context,
            "occurrences": occurrences,
            "patternability": round(pattern, 6),
            "semantic_similarity": round(sem_score, 6),
            "coherence": round(coherence, 6),
            "stability": round(stability, 6),
            "entropy": round(entropy, 6),
            "rupture": round(rupture, 6),
            "hazard": round(hazard, 6),
            "naive_semantic_alert": naive_semantic,
            "naive_structural_alert": naive_structural,
            "hybrid_guardrail_alert": hybrid,
            "repeat_ok": repeat_ok,
            "hazard_ok": hazard_ok,
            "semantic_ok": semantic_ok,
            "cluster": cluster,
            "twins": [
                {
                    "string": twin.string,
                    "occurrences": twin.occurrences,
                    "patternability": round(twin.patternability, 6),
                    "semantic_similarity": round(twin.semantic_similarity, 6),
                    "hazard": round(twin.hazard, 6),
                }
                for twin in twins
            ],
            "repair_suggestion": repair_suggestion,
            "repair_applied": repair_applied,
            "source": self.pack_name,
            "latency_ms": latency_ms,
        }


def load_spans(path: Path) -> List[Dict[str, object]]:
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "spans" in data:
        items = data["spans"]
    else:
        items = data
    if not isinstance(items, list):
        raise ValueError("Spans file must be a list or contain a 'spans' list")
    results: List[Dict[str, object]] = []
    for idx, item in enumerate(items):
        if isinstance(item, str):
            results.append({"span": item})
        elif isinstance(item, dict) and "span" in item:
            results.append(item)
        else:
            raise ValueError(f"Invalid span entry at index {idx}: {item}")
    return results


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path, help="Manifest JSON from reality_filter_pack.py")
    parser.add_argument("--spans", required=True, type=Path, help="JSON file listing candidate spans")
    parser.add_argument("--output", type=Path, default=Path("results/semantic_guardrail_stream.jsonl"))
    parser.add_argument("--metrics-output", type=Path, default=Path("results/semantic_guardrail_metrics.json"))
    parser.add_argument("--seeds", nargs="*", help="Optional override for semantic seeds")
    parser.add_argument("--embedding-method", choices=["auto", "transformer", "hash"], default="transformer")
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--hash-dims", type=int, default=256)
    parser.add_argument("--embedding-min-occ", type=int, default=1)
    parser.add_argument("--semantic-threshold", type=float, default=0.25)
    parser.add_argument("--structural-threshold", type=float, default=0.47)
    parser.add_argument("--r-min", type=int, default=2)
    parser.add_argument("--hazard-max", type=float, default=0.25)
    parser.add_argument("--sigma-min", type=float, default=0.28)
    parser.add_argument("--seed", type=int, default=17, help="RNG seed for latency simulation")
    args = parser.parse_args()

    manifest_path = args.manifest
    spans = load_spans(args.spans)
    manifest = json.loads(manifest_path.read_text())
    seeds = args.seeds or manifest.get("seeds", [])

    engine = TruthPackEngine(
        manifest_path,
        seeds=seeds,
        embedding_method=args.embedding_method,
        model_name=args.model,
        hash_dims=args.hash_dims,
        embedding_min_occ=args.embedding_min_occ,
    )

    rng = random.Random(args.seed)

    events: List[Dict[str, object]] = []
    with args.output.open("w", encoding="utf-8") as fh:
        for idx, item in enumerate(spans):
            span = str(item["span"]).strip()
            if not span:
                continue
            context = item.get("question") or item.get("context")
            record = engine.evaluate_span(
                span,
                context=context if isinstance(context, str) else None,
                semantic_threshold=args.semantic_threshold,
                structural_threshold=args.structural_threshold,
                r_min=args.r_min,
                hazard_max=args.hazard_max,
                sigma_min=args.sigma_min,
                rng=rng,
            )
            record.update(
                {
                    "step": idx,
                    "label": item.get("label"),
                    "question": context,
                }
            )
            fh.write(json.dumps(record) + "\n")
            events.append(record)

    total = len(events)
    semantic_alerts = sum(1 for e in events if e["naive_semantic_alert"])
    structural_alerts = sum(1 for e in events if e["naive_structural_alert"])
    hybrid_alerts = sum(1 for e in events if e["hybrid_guardrail_alert"])
    blocked = total - hybrid_alerts
    repairs = sum(1 for e in events if not e["hybrid_guardrail_alert"] and e["repair_applied"])
    citations = sum(1 for e in events if e["hybrid_guardrail_alert"] and e.get("twins"))
    hall_rate = blocked / total if total else 0.0
    repair_yield = repairs / blocked if blocked else 0.0
    citation_coverage = citations / hybrid_alerts if hybrid_alerts else 0.0
    latency_mean = float(np.mean([e["latency_ms"] for e in events])) if events else 0.0
    latency_p95 = float(np.percentile([e["latency_ms"] for e in events], 95)) if events else 0.0

    metrics_payload = {
        "total_events": total,
        "semantic_alerts": semantic_alerts,
        "structural_alerts": structural_alerts,
        "hybrid_alerts": hybrid_alerts,
        "blocked": blocked,
        "repairs": repairs,
        "citations": citations,
        "hallucination_rate": hall_rate,
        "repair_yield": repair_yield,
        "citation_coverage": citation_coverage,
        "latency_ms_mean": latency_mean,
        "latency_ms_p95": latency_p95,
    }
    write_json(args.metrics_output, metrics_payload)

    print(f"Wrote {total} events to {args.output}")
    print(json.dumps(metrics_payload, indent=2))


if __name__ == "__main__":
    main()
