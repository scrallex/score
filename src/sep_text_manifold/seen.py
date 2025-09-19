from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import hnswlib
import numpy as np


class SeenEngine:
    """Routes trigger strings to foreground/deferred manifold windows."""

    def __init__(self, base_path: Path | str = Path("analysis")) -> None:
        base = Path(base_path)
        self.state: Dict[str, object] = json.loads((base / "score_state_native.json").read_text())
        self.router: Dict[str, object] = json.loads((base / "router_config.json").read_text())

        # Window map (id -> signal)
        ordered_signals = sorted(self.state.get("signals", []), key=lambda s: s["id"])  # type: ignore[arg-type]
        self.windows: Dict[int, Dict[str, object]] = {int(sig["id"]): sig for sig in ordered_signals}
        self.id_order: List[int] = [int(sig["id"]) for sig in ordered_signals]
        self.signature_sequence: List[str] = [sig.get("signature", "") for sig in ordered_signals]
        self.id_to_pos: Dict[int, int] = {wid: idx for idx, wid in enumerate(self.id_order)}
        # Map string -> score payload
        self.strings: Dict[str, Dict[str, object]] = self.state.get("string_scores", {})  # type: ignore[assignment]

        # Signature postings (q-gram -> window ids)
        postings_raw = json.loads((base / "signature_postings.json").read_text())
        self.postings: Dict[Tuple[str, ...], set[int]] = {
            tuple(key.split("\u001f")): set(map(int, ids)) for key, ids in postings_raw.items()
        }

        # ANN index
        meta = json.loads((base / "ann.meta").read_text())
        dim = int(meta["dim"])
        self.ann = hnswlib.Index(space="l2", dim=dim)
        self.ann.load_index(str(base / "ann.hnsw"))
        self.ann.set_ef(100)

    # --- helpers ---------------------------------------------------------

    def locate_trigger_windows(self, trigger: str) -> set[int]:
        entry = self.strings.get(trigger)
        if not entry:
            return set()
        return set(entry.get("window_ids", []))

    def window_vector(self, wid: int) -> np.ndarray | None:
        sig = self.windows.get(wid)
        if not sig:
            return None
        metrics = sig.get("metrics", {})
        return np.array([
            float(metrics.get("coherence", 0.0)),
            float(metrics.get("stability", 0.0)),
            float(metrics.get("entropy", 1.0)),
            float(metrics.get("rupture", 0.0)),
            float(sig.get("lambda_hazard", metrics.get("rupture", 0.0))),
        ], dtype=np.float32)

    def centroid(self, wids: Iterable[int]) -> np.ndarray | None:
        vecs = [self.window_vector(wid) for wid in wids]
        vecs = [v for v in vecs if v is not None]
        if not vecs:
            return None
        return np.vstack(vecs).mean(axis=0)

    def qgram_hits(self, seeds: Iterable[int], q: int = 3, limit: int = 200) -> set[int]:
        counts: Dict[int, int] = {}
        n = len(self.signature_sequence)
        if n == 0:
            return set()
        if n < q:
            key = tuple(self.signature_sequence)
            for wid in self.postings.get(key, set()):
                counts[wid] = counts.get(wid, 0) + 1
            return set(wid for wid, _ in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:limit])
        for wid in seeds:
            pos = self.id_to_pos.get(wid)
            if pos is None:
                continue
            start_min = max(0, pos - q + 1)
            start_max = min(pos, n - q)
            if start_min > start_max:
                start_min = start_max = pos
            for start in range(start_min, start_max + 1):
                qg = tuple(self.signature_sequence[start : start + q])
                for hit in self.postings.get(qg, set()):
                    counts[hit] = counts.get(hit, 0) + 1
        ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        return set(wid for wid, _ in ranked[:limit])

    def ann_hits(self, target: np.ndarray, k: int = 200, max_dist: float = 0.20) -> set[int]:
        if target is None:
            return set()
        labels, distances = self.ann.knn_query(target.reshape(1, -1), k=k)
        return {int(lbl) for lbl, dist in zip(labels[0], distances[0]) if dist <= max_dist}

    def route(self, candidates: Iterable[int]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
        cfg = self.router["router"]["foreground"]  # type: ignore[index]
        min_coh = float(cfg.get("min_coh", 0.8))
        max_ent = float(cfg.get("max_ent", 0.35))
        foreground: List[Dict[str, object]] = []
        deferred: List[Dict[str, object]] = []
        for wid in candidates:
            sig = self.windows.get(wid)
            if not sig:
                continue
            metrics = sig.get("metrics", {})
            item = {
                "window_id": wid,
                "coherence": metrics.get("coherence", 0.0),
                "entropy": metrics.get("entropy", 1.0),
                "stability": metrics.get("stability", 0.0),
                "rupture": metrics.get("rupture", 0.0),
                "signature": sig.get("signature"),
            }
            if item["coherence"] >= min_coh and item["entropy"] <= max_ent:
                foreground.append(item)
            else:
                deferred.append(item)
        foreground.sort(key=lambda x: (-x["coherence"], x["entropy"]))
        deferred.sort(key=lambda x: (-x["coherence"], x["entropy"]))
        return foreground[:100], deferred[:300]

    # --- public entry ----------------------------------------------------

    def seen(self, trigger: str) -> Dict[str, List[Dict[str, object]]]:
        seeds = self.locate_trigger_windows(trigger)
        if not seeds:
            return {"foreground": [], "deferred": []}
        centroid = self.centroid(seeds)
        trig_cfg = self.router["router"]["triggers"]  # type: ignore[index]
        max_dist = float(trig_cfg.get("max_ann_dist", 0.20))
        q_hits = self.qgram_hits(seeds, q=int(trig_cfg.get("min_sig_qgrams", 2)))
        ann_hits = self.ann_hits(centroid, max_dist=max_dist) if centroid is not None else set()
        candidates = seeds | q_hits | ann_hits
        fg, df = self.route(candidates)
        return {"foreground": fg, "deferred": df}


_engine: SeenEngine | None = None


def get_engine() -> SeenEngine:
    global _engine
    if _engine is None:
        _engine = SeenEngine()
    return _engine
