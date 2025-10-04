#!/usr/bin/env python3
"""Prepare a target manifold + semantic artefacts for the reality-filter demo.

Given a truth-pack directory this script:
  1. Runs STM ingestion (native path) to produce a state with signals.
  2. Builds signature postings + ANN index for twin lookup.
  3. Generates semantic bridge reports and scatter plots against provided seeds.
  4. Emits a manifest JSON describing the outputs for downstream tooling.

Example
-------

    PYTHONPATH=src .venv/bin/python scripts/reality_filter_pack.py \
        docs/truth_pack --seeds policy compliance
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from sep_text_manifold.cli import analyse_directory, build_indices

from scripts.semantic_bridge_demo import run_demo
from scripts.semantic_bridge_plot import build_dataframe, plot_bridge
from sep_text_manifold.semantic import EmbeddingConfig, SemanticEmbedder
import pyarrow as pa
import pyarrow.parquet as pq
from bloom_filter2 import BloomFilter

try:  # optional dependency
    import numpy as np
except ImportError:  # pragma: no cover - NumPy is installed in dev env
    np = None  # type: ignore[assignment]


def _normalise_span(span: str) -> str:
    return " ".join(span.lower().strip().split())


@dataclass
class PackManifest:
    name: str
    pack_id: str
    pack_path: str
    output_root: str
    state_path: str
    summary_path: str
    postings_path: str
    ann_path: str
    ann_meta_path: str
    semantic_report: Optional[str]
    scatter_plot: Optional[str]
    seeds: List[str]
    seed_families: Dict[str, List[str]]
    source_globs: List[str]
    content_hash: str
    norm_metrics_path: Optional[str]
    signature_table: Optional[str]
    signature_bloom: Optional[str]
    min_occurrences: int
    window_bytes: int
    stride: int
    generated_at: str


def _json_default(obj: object):
    if np is not None:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default))


def ingest_pack(
    pack_path: Path,
    *,
    window_bytes: int,
    stride: int,
    min_token_len: int,
    drop_numeric: bool,
    min_occurrences: int,
    extensions: Optional[Sequence[str]],
) -> tuple[dict, dict]:
    result = analyse_directory(
        str(pack_path),
        window_bytes=window_bytes,
        stride=stride,
        extensions=extensions,
        verbose=False,
        min_token_length=min_token_len,
        drop_numeric=drop_numeric,
        min_occurrences=min_occurrences,
    )
    summary = result.summary(top=25)
    state = result.to_state(include_signals=True, include_occurrences=True, include_profiles=False)
    state["summary"] = summary
    state["generated_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return state, summary


def compute_content_hash(root: Path, extensions: Optional[Sequence[str]]) -> str:
    allowed = None
    if extensions:
        allowed = {ext.lower().lstrip(".") for ext in extensions}
    hasher = hashlib.blake2b(digest_size=16)
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if allowed:
            suffix = path.suffix.lstrip(".").lower()
            if suffix not in allowed:
                continue
        rel = path.relative_to(root).as_posix().encode("utf-8")
        hasher.update(rel)
        hasher.update(b"\0")
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(8192), b""):
                hasher.update(chunk)
    return f"blake2b:{hasher.hexdigest()}"


def build_norm_metrics(
    state: Dict[str, object],
    *,
    signals: List[Dict[str, object]],
    embedder: SemanticEmbedder,
    min_occurrences: int,
    seeds: Sequence[str],
) -> Dict[str, Dict[str, object]]:
    signal_by_id: Dict[int, Dict[str, object]] = {}
    sig_hazard: Dict[int, float] = {}
    for sig in signals:
        sig_id = int(sig["id"]) if "id" in sig else None
        if sig_id is None:
            continue
        signal_by_id[sig_id] = sig
        metrics = sig.get("metrics", {})
        hazard = sig.get("lambda_hazard")
        if hazard is None:
            hazard = metrics.get("lambda_hazard", metrics.get("rupture", 1.0))
        sig_hazard[sig_id] = float(hazard)

    if seeds:
        seed_vectors = embedder.encode(list(seeds))
        seed_centroid = seed_vectors.mean(axis=0)
        norm = np.linalg.norm(seed_centroid)
        if norm > 0:
            seed_centroid = seed_centroid / norm
    else:
        seed_centroid = None

    norm_metrics: Dict[str, Dict[str, object]] = {}
    strings: Dict[str, Dict[str, object]] = state.get("string_scores", {})  # type: ignore[assignment]
    for string, payload in strings.items():
        occ = int(payload.get("occurrences", 0))
        if occ < min_occurrences:
            continue
        norm = _normalise_span(string)
        metrics = payload.get("metrics", {})
        vector = embedder.encode([string])[0]
        if seed_centroid is not None:
            semantic_sim = float(np.dot(vector, seed_centroid))
        else:
            semantic_sim = 0.0
        hazard_values = [sig_hazard.get(int(wid), 1.0) for wid in payload.get("window_ids", [])]
        hazard = float(np.mean(hazard_values)) if hazard_values else float(metrics.get("lambda", metrics.get("rupture", 1.0)))
        signature = None
        for wid in payload.get("window_ids", []):
            sig = signal_by_id.get(int(wid))
            if sig and sig.get("signature"):
                signature = sig.get("signature")
                break
        norm_metrics[norm] = {
            "string": string,
            "occurrences": occ,
            "patternability": float(payload.get("patternability", metrics.get("coherence", 0.0))),
            "coherence": float(metrics.get("coherence", 0.0)),
            "stability": float(metrics.get("stability", 0.0)),
            "entropy": float(metrics.get("entropy", 1.0)),
            "rupture": float(metrics.get("rupture", 0.0)),
            "hazard": hazard,
            "signature": signature,
            "semantic": semantic_sim,
            "vector": vector.tolist(),
        }
    return norm_metrics


def build_signature_table(
    state: Dict[str, object],
    signals: List[Dict[str, object]],
    *,
    min_occurrences: int,
) -> Tuple[Dict[bytes, Dict[str, float]], BloomFilter, pa.Table]:
    signal_by_id: Dict[int, Dict[str, object]] = {}
    for sig in signals:
        sig_id = sig.get("id")
        if sig_id is None:
            continue
        sig_id = int(sig_id)
        signal_by_id[sig_id] = sig

    sig_dict: Dict[bytes, Dict[str, float]] = {}
    bloom = BloomFilter(max_elements=2000000, error_rate=0.001)

    strings: Dict[str, Dict[str, object]] = state.get("string_scores", {})  # type: ignore[assignment]
    sig_list: List[bytes] = []
    reps_list: List[int] = []
    lam_list: List[float] = []
    coh_list: List[float] = []
    stab_list: List[float] = []
    pattern_list: List[float] = []

    for string, payload in strings.items():
        occ = int(payload.get("occurrences", 0))
        if occ < min_occurrences:
            continue
        norm = _normalise_span(string)
        sig = hashlib.blake2b(norm.encode("utf-8"), digest_size=16).digest()
        metrics = payload.get("metrics", {})
        hazard_values = []
        for wid in payload.get("window_ids", []):
            sig_info = signal_by_id.get(int(wid))
            if sig_info:
                metrics_sig = sig_info.get("metrics", {})
                hazard = sig_info.get("lambda_hazard", metrics_sig.get("lambda_hazard", metrics_sig.get("rupture", 1.0)))
                hazard_values.append(float(hazard))
        lam = float(np.mean(hazard_values)) if hazard_values else float(metrics.get("lambda", metrics.get("rupture", 1.0)))
        coherence = float(metrics.get("coherence", 0.0))
        stability = float(metrics.get("stability", 0.0))
        patternability = float(payload.get("patternability", coherence))

        sig_dict[sig] = {
            "repetitions": occ,
            "lambda": lam,
            "coherence": coherence,
            "stability": stability,
            "patternability": patternability,
        }
        bloom.add(sig)
        sig_list.append(sig)
        reps_list.append(occ)
        lam_list.append(lam)
        coh_list.append(coherence)
        stab_list.append(stability)
        pattern_list.append(patternability)

    table = pa.Table.from_arrays(
        [
            pa.array(sig_list, type=pa.binary(16)),
            pa.array(reps_list, type=pa.uint32()),
            pa.array(lam_list, type=pa.float32()),
            pa.array(coh_list, type=pa.float32()),
            pa.array(stab_list, type=pa.float32()),
            pa.array(pattern_list, type=pa.float32()),
        ],
        names=["sig", "repetitions", "lambda", "coherence", "stability", "patternability"],
    )
    return sig_dict, bloom, table


def build_semantic_outputs(
    state_path: Path,
    *,
    seeds: Sequence[str],
    min_occurrences: int,
    embedding_method: str,
    model_name: str,
    hash_dims: int,
    scatter_output: Path,
) -> tuple[dict, str]:
    report = run_demo(
        state_path=state_path,
        seeds=seeds,
        top_k=25,
        min_occurrences=min_occurrences,
        embedding_method=embedding_method,
        model_name=model_name,
        hash_dims=hash_dims,
    )
    state = json.loads(state_path.read_text())
    embedder = SemanticEmbedder(EmbeddingConfig(method=embedding_method, model_name=model_name, dims=hash_dims))
    df = build_dataframe(state, seeds=seeds, embedder=embedder, min_occurrences=min_occurrences)
    plot_bridge(df, seeds=seeds, output=scatter_output, top_labels=15)
    return report, str(scatter_output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pack", type=Path, help="Directory containing the truth-pack corpus")
    parser.add_argument("--output-root", type=Path, help="Directory to write manifold + semantic outputs")
    parser.add_argument("--name", help="Optional explicit pack name for manifest")
    parser.add_argument("--extensions", nargs="*", help="Optional file extensions to include (e.g. md txt)")
    parser.add_argument("--window-bytes", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--min-token-len", type=int, default=3)
    parser.add_argument("--drop-numeric", action="store_true", help="Drop purely numeric tokens during ingest")
    parser.add_argument("--min-occurrences", type=int, default=1, help="Minimum occurrences to retain strings during ingest")
    parser.add_argument("--seeds", nargs="+", help="Semantic seeds for bridge analysis")
    parser.add_argument("--embedding-method", choices=["auto", "transformer", "hash"], default="transformer")
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--hash-dims", type=int, default=256)
    parser.add_argument("--semantic-min-occ", type=int, default=2, help="Minimum occurrences for semantic bridge plots")
    parser.add_argument("--manifest", type=Path, help="Optional path for manifest JSON (defaults to output-root/manifest.json)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pack_path: Path = args.pack
    if not pack_path.exists():
        raise FileNotFoundError(f"Pack path not found: {pack_path}")
    if not pack_path.is_dir():
        raise ValueError("Pack path must be a directory")

    pack_name = args.name or pack_path.name
    output_root = args.output_root or Path("analysis/truth_packs") / pack_name
    output_root.mkdir(parents=True, exist_ok=True)

    state, summary = ingest_pack(
        pack_path,
        window_bytes=args.window_bytes,
        stride=args.stride,
        min_token_len=args.min_token_len,
        drop_numeric=args.drop_numeric,
        min_occurrences=args.min_occurrences,
        extensions=args.extensions,
    )

    state_path = output_root / "manifold_state.json"
    summary_path = output_root / "summary.json"
    write_json(state_path, state)
    write_json(summary_path, summary)

    postings_path = output_root / "signature_postings.json"
    ann_path = output_root / "ann.hnsw"
    ann_meta_path = output_root / "ann.meta"
    build_indices(
        state_path=state_path,
        postings_path=postings_path,
        ann_path=ann_path,
        ann_meta_path=ann_meta_path,
        q=3,
    )

    seeds = args.seeds or []
    signals = state.get("signals", [])  # type: ignore[assignment]
    embedder = SemanticEmbedder(EmbeddingConfig(method="hash", model_name=args.model, dims=args.hash_dims))
    norm_metrics = build_norm_metrics(
        state,
        signals=list(signals) if isinstance(signals, list) else [],
        embedder=embedder,
        min_occurrences=args.min_occurrences,
        seeds=seeds,
    )
    norm_metrics_path = output_root / "norm_metrics.json"
    write_json(norm_metrics_path, norm_metrics)

    sig_dict, bloom, sig_table = build_signature_table(
        state,
        list(signals) if isinstance(signals, list) else [],
        min_occurrences=args.min_occurrences,
    )
    signature_table_path = output_root / "signature_table.parquet"
    pq.write_table(sig_table, signature_table_path, compression="zstd")
    bloom_path = output_root / "signatures.bloom"
    with bloom_path.open("wb") as fh:
        pickle.dump(bloom, fh)

    semantic_report_path: Optional[Path] = None
    scatter_path: Optional[Path] = None
    if seeds:
        semantic_report, scatter_output = build_semantic_outputs(
            state_path,
            seeds=seeds,
            min_occurrences=args.semantic_min_occ,
            embedding_method=args.embedding_method,
            model_name=args.model,
            hash_dims=args.hash_dims,
            scatter_output=output_root / "semantic_scatter.png",
        )
        semantic_report_path = output_root / "semantic_bridge.json"
        write_json(semantic_report_path, semantic_report)
        scatter_path = Path(scatter_output)

    manifest_path = args.manifest or (output_root / "manifest.json")
    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    pack_id = f"{pack_name}_{generated_at.replace('-', '').replace(':', '')}"
    content_hash = compute_content_hash(pack_path, args.extensions)
    source_globs = [f"**/*.{ext}" for ext in (args.extensions or [])] or ["**/*"]
    seed_families = {
        "factual": list(seeds),
        "novelty": [],
    }
    manifest = PackManifest(
        name=pack_name,
        pack_id=pack_id,
        pack_path=str(pack_path),
        output_root=str(output_root),
        state_path=str(state_path),
        summary_path=str(summary_path),
        postings_path=str(postings_path),
        ann_path=str(ann_path),
        ann_meta_path=str(ann_meta_path),
        semantic_report=str(semantic_report_path) if semantic_report_path else None,
        scatter_plot=str(scatter_path) if scatter_path else None,
        seeds=list(seeds),
        seed_families=seed_families,
        source_globs=source_globs,
        content_hash=content_hash,
        norm_metrics_path=str(norm_metrics_path),
        signature_table=str(signature_table_path),
        signature_bloom=str(bloom_path),
        min_occurrences=args.min_occurrences,
        window_bytes=args.window_bytes,
        stride=args.stride,
        generated_at=generated_at,
    )
    write_json(manifest_path, asdict(manifest))
    print(f"Truth-pack prepared under {output_root}")
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
