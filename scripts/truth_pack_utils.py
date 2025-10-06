"""Utilities for building truth-pack artefacts from in-memory text corpora."""

from __future__ import annotations

import pickle
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Mapping, Optional, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from sep_text_manifold.semantic import EmbeddingConfig, SemanticEmbedder

from scripts.reality_filter_pack import (
    HASH_DIM,
    DEFAULT_NOVELTY_SEEDS,
    PackManifest,
    build_indices,
    build_norm_metrics,
    build_semantic_outputs,
    build_signature_table,
    compute_content_hash,
    ingest_pack,
    write_json,
)


_FILENAME_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def _sanitise_name(name: str) -> str:
    cleaned = _FILENAME_SAFE.sub("_", name.strip())
    truncated = cleaned[:120]
    return truncated or "doc"


def build_truth_pack_from_texts(
    *,
    pack_name: str,
    texts: Mapping[str, str],
    output_root: Path,
    extensions: Sequence[str] = ("txt",),
    seeds: Optional[Sequence[str]] = None,
    novelty_seeds: Optional[Sequence[str]] = None,
    window_bytes: int = 2048,
    stride: int = 1024,
    min_token_len: int = 3,
    min_occurrences: int = 1,
    workers: int = 1,
) -> Path:
    """Ingest the provided documents and write a truth-pack manifest."""

    if not texts:
        raise ValueError("Cannot build truth pack from empty corpus")

    seeds = list(seeds or [])
    novelty = list(novelty_seeds or DEFAULT_NOVELTY_SEEDS)

    with TemporaryDirectory(prefix=f"{pack_name}_src_") as tmp_dir:
        source_dir = Path(tmp_dir)
        name_counts: Dict[str, int] = {}
        for key, content in texts.items():
            base = _sanitise_name(key)
            count = name_counts.get(base, 0)
            name_counts[base] = count + 1
            filename = base if count == 0 else f"{base}_{count}"
            (source_dir / f"{filename}.txt").write_text(content.strip() + "\n", encoding="utf-8")

        state, summary = ingest_pack(
            source_dir,
            window_bytes=window_bytes,
            stride=stride,
            min_token_len=min_token_len,
            drop_numeric=False,
            min_occurrences=min_occurrences,
            extensions=list(extensions),
            workers=workers,
        )

    output_root.mkdir(parents=True, exist_ok=True)

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

    signals = state.get("signals", [])  # type: ignore[assignment]
    embedder = None  # semantic bridge uses hash embedding under the hood
    seed_embedder = SemanticEmbedder(EmbeddingConfig(method="hash", dims=HASH_DIM))

    norm_metrics = build_norm_metrics(
        state,
        signals=list(signals) if isinstance(signals, list) else [],
        embedder=seed_embedder,
        min_occurrences=min_occurrences,
        seeds=seeds,
    )
    norm_metrics_path = output_root / "norm_metrics.json"
    write_json(norm_metrics_path, norm_metrics)

    _, bloom_filter, sig_table = build_signature_table(
        state,
        list(signals) if isinstance(signals, list) else [],
        min_occurrences=min_occurrences,
    )
    signature_table_path = output_root / "signature_table.parquet"
    pq.write_table(sig_table, signature_table_path, compression="zstd")
    bloom_path = output_root / "signatures.bloom"
    with bloom_path.open("wb") as fh:
        pickle.dump(bloom_filter, fh)

    seed_centroids_path: Optional[Path] = None

    semantic_report_path: Optional[Path] = None
    scatter_path: Optional[Path] = None
    if seeds:
        semantic_report, scatter = build_semantic_outputs(
            state_path,
            seeds=seeds,
            min_occurrences=min_occurrences,
            embedding_method="hash",
            model_name="all-MiniLM-L6-v2",
            hash_dims=HASH_DIM,
            scatter_output=output_root / "semantic_scatter.png",
        )
        semantic_report_path = output_root / "semantic_bridge.json"
        write_json(semantic_report_path, semantic_report)
        scatter_path = Path(scatter)

    manifest_path = output_root / "manifest.json"
    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    pack_id = f"{pack_name}_{generated_at.replace('-', '').replace(':', '')}"
    content_hash = compute_content_hash(output_root, extensions=None)
    source_globs = [f"**/*.{ext}" for ext in extensions] or ["**/*"]
    seed_families = {
        "factual": list(seeds),
        "novelty": list(novelty),
    }

    manifest = PackManifest(
        name=pack_name,
        pack_id=pack_id,
        pack_path=str(output_root),
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
        seed_centroids=str(seed_centroids_path) if seed_centroids_path else None,
        hash_dim=HASH_DIM,
        min_occurrences=min_occurrences,
        window_bytes=window_bytes,
        stride=stride,
        generated_at=generated_at,
    )
    write_json(manifest_path, asdict(manifest))
    return manifest_path
