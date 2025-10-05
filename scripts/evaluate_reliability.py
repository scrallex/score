#!/usr/bin/env python3
"""Evaluate a reliability checkpoint and log probability/attention statistics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency in some environments
    plt = None

from sep_text_manifold.attn_ospace import OspaceTransformer, OspaceTransformerConfig

# Pull shared dataset and utility helpers from the training script so the
# evaluation path stays aligned with the fine-tuning harness.
from scripts.train_reliability_attn import (
    EvalDetailDataset,
    collate_batch,
    compute_precision_recall,
    expected_calibration_error,
    map_ids_to_indices,
    parse_grid,
    read_split_file,
    run_model,
    sweep_thresholds,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("detail_path", type=Path, help="Path to eval_detail.jsonl file")
    parser.add_argument("checkpoint", type=Path, help="Reliability checkpoint (.pt)")
    parser.add_argument(
        "--index",
        type=Path,
        help="Optional newline-delimited record ids describing the evaluation subset",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device (default: auto-detect)",
    )
    parser.add_argument("--ece-bins", type=int, default=10, help="Number of bins for ECE")
    parser.add_argument(
        "--histogram-bins",
        type=int,
        default=20,
        help="Number of bins for the probability histogram",
    )
    parser.add_argument("--admit-threshold", type=float, default=0.5)
    parser.add_argument("--margin-threshold", type=float, default=0.25)
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Run admit/margin threshold sweep on predictions",
    )
    parser.add_argument(
        "--admit-grid",
        type=str,
        default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
        help="Comma-separated admit threshold grid when --calibrate is set",
    )
    parser.add_argument(
        "--margin-grid",
        type=str,
        default="-0.5,-0.25,0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1.0",
        help="Comma-separated margin threshold grid when --calibrate is set",
    )
    parser.add_argument(
        "--calibration-plot",
        type=Path,
        help="Optional path to write the reliability diagram (PNG)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Write evaluation summary (including flattened metrics) to this JSON file",
    )
    parser.add_argument(
        "--metrics-key",
        type=str,
        help="Optional key name for the flattened metrics block (useful when updating aggregated files)",
    )
    return parser.parse_args()


def safe_numpy(values: Sequence[float]) -> np.ndarray:
    if isinstance(values, np.ndarray):
        return values
    return np.asarray(list(values), dtype=np.float64)


def safe_corr(a: Sequence[float], b: Sequence[float]) -> float:
    arr_a = safe_numpy(a)
    arr_b = safe_numpy(b)
    if arr_a.size < 2 or arr_b.size < 2:
        return float("nan")
    std_a = arr_a.std(ddof=0)
    std_b = arr_b.std(ddof=0)
    if std_a == 0.0 or std_b == 0.0:
        return float("nan")
    corr = float(np.corrcoef(arr_a, arr_b)[0, 1])
    if math.isnan(corr):
        return float("nan")
    return corr


def probability_histogram(probs: Sequence[float], bins: int) -> Dict[str, List[float]]:
    if not probs:
        return {"edges": [], "density": []}
    hist, edges = np.histogram(probs, bins=bins, range=(0.0, 1.0))
    density = (hist / max(1, len(probs))).tolist()
    return {"edges": edges.tolist(), "density": density}


def compute_attention_entropy(weights: np.ndarray) -> float:
    clipped = np.clip(weights, 1e-12, 1.0)
    return float(-(clipped * np.log(clipped)).sum())


def reliability_diagram(
    probs: Sequence[float],
    labels: Sequence[float],
    bins: int,
    path: Path,
) -> None:
    if plt is None:
        print("[evaluate_reliability] matplotlib unavailable; skipping calibration plot.")
        return
    if not probs:
        print("[evaluate_reliability] empty predictions; skipping calibration plot.")
        return
    edges = np.linspace(0.0, 1.0, bins + 1)
    indices = np.digitize(probs, edges, right=True)
    bucket_conf = []
    bucket_acc = []
    bucket_centers = []
    for bucket in range(bins):
        mask = indices == bucket
        if not mask.any():
            continue
        bucket_probs = np.asarray(probs)[mask]
        bucket_labels = np.asarray(labels)[mask]
        bucket_conf.append(bucket_probs.mean())
        bucket_acc.append(bucket_labels.mean())
        bucket_centers.append((edges[bucket] + edges[bucket + 1]) / 2.0)
    fig, ax = plt.subplots(figsize=(4.0, 4.0))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Ideal")
    ax.plot(bucket_conf, bucket_acc, marker="o", label="Model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction supported")
    ax.set_title("Reliability diagram")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def load_model(
    checkpoint_path: Path,
    dataset: EvalDetailDataset,
    device: torch.device,
    payload: Optional[Dict[str, object]] = None,
) -> Tuple[OspaceTransformer, OspaceTransformerConfig]:
    if payload is None:
        payload = torch.load(checkpoint_path, map_location=device)
    config_blob = payload.get("config")
    if isinstance(config_blob, dict):
        config = OspaceTransformerConfig(**config_blob)
    elif isinstance(config_blob, OspaceTransformerConfig):
        config = config_blob
    else:
        raise ValueError(f"Checkpoint {checkpoint_path} missing config payload")
    model = OspaceTransformer(config).to(device)
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError(f"Checkpoint {checkpoint_path} missing state_dict")
    model.load_state_dict(state_dict)
    vocab = payload.get("tokenizer_vocab")
    if isinstance(vocab, dict):
        dataset.tokenizer.vocab = {str(key): int(value) for key, value in vocab.items()}
    model.eval()
    return model, config


def build_loader(
    dataset: EvalDetailDataset,
    *,
    index_path: Optional[Path],
    batch_size: int,
) -> Tuple[Union[EvalDetailDataset, Subset[EvalDetailDataset]], DataLoader]:
    if index_path is None:
        subset: Union[EvalDetailDataset, Subset[EvalDetailDataset]] = dataset
    else:
        record_ids = read_split_file(index_path)
        indices = map_ids_to_indices(dataset, record_ids, index_path)
        if not indices:
            raise ValueError(f"Split file {index_path} produced no indices")
        subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    return subset, loader


def evaluate(
    model: OspaceTransformer,
    config: OspaceTransformerConfig,
    loader: DataLoader,
    *,
    device: torch.device,
    metric_names: Sequence[str],
) -> Dict[str, object]:
    all_probs: List[float] = []
    all_labels: List[float] = []
    all_margins: List[float] = []
    attention_entropy_values: List[float] = []
    attention_max_values: List[float] = []
    metric_samples: Dict[str, List[float]] = {name: [] for name in metric_names}

    with torch.no_grad():
        for batch in loader:
            output = run_model(model, batch, device, config)
            probs = torch.sigmoid(output.admit_logits).squeeze(-1)
            margins = output.support_margin.squeeze(-1)

            all_probs.extend(probs.detach().cpu().tolist())
            all_labels.extend(batch["labels"].detach().cpu().tolist())
            all_margins.extend(margins.detach().cpu().tolist())

            if output.evidence_attention is None:
                continue

            attn_weights = output.evidence_attention.detach().cpu().numpy()  # (batch, 1, mem_len)
            evidence_mask = batch["evidence_mask"].detach().cpu().numpy().astype(bool)
            evidence_features = batch["evidence_features"].detach().cpu().numpy()

            batch_size = attn_weights.shape[0]
            for idx in range(batch_size):
                mask = evidence_mask[idx]
                if not mask.any():
                    continue
                weights = attn_weights[idx, 0, : mask.shape[0]]
                weights = weights[mask]
                if weights.size == 0:
                    continue
                max_weight = float(weights.max())
                attention_max_values.append(max_weight)
                weight_sum = float(weights.sum())
                if weight_sum > 0.0:
                    normalized = weights / weight_sum
                else:
                    normalized = weights
                entropy = compute_attention_entropy(normalized)
                attention_entropy_values.append(entropy)

                features = evidence_features[idx, mask, : len(metric_names)]
                if features.size == 0:
                    continue
                weighted = normalized @ features
                for metric_idx, metric_name in enumerate(metric_names):
                    if metric_idx < weighted.shape[-1]:
                        metric_samples[metric_name].append(float(weighted[metric_idx]))

    result: Dict[str, object] = {
        "probabilities": all_probs,
        "labels": all_labels,
        "margins": all_margins,
        "attention_entropy": attention_entropy_values,
        "attention_max": attention_max_values,
        "metric_samples": metric_samples,
    }
    return result


def summarise(
    raw: Dict[str, object],
    *,
    ece_bins: int,
    admit_threshold: float,
    margin_threshold: float,
    calibrate: bool,
    admit_grid: Sequence[float],
    margin_grid: Sequence[float],
    hist_bins: int,
) -> Dict[str, object]:
    probs = safe_numpy(raw["probabilities"])
    labels = safe_numpy(raw["labels"])
    margins = safe_numpy(raw["margins"])
    attention_entropy = safe_numpy(raw["attention_entropy"])
    attention_max = safe_numpy(raw["attention_max"])
    metric_samples: Dict[str, Sequence[float]] = raw["metric_samples"]  # type: ignore[assignment]

    metrics: Dict[str, object] = {}
    precision, recall = compute_precision_recall(
        probs.tolist(),
        labels.tolist(),
        margins.tolist(),
        admit_threshold,
        margin_threshold,
    )
    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    metrics.update(
        {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "brier": float(np.mean((probs - labels) ** 2)) if probs.size else float("nan"),
            "ece": expected_calibration_error(probs.tolist(), labels.tolist(), num_bins=ece_bins),
        }
    )

    calibration_summary: Optional[Dict[str, float]] = None
    if calibrate and probs.size:
        calibration_summary = sweep_thresholds(
            probs.tolist(),
            margins.tolist(),
            labels.tolist(),
            admit_grid,
            margin_grid,
        )

    positive_rate = float(labels.mean()) if labels.size else float("nan")
    prob_hist = probability_histogram(probs.tolist(), bins=hist_bins)

    attention_summary: Dict[str, float] = {
        "count": float(attention_entropy.size),
        "attention_entropy_mean": float(attention_entropy.mean()) if attention_entropy.size else float("nan"),
        "attention_entropy_std": float(attention_entropy.std(ddof=0)) if attention_entropy.size else float("nan"),
        "max_attention_mean": float(attention_max.mean()) if attention_max.size else float("nan"),
        "max_attention_std": float(attention_max.std(ddof=0)) if attention_max.size else float("nan"),
        "max_attention_q90": float(np.quantile(attention_max, 0.9)) if attention_max.size else float("nan"),
    }

    structural_summary: Dict[str, Dict[str, float]] = {}
    flattened: Dict[str, float] = {
        "count": float(probs.size),
        "positive_rate": positive_rate,
        "mean_probability": float(probs.mean()) if probs.size else float("nan"),
        "prob_std": float(probs.std(ddof=0)) if probs.size else float("nan"),
    }
    flattened.update(attention_summary)

    for metric_name, values in metric_samples.items():
        arr = safe_numpy(values)
        entry = {
            "mean": float(arr.mean()) if arr.size else float("nan"),
            "std": float(arr.std(ddof=0)) if arr.size else float("nan"),
            "corr_prob": safe_corr(probs, arr),
        }
        structural_summary[metric_name] = entry
        flattened[f"{metric_name}_mean"] = entry["mean"]
        flattened[f"{metric_name}_std"] = entry["std"]
        flattened[f"corr_prob_{metric_name}"] = entry["corr_prob"]

    summary: Dict[str, object] = {
        "metrics": metrics,
        "positive_rate": positive_rate,
        "probability_histogram": prob_hist,
        "attention": attention_summary,
        "structural": structural_summary,
        "flat_metrics": flattened,
    }
    if calibration_summary is not None:
        summary["calibration"] = calibration_summary
    return summary


def main() -> None:
    args = parse_args()

    device = torch.device(args.device)

    # Load checkpoint to obtain configuration, then initialise dataset with the
    # correct feature dimensionality before building the evaluation subset.
    checkpoint_obj = torch.load(args.checkpoint, map_location=device)
    config_blob = checkpoint_obj.get("config")
    if isinstance(config_blob, dict):
        config = OspaceTransformerConfig(**config_blob)
    elif isinstance(config_blob, OspaceTransformerConfig):
        config = config_blob
    else:
        raise ValueError(f"Checkpoint {args.checkpoint} missing config payload")
    feature_dim_override = config.evidence_feature_dim if config.evidence_feature_dim > 0 else None

    dataset = EvalDetailDataset(args.detail_path, feature_dim_override=feature_dim_override)
    model, config = load_model(args.checkpoint, dataset, device, payload=checkpoint_obj)

    subset, loader = build_loader(dataset, index_path=args.index, batch_size=args.batch_size)
    if isinstance(subset, Subset):
        metric_names = subset.dataset.metric_keys[: subset.dataset.feature_dim]
    else:
        metric_names = subset.metric_keys[: subset.feature_dim]

    raw = evaluate(model, config, loader, device=device, metric_names=metric_names)

    admit_grid = parse_grid(args.admit_grid) if args.calibrate else []
    margin_grid = parse_grid(args.margin_grid) if args.calibrate else []

    summary = summarise(
        raw,
        ece_bins=args.ece_bins,
        admit_threshold=args.admit_threshold,
        margin_threshold=args.margin_threshold,
        calibrate=args.calibrate,
        admit_grid=admit_grid,
        margin_grid=margin_grid,
        hist_bins=args.histogram_bins,
    )

    probabilities = raw["probabilities"]
    labels = raw["labels"]
    if args.calibration_plot is not None:
        reliability_diagram(probabilities, labels, args.ece_bins, args.calibration_plot)

    payload = {
        "detail_path": str(args.detail_path),
        "checkpoint": str(args.checkpoint),
        "subset_size": len(probabilities),
        "summary": summary,
    }

    if args.metrics_key:
        payload["metrics_key"] = args.metrics_key

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
