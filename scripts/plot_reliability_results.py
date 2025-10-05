#!/usr/bin/env python3
"""Generate reliability comparison plots and summary tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit("matplotlib is required for plot generation; pip install matplotlib") from exc
import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.train_reliability_attn import EvalDetailDataset, collate_batch, run_model
from sep_text_manifold.attn_ospace import OspaceTransformer, OspaceTransformerConfig


def load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def canonical_label(name: str) -> str:
    mapping = {
        "transformer_base": "Transformer",
        "transformer_no_cross_attention": "No cross-attention",
        "transformer_no_phase": "No phase",
        "transformer_feature_dim_16": "Feature dim = 16",
        "mlp_baseline": "MLP baseline",
    }
    return mapping.get(name, name.replace("_", " ").title())


def gather_fever_metrics(runs: Sequence[Dict[str, object]]) -> Tuple[List[str], List[float], List[float]]:
    labels: List[str] = []
    f1_values: List[float] = []
    ece_values: List[float] = []
    for entry in runs:
        name = str(entry.get("name", "unknown"))
        test = entry.get("test") if isinstance(entry.get("test"), dict) else {}
        validation = entry.get("validation") if isinstance(entry.get("validation"), dict) else {}
        f1 = test.get("f1") or validation.get("f1")
        ece = test.get("ece") if isinstance(test.get("ece"), (int, float)) else validation.get("ece")
        if f1 is None:
            continue
        labels.append(canonical_label(name))
        f1_values.append(float(f1))
        ece_values.append(float(ece) if isinstance(ece, (int, float)) else float("nan"))
    return labels, f1_values, ece_values


def plot_fever_comparison(labels: Sequence[str], f1_values: Sequence[float], ece_values: Sequence[float], output_path: Path) -> None:
    positions = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(positions - width / 2, f1_values, width, label="Macro F1")
    ax.bar(positions + width / 2, ece_values, width, label="ECE")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("FEVER reliability configurations")
    ax.legend()
    ax.grid(axis="y", linestyle=":", linewidth=0.5)
    fig.tight_layout()
    ensure_dir(output_path)
    fig.savefig(output_path)
    plt.close(fig)


def collect_predictions(
    detail_path: Path,
    checkpoint_path: Path,
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, torch.Tensor, OspaceTransformerConfig]:
    payload = torch.load(checkpoint_path, map_location=device)
    config_payload = payload.get("config")
    if isinstance(config_payload, dict):
        config = OspaceTransformerConfig(**config_payload)
    elif isinstance(config_payload, OspaceTransformerConfig):
        config = config_payload
    else:
        raise ValueError(f"Checkpoint {checkpoint_path} missing configuration payload")
    model = OspaceTransformer(config).to(device)
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError(f"Checkpoint {checkpoint_path} missing state_dict")
    model.load_state_dict(state_dict)

    dataset = EvalDetailDataset(detail_path)
    vocab = payload.get("tokenizer_vocab")
    if isinstance(vocab, dict):
        dataset.tokenizer.vocab = {str(key): int(value) for key, value in vocab.items()}
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    logits: List[float] = []
    margins: List[float] = []
    labels: List[float] = []

    model.eval()
    with torch.inference_mode():
        for batch in loader:
            tensor_batch = {key: value.to(device) for key, value in batch.items() if isinstance(value, torch.Tensor)}
            output = run_model(model, tensor_batch, device, config)
            logits.extend(output.admit_logits.view(-1).cpu().tolist())
            margins.extend(output.support_margin.view(-1).cpu().tolist())
            labels.extend(batch["labels"].view(-1).tolist())

    return (
        np.asarray(logits, dtype=np.float64),
        np.asarray(margins, dtype=np.float64),
        np.asarray(labels, dtype=np.float64),
        dataset.tokenizer.vocab,
        config,
    )


def precision_recall(status: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    tp = float(np.logical_and(status, labels).sum())
    fp = float(np.logical_and(status, np.logical_not(labels)).sum())
    fn = float(np.logical_and(np.logical_not(status), labels).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return precision, recall


def margin_sweep(
    logits: np.ndarray,
    margins: np.ndarray,
    labels: np.ndarray,
    admit_threshold: float,
    margin_thresholds: Sequence[float],
    temperature: float,
) -> Dict[str, List[float]]:
    scaled_logits = logits / max(temperature, 1e-6)
    probabilities = 1.0 / (1.0 + np.exp(-scaled_logits))
    precisions: List[float] = []
    recalls: List[float] = []
    for threshold in margin_thresholds:
        status = np.logical_and(probabilities >= admit_threshold, margins >= threshold)
        precision, recall = precision_recall(status, labels >= 0.5)
        precisions.append(precision)
        recalls.append(recall)
    return {"precision": precisions, "recall": recalls}


def plot_scifact_margins(
    margin_thresholds: Sequence[float],
    base_metrics: Dict[str, List[float]],
    scaled_metrics: Dict[str, List[float]],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(margin_thresholds, base_metrics["precision"], marker="o", label="Precision (no temperature)")
    ax.plot(margin_thresholds, base_metrics["recall"], marker="s", label="Recall (no temperature)")
    ax.plot(margin_thresholds, scaled_metrics["precision"], linestyle="--", marker="o", label="Precision (temperature)")
    ax.plot(margin_thresholds, scaled_metrics["recall"], linestyle="--", marker="s", label="Recall (temperature)")
    ax.set_xlabel("Margin threshold")
    ax.set_ylabel("Score")
    ax.set_title("SciFact admit precision/recall vs. margin threshold")
    ax.legend(loc="best")
    ax.grid(True, linestyle=":", linewidth=0.5)
    fig.tight_layout()
    ensure_dir(output_path)
    fig.savefig(output_path)
    plt.close(fig)


def safe_metric(block: Optional[Dict[str, object]], key: str) -> Optional[float]:
    if not isinstance(block, dict):
        return None
    value = block.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def extend_table(
    rows: List[Tuple[str, str, Optional[float], Optional[float], Optional[float], Optional[float]]],
    dataset: str,
    configuration: str,
    validation: Optional[Dict[str, object]],
    test: Optional[Dict[str, object]],
) -> None:
    rows.append(
        (
            dataset,
            configuration,
            safe_metric(validation, "f1"),
            safe_metric(validation, "brier"),
            safe_metric(test, "f1"),
            safe_metric(test, "brier"),
        )
    )


def format_metric(value: Optional[float]) -> str:
    if value is None or np.isnan(value):
        return "-"
    return f"{value:.3f}"


def write_metrics_table(
    output_path: Path,
    fever_runs: Sequence[Dict[str, object]],
    scifact_curriculum: Dict[str, object],
    scifact_finetune: Dict[str, object],
    hover_adapt: Dict[str, object],
    hover_base: Dict[str, object],
    hover_fever: Dict[str, object],
) -> None:
    rows: List[Tuple[str, str, Optional[float], Optional[float], Optional[float], Optional[float]]] = []

    for entry in fever_runs:
        name = canonical_label(str(entry.get("name", "unknown")))
        validation = entry.get("validation") if isinstance(entry.get("validation"), dict) else None
        test = entry.get("test") if isinstance(entry.get("test"), dict) else None
        extend_table(rows, "FEVER", name, validation, test)

    extend_table(
        rows,
        "SciFact",
        "Curriculum",
        scifact_curriculum.get("validation", {}).get("scifact") if isinstance(scifact_curriculum.get("validation"), dict) else scifact_curriculum.get("validation"),
        scifact_curriculum.get("test", {}).get("scifact") if isinstance(scifact_curriculum.get("test"), dict) else scifact_curriculum.get("test"),
    )

    extend_table(
        rows,
        "SciFact",
        "Finetune",
        scifact_finetune.get("validation") if isinstance(scifact_finetune.get("validation"), dict) else None,
        scifact_finetune.get("test") if isinstance(scifact_finetune.get("test"), dict) else None,
    )

    extend_table(
        rows,
        "HoVer",
        "FEVER adapt",
        hover_adapt.get("summary", {}).get("metrics") if isinstance(hover_adapt.get("summary"), dict) else None,
        None,
    )

    extend_table(
        rows,
        "HoVer",
        "FEVER base",
        hover_base.get("summary", {}).get("metrics") if isinstance(hover_base.get("summary"), dict) else None,
        None,
    )

    extend_table(
        rows,
        "HoVer",
        "Transformer",
        hover_fever.get("validation") if isinstance(hover_fever.get("validation"), dict) else None,
        hover_fever.get("test") if isinstance(hover_fever.get("test"), dict) else None,
    )

    ensure_dir(output_path)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("| Dataset | Configuration | Val F1 | Val Brier | Test F1 | Test Brier |\n")
        handle.write("| --- | --- | --- | --- | --- | --- |\n")
        for dataset, config, val_f1, val_brier, test_f1, test_brier in rows:
            handle.write(
                f"| {dataset} | {config} | {format_metric(val_f1)} | {format_metric(val_brier)} | "
                f"{format_metric(test_f1)} | {format_metric(test_brier)} |\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fever-runs", type=Path, default=Path("results/experiments/fever_reliability_runs.json"))
    parser.add_argument("--scifact-detail", type=Path, default=Path("results/eval/scifact_dev/eval_detail.jsonl"))
    parser.add_argument("--scifact-checkpoint", type=Path, default=Path("models/reliability_fever_scifact_curriculum.pt"))
    parser.add_argument("--temperature-summary", type=Path, default=Path("results/analysis/scifact_temperature_finetune.json"))
    parser.add_argument("--calibration-summary", type=Path, default=Path("results/analysis/calibration_summary.json"))
    parser.add_argument("--scifact-finetune", type=Path, default=Path("results/experiments/scifact_finetune.json"))
    parser.add_argument("--scifact-curriculum", type=Path, default=Path("results/experiments/fever_scifact_curriculum.json"))
    parser.add_argument("--hover-adapt", type=Path, default=Path("results/experiments/hover_val_fever_adapt_eval.json"))
    parser.add_argument("--hover-base", type=Path, default=Path("results/experiments/hover_val_fever_base_eval.json"))
    parser.add_argument("--hover-fever", type=Path, default=Path("results/experiments/hover_eval_from_fever.json"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--fever-plot", type=Path, default=Path("results/figures/fever_reliability_comparison.png"))
    parser.add_argument("--scifact-plot", type=Path, default=Path("results/figures/scifact_margin_curves.png"))
    parser.add_argument("--table-output", type=Path, default=Path("results/tables/metrics_summary.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    fever_runs_data = load_json(args.fever_runs)
    if not isinstance(fever_runs_data, list):
        raise ValueError(f"Expected list in {args.fever_runs}")
    labels, f1_values, ece_values = gather_fever_metrics(fever_runs_data)
    plot_fever_comparison(labels, f1_values, ece_values, args.fever_plot)

    temperature_payload = load_json(args.temperature_summary)
    if not isinstance(temperature_payload, dict):
        raise ValueError(f"Expected JSON object in {args.temperature_summary}")
    best_temperature = float(temperature_payload.get("best_temperature", 1.0))

    calibration_summary = load_json(args.calibration_summary)
    admit_threshold = 0.1
    scifact_block = calibration_summary.get("scifact_val_curriculum") if isinstance(calibration_summary, dict) else None
    if isinstance(scifact_block, dict):
        calib = scifact_block.get("calibration")
        if isinstance(calib, dict):
            admit_threshold = float(calib.get("admit_threshold", admit_threshold))

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    logits, margins, labels_array, _, _ = collect_predictions(
        args.scifact_detail,
        args.scifact_checkpoint,
        args.batch_size,
        device,
    )
    margin_thresholds = [-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
    base_metrics = margin_sweep(logits, margins, labels_array, admit_threshold, margin_thresholds, temperature=1.0)
    scaled_metrics = margin_sweep(logits, margins, labels_array, admit_threshold, margin_thresholds, temperature=best_temperature)
    plot_scifact_margins(margin_thresholds, base_metrics, scaled_metrics, args.scifact_plot)

    scifact_curriculum_data = load_json(args.scifact_curriculum)
    scifact_finetune_data = load_json(args.scifact_finetune)
    hover_adapt_data = load_json(args.hover_adapt)
    hover_base_data = load_json(args.hover_base)
    hover_fever_data = load_json(args.hover_fever)

    write_metrics_table(
        args.table_output,
        fever_runs_data,
        scifact_curriculum_data,
        scifact_finetune_data,
        hover_adapt_data,
        hover_base_data,
        hover_fever_data,
    )


if __name__ == "__main__":
    main()
