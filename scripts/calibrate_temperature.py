#!/usr/bin/env python3
"""Temperature scaling utility for the reliability transformer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from train_reliability_attn import (
    EvalDetailDataset,
    OspaceTransformer,
    OspaceTransformerConfig,
    collate_batch,
    expected_calibration_error,
    map_ids_to_indices,
    read_split_file,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("detail_path", type=Path, help="Path to eval_detail.jsonl")
    parser.add_argument("checkpoint", type=Path, help="Model checkpoint to calibrate")
    parser.add_argument("--val-split", type=Path, required=True, help="Validation split id file")
    parser.add_argument("--test-split", type=Path, required=True, help="Test split id file")
    parser.add_argument(
        "--temperatures",
        type=str,
        default="0.25,0.33,0.5,0.75,1.0,1.25,1.5,2.0,3.0,4.0",
        help="Comma-separated list of candidate temperatures (default spans 0.25–4.0)",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=15,
        help="Bins for expected calibration error",
    )
    parser.add_argument("--output", type=Path, required=True, help="Path to write calibration summary JSON")
    return parser.parse_args()


def load_model(checkpoint: Path) -> tuple[OspaceTransformer, OspaceTransformerConfig, Dict[str, int]]:
    payload = torch.load(checkpoint, map_location="cpu")
    config = OspaceTransformerConfig(**payload["config"])
    model = OspaceTransformer(config)
    model.load_state_dict(payload["state_dict"])
    model.to(DEVICE)
    model.eval()
    vocab = {str(k): int(v) for k, v in payload["tokenizer_vocab"].items()}
    return model, config, vocab


def gather_logits(
    dataset: EvalDetailDataset,
    indices: Sequence[int],
    model: OspaceTransformer,
    config: OspaceTransformerConfig,
) -> tuple[np.ndarray, np.ndarray]:
    subset = Subset(dataset, list(indices))
    loader = DataLoader(subset, batch_size=64, shuffle=False, collate_fn=collate_batch)
    logits: List[float] = []
    labels: List[float] = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            phase_values = batch["phase_values"].to(DEVICE)
            phase_features = phase_values.unsqueeze(-1).repeat(1, 1, config.d_model)
            evidence_tokens = batch["evidence_ids"].to(DEVICE)
            evidence_token_mask = batch["evidence_token_mask"].to(DEVICE)
            evidence_features = batch["evidence_features"].to(DEVICE)

            output = model(
                input_ids,
                attention_mask=attention_mask,
                phase_features=phase_features,
                evidence_memory=None,
                evidence_mask=None,
                evidence_tokens=evidence_tokens,
                evidence_token_mask=evidence_token_mask,
                evidence_features=evidence_features,
            )
            logits.extend(output.admit_logits.squeeze(-1).cpu().tolist())
            labels.extend(batch["labels"].cpu().tolist())
    return np.array(logits, dtype=np.float64), np.array(labels, dtype=np.float64)


def evaluate_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    temperature: float,
    *,
    num_bins: int,
) -> Dict[str, float]:
    scaled = logits / max(temperature, 1e-6)
    probs = 1.0 / (1.0 + np.exp(-scaled))
    preds = probs >= 0.5
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    brier = float(np.mean((probs - labels) ** 2))
    ece = float(expected_calibration_error(probs.tolist(), labels.tolist(), num_bins=num_bins))
    return {
        "temperature": temperature,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "brier": brier,
        "ece": ece,
    }


def main() -> None:
    args = parse_args()
    temps = [float(token.strip()) for token in args.temperatures.split(",") if token.strip()]
    if not temps:
        raise ValueError("No temperatures parsed from --temperatures")

    model, config, vocab = load_model(args.checkpoint)
    dataset = EvalDetailDataset(args.detail_path)
    dataset.tokenizer.vocab = vocab

    val_ids = map_ids_to_indices(dataset, read_split_file(args.val_split), args.val_split)
    test_ids = map_ids_to_indices(dataset, read_split_file(args.test_split), args.test_split)

    val_logits, val_labels = gather_logits(dataset, val_ids, model, config)
    test_logits, test_labels = gather_logits(dataset, test_ids, model, config)

    summaries = {
        "validation": [],
        "test": [],
    }
    best_temp = 1.0
    best_brier = float("inf")
    for temp in temps:
        stats = evaluate_temperature(val_logits, val_labels, temp, num_bins=args.num_bins)
        summaries["validation"].append(stats)
        if stats["brier"] < best_brier:
            best_brier = stats["brier"]
            best_temp = temp

    baseline_stats = evaluate_temperature(val_logits, val_labels, 1.0, num_bins=args.num_bins)
    test_stats = evaluate_temperature(test_logits, test_labels, best_temp, num_bins=args.num_bins)
    baseline_test = evaluate_temperature(test_logits, test_labels, 1.0, num_bins=args.num_bins)
    summaries["test"].append({"temperature": 1.0, **baseline_test})
    summaries["test"].append({"temperature": best_temp, **test_stats})

    output = {
        "detail_path": str(args.detail_path),
        "checkpoint": str(args.checkpoint),
        "val_split": str(args.val_split),
        "test_split": str(args.test_split),
        "temperatures": temps,
        "best_temperature": best_temp,
        "validation_metrics": summaries["validation"],
        "test_metrics": summaries["test"],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
