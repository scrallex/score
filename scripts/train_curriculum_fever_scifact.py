#!/usr/bin/env python3
"""Curriculum fine-tune that interleaves FEVER and SciFact batches."""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from sep_text_manifold.attn_ospace import OspaceTransformer, OspaceTransformerConfig

from scripts.train_reliability_attn import (
    EvalDetailDataset,
    METRIC_KEYS,
    WhitespaceTokenizer,
    collate_batch,
    compute_precision_recall,
    ensure_disjoint,
    expected_calibration_error,
    map_ids_to_indices,
    parse_grid,
    read_split_file,
    run_model,
    save_checkpoint,
    sweep_thresholds,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fever-detail", type=Path, default=Path("results/eval/fever_train/eval_detail.jsonl"))
    parser.add_argument(
        "--secondary-detail",
        dest="secondary_detail",
        type=Path,
        help="Eval detail file for the secondary dataset (overrides the SciFact default).",
    )
    parser.add_argument(
        "--scifact-detail",
        dest="secondary_detail",
        type=Path,
        default=Path("results/eval/scifact_train/eval_detail.jsonl"),
        help="Alias for --secondary-detail; defaults to the SciFact curriculum artefact.",
    )
    parser.add_argument("--fever-train-index", type=Path, default=Path("data/splits/fever_train_ids.txt"))
    parser.add_argument("--fever-val-index", type=Path, default=Path("data/splits/fever_val_ids.txt"))
    parser.add_argument("--fever-test-index", type=Path, default=Path("data/splits/fever_test_ids.txt"))
    parser.add_argument(
        "--secondary-train-index",
        dest="secondary_train_index",
        type=Path,
        help="Training split ids for the secondary dataset (overrides SciFact default).",
    )
    parser.add_argument(
        "--secondary-val-index",
        dest="secondary_val_index",
        type=Path,
        help="Validation split ids for the secondary dataset (overrides SciFact default).",
    )
    parser.add_argument(
        "--secondary-test-index",
        dest="secondary_test_index",
        type=Path,
        help="Test split ids for the secondary dataset (overrides SciFact default).",
    )
    parser.add_argument(
        "--scifact-train-index",
        dest="secondary_train_index",
        type=Path,
        default=Path("data/splits/scifact_train_ids.txt"),
        help="Alias for --secondary-train-index; defaults to SciFact ids.",
    )
    parser.add_argument(
        "--scifact-val-index",
        dest="secondary_val_index",
        type=Path,
        default=Path("data/splits/scifact_val_ids.txt"),
        help="Alias for --secondary-val-index; defaults to SciFact ids.",
    )
    parser.add_argument(
        "--scifact-test-index",
        dest="secondary_test_index",
        type=Path,
        default=Path("data/splits/scifact_test_ids.txt"),
        help="Alias for --secondary-test-index; defaults to SciFact ids.",
    )
    parser.add_argument(
        "--secondary-label",
        type=str,
        default="scifact",
        help="Human-friendly label for the secondary dataset (used in logs and summaries).",
    )
    parser.add_argument(
        "--ratio",
        type=str,
        default="3:1",
        help="FEVER:secondary batch ratio (e.g. 3:1)",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device",
    )
    parser.add_argument("--load-checkpoint", type=Path, default=Path("models/reliability_fever_base.pt"))
    parser.add_argument("--output-checkpoint", type=Path, help="Where to write the adapted checkpoint")
    parser.add_argument("--experiment-json", type=Path, help="Path to store metrics summary JSON")
    parser.add_argument("--ece-bins", type=int, default=15)
    parser.add_argument("--calibrate", action="store_true", help="Run threshold sweeps for both validation splits")
    parser.add_argument("--admit-grid", type=str, default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--margin-grid", type=str, default="-0.5,-0.25,0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1.0")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Optional cap on FEVER batches per epoch for quicker curricula debugging (0 = full epoch).",
    )
    return parser.parse_args()


def parse_ratio(spec: str) -> Tuple[int, int]:
    try:
        fever_part, scifact_part = spec.split(":", 1)
        fever_val = int(fever_part)
        scifact_val = int(scifact_part)
    except ValueError as exc:  # pragma: no cover - user input error
        raise ValueError(f"Invalid ratio spec '{spec}' (expected 'A:B')") from exc
    if fever_val <= 0 or scifact_val <= 0:
        raise ValueError("Ratio components must be positive integers")
    return fever_val, scifact_val


def build_loader(
    dataset: EvalDetailDataset,
    *,
    train_index: Path,
    val_index: Path,
    test_index: Path,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader, DataLoader, Subset, Subset, Subset]:
    train_ids = read_split_file(train_index)
    val_ids = read_split_file(val_index)
    test_ids = read_split_file(test_index)

    train_idx = map_ids_to_indices(dataset, train_ids, train_index)
    val_idx = map_ids_to_indices(dataset, val_ids, val_index)
    test_idx = map_ids_to_indices(dataset, test_ids, test_index)

    ensure_disjoint({"train": train_idx, "validation": val_idx, "test": test_idx})

    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    test_subset = Subset(dataset, test_idx)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    return train_loader, val_loader, test_loader, train_subset, val_subset, test_subset


def extend_embedding(state_dict: Dict[str, torch.Tensor], new_vocab: int) -> None:
    weight = state_dict.get("token_embed.weight")
    if not isinstance(weight, torch.Tensor):
        raise ValueError("Checkpoint missing token_embed.weight")
    old_vocab, dim = weight.shape
    if new_vocab == old_vocab:
        return
    if new_vocab < old_vocab:
        state_dict["token_embed.weight"] = weight[:new_vocab].clone()
        return
    # Grow embedding for new vocabulary entries.
    new_weight = torch.empty(new_vocab, dim)
    new_weight[:old_vocab] = weight
    torch.nn.init.normal_(new_weight[old_vocab:], mean=0.0, std=0.02)
    state_dict["token_embed.weight"] = new_weight


def run_step(
    model: OspaceTransformer,
    batch: Dict[str, torch.Tensor],
    config: OspaceTransformerConfig,
    device: torch.device,
    bce_loss: nn.Module,
    mse_loss: nn.Module,
) -> Tuple[torch.Tensor, float, float]:
    labels = batch["labels"].to(device)
    margins = batch["margins"].to(device)
    output = run_model(model, batch, device, config)
    admit_loss = bce_loss(output.admit_logits, labels)
    margin_loss = mse_loss(output.support_margin, margins)
    total_loss = admit_loss + 0.1 * margin_loss
    return total_loss, admit_loss.item(), margin_loss.item()


def evaluate_loader(
    model: OspaceTransformer,
    loader: DataLoader,
    config: OspaceTransformerConfig,
    device: torch.device,
    *,
    admit_threshold: float,
    margin_threshold: float,
    ece_bins: int,
) -> Dict[str, float]:
    if len(loader.dataset) == 0:
        return {
            "loss": float("nan"),
            "admit_loss": float("nan"),
            "margin_loss": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "brier": float("nan"),
            "ece": float("nan"),
        }

    bce_loss = nn.BCEWithLogitsLoss()
    mse_loss = nn.MSELoss()
    total = 0
    admit_total = 0
    margin_total = 0
    probs: List[float] = []
    labels: List[float] = []
    margins: List[float] = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            output = run_model(model, batch, device, config)
            label_tensor = batch["labels"].to(device)
            margin_tensor = batch["margins"].to(device)
            admit_val = bce_loss(output.admit_logits, label_tensor).item()
            margin_val = mse_loss(output.support_margin, margin_tensor).item()
            loss_val = admit_val + 0.1 * margin_val
            total += loss_val
            admit_total += admit_val
            margin_total += margin_val
            probs.extend(torch.sigmoid(output.admit_logits).detach().cpu().tolist())
            labels.extend(batch["labels"].tolist())
            margins.extend(output.support_margin.detach().cpu().tolist())

    precision, recall = compute_precision_recall(probs, labels, margins, admit_threshold, margin_threshold)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    brier = sum((p - y) ** 2 for p, y in zip(probs, labels)) / len(probs)
    ece = expected_calibration_error(probs, labels, num_bins=ece_bins)

    batches = len(loader)
    return {
        "loss": total / batches,
        "admit_loss": admit_total / batches,
        "margin_loss": margin_total / batches,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "brier": brier,
        "ece": ece,
    }


def calibrate_split(
    model: OspaceTransformer,
    loader: DataLoader,
    config: OspaceTransformerConfig,
    device: torch.device,
    admit_grid: Sequence[float],
    margin_grid: Sequence[float],
) -> Dict[str, float]:
    probs: List[float] = []
    labels: List[float] = []
    margins: List[float] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            output = run_model(model, batch, device, config)
            probs.extend(torch.sigmoid(output.admit_logits).detach().cpu().tolist())
            labels.extend(batch["labels"].tolist())
            margins.extend(output.support_margin.detach().cpu().tolist())
    if not probs:
        return {}
    return sweep_thresholds(probs, margins, labels, admit_grid, margin_grid)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    fever_ratio, secondary_ratio = parse_ratio(args.ratio)
    secondary_label_raw = (args.secondary_label or "secondary").strip()
    secondary_label = secondary_label_raw or "secondary"
    secondary_slug = secondary_label.lower().replace(" ", "_")

    checkpoint_payload = torch.load(args.load_checkpoint, map_location=device)
    vocab_payload = checkpoint_payload.get("tokenizer_vocab")
    tokenizer = WhitespaceTokenizer()
    if isinstance(vocab_payload, dict):
        # Seed the tokenizer with checkpoint vocabulary before datasets add new tokens.
        tokenizer.vocab = {str(k): int(v) for k, v in vocab_payload.items()}

    feature_dim = len(METRIC_KEYS)

    fever_dataset = EvalDetailDataset(
        args.fever_detail,
        tokenizer=tokenizer,
        feature_dim_override=feature_dim,
    )
    secondary_dataset = EvalDetailDataset(
        args.secondary_detail,
        tokenizer=tokenizer,
        feature_dim_override=feature_dim,
    )

    combined_vocab = tokenizer.vocab_size

    fever_train_loader, fever_val_loader, fever_test_loader, _, _, _ = build_loader(
        fever_dataset,
        train_index=args.fever_train_index,
        val_index=args.fever_val_index,
        test_index=args.fever_test_index,
        batch_size=args.batch_size,
    )
    secondary_train_loader, secondary_val_loader, secondary_test_loader, _, _, _ = build_loader(
        secondary_dataset,
        train_index=args.secondary_train_index,
        val_index=args.secondary_val_index,
        test_index=args.secondary_test_index,
        batch_size=args.batch_size,
    )

    config_blob = checkpoint_payload.get("config")
    if isinstance(config_blob, dict):
        config = OspaceTransformerConfig(**config_blob)
    elif isinstance(config_blob, OspaceTransformerConfig):
        config = config_blob
    else:
        raise ValueError("Checkpoint missing config payload")
    config.vocab_size = combined_vocab
    config.evidence_feature_dim = max(fever_dataset.feature_dim, secondary_dataset.feature_dim)

    state_dict = checkpoint_payload.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint missing state_dict")
    extend_embedding(state_dict, config.vocab_size)

    model = OspaceTransformer(config).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    bce_loss = nn.BCEWithLogitsLoss()
    mse_loss = nn.MSELoss()

    secondary_iterator: Iterator[Dict[str, torch.Tensor]] = iter(secondary_train_loader)

    def next_secondary_batch() -> Dict[str, torch.Tensor]:
        nonlocal secondary_iterator
        try:
            return next(secondary_iterator)
        except StopIteration:
            secondary_iterator = iter(secondary_train_loader)
            return next(secondary_iterator)

    admit_threshold = 0.5
    margin_threshold = 0.25

    history: List[Dict[str, object]] = []
    max_steps = max(0, args.max_steps)
    for epoch in range(1, args.epochs + 1):
        model.train()
        running: Dict[str, Deque[float]] = {
            "fever_loss": deque(maxlen=100),
            "fever_admit": deque(maxlen=100),
            "fever_margin": deque(maxlen=100),
            "secondary_loss": deque(maxlen=100),
            "secondary_admit": deque(maxlen=100),
            "secondary_margin": deque(maxlen=100),
        }
        secondary_credit = 0

        for step, fever_batch in enumerate(fever_train_loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            loss, admit_l, margin_l = run_step(model, fever_batch, config, device, bce_loss, mse_loss)
            loss.backward()
            optimizer.step()
            running["fever_loss"].append(float(loss.item()))
            running["fever_admit"].append(admit_l)
            running["fever_margin"].append(margin_l)

            secondary_credit += secondary_ratio
            while secondary_credit >= fever_ratio:
                secondary_batch = next_secondary_batch()
                optimizer.zero_grad(set_to_none=True)
                loss_secondary, admit_secondary, margin_secondary = run_step(
                    model,
                    secondary_batch,
                    config,
                    device,
                    bce_loss,
                    mse_loss,
                )
                loss_secondary.backward()
                optimizer.step()
                running["secondary_loss"].append(float(loss_secondary.item()))
                running["secondary_admit"].append(admit_secondary)
                running["secondary_margin"].append(margin_secondary)
                secondary_credit -= fever_ratio

            if step % 100 == 0:
                fever_avg = sum(running["fever_loss"]) / max(1, len(running["fever_loss"]))
                secondary_avg = sum(running["secondary_loss"]) / max(1, len(running["secondary_loss"]))
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "step": step,
                            "fever_loss": fever_avg,
                            f"{secondary_slug}_loss": secondary_avg,
                        }
                    )
                )

            if max_steps and step >= max_steps:
                break

        fever_val_metrics = evaluate_loader(
            model,
            fever_val_loader,
            config,
            device,
            admit_threshold=admit_threshold,
            margin_threshold=margin_threshold,
            ece_bins=args.ece_bins,
        )
        secondary_val_metrics = evaluate_loader(
            model,
            secondary_val_loader,
            config,
            device,
            admit_threshold=admit_threshold,
            margin_threshold=margin_threshold,
            ece_bins=args.ece_bins,
        )
        snapshot = {
            "epoch": epoch,
            "fever_val": fever_val_metrics,
            "secondary_val": secondary_val_metrics,
            f"{secondary_slug}_val": secondary_val_metrics,
        }
        history.append(snapshot)
        print(json.dumps(snapshot, indent=2))

    model.eval()

    admit_grid = parse_grid(args.admit_grid) if args.calibrate else []
    margin_grid = parse_grid(args.margin_grid) if args.calibrate else []
    fever_calibration = calibrate_split(model, fever_val_loader, config, device, admit_grid, margin_grid) if args.calibrate else {}
    secondary_calibration = (
        calibrate_split(model, secondary_val_loader, config, device, admit_grid, margin_grid)
        if args.calibrate
        else {}
    )

    fever_test_metrics = evaluate_loader(
        model,
        fever_test_loader,
        config,
        device,
        admit_threshold=fever_calibration.get("admit_threshold", admit_threshold),
        margin_threshold=fever_calibration.get("margin_threshold", margin_threshold),
        ece_bins=args.ece_bins,
    )
    secondary_test_metrics = evaluate_loader(
        model,
        secondary_test_loader,
        config,
        device,
        admit_threshold=secondary_calibration.get("admit_threshold", admit_threshold),
        margin_threshold=secondary_calibration.get("margin_threshold", margin_threshold),
        ece_bins=args.ece_bins,
    )

    summary = {
        "schedule": {
            "ratio": args.ratio,
            "fever_batches_per_epoch": len(fever_train_loader),
            "secondary_batches_per_epoch": len(secondary_train_loader),
            "secondary_label": secondary_label,
            "max_steps": max_steps,
        },
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "history": history,
        "validation": {
            "fever": history[-1]["fever_val"] if history else {},
            "secondary": history[-1]["secondary_val"] if history else {},
            secondary_slug: history[-1].get(f"{secondary_slug}_val", {}) if history else {},
        },
        "calibration": {
            "fever": fever_calibration,
            "secondary": secondary_calibration,
            secondary_slug: secondary_calibration,
        },
        "test": {
            "fever": fever_test_metrics,
            "secondary": secondary_test_metrics,
            secondary_slug: secondary_test_metrics,
        },
        "secondary_label": secondary_label,
        "checkpoint": str(args.output_checkpoint) if args.output_checkpoint else None,
    }

    if args.output_checkpoint:
        args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        save_checkpoint(
            args.output_checkpoint,
            model=model,
            config=config,
            vocab=tokenizer.vocab,
            metrics=summary,
        )
        print(f"[curriculum] checkpoint written to {args.output_checkpoint}")

    if args.experiment_json:
        args.experiment_json.parent.mkdir(parents=True, exist_ok=True)
        args.experiment_json.write_text(json.dumps(summary, indent=2, sort_keys=True))
        print(f"[curriculum] summary written to {args.experiment_json}")
    else:
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
