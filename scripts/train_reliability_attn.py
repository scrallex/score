#!/usr/bin/env python3
"""Skeleton training harness for the O-space Transformer reliability head."""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Subset

from sep_text_manifold.attn_ospace import (
    OspaceTransformer,
    OspaceTransformerConfig,
    OspaceTransformerOutput,
)

METRIC_KEYS: Sequence[str] = (
    "patternability",
    "semantic",
    "coherence",
    "stability",
    "entropy",
    "rupture",
    "lambda",
)

LABEL_TO_MARGIN: Dict[str, float] = {
    "SUPPORTED": 1.0,
    "REFUTED": -1.0,
    "UNVERIFIABLE": 0.0,
}


class WhitespaceTokenizer:
    """Minimal tokenizer that builds a vocab over whitespace-delimited tokens."""

    def __init__(self) -> None:
        self.vocab: Dict[str, int] = {"<pad>": 0, "<unk>": 1}

    def add_sentence(self, text: str) -> None:
        for token in text.lower().split():
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)

    def encode(self, text: str) -> List[int]:
        return [self.vocab.get(token, 1) for token in text.lower().split()] or [self.vocab["<unk>"]]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


class EvalDetailDataset(Dataset[Dict[str, object]]):
    """Wrap evaluation detail records for training."""

    def __init__(self, path: Path, tokenizer: Optional[WhitespaceTokenizer] = None) -> None:
        self.path = path
        self.records = self._load_records(path)
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        self.metric_keys: Sequence[str] = tuple(METRIC_KEYS)
        self.feature_dim = len(self.metric_keys)
        self.label_counts: Dict[str, int] = {}
        self._build_vocab()

    @staticmethod
    def _load_records(path: Path) -> List[Dict[str, object]]:
        if not path.exists():
            raise FileNotFoundError(f"Evaluation detail file not found: {path}")
        records: List[Dict[str, object]] = []
        with path.open() as fh:
            for line in fh:
                if line.strip():
                    records.append(json.loads(line))
        if not records:
            raise ValueError(f"No records loaded from {path}")
        return records

    def _build_vocab(self) -> None:
        for record in self.records:
            question = record.get("question", "")
            final_answer = record.get("final_answer", "")
            baseline = record.get("baseline_answer", "")
            self.tokenizer.add_sentence(question)
            self.tokenizer.add_sentence(final_answer)
            self.tokenizer.add_sentence(baseline)
            for entry in self._collect_evidence_entries(record):
                self.tokenizer.add_sentence(entry["text"])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        record = self.records[idx]
        question = record.get("question", "")
        final_answer = record.get("final_answer", "")
        baseline = record.get("baseline_answer", "")
        aggregate_text = " \u25cb ".join(token for token in [question, final_answer, baseline] if token)
        tokens = self.tokenizer.encode(aggregate_text)
        phase_values = self._extract_phase_vector(record, len(tokens))

        label_text = self._label_text(record)
        if label_text:
            self.label_counts[label_text] = self.label_counts.get(label_text, 0) + 1

        evidence_entries = self._collect_evidence_entries(record)
        evidence_payload = []
        for entry in evidence_entries:
            encoded = self.tokenizer.encode(entry["text"])
            if not encoded:
                continue
            evidence_payload.append({"tokens": encoded, "metrics": entry["metrics"]})

        if not evidence_payload:
            evidence_payload = [
                {
                    "tokens": tokens,
                    "metrics": [0.0] * self.feature_dim,
                }
            ]

        margin_target = self._derive_margin(record, label_text)
        label = 1.0 if self._is_supported(record, label_text) else 0.0
        return {
            "tokens": tokens,
            "label": label,
            "margin": margin_target,
            "phase": phase_values,
            "evidence": evidence_payload,
        }

    def _collect_evidence_entries(self, record: Dict[str, object]) -> List[Dict[str, object]]:
        entries: List[Dict[str, object]] = []

        def metrics_vector(metrics: Dict[str, object]) -> List[float]:
            return [float(metrics.get(key, 0.0)) for key in self.metric_keys]

        for sentence in record.get("sentences", []) or []:
            text = sentence.get("sentence", "")
            metrics = sentence.get("metrics", {}) if isinstance(sentence, dict) else {}
            if text:
                entries.append({"text": text, "metrics": metrics_vector(metrics)})
            for twin in sentence.get("twins", []) or []:
                if not isinstance(twin, dict):
                    continue
                twin_text = twin.get("string", "")
                if not twin_text:
                    continue
                twin_metrics = {
                    "patternability": twin.get("patternability", 0.0),
                    "semantic": twin.get("semantic_similarity", 0.0),
                    "lambda": twin.get("hazard", 0.0),
                }
                entries.append({"text": twin_text, "metrics": metrics_vector(twin_metrics)})

        citations = record.get("citations")
        if isinstance(citations, list):
            for item in citations:
                if isinstance(item, str) and item:
                    entries.append({"text": item, "metrics": [0.0] * self.feature_dim})

        # Allow derived evidence collections (e.g. FEVER ingest) under "evidence"
        extra_evidence = record.get("evidence")
        if isinstance(extra_evidence, list):
            for payload in extra_evidence:
                if isinstance(payload, dict):
                    text = str(payload.get("text") or "").strip()
                    metrics = payload.get("metrics", {}) if isinstance(payload.get("metrics"), dict) else {}
                    if text:
                        entries.append({"text": text, "metrics": metrics_vector(metrics)})

        return entries

    def _derive_margin(self, record: Dict[str, object], label_text: str) -> float:
        margin = 0.0
        for sentence in record.get("sentences", []) or []:
            if not isinstance(sentence, dict):
                continue
            metrics = sentence.get("metrics", {}) if isinstance(sentence.get("metrics"), dict) else {}
            margin = max(margin, float(metrics.get("semantic", 0.0)))
        if margin > 0.0:
            return margin
        return LABEL_TO_MARGIN.get(label_text, 0.0)

    def _label_text(self, record: Dict[str, object]) -> str:
        return str(record.get("expected") or record.get("label") or "").upper()

    def _is_supported(self, record: Dict[str, object], label_text: str) -> bool:
        if bool(record.get("supported")):
            return True
        return label_text == "SUPPORTED"

    @staticmethod
    def _extract_phase_vector(record: Dict[str, object], length: int) -> List[float]:
        phase_raw = record.get("phase_values")
        if isinstance(phase_raw, list) and all(isinstance(val, (int, float)) for val in phase_raw):
            floats = [float(val) for val in phase_raw]
            if len(floats) >= length:
                return floats[:length]
            return floats + [0.0] * (length - len(floats))
        return [0.0] * length


def collate_batch(batch: Sequence[Dict[str, object]]) -> Dict[str, torch.Tensor]:
    token_tensors = [torch.tensor(sample["tokens"], dtype=torch.long) for sample in batch]
    input_ids = pad_sequence(token_tensors, batch_first=True, padding_value=0)
    attention_mask = (input_ids != 0).long()
    labels = torch.tensor([sample["label"] for sample in batch], dtype=torch.float32)
    margins = torch.tensor([sample["margin"] for sample in batch], dtype=torch.float32)

    phase_tensors = [torch.tensor(sample["phase"], dtype=torch.float32) for sample in batch]
    phase_values = pad_sequence(phase_tensors, batch_first=True, padding_value=0.0)

    max_evidence_count = max(1, max(len(sample["evidence"]) for sample in batch))
    evidence_lengths = [
        len(entry["tokens"])
        for sample in batch
        for entry in (sample["evidence"] if sample["evidence"] else [])
    ]
    default_len = max(len(sample["tokens"]) for sample in batch)
    max_evidence_len = max(1, max(evidence_lengths) if evidence_lengths else default_len, default_len)
    feature_dim = len(METRIC_KEYS)
    evidence_ids = torch.zeros(len(batch), max_evidence_count, max_evidence_len, dtype=torch.long)
    evidence_token_mask = torch.zeros_like(evidence_ids)
    evidence_features = torch.zeros(len(batch), max_evidence_count, feature_dim, dtype=torch.float32)

    for batch_idx, sample in enumerate(batch):
        entries = sample["evidence"]
        if not entries:
            entries = [
                {
                    "tokens": sample["tokens"],
                    "metrics": [0.0] * feature_dim,
                }
            ]
        for evidence_idx, entry in enumerate(entries[:max_evidence_count]):
            seq = entry["tokens"]
            metrics = entry.get("metrics") or []
            seq_len = min(len(seq), max_evidence_len)
            if seq_len:
                evidence_ids[batch_idx, evidence_idx, :seq_len] = torch.tensor(
                    seq[:seq_len], dtype=torch.long
                )
                evidence_token_mask[batch_idx, evidence_idx, :seq_len] = 1
            if metrics:
                metrics_tensor = torch.tensor(metrics[:feature_dim], dtype=torch.float32)
                if metrics_tensor.numel() < feature_dim:
                    pad = torch.zeros(feature_dim - metrics_tensor.numel(), dtype=torch.float32)
                    metrics_tensor = torch.cat([metrics_tensor, pad])
            else:
                metrics_tensor = torch.zeros(feature_dim, dtype=torch.float32)
            evidence_features[batch_idx, evidence_idx] = metrics_tensor

    evidence_mask = (evidence_token_mask.sum(dim=-1) > 0).long()

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "margins": margins,
        "phase_values": phase_values,
        "evidence_ids": evidence_ids,
        "evidence_mask": evidence_mask,
        "evidence_token_mask": evidence_token_mask,
        "evidence_features": evidence_features,
    }


def train_val_loaders(
    path: Path,
    batch_size: int,
    val_ratio: float,
    seed: int,
) -> Tuple[EvalDetailDataset, DataLoader[Dict[str, torch.Tensor]], DataLoader[Dict[str, torch.Tensor]]]:
    dataset = EvalDetailDataset(path)
    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    if len(indices) <= 1 or val_ratio <= 0.0:
        train_subset = Subset(dataset, indices)
        val_subset = Subset(dataset, [])
    else:
        val_size = max(1, int(len(indices) * val_ratio))
        if val_size >= len(indices):
            val_size = len(indices) - 1
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    return dataset, train_loader, val_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("detail_path", type=Path, help="Path to eval_detail.jsonl used for training")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dry-run", action="store_true", help="Load data and run a single forward pass only")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Fraction of data used for validation")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for train/val split")
    parser.add_argument("--admit-threshold", type=float, default=0.5, help="Decision threshold for admit probability")
    parser.add_argument("--margin-threshold", type=float, default=0.25, help="Decision threshold for support margin")
    parser.add_argument(
        "--attention-entropy-weight",
        type=float,
        default=0.0,
        help="Regularisation weight encouraging low-entropy attention",
    )
    parser.add_argument(
        "--output-checkpoint",
        type=Path,
        help="Path to write the trained reliability checkpoint (config, vocab, state_dict)",
    )
    parser.add_argument(
        "--disable-phase-channel",
        action="store_true",
        help="Disable phase channel injection when embedding tokens",
    )
    parser.add_argument(
        "--disable-cross-attention",
        action="store_true",
        help="Skip evidence cross-attention when set (ablations)",
    )
    parser.add_argument(
        "--max-evidence-len",
        type=int,
        default=256,
        help="Maximum evidence token length to encode",
    )
    parser.add_argument(
        "--evidence-encoder-layers",
        type=int,
        default=1,
        help="Number of self-attention layers used to encode evidence sentences",
    )
    parser.add_argument(
        "--evidence-encoder-heads",
        type=int,
        default=4,
        help="Number of attention heads for the evidence encoder",
    )
    parser.add_argument(
        "--calibrate-thresholds",
        action="store_true",
        help="Sweep admit/margin thresholds on the validation set to maximise F1",
    )
    parser.add_argument(
        "--admit-grid",
        type=str,
        default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
        help="Comma-separated admit thresholds used during calibration sweeps",
    )
    parser.add_argument(
        "--margin-grid",
        type=str,
        default="-0.5,-0.25,0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1.0",
        help="Comma-separated margin thresholds used during calibration sweeps",
    )
    parser.add_argument(
        "--ece-bins",
        type=int,
        default=10,
        help="Number of bins used when computing Expected Calibration Error",
    )
    return parser.parse_args()


def run_model(
    model: OspaceTransformer,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    config: OspaceTransformerConfig,
) -> OspaceTransformerOutput:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    phase_values = batch.get("phase_values")
    if phase_values is not None:
        phase_tensor = phase_values.to(device)
        phase_features = phase_tensor.unsqueeze(-1).repeat(1, 1, config.d_model)
    else:
        phase_features = None

    evidence_ids = batch.get("evidence_ids")
    evidence_mask = batch.get("evidence_mask")
    evidence_token_mask = batch.get("evidence_token_mask")
    evidence_features = batch.get("evidence_features")

    if evidence_ids is not None:
        evidence_ids = evidence_ids.to(device)
        evidence_tokens = evidence_ids
        token_mask_tensor = evidence_token_mask.to(device) if evidence_token_mask is not None else None
        feature_tensor = evidence_features.to(device) if evidence_features is not None else None
        memory_mask = None
    else:
        evidence_tokens = None
        token_mask_tensor = None
        feature_tensor = None
        memory_mask = None

    return model(
        input_ids,
        attention_mask=attention_mask,
        phase_features=phase_features,
        evidence_memory=None,
        evidence_mask=memory_mask,
        evidence_tokens=evidence_tokens,
        evidence_token_mask=token_mask_tensor,
        evidence_features=feature_tensor,
    )


def save_checkpoint(
    path: Path,
    *,
    model: OspaceTransformer,
    config: OspaceTransformerConfig,
    vocab: Dict[str, int],
    metrics: Dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    payload = {
        "config": dataclasses.asdict(config),
        "state_dict": state_dict,
        "tokenizer_vocab": vocab,
        "metrics": metrics,
    }
    torch.save(payload, path)


def train() -> None:
    args = parse_args()
    dataset, train_loader, val_loader = train_val_loaders(
        args.detail_path, args.batch_size, args.val_ratio, args.seed
    )

    device = torch.device(args.device)

    if args.ece_bins <= 0:
        raise ValueError("--ece-bins must be a positive integer")

    config = OspaceTransformerConfig(
        vocab_size=dataset.tokenizer.vocab_size,
        use_phase_channel=not args.disable_phase_channel,
        use_cross_attention=not args.disable_cross_attention,
        max_evidence_len=args.max_evidence_len,
        evidence_encoder_layers=args.evidence_encoder_layers,
        evidence_encoder_heads=args.evidence_encoder_heads,
        evidence_feature_dim=dataset.feature_dim,
    )
    model = OspaceTransformer(config).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    bce_loss = nn.BCEWithLogitsLoss()
    mse_loss = nn.MSELoss()

    def run_step(batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, float, float, float]:
        labels = batch["labels"].to(device)
        margins = batch["margins"].to(device)
        output = run_model(model, batch, device, config)
        admit_loss = bce_loss(output.admit_logits, labels)
        margin_loss = mse_loss(output.support_margin, margins)
        attention_reg = 0.0
        if args.attention_entropy_weight > 0.0 and output.evidence_attention is not None:
            attn = torch.clamp(output.evidence_attention.squeeze(1), min=1e-8)
            attention_reg = -(attn * attn.log()).sum(dim=-1).mean()
        total_loss = admit_loss + 0.1 * margin_loss + args.attention_entropy_weight * attention_reg
        return total_loss, admit_loss.item(), margin_loss.item(), float(attention_reg)

    @torch.no_grad()
    def evaluate(
        loader: DataLoader[Dict[str, torch.Tensor]]
    ) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
        if len(loader.dataset) == 0:
            metrics = {
                "loss": float("nan"),
                "admit_loss": float("nan"),
                "margin_loss": float("nan"),
                "attention_reg": float("nan"),
                "precision": float("nan"),
                "recall": float("nan"),
                "f1": float("nan"),
                "brier": float("nan"),
                "ece": float("nan"),
            }
            return metrics, {"probs": [], "labels": [], "margins": []}

        model.eval()
        cumulative = {"loss": 0.0, "admit": 0.0, "margin": 0.0, "attn": 0.0}
        probs: List[float] = []
        labels_list: List[float] = []
        margin_preds: List[float] = []
        batches = 0

        for batch in loader:
            output = run_model(model, batch, device, config)
            label_tensor = batch["labels"].to(device)
            margin_tensor = batch["margins"].to(device)

            admit_loss_val = bce_loss(output.admit_logits, label_tensor).item()
            margin_loss_val = mse_loss(output.support_margin, margin_tensor).item()
            attention_reg = 0.0
            if args.attention_entropy_weight > 0.0 and output.evidence_attention is not None:
                attn = torch.clamp(output.evidence_attention.squeeze(1), min=1e-8)
                attention_reg = float(-(attn * attn.log()).sum(dim=-1).mean().item())

            total_loss = admit_loss_val + 0.1 * margin_loss_val + args.attention_entropy_weight * attention_reg
            cumulative["loss"] += total_loss
            cumulative["admit"] += admit_loss_val
            cumulative["margin"] += margin_loss_val
            cumulative["attn"] += attention_reg
            batches += 1

            probs.extend(torch.sigmoid(output.admit_logits).detach().cpu().tolist())
            labels_list.extend(batch["labels"].tolist())
            margin_preds.extend(output.support_margin.detach().cpu().tolist())

        precision, recall = compute_precision_recall(
            probs, labels_list, margin_preds, args.admit_threshold, args.margin_threshold
        )
        if precision + recall == 0.0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        brier = sum((p - y) ** 2 for p, y in zip(probs, labels_list)) / len(probs)
        ece = expected_calibration_error(probs, labels_list, num_bins=args.ece_bins)

        metrics = {
            "loss": cumulative["loss"] / batches,
            "admit_loss": cumulative["admit"] / batches,
            "margin_loss": cumulative["margin"] / batches,
            "attention_reg": cumulative["attn"] / batches,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "brier": brier,
            "ece": ece,
        }
        predictions = {"probs": probs, "labels": labels_list, "margins": margin_preds}
        return metrics, predictions

    if args.dry_run:
        batch = next(iter(train_loader))
        loss, admit_l, margin_l, attn_reg = run_step(batch)
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "loss": float(loss.item()),
                    "admit_loss": admit_l,
                    "margin_loss": margin_l,
                    "attention_reg": attn_reg,
                    "vocab_size": dataset.tokenizer.vocab_size,
                },
                indent=2,
            )
        )
        return

    final_val_metrics: Optional[Dict[str, float]] = None
    last_val_predictions: Dict[str, List[float]] = {"probs": [], "labels": [], "margins": []}
    calibration_summary: Optional[Dict[str, float]] = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_admit = 0.0
        running_margin = 0.0
        running_attn = 0.0
        for batch_idx, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            loss, admit_l, margin_l, attn_reg = run_step(batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_admit += admit_l
            running_margin += margin_l
            running_attn += attn_reg
            if batch_idx % 10 == 0:
                total = batch_idx
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "step": batch_idx,
                            "loss": running_loss / total,
                            "admit_loss": running_admit / total,
                            "margin_loss": running_margin / total,
                            "attention_reg": running_attn / total,
                        }
                    )
                )

        train_metrics = {
            "epoch": epoch,
            "train_loss": running_loss / max(1, len(train_loader)),
            "train_admit_loss": running_admit / max(1, len(train_loader)),
            "train_margin_loss": running_margin / max(1, len(train_loader)),
            "train_attention_reg": running_attn / max(1, len(train_loader)),
        }
        val_metrics, val_predictions = evaluate(val_loader)
        final_val_metrics = val_metrics
        last_val_predictions = val_predictions
        train_metrics.update({f"val_{key}": value for key, value in val_metrics.items()})
        print(json.dumps(train_metrics, indent=2))

    if args.calibrate_thresholds and last_val_predictions["labels"]:
        admit_grid = parse_grid(args.admit_grid)
        margin_grid = parse_grid(args.margin_grid)
        calibration_summary = sweep_thresholds(
            last_val_predictions["probs"],
            last_val_predictions["margins"],
            last_val_predictions["labels"],
            admit_grid,
            margin_grid,
        )
        print(
            json.dumps(
                {
                    "calibration": calibration_summary,
                    "admit_grid": admit_grid,
                    "margin_grid": margin_grid,
                },
                indent=2,
            )
        )
    elif args.calibrate_thresholds:
        print("[train_reliability_attn] Skipping calibration sweep (no validation data).")

    if args.output_checkpoint:
        metrics = final_val_metrics or {}
        if calibration_summary is not None:
            metrics = {**metrics, "calibration": calibration_summary}
        save_checkpoint(
            args.output_checkpoint,
            model=model,
            config=config,
            vocab=dataset.tokenizer.vocab,
            metrics=metrics,
        )
        print(f"[train_reliability_attn] Checkpoint written to {args.output_checkpoint}")


def compute_precision_recall(
    probs: Sequence[float],
    labels: Sequence[float],
    margin_preds: Sequence[float],
    admit_threshold: float,
    margin_threshold: float,
) -> Tuple[float, float]:
    tp = 0
    fp = 0
    fn = 0
    for prob, label, margin in zip(probs, labels, margin_preds):
        predicted = (prob >= admit_threshold) and (margin >= margin_threshold)
        label_bool = label >= 0.5
        if predicted and label_bool:
            tp += 1
        elif predicted and not label_bool:
            fp += 1
        elif (not predicted) and label_bool:
            fn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return precision, recall


def parse_grid(spec: str) -> List[float]:
    values: List[float] = []
    for raw in spec.split(','):
        item = raw.strip()
        if not item:
            continue
        try:
            values.append(float(item))
        except ValueError as exc:  # pragma: no cover - user configuration error
            raise ValueError(f"Invalid grid value '{item}' in '{spec}'") from exc
    if not values:
        raise ValueError(f"Calibration grid '{spec}' produced no values")
    return values


def expected_calibration_error(
    probs: Sequence[float],
    labels: Sequence[float],
    *,
    num_bins: int,
) -> float:
    if not probs:
        return float('nan')
    bin_totals = [0 for _ in range(num_bins)]
    bin_confidence = [0.0 for _ in range(num_bins)]
    bin_accuracy = [0.0 for _ in range(num_bins)]
    for prob, label in zip(probs, labels):
        clipped = min(max(prob, 0.0), 1.0)
        index = min(num_bins - 1, int(clipped * num_bins))
        bin_totals[index] += 1
        bin_confidence[index] += clipped
        bin_accuracy[index] += float(label)
    total = float(len(probs))
    ece = 0.0
    for count, conf_sum, acc_sum in zip(bin_totals, bin_confidence, bin_accuracy):
        if count == 0:
            continue
        avg_conf = conf_sum / count
        avg_acc = acc_sum / count
        ece += (count / total) * abs(avg_acc - avg_conf)
    return ece


def sweep_thresholds(
    probs: Sequence[float],
    margins: Sequence[float],
    labels: Sequence[float],
    admit_grid: Sequence[float],
    margin_grid: Sequence[float],
) -> Dict[str, float]:
    best = {
        "f1": 0.0,
        "admit_threshold": admit_grid[0],
        "margin_threshold": margin_grid[0],
        "precision": 0.0,
        "recall": 0.0,
    }
    for admit in admit_grid:
        for margin in margin_grid:
            precision, recall = compute_precision_recall(probs, labels, margins, admit, margin)
            if precision + recall == 0.0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)
            if f1 > best["f1"]:
                best.update(
                    {
                        "f1": f1,
                        "admit_threshold": admit,
                        "margin_threshold": margin,
                        "precision": precision,
                        "recall": recall,
                    }
                )
    return best

    if args.dry_run:
        batch = next(iter(loader))
        loss, admit_l, margin_l = run_step(batch)
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "loss": float(loss.item()),
                    "admit_loss": admit_l,
                    "margin_loss": margin_l,
                    "vocab_size": dataset.tokenizer.vocab_size,
                },
                indent=2,
            )
        )
        return

    model.train()
    for epoch in range(1, args.epochs + 1):
        running_loss = 0.0
        running_admit = 0.0
        running_margin = 0.0
        for batch_idx, batch in enumerate(loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            loss, admit_l, margin_l = run_step(batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_admit += admit_l
            running_margin += margin_l

            if batch_idx % 10 == 0:
                total = batch_idx
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "step": batch_idx,
                            "loss": running_loss / total,
                            "admit_loss": running_admit / total,
                            "margin_loss": running_margin / total,
                        }
                    )
                )

if __name__ == "__main__":  # pragma: no cover - script entrypoint
    train()
