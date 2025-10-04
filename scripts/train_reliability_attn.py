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
            for evidence in self._extract_evidence_strings(record):
                self.tokenizer.add_sentence(evidence)

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
        evidence_tokens = [self.tokenizer.encode(text) for text in self._extract_evidence_strings(record)]
        if not evidence_tokens:
            evidence_tokens = [tokens]

        # Use the maximum semantic margin from the sentences as a proxy target.
        margin_target = 0.0
        for sentence in record.get("sentences", []):
            metrics = sentence.get("metrics", {})
            margin_target = max(margin_target, float(metrics.get("semantic", 0.0)))

        label = float(bool(record.get("supported", False)))
        return {
            "tokens": tokens,
            "label": label,
            "margin": margin_target,
            "phase": phase_values,
            "evidence": evidence_tokens,
        }

    @staticmethod
    def _extract_evidence_strings(record: Dict[str, object]) -> List[str]:
        evidence: List[str] = []
        for sentence in record.get("sentences", []):
            text = sentence.get("sentence", "")
            if text:
                evidence.append(text)
            for twin in sentence.get("twins", []):
                twin_str = twin.get("string", "")
                if twin_str:
                    evidence.append(twin_str)
        citations = record.get("citations")
        if isinstance(citations, list):
            for item in citations:
                if isinstance(item, str) and item:
                    evidence.append(item)
        return evidence

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
    max_evidence_len = max(
        1,
        max(
            len(sequence)
            for sample in batch
            for sequence in (sample["evidence"] if sample["evidence"] else [[0]])
        ),
    )
    evidence_ids = torch.zeros(len(batch), max_evidence_count, max_evidence_len, dtype=torch.long)
    evidence_mask = torch.zeros(len(batch), max_evidence_count, dtype=torch.long)

    for batch_idx, sample in enumerate(batch):
        evidence_sequences = sample["evidence"] or [sample["tokens"]]
        for evidence_idx, sequence in enumerate(evidence_sequences[:max_evidence_count]):
            seq_len = min(len(sequence), max_evidence_len)
            if seq_len:
                evidence_ids[batch_idx, evidence_idx, :seq_len] = torch.tensor(
                    sequence[:seq_len], dtype=torch.long
                )
                evidence_mask[batch_idx, evidence_idx] = 1

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "margins": margins,
        "phase_values": phase_values,
        "evidence_ids": evidence_ids,
        "evidence_mask": evidence_mask,
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
    if evidence_ids is not None:
        evidence_ids = evidence_ids.to(device)
        evidence_mask = evidence_mask.to(device) if evidence_mask is not None else None
        batch_size, mem_count, mem_len = evidence_ids.shape
        flat_ids = evidence_ids.view(batch_size * mem_count, mem_len)
        token_mask = (flat_ids != 0).unsqueeze(-1)
        embedded = model.token_embed(flat_ids)
        summed = (embedded * token_mask).sum(dim=1)
        lengths = token_mask.sum(dim=1).clamp_min(1)
        mean_embeddings = summed / lengths
        evidence_memory = mean_embeddings.view(batch_size, mem_count, -1)
    else:
        evidence_memory = None
        evidence_mask = None

    return model(
        input_ids,
        attention_mask=attention_mask,
        phase_features=phase_features,
        evidence_memory=evidence_memory,
        evidence_mask=evidence_mask,
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

    config = OspaceTransformerConfig(vocab_size=dataset.tokenizer.vocab_size)
    model = OspaceTransformer(config).to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    bce_loss = nn.BCEWithLogitsLoss()
    mse_loss = nn.MSELoss()

    def run_step(batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, float, float, float]:
        labels = batch["labels"].to(args.device)
        margins = batch["margins"].to(args.device)
        output = run_model(model, batch, args.device, config)
        admit_loss = bce_loss(output.admit_logits, labels)
        margin_loss = mse_loss(output.support_margin, margins)
        attention_reg = 0.0
        if args.attention_entropy_weight > 0.0 and output.evidence_attention is not None:
            attn = torch.clamp(output.evidence_attention.squeeze(1), min=1e-8)
            attention_reg = -(attn * attn.log()).sum(dim=-1).mean()
        total_loss = admit_loss + 0.1 * margin_loss + args.attention_entropy_weight * attention_reg
        return total_loss, admit_loss.item(), margin_loss.item(), float(attention_reg)

    @torch.no_grad()
    def evaluate(loader: DataLoader[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        if len(loader.dataset) == 0:
            return {key: float("nan") for key in [
                "loss",
                "admit_loss",
                "margin_loss",
                "attention_reg",
                "precision",
                "recall",
                "brier",
            ]}
        model.eval()
        total = 0
        cumulative = {"loss": 0.0, "admit": 0.0, "margin": 0.0, "attn": 0.0}
        probs: List[float] = []
        labels: List[float] = []
        margin_preds: List[float] = []
        for batch in loader:
            loss, admit_l, margin_l, attn_reg = run_step(batch)
            cumulative["loss"] += loss.item()
            cumulative["admit"] += admit_l
            cumulative["margin"] += margin_l
            cumulative["attn"] += attn_reg
            total += 1

            output = run_model(model, batch, args.device, config)
            probs.extend(torch.sigmoid(output.admit_logits).detach().cpu().tolist())
            labels.extend(batch["labels"].tolist())
            margin_preds.extend(output.support_margin.detach().cpu().tolist())

        precision, recall = compute_precision_recall(
            probs, labels, margin_preds, args.admit_threshold, args.margin_threshold
        )
        brier = sum((p - y) ** 2 for p, y in zip(probs, labels)) / len(probs)
        return {
            "loss": cumulative["loss"] / total,
            "admit_loss": cumulative["admit"] / total,
            "margin_loss": cumulative["margin"] / total,
            "attention_reg": cumulative["attn"] / total,
            "precision": precision,
            "recall": recall,
            "brier": brier,
        }

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
        val_metrics = evaluate(val_loader)
        final_val_metrics = val_metrics
        train_metrics.update({f"val_{key}": value for key, value in val_metrics.items()})
        print(json.dumps(train_metrics, indent=2))

    if args.output_checkpoint:
        metrics = final_val_metrics or {}
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
