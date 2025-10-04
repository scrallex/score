#!/usr/bin/env python3
"""Skeleton training harness for the O-space Transformer reliability head."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from sep_text_manifold.attn_ospace import OspaceTransformer, OspaceTransformerConfig


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
            for sentence in record.get("sentences", []):
                text = sentence.get("sentence", "")
                self.tokenizer.add_sentence(text)
                for twin in sentence.get("twins", []):
                    self.tokenizer.add_sentence(twin.get("string", ""))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        record = self.records[idx]
        question = record.get("question", "")
        final_answer = record.get("final_answer", "")
        baseline = record.get("baseline_answer", "")
        aggregate_text = " \u25cb ".join(token for token in [question, final_answer, baseline] if token)
        tokens = self.tokenizer.encode(aggregate_text)

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
        }


def collate_batch(batch: Sequence[Dict[str, object]]) -> Dict[str, torch.Tensor]:
    token_tensors = [torch.tensor(sample["tokens"], dtype=torch.long) for sample in batch]
    input_ids = pad_sequence(token_tensors, batch_first=True, padding_value=0)
    attention_mask = (input_ids != 0).long()
    labels = torch.tensor([sample["label"] for sample in batch], dtype=torch.float32)
    margins = torch.tensor([sample["margin"] for sample in batch], dtype=torch.float32)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "margins": margins,
    }


def build_dataloader(path: Path, batch_size: int) -> Tuple[EvalDetailDataset, DataLoader[Dict[str, torch.Tensor]]]:
    dataset = EvalDetailDataset(path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    return dataset, loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("detail_path", type=Path, help="Path to eval_detail.jsonl used for training")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dry-run", action="store_true", help="Load data and run a single forward pass only")
    return parser.parse_args()


def train() -> None:
    args = parse_args()
    dataset, loader = build_dataloader(args.detail_path, args.batch_size)

    config = OspaceTransformerConfig(vocab_size=dataset.tokenizer.vocab_size)
    model = OspaceTransformer(config).to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    bce_loss = nn.BCEWithLogitsLoss()
    mse_loss = nn.MSELoss()

    def run_step(batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, float, float]:
        input_ids = batch["input_ids"].to(args.device)
        attention_mask = batch["attention_mask"].to(args.device)
        labels = batch["labels"].to(args.device)
        margins = batch["margins"].to(args.device)

        output = model(input_ids, attention_mask=attention_mask)
        admit_loss = bce_loss(output.admit_logits, labels)
        margin_loss = mse_loss(output.support_margin, margins)
        loss = admit_loss + 0.1 * margin_loss
        return loss, admit_loss.item(), margin_loss.item()

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

        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "loss": running_loss / max(1, len(loader)),
                    "admit_loss": running_admit / max(1, len(loader)),
                    "margin_loss": running_margin / max(1, len(loader)),
                },
                indent=2,
            )
        )


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    train()

