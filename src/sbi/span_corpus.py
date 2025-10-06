"""Utilities to extract span-level data from truth-pack corpora."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

LOGGER = logging.getLogger(__name__)


def _normalise_text(text: str) -> str:
    return " ".join(text.strip().split())


def compute_span_id(text: str) -> str:
    """Compute deterministic span identifier (hex) from normalised text."""
    norm = _normalise_text(text).lower()
    digest = hashlib.blake2b(norm.encode("utf-8"), digest_size=16).hexdigest()
    return digest


@dataclasses.dataclass(slots=True)
class ContextAccumulator:
    count: int = 0
    uri_set: set[str] = dataclasses.field(default_factory=set)
    domain_set: set[str] = dataclasses.field(default_factory=set)
    timebin_set: set[str] = dataclasses.field(default_factory=set)
    unique_uris: int = 0
    unique_domains: int = 0
    unique_timebins: int = 0

    def add(self, *, uri: Optional[str], domain: Optional[str], timebin: Optional[str]) -> None:
        self.count += 1
        if uri:
            self.uri_set.add(uri)
        if domain:
            self.domain_set.add(domain)
        if timebin:
            self.timebin_set.add(timebin)
        self.unique_uris = len(self.uri_set)
        self.unique_domains = len(self.domain_set)
        self.unique_timebins = len(self.timebin_set)

    def as_dict(self) -> Dict[str, object]:
        return {
            "count": self.count,
            "unique_uris": self.unique_uris,
            "unique_domains": self.unique_domains,
            "unique_timebins": self.unique_timebins,
            "reinforcement": self.unique_uris + self.unique_domains,
        }


@dataclasses.dataclass(slots=True)
class SpanAggregate:
    span_id: str
    text: str
    occurrences: int
    sources: Dict[str, int]
    span_types: Dict[str, int]
    citations: Dict[str, int]
    left_contexts: Dict[str, ContextAccumulator]
    right_contexts: Dict[str, ContextAccumulator]

    def to_jsonable(self, *, k_ctx: int | None = None) -> Dict[str, object]:
        def top_k(contexts: Dict[str, ContextAccumulator]) -> List[Dict[str, object]]:
            items = sorted(
                contexts.items(),
                key=lambda item: (item[1].count, item[0]),
                reverse=True,
            )
            if k_ctx is not None:
                items = items[:k_ctx]
            return [
                {
                    "text": text,
                    **acc.as_dict(),
                }
                for text, acc in items
            ]

        return {
            "span_id": self.span_id,
            "text": self.text,
            "occurrences": self.occurrences,
            "sources": self.sources,
            "span_types": self.span_types,
            "citations": self.citations,
            "left_contexts": top_k(self.left_contexts),
            "right_contexts": top_k(self.right_contexts),
        }


@dataclasses.dataclass(slots=True)
class _SpanAggregateMutable:
    text: str
    occurrences: int = 0
    sources: Counter = dataclasses.field(default_factory=Counter)
    span_types: Counter = dataclasses.field(default_factory=Counter)
    citations: Counter = dataclasses.field(default_factory=Counter)
    left_contexts: Dict[str, ContextAccumulator] = dataclasses.field(default_factory=dict)
    right_contexts: Dict[str, ContextAccumulator] = dataclasses.field(default_factory=dict)

    def freeze(self, span_id: str) -> SpanAggregate:
        return SpanAggregate(
            span_id=span_id,
            text=self.text,
            occurrences=self.occurrences,
            sources=dict(self.sources),
            span_types=dict(self.span_types),
            citations=dict(self.citations),
            left_contexts={k: v for k, v in self.left_contexts.items()},
            right_contexts={k: v for k, v in self.right_contexts.items()},
        )


@dataclasses.dataclass(slots=True)
class EvidenceEntry:
    text: str
    citations: Tuple[str, ...]


@dataclasses.dataclass(slots=True)
class RecordSpans:
    record_id: str
    claim: str
    final_answer: str
    evidences: Tuple[EvidenceEntry, ...]


def load_manifest(path: Path) -> Dict[str, object]:
    data = json.loads(path.read_text())
    return data


def iter_pack_records(manifest_path: Path) -> Iterator[RecordSpans]:
    manifest = load_manifest(manifest_path)
    yield from _iter_pack_records(manifest)


def _iter_pack_records(manifest: Dict[str, object]) -> Iterator[RecordSpans]:
    corpus_path = Path(manifest["pack_path"]).resolve()
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus directory missing: {corpus_path}")
    chunk_paths = sorted(corpus_path.glob("*.txt"))
    if not chunk_paths:
        raise FileNotFoundError(f"No chunk files found under {corpus_path}")

    for chunk_path in chunk_paths:
        with chunk_path.open("r", encoding="utf-8") as handle:
            lines: List[str] = []
            for raw in handle:
                line = raw.rstrip("\n")
                stripped = line.strip()
                if stripped == "---":
                    if lines:
                        record = _parse_record_lines(lines)
                        if record is not None:
                            yield record
                    lines = []
                    continue
                if not stripped and not lines:
                    continue  # skip leading blanks
                lines.append(line)
            if lines:
                record = _parse_record_lines(lines)
                if record is not None:
                    yield record


def _parse_record_lines(lines: Sequence[str]) -> Optional[RecordSpans]:
    record_id = ""
    claim = ""
    final_answer = ""
    evidences: List[EvidenceEntry] = []
    current_evidence: Optional[List[str]] = None
    current_citations: List[str] = []
    in_evidence = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("ID:"):
            record_id = stripped.split(":", 1)[1].strip()
            continue
        if stripped.startswith("Claim:"):
            claim = stripped.split(":", 1)[1].strip()
            continue
        if stripped.startswith("Final Answer:"):
            final_answer = stripped.split(":", 1)[1].strip()
            continue
        if stripped.startswith("Evidence:"):
            in_evidence = True
            if current_evidence is not None and current_evidence:
                evidences.append(
                    EvidenceEntry(text=" ".join(current_evidence).strip(), citations=tuple(current_citations))
                )
            current_evidence = []
            current_citations = []
            continue

        if not in_evidence:
            continue

        if "(sentence" in stripped:
            if current_evidence is not None:
                current_citations.append(stripped)
            continue

        if current_evidence is None:
            current_evidence = []
            current_citations = []
        current_evidence.append(stripped)

    if current_evidence is not None and current_evidence:
        evidences.append(EvidenceEntry(text=" ".join(current_evidence).strip(), citations=tuple(current_citations)))

    if not record_id:
        return None
    return RecordSpans(
        record_id=record_id,
        claim=claim,
        final_answer=final_answer,
        evidences=tuple(evidences),
    )


def build_span_aggregates(manifest_path: Path) -> Dict[str, SpanAggregate]:
    manifest = load_manifest(manifest_path)
    aggregates: Dict[str, _SpanAggregateMutable] = {}

    for record in _iter_pack_records(manifest):
        ordered_spans: List[Tuple[str, str, Tuple[str, ...]]] = []
        if record.claim:
            ordered_spans.append((record.claim, "claim", ()))
        if record.final_answer:
            ordered_spans.append((record.final_answer, "final_answer", ()))
        for evidence in record.evidences:
            text = evidence.text.strip()
            if not text:
                continue
            ordered_spans.append((text, "evidence", evidence.citations))

        for idx, (text, span_type, citations) in enumerate(ordered_spans):
            norm_text = _normalise_text(text)
            if not norm_text:
                continue
            span_id = compute_span_id(text)
            entry = aggregates.get(span_id)
            if entry is None:
                entry = aggregates[span_id] = _SpanAggregateMutable(text=text)
            entry.occurrences += 1
            entry.sources[record.record_id] += 1
            entry.span_types[span_type] += 1
            for citation in citations:
                entry.citations[citation] += 1

            uri = f"fever://{record.record_id}#{span_type}"
            domain = citations[0] if citations else record.record_id
            timebin = "na"

            left_text: Optional[str] = None
            if idx > 0:
                left_text = ordered_spans[idx - 1][0]
            if left_text:
                left_norm = _normalise_text(left_text)
                if left_norm:
                    acc = entry.left_contexts.get(left_norm)
                    if acc is None:
                        acc = entry.left_contexts[left_norm] = ContextAccumulator()
                    acc.add(uri=uri, domain=domain, timebin=timebin)

            right_text: Optional[str] = None
            if idx + 1 < len(ordered_spans):
                right_text = ordered_spans[idx + 1][0]
            if right_text:
                right_norm = _normalise_text(right_text)
                if right_norm:
                    acc = entry.right_contexts.get(right_norm)
                    if acc is None:
                        acc = entry.right_contexts[right_norm] = ContextAccumulator()
                    acc.add(uri=uri, domain=domain, timebin=timebin)

    frozen: Dict[str, SpanAggregate] = {}
    for span_id, mutable in aggregates.items():
        frozen[span_id] = mutable.freeze(span_id)
    LOGGER.info("Constructed %d unique spans", len(frozen))
    return frozen


def write_span_inventory(
    aggregates: Dict[str, SpanAggregate],
    *,
    output_path: Path,
    k_ctx: int = 64,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for span in aggregates.values():
            payload = span.to_jsonable(k_ctx=k_ctx)
            handle.write(json.dumps(payload, ensure_ascii=False))
            handle.write("\n")


def load_span_inventory(path: Path) -> Dict[str, SpanAggregate]:
    spans: Dict[str, SpanAggregate] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            left_contexts = {}
            for ctx in payload.get("left_contexts", []):
                acc = ContextAccumulator(
                    count=int(ctx.get("count", 0)),
                    uri_set=set(),
                    domain_set=set(),
                    timebin_set=set(),
                    unique_uris=int(ctx.get("unique_uris", 0)),
                    unique_domains=int(ctx.get("unique_domains", 0)),
                    unique_timebins=int(ctx.get("unique_timebins", 0)),
                )
                left_contexts[ctx["text"]] = acc

            right_contexts = {}
            for ctx in payload.get("right_contexts", []):
                acc = ContextAccumulator(
                    count=int(ctx.get("count", 0)),
                    uri_set=set(),
                    domain_set=set(),
                    timebin_set=set(),
                    unique_uris=int(ctx.get("unique_uris", 0)),
                    unique_domains=int(ctx.get("unique_domains", 0)),
                    unique_timebins=int(ctx.get("unique_timebins", 0)),
                )
                right_contexts[ctx["text"]] = acc
            spans[payload["span_id"]] = SpanAggregate(
                span_id=payload["span_id"],
                text=payload["text"],
                occurrences=int(payload.get("occurrences", 0)),
                sources={k: int(v) for k, v in payload.get("sources", {}).items()},
                span_types={k: int(v) for k, v in payload.get("span_types", {}).items()},
                citations={k: int(v) for k, v in payload.get("citations", {}).items()},
                left_contexts=left_contexts,
                right_contexts=right_contexts,
            )
    return spans


def write_context_table(
    aggregates: Dict[str, SpanAggregate],
    *,
    output_path: Path,
    k_ctx: int = 64,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def top_contexts(contexts: Dict[str, ContextAccumulator]) -> List[Dict[str, object]]:
        items = sorted(contexts.items(), key=lambda item: (item[1].count, item[0]), reverse=True)
        if k_ctx > 0:
            items = items[:k_ctx]
        return [
            {
                "text": text,
                **acc.as_dict(),
            }
            for text, acc in items
        ]

    with output_path.open("w", encoding="utf-8") as handle:
        for span in aggregates.values():
            payload = {
                "span_id": span.span_id,
                "span": span.text,
                "left": top_contexts(span.left_contexts),
                "right": top_contexts(span.right_contexts),
            }
            handle.write(json.dumps(payload, ensure_ascii=False))
            handle.write("\n")


__all__ = [
    "SpanAggregate",
    "ContextAccumulator",
    "RecordSpans",
    "iter_pack_records",
    "build_span_aggregates",
    "write_span_inventory",
    "write_context_table",
    "load_span_inventory",
    "compute_span_id",
]
