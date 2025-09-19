"""
String extraction and aggregation for the Sep Text Manifold.

This module provides utilities to extract tokens from text and to
aggregate quantum metrics over the occurrences of each string.  It
aligns each token back onto the byte windows produced by the manifold
builder and computes mean values of coherence, stability, entropy and
rupture for each string.

These functions are intentionally simple and can be refined.  For
example, you may wish to support n‑grams, phrase detection or more
sophisticated tokenisation, and to compute additional statistics
(standard deviation, 95th percentiles, etc.).
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict, Dict, Iterable, List, Optional, Set, Tuple


@dataclass
class StringOccurrence:
    """Represents a single occurrence of a string in a document."""
    string: str
    file_id: str
    char_start: int
    char_end: int
    byte_start: int
    byte_end: int


def extract_strings(text: str, file_id: str) -> List[StringOccurrence]:
    """Extract simple word tokens from *text*.

    This function uses a regular expression to find sequences of
    alphanumeric characters.  It returns a list of ``StringOccurrence``
    objects with both character and byte offsets.  Byte offsets are
    calculated relative to the UTF‑8 encoding of the entire text.
    """
    occurrences: List[StringOccurrence] = []
    # Precompute cumulative byte lengths for each character index to
    # avoid repeatedly encoding prefixes.  `byte_offsets[i]` will be
    # the number of bytes in text[:i].
    # Build mapping from char index to byte index.  We step through the
    # encoded bytes and note when the next character begins.
    byte_offsets = [0] * (len(text) + 1)
    byte_index = 0
    for char_index, ch in enumerate(text):
        byte_offsets[char_index] = byte_index
        byte_index += len(ch.encode("utf-8"))
    byte_offsets[len(text)] = byte_index
    # Use regex to extract words
    for match in re.finditer(r"\b\w+\b", text, flags=re.UNICODE):
        word = match.group(0)
        cs = match.start()
        ce = match.end()
        bs = byte_offsets[cs]
        be = byte_offsets[ce]
        occurrences.append(StringOccurrence(word.lower(), file_id, cs, ce, bs, be))
    return occurrences


def aggregate_string_metrics(
    occurrences: Iterable[StringOccurrence],
    signals: List[Dict[str, object]],
    *,
    window_bytes: int,
    stride: int,
    aggregated_fields: Optional[Iterable[str]] = None,
) -> Dict[str, Dict[str, object]]:
    """Aggregate manifold metrics over occurrences of each string.

    Parameters
    ----------
    occurrences:
        Iterable of ``StringOccurrence`` objects.
    signals:
        List of signal dictionaries returned from `build_manifold`.  Each
        signal must contain an ``index`` (the end byte position of the
        window) and a ``metrics`` dict.
    window_bytes:
        Width of each window in bytes – must match what was passed to
        `build_manifold`.
    stride:
        Stride between windows in bytes – must match what was passed to
        `build_manifold`.
    aggregated_fields:
        Optional list of metric field names to aggregate.  Defaults to
        ``["coherence", "stability", "entropy", "rupture"]``.

    Returns
    -------
    Dict[str, Dict[str, object]]
        A mapping from string to a dictionary containing:
        ``metrics`` – the mean metric values over covering windows;
        ``occurrences`` – how many times the string was observed; and
        ``window_ids`` – the sorted list of window identifiers that
        covered at least one occurrence.
    """
    if aggregated_fields is None:
        aggregated_fields = ["coherence", "stability", "entropy", "rupture"]
    # Build list of tuples: (window_id, window_start, window_end, metrics)
    window_ranges: List[Tuple[int, int, int, Dict[str, float]]] = []
    for fallback_id, sig in enumerate(signals):
        end = int(sig.get("window_end", sig.get("index", 0)))
        start = int(sig.get("window_start", end - window_bytes))
        if start < 0:
            start = 0
        metrics = sig.get("metrics", {})
        window_id = int(sig.get("id", fallback_id))
        window_ranges.append((window_id, start, end, metrics))
    metric_samples: DefaultDict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    window_hits: DefaultDict[str, Set[int]] = defaultdict(set)
    occurrence_counts: DefaultDict[str, int] = defaultdict(int)
    for occ in occurrences:
        occurrence_counts[occ.string] += 1
        for wid, ws, we, metrics in window_ranges:
            if occ.byte_start >= ws and occ.byte_end <= we:
                window_hits[occ.string].add(wid)
                for field in aggregated_fields:
                    value = metrics.get(field)
                    if value is not None:
                        metric_samples[occ.string][field].append(float(value))
    result: Dict[str, Dict[str, object]] = {}
    for s, field_values in metric_samples.items():
        metric_means: Dict[str, float] = {}
        for field, values in field_values.items():
            if values:
                metric_means[field] = sum(values) / len(values)
        result[s] = {
            "metrics": metric_means,
            "occurrences": occurrence_counts.get(s, 0),
            "window_ids": sorted(window_hits.get(s, set())),
        }
    for s, count in occurrence_counts.items():
        if s not in result:
            result[s] = {
                "metrics": {},
                "occurrences": count,
                "window_ids": [],
            }
    return result
