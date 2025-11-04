#!/usr/bin/env python3
"""Generate canonical manifold metric expectations on synthetic bitstreams."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

try:
    import sep_quantum
except ImportError as exc:  # pragma: no cover - native module is optional
    raise SystemExit(
        "sep_quantum native module is required. Build it with 'pip install .[native]' first."
    ) from exc


@dataclass
class SyntheticSpec:
    name: str
    description: str
    generator: callable
    samples: int


def constant_bits(length: int, value: int) -> List[int]:
    return [value] * length


def alternating_bits(length: int) -> List[int]:
    return [(i % 2) for i in range(length)]


def biased_random_walk(length: int, flip_bias: float, seed: int) -> List[int]:
    rng = random.Random(seed)
    bits = [rng.choice([0, 1])]
    for _ in range(1, length):
        if rng.random() < flip_bias:
            bits.append(1 - bits[-1])
        else:
            bits.append(bits[-1])
    return bits


def noise_with_bursts(length: int, burst_probability: float, burst_length: int, seed: int) -> List[int]:
    rng = random.Random(seed)
    bits = [0] * length
    idx = 0
    while idx < length:
        if rng.random() < burst_probability:
            burst_end = min(length, idx + rng.randint(1, burst_length))
            for j in range(idx, burst_end):
                bits[j] = 1
            idx = burst_end
        else:
            idx += 1
    return bits


def random_noise(length: int, seed: int) -> List[int]:
    rng = random.Random(seed)
    return [rng.getrandbits(1) for _ in range(length)]


def analyze_bits(bits: Sequence[int]) -> Dict[str, object]:
    result = sep_quantum.analyze_window(list(bits))
    events = list(sep_quantum.transform_rich(list(bits)))
    aggregates = list(sep_quantum.aggregate_events(events))

    event_counter = Counter(evt.state.name for evt in events)
    aggregate_rows = [
        {
            "state": agg.state.name,
            "count": int(agg.count),
        }
        for agg in aggregates
    ]

    return {
        "metrics": {
            "coherence": float(result.coherence),
            "stability": float(result.stability),
            "entropy": float(result.entropy),
            "rupture_ratio": float(result.rupture_ratio),
            "flip_ratio": float(result.flip_ratio),
        },
        "counts": {
            "null_state": int(result.null_state_count),
            "flip": int(result.flip_count),
            "rupture": int(result.rupture_count),
        },
        "event_histogram": dict(event_counter),
        "aggregate_events": aggregate_rows,
    }


def build_specs(length: int, samples: int) -> Iterable[SyntheticSpec]:
    return [
        SyntheticSpec(
            name="constant_zero",
            description="All-zero bitstream",
            generator=lambda idx: constant_bits(length, 0),
            samples=1,
        ),
        SyntheticSpec(
            name="constant_one",
            description="All-one bitstream",
            generator=lambda idx: constant_bits(length, 1),
            samples=1,
        ),
        SyntheticSpec(
            name="alternating",
            description="Alternating 0/1 pattern",
            generator=lambda idx: alternating_bits(length),
            samples=1,
        ),
        SyntheticSpec(
            name="biased_random_walk",
            description="Random walk with mild flip bias",
            generator=lambda idx: biased_random_walk(length, 0.35, seed=10_000 + idx),
            samples=samples,
        ),
        SyntheticSpec(
            name="noise_with_bursts",
            description="Mostly zeros with occasional bursts of ones",
            generator=lambda idx: noise_with_bursts(length, 0.015, 12, seed=20_000 + idx),
            samples=samples,
        ),
        SyntheticSpec(
            name="random_noise",
            description="Uniform random bits",
            generator=lambda idx: random_noise(length, seed=30_000 + idx),
            samples=samples,
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic manifold datasets")
    parser.add_argument("--length", type=int, default=256, help="Bitstream length per sample")
    parser.add_argument("--samples", type=int, default=5, help="Samples per stochastic pattern")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/qfh_synthetic_native.json"),
        help="Destination JSON file",
    )
    args = parser.parse_args()

    records: List[Dict[str, object]] = []
    for spec in build_specs(args.length, args.samples):
        for idx in range(spec.samples):
            bits = spec.generator(idx)
            analysis = analyze_bits(bits)
            records.append(
                {
                    "pattern": spec.name,
                    "description": spec.description,
                    "sample_index": idx,
                    "length": len(bits),
                    "analysis": analysis,
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "records": len(records)}, indent=2))


if __name__ == "__main__":
    main()
