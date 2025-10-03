#!/usr/bin/env python3
"""Generate deterministic evaluation claims for a truth-pack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


DEFAULT_TOTAL = 120
SUPPORTED_RATIO = 0.4
UNVERIFIABLE_RATIO = 0.4
REFUTED_RATIO = 0.2


def load_state(manifest_path: Path) -> Dict[str, object]:
    manifest = json.loads(manifest_path.read_text())
    state_path = Path(manifest["state_path"])
    if not state_path.exists():
        raise FileNotFoundError(f"State file not found: {state_path}")
    return manifest, json.loads(state_path.read_text())


def pick_strings(state: Dict[str, object], min_occ: int) -> List[Tuple[str, Dict[str, object]]]:
    strings = state.get("string_scores", {})  # type: ignore[assignment]
    items = [
        (name, data)
        for name, data in strings.items()
        if int(data.get("occurrences", 0)) >= min_occ and len(name.split()) <= 6
    ]
    items.sort(key=lambda kv: (-int(kv[1].get("occurrences", 0)), kv[0]))
    return items


def build_claims(
    items: List[Tuple[str, Dict[str, object]]],
    *,
    pack_name: str,
    total: int,
    seed_prefix: str,
) -> List[Dict[str, object]]:
    if not items:
        raise ValueError("No candidate strings available; lower --min-occurrences")

    supported_n = max(1, int(total * SUPPORTED_RATIO))
    unverified_n = max(1, int(total * UNVERIFIABLE_RATIO))
    refuted_n = max(1, total - supported_n - unverified_n)

    supported = items[:supported_n]
    refuted_base = items[supported_n:supported_n + refuted_n]
    if len(refuted_base) < refuted_n:
        refuted_base = items[:refuted_n]

    claims: List[Dict[str, object]] = []

    for idx, (token, data) in enumerate(supported, 1):
        qid = f"{seed_prefix}S{idx:03d}"
        claims.append(
            {
                "id": qid,
                "question": f"What does the documentation state about '{token}'?",
                "expected": "SUPPORTED",
                "gold_uris": [f"doc://{pack_name}#{token}"],
                "notes": f"Auto-generated supported claim for token '{token}'",
            }
        )

    for idx, (token, _) in enumerate(refuted_base, 1):
        qid = f"{seed_prefix}R{idx:03d}"
        claims.append(
            {
                "id": qid,
                "question": f"Is it correct that '{token}' is explicitly denied in the pack?",
                "expected": "REFUTED",
                "gold_uris": [f"doc://{pack_name}#{token}"],
                "notes": f"Auto-generated refuted claim built from token '{token}'",
            }
        )

    for idx in range(1, unverified_n + 1):
        token = f"unverified_claim_{idx:03d}"
        qid = f"{seed_prefix}U{idx:03d}"
        claims.append(
            {
                "id": qid,
                "question": f"What policy covers '{token}'?",
                "expected": "UNVERIFIABLE",
                "gold_uris": [],
                "notes": "Synthetic unverified claim",
            }
        )

    claims.sort(key=lambda c: c["id"])
    return claims[:total]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--total", type=int, default=DEFAULT_TOTAL)
    parser.add_argument("--min-occurrences", type=int, default=3)
    args = parser.parse_args()

    manifest, state = load_state(args.manifest)
    items = pick_strings(state, min_occ=args.min_occurrences)
    pack_name = manifest.get("name") or Path(manifest["pack_path"]).name
    seed_prefix = (pack_name or "PACK").upper()[:4]

    claims = build_claims(items, pack_name=pack_name, total=args.total, seed_prefix=seed_prefix)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for claim in claims:
            fh.write(json.dumps(claim, ensure_ascii=False) + "\n")

    print(f"Wrote {len(claims)} claims to {args.output}")
    print("Review checklist:")
    print("- Verify SUPPORTED claims map to real content")
    print("- Spot-check REFUTED/UNVERIFIABLE for phrasing")


if __name__ == "__main__":
    main()
