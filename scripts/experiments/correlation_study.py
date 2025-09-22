#!/usr/bin/env python3
"""Compute proxy correlations between QFH metrics and domain-level success."""

from __future__ import annotations

import argparse
import csv
import json
from math import sqrt
from pathlib import Path
from typing import Dict, List

FIELDS = [
    'id',
    'window_start',
    'window_end',
    'index',
    'coherence',
    'stability',
    'entropy',
    'rupture',
    'lambda_hazard',
]


def load_domain_summary(summary_dir: Path, domain: str) -> Dict[str, object]:
    summary_path = summary_dir / f"{domain}_state_native.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    return json.loads(summary_path.read_text(encoding='utf-8'))


def load_domain_csv(summary_dir: Path, domain: str) -> List[Dict[str, float]]:
    csv_path = summary_dir / f"{domain}_state_native.csv"
    rows: List[Dict[str, float]] = []
    if not csv_path.exists():
        return rows
    with csv_path.open(encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append({
                'coherence': float(row['coherence']) if row['coherence'] else 0.0,
                'stability': float(row['stability']) if row['stability'] else 0.0,
                'entropy': float(row['entropy']) if row['entropy'] else 0.0,
                'lambda_hazard': float(row['lambda_hazard']) if row['lambda_hazard'] else 0.0,
            })
    return rows


def pearson(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    if n == 0 or len(ys) != n:
        return 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    sd_x = sqrt(sum((x - mean_x) ** 2 for x in xs))
    sd_y = sqrt(sum((y - mean_y) ** 2 for y in ys))
    return cov / (sd_x * sd_y) if sd_x and sd_y else 0.0


def kendall(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    if n < 2 or len(ys) != n:
        return 0.0
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            sign = (xs[i] - xs[j]) * (ys[i] - ys[j])
            if sign > 0:
                concordant += 1
            elif sign < 0:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total else 0.0


def compute_domain_correlations(summary_dir: Path, domains: List[str]) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, List[float]] = {}
    success_rates: List[float] = []
    for domain in domains:
        summary = load_domain_summary(summary_dir, domain)
        success_rates.append(float(summary['success_rate_proxy']))
        for key, value in summary['metric_means'].items():
            metrics.setdefault(key, []).append(float(value))
    correlations: Dict[str, Dict[str, float]] = {}
    for metric, values in metrics.items():
        correlations[metric] = {
            'pearson': pearson(values, success_rates),
            'kendall': kendall(values, success_rates),
        }
    return correlations


def scatter_points(summary_dir: Path, domains: List[str]) -> List[Dict[str, float]]:
    points = []
    for domain in domains:
        summary = load_domain_summary(summary_dir, domain)
        points.append({
            'domain': domain,
            'lambda_mean': float(summary['metric_means']['lambda_mean']),
            'success_proxy': float(summary['success_rate_proxy']),
        })
    return points


def main() -> None:
    parser = argparse.ArgumentParser(description='Compute proxy correlations for PlanBench domains')
    parser.add_argument('--summary-dir', default='score/output/planbench_native_summary')
    parser.add_argument('--domains', default='blocksworld,mystery_bw,logistics')
    parser.add_argument('--out', default='score/output/planbench_native_summary/domain_correlations.json')
    args = parser.parse_args()

    summary_dir = Path(args.summary_dir)
    domains = [d.strip() for d in args.domains.split(',') if d.strip()]

    correlations = compute_domain_correlations(summary_dir, domains)
    points = scatter_points(summary_dir, domains)

    payload = {
        'domains': domains,
        'correlations': correlations,
        'lambda_vs_success': points,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
