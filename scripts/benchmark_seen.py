#!/usr/bin/env python3
"""Benchmark the /seen FastAPI service to confirm throughput >= 1k rps."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import httpx
import asyncio

APP_PATH = "scripts.reality_filter_service:app"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("analysis/truth_packs/docs_demo/manifest.json"))
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--concurrency", type=int, default=200)
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--timeout", type=float, default=60.0)
    return parser.parse_args()


async def send_requests(base_url: str, payload: dict, total: int, concurrency: int, timeout: float):
    latencies = []
    errors = 0

    requests_per_worker = max(1, total // concurrency)
    remainder = total % concurrency

    async def worker(n_requests: int):
        nonlocal errors
        async with httpx.AsyncClient(base_url=base_url, timeout=timeout) as client:
            for _ in range(n_requests):
                start = time.perf_counter()
                try:
                    resp = await client.post("/seen", json=payload)
                    if resp.status_code != 200:
                        errors += 1
                    else:
                        _ = resp.json()
                except Exception:
                    errors += 1
                else:
                    latencies.append((time.perf_counter() - start) * 1000)

    tasks = []
    for i in range(concurrency):
        n = requests_per_worker + (1 if i < remainder else 0)
        if n == 0:
            continue
        tasks.append(asyncio.create_task(worker(n)))
    await asyncio.gather(*tasks)
    return latencies, errors


def main() -> None:
    args = parse_args()
    manifest = args.manifest.resolve()
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    env = os.environ.copy()
    cmd = [
        "python",
        "-m",
        "uvicorn",
        APP_PATH,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--log-level",
        "error",
    ]
    server = subprocess.Popen(cmd, env=env)
    try:
        base_url = f"http://127.0.0.1:{args.port}"
        deadline = time.time() + 10
        ready = False
        while time.time() < deadline:
            try:
                response = httpx.get(f"{base_url}/healthz", timeout=1.0)
                if response.status_code == 200:
                    ready = True
                    break
            except Exception:
                time.sleep(0.1)
        if not ready:
            raise RuntimeError("Service did not start in time")

        payload = {
            "text": "guardrail",
            "question": None,
            "pack_manifest": str(manifest),
            "embedding_method": "hash",
            "hash_dims": 256,
        }

        # Warmup to load embeddings/caches.
        for _ in range(5):
            httpx.post(f"{base_url}/seen", json=payload, timeout=args.timeout)

        start = time.perf_counter()
        latencies, errors = asyncio.run(
            send_requests(base_url, payload, args.requests, args.concurrency, args.timeout)
        )
        elapsed = time.perf_counter() - start

        successful = len(latencies)
        throughput = successful / elapsed if elapsed else 0.0
        p50 = statistics.median(latencies) if latencies else 0.0
        p90 = statistics.quantiles(latencies, n=10)[8] if len(latencies) >= 10 else p50

        print(json.dumps(
            {
                "total_requests": args.requests,
                "successful": successful,
                "errors": errors,
                "elapsed_sec": elapsed,
                "throughput_rps": throughput,
                "latency_ms_p50": p50,
                "latency_ms_p90": p90,
            },
            indent=2,
        ))
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()


if __name__ == "__main__":
    main()
