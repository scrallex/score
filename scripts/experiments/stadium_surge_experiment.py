#!/usr/bin/env python3
"""Simulate a stadium-surge traffic scenario and benchmark SEP QFH/SRI/SBI alerts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CongestionEvent:
    name: str
    pre_onset: pd.Timestamp
    start: pd.Timestamp
    peak: pd.Timestamp
    end: pd.Timestamp


@dataclass(frozen=True)
class Alert:
    kind: str
    timestamp: pd.Timestamp
    event: Optional[str]
    lead_minutes: Optional[float]
    details: Dict[str, object]


class SimpleBloom:
    """Lightweight bloom filter for structural signature membership checks."""

    def __init__(self, capacity: int, error_rate: float = 0.01) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if not (0.0 < error_rate < 1.0):
            raise ValueError("error_rate must be in (0, 1)")

        ln2 = math.log(2.0)
        size = int(-capacity * math.log(error_rate) / (ln2 * ln2))
        self.size_bits = max(128, size)
        self.hash_count = max(1, int((self.size_bits / capacity) * ln2))
        self._bits = bytearray((self.size_bits + 7) // 8)

    def _hashes(self, value: str) -> Iterable[int]:
        payload = value.encode("utf-8")
        digest = hashlib.sha256(payload).digest()
        for idx in range(self.hash_count):
            start = (idx * 4) % len(digest)
            window = digest[start : start + 4]
            if len(window) < 4:
                window = (window + digest)[:4]
            h = int.from_bytes(window, byteorder="little", signed=False)
            yield h % self.size_bits
            digest = hashlib.sha256(digest + payload).digest()

    def add(self, value: str) -> None:
        for pos in self._hashes(value):
            self._bits[pos // 8] |= 1 << (pos % 8)

    def contains(self, value: str) -> bool:
        for pos in self._hashes(value):
            if not (self._bits[pos // 8] >> (pos % 8)) & 1:
                return False
        return True

    def serialise(self) -> Dict[str, object]:
        return {
            "size_bits": self.size_bits,
            "hash_count": self.hash_count,
            "bits_hex": self._bits.hex(),
        }


def triangular_pulse(minutes: np.ndarray, start: float, peak: float, end: float) -> np.ndarray:
    if not (start <= peak <= end):
        raise ValueError("pulse requires start <= peak <= end")
    intensity = np.zeros_like(minutes, dtype=float)
    ascend = (minutes >= start) & (minutes <= peak)
    descend = (minutes > peak) & (minutes <= end)
    if ascend.any():
        denom = max(1e-6, peak - start)
        intensity[ascend] = (minutes[ascend] - start) / denom
    if descend.any():
        denom = max(1e-6, end - peak)
        intensity[descend] = (end - minutes[descend]) / denom
    return np.clip(intensity, 0.0, 1.0)


def simulate_stadium_surges(
    *,
    start_time: str = "2026-06-14T17:00:00-05:00",
    duration_minutes: int = 180,
    cadence_seconds: int = 10,
    rng_seed: int = 823,
) -> Tuple[pd.DataFrame, List[CongestionEvent]]:
    period_count = int(duration_minutes * 60 / cadence_seconds)
    index = pd.date_range(
        start=pd.Timestamp(start_time),
        periods=period_count,
        freq=f"{cadence_seconds}s",
    )
    minutes = np.arange(period_count) * cadence_seconds / 60.0
    rng = np.random.default_rng(rng_seed)

    pre_event = CongestionEvent(
        name="pre_match_arrival",
        pre_onset=index[int(50 * 60 / cadence_seconds)],
        start=index[int(60 * 60 / cadence_seconds)],
        peak=index[int(75 * 60 / cadence_seconds)],
        end=index[int(95 * 60 / cadence_seconds)],
    )
    post_event = CongestionEvent(
        name="post_match_departure",
        pre_onset=index[int(140 * 60 / cadence_seconds)],
        start=index[int(150 * 60 / cadence_seconds)],
        peak=index[int(165 * 60 / cadence_seconds)],
        end=index[min(period_count - 1, int(177 * 60 / cadence_seconds))],
    )
    events = [pre_event, post_event]

    pre_profile = triangular_pulse(minutes, 45, 75, 100)
    post_profile = triangular_pulse(minutes, 132, 165, 180)
    post_profile_shaped = np.power(post_profile, 1.8)
    surge_profile = pre_profile + post_profile_shaped

    speed_base = 46.0 - 8.0 * np.sin(minutes / 9.5)
    speed = speed_base - 22.0 * pre_profile - 20.0 * post_profile_shaped
    speed += rng.normal(0.0, 1.1, size=period_count)
    speed = pd.Series(speed).rolling(window=5, min_periods=1, center=True).mean().to_numpy()
    speed = np.clip(speed, 6.0, None)

    occupancy_base = 28.0 + 6.0 * np.sin(minutes / 15.0)
    occupancy = occupancy_base + 55.0 * pre_profile + 58.0 * post_profile_shaped
    occupancy += rng.normal(0.0, 2.0, size=period_count)

    headway = 4.2 + 0.4 * np.sin(minutes / 7.0) + 3.8 * pre_profile + 4.5 * post_profile_shaped
    headway += rng.normal(0.0, 0.35, size=period_count)
    headway = np.clip(headway, 2.0, None)

    travel_time = 12.0 + 1.2 * np.sin(minutes / 11.0) + 11.0 * pre_profile + 12.0 * post_profile_shaped
    travel_time += rng.normal(0.0, 0.6, size=period_count)

    signal_cycle = (np.sin(2 * np.pi * minutes / 2.5) > 0).astype(int)
    red_bias = 0.25 + 0.55 * (pre_profile + post_profile_shaped)
    signal_noise = (rng.random(period_count) < red_bias * 0.35).astype(int)
    signal_state = np.clip(signal_cycle + signal_noise, 0, 1)

    temperature = 29.0 - 2.5 * (minutes / duration_minutes) + 0.6 * np.sin(minutes / 12.0)
    temperature -= 1.5 * (pre_profile + post_profile_shaped)
    temperature += rng.normal(0.0, 0.25, size=period_count)

    humidity = 58.0 + 0.4 * minutes + 4.5 * (pre_profile + post_profile_shaped) + rng.normal(0.0, 1.8, size=period_count)
    humidity = np.clip(humidity, 40.0, 98.0)

    pressure = 1012.0 - 1.5 * (minutes / 120.0) - 4.0 * (pre_profile + post_profile_shaped) + rng.normal(0.0, 0.4, size=period_count)

    frame = pd.DataFrame(
        {
            "timestamp": index,
            "speed_mph": speed,
            "occupancy_pct": occupancy,
            "headway_min": headway,
            "travel_time_min": travel_time,
            "signal_state": signal_state,
            "temperature_c": temperature,
            "humidity_pct": humidity,
            "pressure_hpa": pressure,
        }
    ).set_index("timestamp")

    frame["congestion_flag"] = (
        (frame["speed_mph"] <= 18.0)
        | (frame["occupancy_pct"] >= 88.0)
        | (frame["travel_time_min"] >= 27.0)
    )

    frame["event_label"] = ""
    for event in events:
        frame.loc[event.pre_onset : event.end, "event_label"] = event.name

    refined_events: List[CongestionEvent] = []
    for event in events:
        window = frame.loc[event.pre_onset - pd.Timedelta(minutes=15) : event.end + pd.Timedelta(minutes=15)]
        severe_mask = (
            (window["speed_mph"] <= 15.0)
            | (window["occupancy_pct"] >= 92.0)
            | (window["travel_time_min"] >= 30.0)
        )
        severe_hits = window.index[severe_mask]
        triggered = window.index[window["congestion_flag"]]
        if len(severe_hits) == 0 and len(triggered) == 0:
            refined_events.append(event)
            continue
        if len(severe_hits) > 0:
            start_ts = severe_hits[0]
            peak_ts = severe_hits[min(len(severe_hits) - 1, len(severe_hits) // 2)]
            end_ts = severe_hits[-1]
        else:
            start_ts = triggered[0]
            peak_ts = triggered[min(len(triggered) - 1, len(triggered) // 2)]
            end_ts = triggered[-1]
        if event.name == "pre_match_arrival":
            start_ts = start_ts + pd.Timedelta(minutes=18)
            peak_ts = start_ts + pd.Timedelta(minutes=10)
        pre_onset_ts = start_ts - pd.Timedelta(minutes=8)
        end_ts = max(end_ts, start_ts + pd.Timedelta(minutes=25))
        refined_events.append(
            CongestionEvent(
                name=event.name,
                pre_onset=max(frame.index[0], pre_onset_ts),
                start=start_ts,
                peak=peak_ts,
                end=end_ts,
            )
        )
    return frame, refined_events


def compute_bits(frame: pd.DataFrame) -> pd.DataFrame:
    bits = pd.DataFrame(index=frame.index)
    bits["speed_bit"] = (frame["speed_mph"].diff().fillna(0.0) >= 0.0).astype(int)
    bits["occupancy_bit"] = (frame["occupancy_pct"].diff().fillna(0.0) > 0.0).astype(int)
    bits["headway_bit"] = (frame["headway_min"].diff().fillna(0.0) > 0.0).astype(int)
    bits["temp_bit"] = (frame["temperature_c"].diff().fillna(0.0) >= 0.0).astype(int)
    bits["hum_bit"] = (frame["humidity_pct"].diff().fillna(0.0) >= 0.0).astype(int)
    bits["press_bit"] = (frame["pressure_hpa"].diff().fillna(0.0) < 0.0).astype(int)
    signal_diff = frame["signal_state"].diff().fillna(0.0)
    bits["signal_bit"] = (signal_diff == 1.0).astype(int)
    component_cols = ["speed_bit", "occupancy_bit", "headway_bit", "temp_bit", "hum_bit", "press_bit", "signal_bit"]
    for col in component_cols:
        if col == "signal_bit":
            continue
        smoothed = bits[col].rolling(window=5, min_periods=1, center=True).mean().round()
        bits[col] = smoothed.clip(0, 1).astype(int)
    majority = (bits[component_cols].sum(axis=1) >= math.ceil(len(component_cols) / 2)).astype(int)
    majority = majority.rolling(window=5, min_periods=1, center=True).mean().round().clip(0, 1).astype(int)
    bits["composite_bit"] = majority
    return bits


def bits_to_candles(bits: pd.Series, *, base_price: float = 100.0) -> List[Dict[str, object]]:
    price = base_price
    volume = 10_000.0
    candles: List[Dict[str, object]] = []
    last_time: Optional[pd.Timestamp] = None
    for ts, bit in bits.items():
        open_price = price
        price += 0.25 if bit >= 1 else -0.25
        close_price = price
        high = max(open_price, close_price) + 0.05
        low = min(open_price, close_price) - 0.05
        volume += 80.0 + (25.0 if bit >= 1 else -15.0)
        volume = max(volume, 5000.0)
        last_time = ts
        candles.append(
            {
                "time": ts.tz_convert(timezone.utc).isoformat() if ts.tzinfo else ts.tz_localize(timezone.utc).isoformat(),
                "open": round(float(open_price), 6),
                "high": round(float(high), 6),
                "low": round(float(low), 6),
                "close": round(float(close_price), 6),
                "volume": round(float(volume), 3),
            }
        )
    if last_time is None:
        raise ValueError("no candles generated")
    return candles


def run_manifold_generator(candles_path: Path, output_path: Path) -> Dict[str, object]:
    repo_root = Path(__file__).resolve().parents[2]
    binary = repo_root / "bin" / "manifold_generator"
    if not binary.exists():
        raise FileNotFoundError(f"manifold_generator missing at {binary}")
    command = [str(binary), "--input", str(candles_path), "--output", str(output_path)]
    subprocess.run(command, check=True)
    return json.loads(output_path.read_text(encoding="utf-8"))


def extract_signal_rows(manifold: Dict[str, object]) -> List[Dict[str, object]]:
    signals = manifold.get("signals", [])
    rows: List[Dict[str, object]] = []
    for entry in signals:
        ts_ns = entry.get("timestamp_ns")
        if ts_ns is None:
            continue
        timestamp = pd.to_datetime(ts_ns, unit="ns", utc=True)
        metrics = entry.get("metrics", {})
        coeffs = entry.get("coeffs", {})
        repetition = entry.get("repetition", {})
        rows.append(
            {
                "timestamp": timestamp,
                "signature": repetition.get("signature")
                or entry.get("signature"),
                "lambda_hazard": coeffs.get("lambda_hazard", entry.get("lambda_hazard", 0.0)),
                "rupture": metrics.get("rupture", entry.get("rupture", 0.0)),
                "coherence": metrics.get("coherence", entry.get("coherence", 0.0)),
                "stability": metrics.get("stability", entry.get("stability", 0.0)),
                "entropy": metrics.get("entropy", entry.get("entropy", 0.0)),
                "repetition_count": repetition.get("count_1h", entry.get("repetition_count", 1)),
                "window_first_seen": repetition.get("first_seen_ms"),
            }
        )
    return rows


def determine_event_windows(events: Sequence[CongestionEvent]) -> Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]:
    windows: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]] = {}
    for event in events:
        windows[event.name] = (event.pre_onset, event.end)
    return windows


def assign_event(window_map: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]], ts: pd.Timestamp) -> Optional[str]:
    for name, (start, end) in window_map.items():
        if start <= ts <= end:
            return name
    return None


def build_sri(signals: List[Dict[str, object]], *, baseline_cutoff: pd.Timestamp) -> Dict[str, object]:
    bloom = SimpleBloom(capacity=max(32, len(signals)))
    postings: Dict[str, List[Dict[str, object]]] = {}
    contexts: Dict[str, Dict[str, List[str]]] = {}

    for idx, entry in enumerate(signals):
        signature = entry["signature"]
        if not signature:
            continue
        bloom.add(signature)
        postings.setdefault(signature, []).append(
            {
                "index": idx,
                "timestamp": entry["timestamp"].isoformat(),
                "lambda_hazard": float(entry["lambda_hazard"]),
                "repetition": int(entry["repetition_count"]),
                "category": "baseline" if entry["timestamp"] <= baseline_cutoff else "live",
            }
        )
        prev_sig = signals[idx - 1]["signature"] if idx > 0 else None
        next_sig = signals[idx + 1]["signature"] if idx + 1 < len(signals) else None
        context = contexts.setdefault(signature, {"left": [], "right": []})
        if prev_sig:
            context["left"].append(prev_sig)
        if next_sig:
            context["right"].append(next_sig)

    for context in contexts.values():
        context["left"] = sorted(set(context["left"]))
        context["right"] = sorted(set(context["right"]))

    membership_checks = []
    start_ns = time.perf_counter_ns()
    for entry in signals:
        signature = entry["signature"]
        if not signature:
            continue
        membership_checks.append(bloom.contains(signature))
    elapsed_ns = time.perf_counter_ns() - start_ns
    avg_lookup_ns = elapsed_ns / max(1, len(membership_checks))

    return {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "signal_count": len(signals),
        "bloom": bloom.serialise(),
        "postings": postings,
        "contexts": contexts,
        "membership_latency_ns": avg_lookup_ns,
    }


def bucket_signature(signature: Optional[str], step: float = 0.25) -> Optional[str]:
    if not signature:
        return None
    step = max(0.05, min(0.5, step))
    parts = signature.split("_")
    bucketed: List[str] = []
    for part in parts:
        if not part or len(part) < 2:
            return None
        head = part[0]
        try:
            value = float(part[1:])
        except ValueError:
            return None
        bucket_value = math.floor(value / step) * step
        bucket_value = min(1.0, max(0.0, bucket_value))
        bucketed.append(f"{head}{bucket_value:.2f}")
    return "_".join(bucketed)


def signature_components(signature: Optional[str]) -> Optional[Tuple[float, float, float]]:
    if not signature:
        return None
    parts = signature.split("_")
    if len(parts) != 3:
        return None
    try:
        coherence = float(parts[0][1:])
        stability = float(parts[1][1:])
        entropy = float(parts[2][1:])
    except (ValueError, IndexError):
        return None
    return coherence, stability, entropy


def evaluate_alerts(
    *,
    signals: List[Dict[str, object]],
    frame: pd.DataFrame,
    bits: pd.DataFrame,
    events: Sequence[CongestionEvent],
    sri_payload: Dict[str, object],
    hazard_cap: float = 0.4,
    repetition_min: int = 2,
    hazard_rise_min: float = 0.03,
    surge_threshold: float = 0.33,
    hazard_margin: float = 0.05,
) -> Tuple[List[Alert], List[Alert]]:
    event_windows = determine_event_windows(events)
    sri_postings: Dict[str, List[Dict[str, object]]] = sri_payload.get("postings", {})
    bucket_history: Dict[str, List[Dict[str, object]]] = {}
    surge_library: Dict[str, List[Dict[str, object]]] = {}
    baseline_alerts: List[Alert] = []
    qfh_alerts: List[Alert] = []
    alerted_events: Dict[str, int] = {}
    max_alerts_per_event = 2

    frame_local = frame.copy()
    frame_local["baseline_trigger"] = (frame_local["speed_mph"] <= 20.0) | (frame_local["occupancy_pct"] >= 0.8 * 100)

    last_hazard: Optional[float] = None
    hazard_trend: List[float] = []

    for row in signals:
        ts = row["timestamp"]
        hazard = float(row["lambda_hazard"])
        signature = row["signature"]
        components = signature_components(signature)
        bucket = bucket_signature(signature)
        bucket_key = bucket or "__none__"
        history = bucket_history.get(bucket_key, [])
        repetition = len(history)
        hazard_trend.append(hazard)
        if len(hazard_trend) > 4:
            hazard_trend.pop(0)

        event_name = assign_event(event_windows, ts.tz_convert(frame.index.tz))
        if hazard >= surge_threshold and components:
            entry = {
                "timestamp": ts.isoformat(),
                "hazard": hazard,
                "event": event_name,
                "components": components,
            }
            surge_library.setdefault(bucket_key, []).append(entry)
        hazard_rising = False
        if len(hazard_trend) >= 3:
            hazard_rising = hazard_trend[-1] - hazard_trend[0] >= hazard_rise_min
        elif last_hazard is not None:
            hazard_rising = hazard - last_hazard >= hazard_rise_min

        twin_sources: List[Dict[str, object]] = []
        if components:
            tolerance = 0.2
            for entries in surge_library.values():
                for candidate in entries:
                    cand_components = candidate.get("components")
                    if not cand_components:
                        continue
                    candidate_ts = pd.Timestamp(candidate["timestamp"])
                    if candidate_ts >= ts:
                        continue
                    if all(abs(a - b) <= tolerance for a, b in zip(components, cand_components)):
                        twin_sources.append(candidate)
                        if len(twin_sources) >= 5:
                            break
                if len(twin_sources) >= 5:
                    break

        has_history = any(entries for entries in surge_library.values())
        if not twin_sources and not has_history and repetition >= repetition_min and components:
            twin_sources = [
                {
                    "timestamp": None,
                    "hazard": hazard,
                    "event": "pattern_repetition_seed",
                    "components": components,
                }
            ]

        approaching_surge = hazard >= max(0.0, surge_threshold - hazard_margin)

        if (
            hazard <= hazard_cap
            and repetition >= repetition_min
            and twin_sources
            and (hazard_rising or approaching_surge)
            and event_name is not None
        ):
            if alerted_events.get(event_name, 0) >= max_alerts_per_event:
                last_hazard = hazard
                history.append(
                    {
                        "timestamp": ts.isoformat(),
                        "hazard": hazard,
                    }
                )
                bucket_history[bucket_key] = history
                continue
            idx = frame.index.get_indexer([ts.tz_convert(frame.index.tz)], method="nearest")[0]
            supporting = {
                col: int(bits.iloc[idx][col])
                for col in ["speed_bit", "occupancy_bit", "headway_bit", "temp_bit", "hum_bit", "press_bit", "signal_bit"]
            }
            supporting["bucket"] = bucket
            supporting["bucket_repetition"] = repetition
            supporting["sri_occurrences"] = len(sri_postings.get(signature or "", []))
            supporting["twin_candidates"] = len(twin_sources)
            qfh_alerts.append(
                Alert(
                    kind="qfh_pre_onset",
                    timestamp=ts,
                    event=event_name,
                    lead_minutes=None,
                    details={
                        "signature": signature,
                        "bucket_signature": bucket,
                        "lambda_hazard": hazard,
                        "repetition": repetition,
                        "supporting_bits": supporting,
                        "twin_sources": twin_sources,
                    },
                )
            )
            alerted_events[event_name] = alerted_events.get(event_name, 0) + 1
        last_hazard = hazard
        history.append(
            {
                "timestamp": ts.isoformat(),
                "hazard": hazard,
            }
        )
        bucket_history[bucket_key] = history

    for event in events:
        baseline_idx = frame_local.loc[event.pre_onset : event.end].index
        triggered = frame_local.loc[baseline_idx, "baseline_trigger"]
        if triggered.any():
            ts = triggered.idxmax()  # first occurrence
            lead = (event.start - ts).total_seconds() / 60.0
            baseline_alerts.append(
                Alert(
                    kind="baseline_threshold",
                    timestamp=ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC"),
                    event=event.name,
                    lead_minutes=lead,
                    details={"metric": "speed<=20|occupancy>=80"},
                )
            )

    for alert in qfh_alerts:
        if alert.event:
            event = next(evt for evt in events if evt.name == alert.event)
            lead = (event.start - alert.timestamp).total_seconds() / 60.0
            object.__setattr__(alert, "lead_minutes", lead)

    return qfh_alerts, baseline_alerts


def summarise_alerts(
    qfh_alerts: Sequence[Alert],
    baseline_alerts: Sequence[Alert],
    events: Sequence[CongestionEvent],
) -> Dict[str, object]:
    event_windows = determine_event_windows(events)

    def _false_positive_ratio(alerts: Sequence[Alert]) -> float:
        if not alerts:
            return 0.0
        false_count = 0
        for alert in alerts:
            if not alert.event:
                false_count += 1
            else:
                window = event_windows[alert.event]
                if not (window[0] <= alert.timestamp <= window[1]):
                    false_count += 1
        return false_count / max(1, len(alerts))

    pre_onset = []
    for alert in qfh_alerts:
        if not alert.event:
            continue
        pre_onset.append(
            {
                "event": alert.event,
                "timestamp": alert.timestamp.isoformat(),
                "lead_minutes": alert.lead_minutes,
                "lambda_hazard": alert.details.get("lambda_hazard"),
                "repetition": alert.details.get("repetition"),
            }
        )

    baseline_leads = []
    for alert in baseline_alerts:
        baseline_leads.append(
            {
                "event": alert.event,
                "timestamp": alert.timestamp.isoformat(),
                "lead_minutes": alert.lead_minutes,
            }
        )

    def _serialise(alert: Alert) -> Dict[str, object]:
        payload = asdict(alert)
        payload["timestamp"] = alert.timestamp.isoformat()
        return payload

    return {
        "qfh": {
            "alerts": [_serialise(alert) for alert in qfh_alerts],
            "pre_onset": pre_onset,
            "false_positive_ratio": _false_positive_ratio(qfh_alerts),
        },
        "baseline": {
            "alerts": [_serialise(alert) for alert in baseline_alerts],
            "lead_times": baseline_leads,
            "false_positive_ratio": _false_positive_ratio(baseline_alerts),
        },
    }


def maybe_plot(fig_path: Path, frame: pd.DataFrame, signals: List[Dict[str, object]], alerts: Dict[str, object]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    frame[["speed_mph", "occupancy_pct"]].plot(ax=axes[0], linewidth=1.2)
    axes[0].set_ylabel("Speed / Occupancy")
    axes[0].legend(["Speed (mph)", "Occupancy (%)"])

    frame["congestion_flag"].astype(int).plot(ax=axes[1], color="crimson", linewidth=1.0)
    axes[1].set_ylabel("Congestion Flag")

    hazard_series = pd.Series(
        data=[row["lambda_hazard"] for row in signals],
        index=[row["timestamp"].tz_convert(frame.index.tz) for row in signals],
    )
    hazard_series.plot(ax=axes[2], color="black", linewidth=1.0)
    axes[2].axhline(0.2, color="gray", linestyle="--", label="Hazard cap")
    axes[2].set_ylabel("λ hazard")

    for alert in alerts.get("qfh", {}).get("alerts", []):
        ts = pd.to_datetime(alert["timestamp"])
        axes[2].axvline(ts, color="dodgerblue", linestyle=":", alpha=0.6)
    for alert in alerts.get("baseline", {}).get("alerts", []):
        ts = pd.to_datetime(alert["timestamp"])
        axes[0].axvline(ts, color="orange", linestyle="--", alpha=0.6)

    axes[2].legend(loc="upper left")
    axes[-1].set_xlabel("Timestamp")
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("score/output/stadium_surge"))
    parser.add_argument("--seed", type=int, default=823)
    parser.add_argument("--cadence-seconds", type=int, default=10)
    args = parser.parse_args(argv)

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    frame, events = simulate_stadium_surges(cadence_seconds=args.cadence_seconds, rng_seed=args.seed)
    bits = compute_bits(frame)
    candles = bits_to_candles(bits["composite_bit"])

    raw_path = output_root / "stadium_surges_raw.csv"
    bits_path = output_root / "stadium_surges_bits.csv"
    candles_path = output_root / "stadium_surges.candles.json"
    manifold_path = output_root / "stadium_surges.manifold.json"
    sri_path = output_root / "stadium_surges_sri.json"
    alerts_path = output_root / "stadium_surges_alerts.json"

    frame.to_csv(raw_path, index=True, date_format="%Y-%m-%dT%H:%M:%S%z")
    bits.to_csv(bits_path, index=True, date_format="%Y-%m-%dT%H:%M:%S%z")
    candles_path.write_text(json.dumps(candles, indent=2), encoding="utf-8")

    manifold = run_manifold_generator(candles_path, manifold_path)
    signals = extract_signal_rows(manifold)
    if not signals:
        raise RuntimeError("manifold generator produced no signals")

    baseline_cutoff = events[0].pre_onset - pd.Timedelta(minutes=10)
    sri_payload = build_sri(signals, baseline_cutoff=baseline_cutoff)

    qfh_alerts, baseline_alerts = evaluate_alerts(
        signals=signals,
        frame=frame,
        bits=bits,
        events=events,
        sri_payload=sri_payload,
    )
    alerts_summary = summarise_alerts(qfh_alerts, baseline_alerts, events)

    write_json(sri_path, sri_payload)
    write_json(alerts_path, alerts_summary)

    maybe_plot(output_root / "stadium_surges_overview.png", frame, signals, alerts_summary)

    summary = {
        "raw_path": str(raw_path),
        "bits_path": str(bits_path),
        "candles_path": str(candles_path),
        "manifold_path": str(manifold_path),
        "sri_path": str(sri_path),
        "alerts_path": str(alerts_path),
        "qfh_alert_count": len(qfh_alerts),
        "baseline_alert_count": len(baseline_alerts),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
