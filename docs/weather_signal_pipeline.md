# NOAA Weather Signal Pipeline

This guide captures the end-to-end workflow for turning public National Weather Service data into SEP/QFH manifolds and hazard metrics. Follow these steps to reproduce the Austin prototype or scale across stations.

## 1. Data sourcing

- **Provider**: National Weather Service (NWS) station feeds exposed through `https://api.weather.gov/stations/{STATION}/observations`.
- **Access**: No API key required. Identify each station via its ICAO identifier (e.g. `KATT` for Camp Mabry, Austin).
- **Cadence**: Most stations publish new observations every hour. Request a 24-hour window by supplying `start` and `end` query parameters in RFC 3339 format (`2025-10-01T00:00:00Z`).
- **Fields used**: temperature (`temperature.value`, °C), relative humidity (`relativeHumidity.value`, %), and barometric pressure (`barometricPressure.value`, Pa). The feed includes many other measurements that can be added later (wind, dew point, visibility).

## 2. Fetching observations

1. Set up a `requests.Session` with a descriptive `User-Agent`. Some NOAA endpoints throttle generic/empty agents.
2. Call the `/observations` endpoint once per station/day with `limit=1000`. The API returns most-recent-first results; sort ascending by `properties.timestamp`.
3. Parse the GeoJSON `features` array into a `pandas.DataFrame`, dropping entries with missing T/RH/P values and deduplicating the timestamps.
4. Resample to a one-hour cadence via `frame.resample("1H").median().ffill()` so small gaps are filled and each hour is represented.

The notebook `notebooks/noaa_weather_manifold.ipynb` includes helper functions (`fetch_noaa_observations`, `observations_to_frame`) that implement this pattern.

## 3. Encoding weather bits

Translate hourly values into domain-aware bits before passing them to QFH:

- `temp_bit`: `1` when temperature is non-decreasing: `T_t >= T_{t-1}`.
- `hum_bit`: `1` when relative humidity is non-decreasing.
- `press_bit`: `1` when barometric pressure is falling (`P_t < P_{t-1}`) to highlight approaching systems.
- `composite_bit`: majority vote across the three indicators (`>= 2` true).

The notebook’s `compute_bits` function derives these flags and drops the first row (which lacks a previous comparison). Extend this module with additional encoders (e.g., pressure-leading-humidity lag bits or dewpoint crosses) without changing downstream tooling.

## 4. Converting bits to manifold-ready candles

`bin/manifold_generator` expects OHLCV candles. For prototyping, synthesise candles whose up/down moves mirror the composite bitstream:

1. Seed an initial candle one hour before the first observation so the bitstream has a baseline.
2. For each hourly step:
   - Open at the previous close.
   - Close one unit higher if the bit is `1`, otherwise one unit lower.
   - Set `high`/`low` to bracket the open/close by ±0.05.
   - Increase volume monotonically to keep the manifold kernel’s volume filters permissive.

The notebook’s `bits_to_candles` helper writes the resulting list to `analysis/weather/<STATION>_<DATE>.candles.json`. In production you can feed actual station metrics as `open/high/low/close` values instead of synthetic ones as long as the bit logic remains encoded in the candle features.

## 5. Running the SEP manifold generator

With the candle JSON in place:

```bash
bin/manifold_generator --input analysis/weather/KATT_2025-10-01.candles.json   --output analysis/weather/KATT_2025-10-01.manifold.json
```

`manifold_generator` automatically loads the native manifold kernel (`src/core/manifold_builder.cpp`) and produces up to 512 signals with coherence, entropy, rupture, and hazard λ. The notebook executes this command via `subprocess.run` and surfaces stdout/stderr for debugging.

## 6. Inspecting λ and rupture

Load the manifold JSON and review the `signals` array:

- `metrics.rupture` (and `lambda_hazard`) increase when the composite bit flips frequently within the analysis window.
- `metrics.coherence` climbs during steady regimes (long runs of identical bits).
- `repetition.signature` groups windows with similar metric triples so you can detect recurring structural states.

The notebook extracts the most recent signal, prints the metrics, and optionally plots composite bits vs hazard using `matplotlib`.

## 7. Scaling beyond a single station

Once the prototype is validated:

1. **Batch ingestion** – Maintain a station registry and loop over `STATION_ID` values, writing per-station candle files. Cache API responses to avoid hitting rate limits.
2. **Valkey export** – Replace the file `--output` with `--output valkey` (see `src/app/manifold_generator_main.cpp`) so manifolds land in `manifold:{station}:{date}` keys alongside existing SEP assets.
3. **Streaming** – Tail Valkey updates or the generated manifold logs using `src/stm_stream/runtime.py` to power dashboards or alerting hooks.
4. **Feature expansion** – Add wind speed, dewpoint, or multi-station relational bits. Adjust window length and stride to cover daily vs multi-day dynamics.
5. **Evaluation** – Compare λ spikes against historical storm reports or grid load events to calibrate alert thresholds.

## 8. Repository artefacts

- Notebook: `notebooks/noaa_weather_manifold.ipynb` – runnable prototype for the full pipeline.
- Native engine: `bin/manifold_generator`, `src/core/manifold_builder.cpp`, `src/core/qfh.cpp`.
- Bit encoding helpers (reference): `scripts/encode_struct_bits.py` shows the same pattern for telemetry data.
- Documentation: This file plus `docs/overview.md` and `docs/integration_with_sep.md` for SEP core integration.

With these ingredients you can cull NOAA data without credentials, encode it into SEP-compatible bitstreams, and generate hazard metrics suitable for dashboards, alerts, or downstream routing.
