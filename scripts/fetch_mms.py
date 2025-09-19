#!/usr/bin/env python3
"""Download MMS1 CDF products for a date range and export CSV slices.

The script targets the public CDAWeb HTTPS archive (anonymous access)
and fetches FGM survey L2 as well as FPI fast L2 moments.  It then
resamples the extracted time series to a configurable cadence and
writes them to ``nasa/mms/csv/<YYYY-MM-DD>/`` so that the STM pipeline
can ingest the folder directly.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import io
import tempfile
import re
from pathlib import Path
from typing import Iterable, Iterator, List

import cdflib
import pandas as pd
import requests
from lxml import html
from tqdm import tqdm


BASES = {
    "fgm": "https://cdaweb.gsfc.nasa.gov/pub/data/mms/mms1/fgm/srvy/l2/{yyyy}/{mm:02d}/",
    "fpi_des": "https://cdaweb.gsfc.nasa.gov/pub/data/mms/mms1/fpi/fast/l2/des-moms/{yyyy}/{mm:02d}/",
    "fpi_dis": "https://cdaweb.gsfc.nasa.gov/pub/data/mms/mms1/fpi/fast/l2/dis-moms/{yyyy}/{mm:02d}/",
}

OUT_ROOT = Path("nasa/mms/csv")


def day_range(start: dt.date, stop: dt.date) -> Iterator[dt.date]:
    current = start
    while current <= stop:
        yield current
        current += dt.timedelta(days=1)


def list_day_files(base_url: str, day_tag: str) -> List[str]:
    url = base_url
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    tree = html.fromstring(response.content)
    links = tree.xpath('//a/@href')
    pattern = re.compile(day_tag)
    files = [href for href in links if pattern.search(href)]
    files = [href for href in files if href.endswith(".cdf") or href.endswith(".cdf.gz")]
    return [url + href for href in files]


def read_cdf(url: str) -> tuple[cdflib.CDF, Path]:
    response = requests.get(url, timeout=90)
    response.raise_for_status()
    payload = response.content
    if url.endswith(".gz"):
        payload = gzip.GzipFile(fileobj=io.BytesIO(payload)).read()
    with tempfile.NamedTemporaryFile(suffix=".cdf", delete=False) as handle:
        handle.write(payload)
        path = handle.name
    return cdflib.CDF(path), Path(path)


def locate_epoch_var(cdf: cdflib.CDF) -> str | None:
    info = cdf.cdf_info()
    # Common names first
    if "Epoch" in info.rVariables or "Epoch" in info.zVariables:
        return "Epoch"
    for var in info.zVariables:
        try:
            dtype = cdf.varinq(var).Data_Type_Description
        except Exception:
            continue
        if dtype.startswith("CDF_EPOCH"):
            return var
    return None


def extract_fgm(cdf: cdflib.CDF) -> pd.DataFrame | None:
    epoch_var = locate_epoch_var(cdf)
    if epoch_var is None:
        return None
    info = cdf.cdf_info()
    candidates: List[str] = []
    for var in info.zVariables:
        if re.search(r"b_.*gse", var, re.IGNORECASE):
            candidates.append(var)
    if not candidates:
        for var in info.zVariables:
            if re.search(r"_b_", var) or re.search(r"^b_", var, re.IGNORECASE):
                candidates.append(var)
    if not candidates:
        return None
    target = candidates[0]
    epoch = cdflib.cdfepoch.to_datetime(cdf.varget(epoch_var))
    values = cdf.varget(target)
    if values.ndim == 1:
        values = values[:, None]
    base_cols = [f"{target}_x", f"{target}_y", f"{target}_z", f"{target}_mag"]
    columns = base_cols[: values.shape[1]]
    return pd.DataFrame({"time": pd.to_datetime(epoch)}).join(pd.DataFrame(values, columns=columns))


def extract_fpi(cdf: cdflib.CDF) -> pd.DataFrame | None:
    epoch_var = locate_epoch_var(cdf)
    if epoch_var is None:
        return None
    info = cdf.cdf_info()
    vectors: List[str] = []
    scalars: List[str] = []
    for var in info.zVariables:
        lower = var.lower()
        if "bulkv_gse" in lower or re.search(r"v_.*gse", lower):
            vectors.append(var)
        if "numberdensity" in lower or lower.endswith("_n"):
            scalars.append(var)
    if not vectors and not scalars:
        return None
    epoch = cdflib.cdfepoch.to_datetime(cdf.varget(epoch_var))
    frame = pd.DataFrame({"time": pd.to_datetime(epoch)})
    if vectors:
        data = cdf.varget(vectors[0])
        if data.ndim == 1:
            data = data[:, None]
        base_cols = [f"{vectors[0]}_x", f"{vectors[0]}_y", f"{vectors[0]}_z", f"{vectors[0]}_mag"]
        cols = base_cols[: data.shape[1]]
        frame = frame.join(pd.DataFrame(data, columns=cols))
    if scalars:
        frame[scalars[0]] = pd.Series(cdf.varget(scalars[0]))
    return frame if len(frame.columns) > 1 else None


def tidy_frames(frames: Iterable[pd.DataFrame], resample: str) -> pd.DataFrame:
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna().sort_values("time").set_index("time")
    combined = combined.resample(resample).median()
    combined = combined.dropna(how="all")
    return combined


def process_day(day: dt.date, instruments: List[str], resample: str) -> None:
    outdir = OUT_ROOT / day.isoformat()
    outdir.mkdir(parents=True, exist_ok=True)
    year = day.year
    month = day.month
    tag = f"{year}{month:02d}{day.day:02d}"
    for inst in instruments:
        base_key = inst
        if inst == "fpi":
            # default to ions (dis) but fall back to electrons if needed
            base_key = "fpi_dis"
        base_url = BASES.get(base_key)
        if base_url is None:
            print(f"[warn] unknown instrument '{inst}'")
            continue
        urls = list_day_files(base_url.format(yyyy=year, mm=month), tag)
        if not urls and inst == "fpi":
            base_url = BASES["fpi_des"]
            urls = list_day_files(base_url.format(yyyy=year, mm=month), tag)
        if not urls:
            print(f"[{inst}] no files found for {day}")
            continue
        frames: List[pd.DataFrame] = []
        for url in tqdm(urls, desc=f"{inst} {day}"):
            try:
                cdf, temp_path = read_cdf(url)
                try:
                    if inst == "fgm":
                        frame = extract_fgm(cdf)
                    else:
                        frame = extract_fpi(cdf)
                    if frame is not None:
                        frames.append(frame)
                finally:
                    del cdf
                    temp_path.unlink(missing_ok=True)
            except Exception as exc:  # pragma: no cover - network errors
                print(f"[{inst}] warn failed {url}: {exc}")
        if not frames:
            print(f"[{inst}] nothing parsed for {day}")
            continue
        tidy = tidy_frames(frames, resample)
        if tidy.empty:
            print(f"[{inst}] empty after resample for {day}")
            continue
        output = outdir / f"mms1_{inst}.csv"
        tidy.to_csv(output)
        print(f"[{inst}] wrote {output} rows={len(tidy)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch MMS1 CDF slices and export CSV")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--stop", required=True, help="Stop date YYYY-MM-DD")
    parser.add_argument("--inst", default="fgm,fpi", help="Comma list of instruments (fgm,fpi)")
    parser.add_argument("--resample", default="1S", help="Pandas offset (default 1S)")
    args = parser.parse_args()

    start = dt.date.fromisoformat(args.start)
    stop = dt.date.fromisoformat(args.stop)
    instruments = [token.strip() for token in args.inst.split(",") if token.strip()]

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for day in day_range(start, stop):
        print(f"=== {day} ===")
        process_day(day, instruments, args.resample)


if __name__ == "__main__":
    main()
