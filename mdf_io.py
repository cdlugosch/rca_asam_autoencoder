#!/usr/bin/env python3
"""
mdf_io.py — Part A: MDF files → parquet + channel inventory

Writes (overwriting existing files):
  timeseries/<run_id>.parquet  — resampled, cleaned signal data (one file per MDF run)
  channel_info.csv             — channel inventory across all runs (for inspection)
  run_metadata.json            — MDF start times needed by Part B for DTC resolution

Example:
    python mdf_io.py \\
      --input-dir ./mdf_files \\
      --intermediate-dir ./intermediate \\
      --sampling 100ms \\
      --channels-file config/channels.txt

    # Channel discovery mode (lists channels, writes channel_info.csv, then exits):
    python mdf_io.py --input-dir ./mdf_files --intermediate-dir ./intermediate --list-channels

Dependencies:
    - asammdf
    - pandas
    - numpy
    - pyarrow  (for parquet)
"""

import argparse
import json
import logging

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from asammdf import MDF


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MDF → parquet (Part A)")
    p.add_argument("--input-dir", required=True, help="Folder with .mf4/.mdf files")
    p.add_argument("--intermediate-dir", required=True, help="Output folder for parquet + metadata")
    p.add_argument("--sampling", default="100ms", help="Resampling interval, e.g. 10ms, 100ms, 1s")
    p.add_argument("--channels-file", default=None, help="Optional text file: one channel name per line")
    p.add_argument("--list-channels", action="store_true",
                   help="Discover all channels, write channel_info.csv, then exit")
    return p.parse_args()


def load_channel_list(path: Optional[str]) -> Optional[List[str]]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        logging.warning("Channels file not found: %s", p)
        return None
    with p.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_mdf_to_df(
    mdf_path: Path,
    channels: Optional[List[str]],
    sampling: str,
) -> Tuple[pd.DataFrame, Optional[object]]:
    logging.info("Loading %s", mdf_path.name)
    mdf = MDF(str(mdf_path))
    start_time = getattr(mdf, "start_time", None)
    df = mdf.to_dataframe(time_from_zero=True, channels=channels)
    df.index.name = "time_s"
    df.index = pd.to_timedelta(df.index, unit="s")
    df = df.resample(sampling).mean()
    df["run_id"] = mdf_path.stem
    return df, start_time


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(axis=1, how="all")

    non_numeric = [c for c in df.select_dtypes(exclude=[np.number]).columns if c != "run_id"]
    if non_numeric:
        logging.info("Dropping non-numeric columns: %s", non_numeric)
        df = df.drop(columns=non_numeric)

    numeric_cols = [c for c in df.columns if c != "run_id"]
    if numeric_cols:
        constant_cols = df[numeric_cols].columns[df[numeric_cols].nunique(dropna=True) <= 1].tolist()
        if constant_cols:
            logging.info("Dropping constant columns: %s", constant_cols)
            df = df.drop(columns=constant_cols)

    return df


def discover_channels(input_dir: Path) -> Dict[str, List[str]]:
    files = sorted(list(input_dir.glob("*.mf4")) + list(input_dir.glob("*.mdf")))
    if not files:
        raise RuntimeError(f"No MDF files found in {input_dir}")
    result: Dict[str, List[str]] = {}
    for p in files:
        mdf = MDF(str(p))
        names = sorted(set(mdf.channels_db.keys()))
        result[p.stem] = names
        logging.info("%s: %d channels", p.name, len(names))
    return result


def write_channel_info(channel_sets: Dict[str, List[str]], out_path: Path) -> None:
    all_channels = sorted(set(ch for chs in channel_sets.values() for ch in chs))
    rows = [
        {
            "channel": ch,
            "present_in_runs": ";".join(rid for rid, chs in channel_sets.items() if ch in chs),
            "run_count": sum(1 for chs in channel_sets.values() if ch in chs),
        }
        for ch in all_channels
    ]
    pd.DataFrame(rows).to_csv(out_path, index=False)
    logging.info("Channel info written to %s (%d channels)", out_path, len(all_channels))


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

    input_dir = Path(args.input_dir)
    intermediate_dir = Path(args.intermediate_dir)

    intermediate_dir.mkdir(parents=True, exist_ok=True)

    # Channel discovery mode — list channels, write inventory, exit
    if args.list_channels:
        run_channels = discover_channels(input_dir)
        write_channel_info(run_channels, intermediate_dir / "channel_info.csv")
        return

    channels = load_channel_list(args.channels_file)
    files = sorted(list(input_dir.glob("*.mf4")) + list(input_dir.glob("*.mdf")))
    if not files:
        raise RuntimeError(f"No MDF files found in {input_dir}")

    timeseries_dir = intermediate_dir / "timeseries"
    timeseries_dir.mkdir(exist_ok=True)

    mdf_start_times: Dict[str, Optional[str]] = {}
    channel_sets: Dict[str, List[str]] = {}

    for p in files:
        df, start_time = load_mdf_to_df(p, channels, args.sampling)
        df = clean_dataframe(df)
        df.to_parquet(timeseries_dir / f"{p.stem}.parquet")
        mdf_start_times[p.stem] = str(start_time) if start_time is not None else None
        channel_sets[p.stem] = [c for c in df.columns if c != "run_id"]
        logging.info("Written %s — %d rows, %d channels", p.stem, len(df), len(channel_sets[p.stem]))

    # run_metadata.json: MDF start times consumed by Part B for DTC resolution
    with open(intermediate_dir / "run_metadata.json", "w") as f:
        json.dump({"start_times": mdf_start_times}, f, indent=2)

    write_channel_info(channel_sets, intermediate_dir / "channel_info.csv")
    logging.info("Done. Intermediate files written to %s", intermediate_dir)


if __name__ == "__main__":
    main()
