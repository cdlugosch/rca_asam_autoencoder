#!/usr/bin/env python3
"""
mdf_io.py — MDF files → parquet + channel inventory

Edit the CONFIG section below, then run:
    python mdf_io.py

Expected input structure:
    mdf_files/<VIN>/<issue_id>/<VIN>_<run>.mf4

Writes (overwriting existing files):
  intermediate/timeseries/<run_id>.parquet  — resampled signal data with VIN/issue_id/testrun columns
  intermediate/channel_info.csv             — channel inventory across all runs
  intermediate/run_metadata.json            — MDF start times needed for DTC resolution

Dependencies:
    - asammdf, pandas, numpy, pyarrow
"""

import json

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from asammdf import MDF


# ─────────────────────────────────────────────────────────────────────────────
# Config — edit these values before running
# ─────────────────────────────────────────────────────────────────────────────

INPUT_DIR        = "./mdf_files"
INTERMEDIATE_DIR = "./intermediate"
SIDECAR_CSV      = "./intermediate/run_metadata.csv"  # run_id, vin, issue_id, testrun
SAMPLING         = "100ms"        # resampling interval: 10ms, 100ms, 1s, …
CHANNELS_FILE    = None           # path to text file with one channel name per line
LIST_CHANNELS    = False          # True: discover channels, write inventory, then exit


def load_channel_list(path: Optional[str]) -> Optional[List[str]]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        print(f"WARNING: Channels file not found: {p}")
        return None
    with p.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_sidecar(path: str) -> Dict[str, Dict[str, str]]:
    p = Path(path)
    if not p.exists():
        print(f"WARNING: Sidecar CSV not found: {p} — vin/issue_id/testrun will be empty")
        return {}
    df = pd.read_csv(p, dtype=str).fillna("")
    return {row["run_id"]: row.to_dict() for _, row in df.iterrows()}


def load_mdf_to_df(
    mdf_path: Path,
    channels: Optional[List[str]],
    sampling: str,
) -> Tuple[pd.DataFrame, Optional[object]]:
    print(f"Loading {mdf_path.name}")
    mdf = MDF(str(mdf_path))
    start_time = getattr(mdf, "start_time", None)
    df = mdf.to_dataframe(time_from_zero=True, channels=channels)
    df.index.name = "time_s"
    df.index = pd.to_timedelta(df.index, unit="s")
    df = df.resample(sampling).mean()
    df["run_id"] = mdf_path.stem
    return df, start_time


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    META_COLS = {"run_id", "vin", "issue_id", "testrun"}
    df = df.dropna(axis=1, how="all")

    non_numeric = [c for c in df.select_dtypes(exclude=[np.number]).columns if c not in META_COLS]
    if non_numeric:
        print(f"Dropping non-numeric columns: {non_numeric}")
        df = df.drop(columns=non_numeric)

    numeric_cols = [c for c in df.columns if c not in META_COLS]
    if numeric_cols:
        constant_cols = df[numeric_cols].columns[df[numeric_cols].nunique(dropna=True) <= 1].tolist()
        if constant_cols:
            print(f"Dropping constant columns: {constant_cols}")
            df = df.drop(columns=constant_cols)

    return df


def discover_channels(input_dir: Path) -> Dict[str, List[str]]:
    files = sorted(input_dir.rglob("*.mf4")) + sorted(input_dir.rglob("*.mdf"))
    if not files:
        raise RuntimeError(f"No MDF files found under {input_dir}")
    result: Dict[str, List[str]] = {}
    for p in files:
        mdf = MDF(str(p))
        names = sorted(set(mdf.channels_db.keys()))
        result[p.stem] = names
        print(f"{p.name}: {len(names)} channels")
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
    print(f"Channel info written to {out_path} ({len(all_channels)} channels)")


def main():
    input_dir        = Path(INPUT_DIR)
    intermediate_dir = Path(INTERMEDIATE_DIR)

    intermediate_dir.mkdir(parents=True, exist_ok=True)

    # Channel discovery mode — list channels, write inventory, exit
    if LIST_CHANNELS:
        run_channels = discover_channels(input_dir)
        write_channel_info(run_channels, intermediate_dir / "channel_info.csv")
        return

    channels = load_channel_list(CHANNELS_FILE)
    sidecar  = load_sidecar(SIDECAR_CSV)

    files = sorted(input_dir.rglob("*.mf4")) + sorted(input_dir.rglob("*.mdf"))
    if not files:
        raise RuntimeError(f"No MDF files found under {input_dir}")

    META_COLS = {"run_id", "vin", "issue_id", "testrun"}

    mdf_start_times: Dict[str, Optional[str]] = {}
    channel_sets: Dict[str, List[str]] = {}

    for p in files:
        # Derive VIN and issue_id from folder hierarchy: <VIN>/<issue_id>/<file>
        issue_id = p.parent.name
        vin      = p.parent.parent.name

        df, start_time = load_mdf_to_df(p, channels, SAMPLING)

        run_id = p.stem
        meta = sidecar.get(run_id, {})
        df["vin"]      = meta.get("vin", vin)
        df["issue_id"] = meta.get("issue_id", issue_id)
        df["testrun"]  = meta.get("testrun", "")

        df = clean_dataframe(df)
        run_dir = intermediate_dir / vin / issue_id
        run_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(run_dir / f"{run_id}.parquet")

        mdf_start_times[run_id] = str(start_time) if start_time is not None else None
        channel_sets[run_id] = [c for c in df.columns if c not in META_COLS]
        print(f"Written {run_id} — {len(df)} rows, {len(channel_sets[run_id])} channels  [vin={df['vin'].iloc[0]}, issue_id={df['issue_id'].iloc[0]}, testrun={df['testrun'].iloc[0]}]")

    # run_metadata.json: MDF start times consumed by ae_anomaly.py for DTC resolution
    with open(intermediate_dir / "run_metadata.json", "w") as f:
        json.dump({"start_times": mdf_start_times}, f, indent=2)

    write_channel_info(channel_sets, intermediate_dir / "channel_info.csv")
    print(f"Done. Intermediate files written to {intermediate_dir}")


if __name__ == "__main__":
    main()
