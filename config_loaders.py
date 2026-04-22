#!/usr/bin/env python3
"""
config_loaders.py — Shared config-loading utilities for Parts B and C.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def load_sensor_ecu_map(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        logging.warning("Sensor-ECU map not found: %s", p)
        return {}
    df = pd.read_csv(p)
    if not {"sensor", "ecu"}.issubset(df.columns):
        raise ValueError("sensor_ecu_map missing columns: {sensor, ecu}")
    return dict(zip(df["sensor"], df["ecu"]))


def load_dtc_log(path: Optional[str]) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    p = Path(path)
    if not p.exists():
        logging.warning("DTC log not found: %s", p)
        return pd.DataFrame()
    df = pd.read_csv(p)
    if "dtc_code" not in df.columns:
        raise ValueError("DTC log must contain column: dtc_code")
    for col, default in [("ecu", ""), ("description", ""), ("run_id", "")]:
        if col not in df.columns:
            df[col] = default
    if "relative_time_s" not in df.columns:
        df["relative_time_s"] = np.nan
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    return df


def infer_ecu_from_name(sensor: str) -> str:
    if "." in sensor:
        return sensor.split(".")[0]
    if "_" in sensor:
        prefix = sensor.split("_")[0]
        if len(prefix) <= 8:
            return prefix
    return "UNKNOWN"


def build_sensor_ecu_table(columns: List[str], sensor_ecu_map: Dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"signal": c, "ecu": sensor_ecu_map.get(c, infer_ecu_from_name(c))}
         for c in columns if c != "run_id"]
    )
