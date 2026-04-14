#!/usr/bin/env python3
"""
rca_analytics.py — Part B: parquet → RCA analytics

Reads timeseries parquet files from <intermediate-dir>/timeseries/ and
run_metadata.json produced by mdf_io.py (Part A).

Produces:
  signals/
    aligned_clean_signals.csv       — resampled timeseries (human-readable)
    signal_ecu_mapping_resolved.csv — signal → ECU mapping

  rule_stat/
    limit_violations.csv            — hard-limit breaches (factual, no scoring)
    dtc_limit_violation_windows.csv — limit violations within each DTC time window

  correlation/
    correlation_matrix[_<run>].csv
    top_correlations[_<run>].csv

  lag/
    lag_vs_<ref>[_<run>].csv        — FFT cross-correlation lag

Anomaly scoring and ECU/DTC ranking are handled by ae_anomaly.py (Part C).

Example:
    python rca_analytics.py \\
      --intermediate-dir ./intermediate \\
      --output-dir ./output \\
      --sensor-ecu-map config/sensor_ecu_map.csv \\
      --rules-file config/anomaly_rules.csv \\
      --dtc-log config/dtc_log.csv \\
      --ref-signal EngineSpeed \\
      --per-run

Dependencies:
    - pandas, numpy, pyarrow
"""

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parquet → RCA analytics (Part B)")
    p.add_argument("--intermediate-dir",      required=True)
    p.add_argument("--output-dir",            required=True)
    p.add_argument("--sensor-ecu-map",        default=None)
    p.add_argument("--rules-file",            default=None)
    p.add_argument("--dtc-log",               default=None)
    p.add_argument("--ref-signal",            default=None)
    p.add_argument("--max-lag",               type=int,   default=30)
    p.add_argument("--dtc-window-before-s",   type=float, default=2.0)
    p.add_argument("--dtc-window-after-s",    type=float, default=2.0)
    p.add_argument("--per-run",               action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------
# Config loaders
# ---------------------------------------------------------------------

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


def load_rules(path: Optional[str]) -> pd.DataFrame:
    empty = pd.DataFrame(columns=["sensor", "min_value", "max_value", "max_abs_slope"])
    if not path:
        return empty
    p = Path(path)
    if not p.exists():
        logging.warning("Rules file not found: %s", p)
        return empty
    df = pd.read_csv(p)
    for col in ["sensor", "min_value", "max_value", "max_abs_slope"]:
        if col not in df.columns:
            df[col] = np.nan
    return df[["sensor", "min_value", "max_value", "max_abs_slope"]]


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


def load_intermediate(intermediate_dir: Path):
    timeseries_dir = intermediate_dir / "timeseries"
    parquet_files = sorted(timeseries_dir.glob("*.parquet"))
    if not parquet_files:
        raise RuntimeError(f"No parquet files found in {timeseries_dir}")

    frames = [pd.read_parquet(f) for f in parquet_files]
    df = pd.concat(frames, axis=0)
    logging.info("Loaded %d runs, %d rows, %d signals",
                 len(frames), len(df),
                 len([c for c in df.columns if c != "run_id"]))

    meta_path = intermediate_dir / "run_metadata.json"
    mdf_start_times: Dict[str, Optional[object]] = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        for run_id, ts_str in meta.get("start_times", {}).items():
            mdf_start_times[run_id] = pd.Timestamp(ts_str) if ts_str else None
    else:
        logging.warning("run_metadata.json not found — DTC timestamp resolution unavailable")

    return df, mdf_start_times


# ---------------------------------------------------------------------
# DTC timestamp resolution
# ---------------------------------------------------------------------

def resolve_dtc_timestamps(
    dtc_df: pd.DataFrame,
    mdf_start_times: Dict[str, Optional[object]],
) -> pd.DataFrame:
    if dtc_df.empty or "timestamp" not in dtc_df.columns:
        return dtc_df

    needs_resolution = dtc_df["relative_time_s"].isna() & dtc_df["timestamp"].notna()
    if not needs_resolution.any():
        return dtc_df

    known_starts = {rid: st for rid, st in mdf_start_times.items() if st is not None}
    if not known_starts:
        logging.warning("No MDF start times — DTC timestamps cannot be resolved")
        return dtc_df

    aware_starts: Dict[str, pd.Timestamp] = {}
    for rid, st in known_starts.items():
        ts = pd.Timestamp(st)
        aware_starts[rid] = ts if ts.tzinfo else ts.tz_localize("UTC")

    dtc_df = dtc_df.copy()
    for idx in dtc_df[needs_resolution].index:
        ts = dtc_df.loc[idx, "timestamp"]
        if pd.isna(ts):
            continue
        run_id = dtc_df.loc[idx, "run_id"]
        if run_id and run_id in aware_starts:
            dtc_df.loc[idx, "relative_time_s"] = (ts - aware_starts[run_id]).total_seconds()
            continue
        best_run, best_rel = None, None
        for rid, start_t in aware_starts.items():
            rel = (ts - start_t).total_seconds()
            if rel >= 0 and (best_rel is None or rel < best_rel):
                best_rel, best_run = rel, rid
        if best_run is not None:
            dtc_df.loc[idx, "relative_time_s"] = best_rel
            if not dtc_df.loc[idx, "run_id"]:
                dtc_df.loc[idx, "run_id"] = best_run
        else:
            logging.warning("DTC %s at %s could not be matched to any run",
                            dtc_df.loc[idx, "dtc_code"], ts)
    return dtc_df


# ---------------------------------------------------------------------
# ECU mapping
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------

def compute_correlation(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [c for c in df.columns if c != "run_id"]
    return df[numeric_cols].corr(method="pearson") if numeric_cols else pd.DataFrame()


def flatten_correlation_matrix(corr: pd.DataFrame, min_abs_corr: float = 0.7) -> pd.DataFrame:
    if corr.empty:
        return pd.DataFrame(columns=["signal_a", "signal_b", "corr", "abs_corr"])
    pairs = corr.stack().reset_index()
    pairs.columns = ["signal_a", "signal_b", "corr"]
    pairs = pairs[pairs["signal_a"] != pairs["signal_b"]].copy()
    pairs["pair_key"] = pairs.apply(
        lambda r: tuple(sorted([r["signal_a"], r["signal_b"]])), axis=1)
    pairs = pairs.drop_duplicates(subset=["pair_key"]).drop(columns=["pair_key"])
    pairs["abs_corr"] = pairs["corr"].abs()
    return pairs[pairs["abs_corr"] >= min_abs_corr].sort_values("abs_corr", ascending=False)


# ---------------------------------------------------------------------
# Lag analysis — FFT-based O(N log N)
# ---------------------------------------------------------------------

def cross_correlation_with_lag(df: pd.DataFrame, ref_signal: str, max_lag: int) -> pd.DataFrame:
    numeric_cols = [c for c in df.columns if c != "run_id"]
    if ref_signal not in numeric_cols:
        raise ValueError(f"Reference signal not found: {ref_signal}")

    ref = df[ref_signal].to_numpy(dtype=float)
    rows = []

    for sensor in numeric_cols:
        if sensor == ref_signal:
            continue
        sig = df[sensor].to_numpy(dtype=float)
        valid = ~(np.isnan(ref) | np.isnan(sig))
        ref_v, sig_v = ref[valid], sig[valid]
        if len(ref_v) < max(2, max_lag + 1):
            continue
        ref_std, sig_std = ref_v.std(), sig_v.std()
        if ref_std == 0 or sig_std == 0:
            continue
        ref_z = (ref_v - ref_v.mean()) / ref_std
        sig_z = (sig_v - sig_v.mean()) / sig_std
        nv = len(ref_z)
        fft_len = 1 << (2 * nv - 1).bit_length()
        xcorr = np.real(
            np.fft.irfft(
                np.fft.rfft(ref_z, fft_len) * np.conj(np.fft.rfft(sig_z, fft_len)),
                fft_len,
            )
        )
        lags = np.arange(-max_lag, max_lag + 1)
        corr_vals = xcorr[lags % fft_len] / nv
        best_idx = int(np.argmax(np.abs(corr_vals)))
        rows.append({"signal": sensor,
                     "best_corr": float(corr_vals[best_idx]),
                     "best_lag_samples": int(lags[best_idx])})

    out = pd.DataFrame(rows)
    if not out.empty:
        out["abs_corr"] = out["best_corr"].abs()
        out = out.sort_values("abs_corr", ascending=False)
    return out


# ---------------------------------------------------------------------
# Hard-limit violation detection
# ---------------------------------------------------------------------

def detect_limit_violations(
    df: pd.DataFrame,
    rules_df: pd.DataFrame,
    sampling_seconds: float,
) -> pd.DataFrame:
    """
    Return one row per sample that breaches a hard limit rule.
    Purely factual — no scoring, physical units throughout.

    Columns: run_id | time_s | signal | violation_type | threshold | value | excess
      violation_type  'min'   — value dropped below threshold
                      'max'   — value exceeded threshold
                      'slope' — |rate of change| exceeded threshold
    """
    empty = pd.DataFrame(columns=["run_id", "time_s", "signal",
                                   "violation_type", "threshold", "value", "excess"])
    if rules_df.empty:
        return empty

    records = []
    for _, rule in rules_df.iterrows():
        sensor = rule["sensor"]
        if sensor not in df.columns:
            continue
        series = df[sensor]

        def _append(mask: pd.Series, vtype: str, thr: float, vals: pd.Series) -> None:
            hits = df.loc[mask, ["run_id"]].copy()
            if hits.empty:
                return
            hits["time_s"]         = hits.index.total_seconds()
            hits["signal"]         = sensor
            hits["violation_type"] = vtype
            hits["threshold"]      = thr
            hits["value"]          = vals.loc[mask].values
            hits["excess"]         = (vals.loc[mask] - thr).abs().values
            records.append(hits)

        min_v     = rule["min_value"]     if pd.notna(rule["min_value"])     else None
        max_v     = rule["max_value"]     if pd.notna(rule["max_value"])     else None
        max_slope = rule["max_abs_slope"] if pd.notna(rule["max_abs_slope"]) else None

        if min_v is not None:
            _append(series < min_v, "min", min_v, series)
        if max_v is not None:
            _append(series > max_v, "max", max_v, series)
        if max_slope is not None:
            slope = df.groupby("run_id")[sensor].diff() / sampling_seconds
            _append(slope.abs() > max_slope, "slope", max_slope, slope.abs())

    return pd.concat(records, ignore_index=True) if records else empty


# ---------------------------------------------------------------------
# DTC window — limit violation audit
# ---------------------------------------------------------------------

def find_dtc_limit_violations(
    dtc_df: pd.DataFrame,
    lv_df: pd.DataFrame,
    sensor_ecu_df: pd.DataFrame,
    before_s: float,
    after_s: float,
) -> pd.DataFrame:
    """
    For each DTC event, list the limit violations that fall within
    ±(before_s / after_s) of the event timestamp.
    """
    if dtc_df.empty or lv_df.empty:
        return pd.DataFrame()

    sensor_to_ecu = dict(zip(sensor_ecu_df["signal"], sensor_ecu_df["ecu"]))
    lv = lv_df.copy()
    lv["ecu"] = lv["signal"].map(sensor_to_ecu).fillna("UNKNOWN")

    rows = []
    for _, dtc in dtc_df.iterrows():
        run_id = dtc.get("run_id", "")
        rel_t  = dtc.get("relative_time_s", np.nan)
        if pd.isna(rel_t):
            continue
        mask = (lv["time_s"] >= rel_t - before_s) & (lv["time_s"] <= rel_t + after_s)
        if run_id:
            mask &= lv["run_id"] == run_id
        win = lv.loc[mask]
        rows.append({
            "run_id":            run_id,
            "relative_time_s":   rel_t,
            "dtc_code":          dtc.get("dtc_code", ""),
            "dtc_ecu":           dtc.get("ecu", ""),
            "description":       dtc.get("description", ""),
            "violation_count":   len(win),
            "signals_in_window": ";".join(sorted(win["signal"].unique())) if not win.empty else "",
            "ecus_in_window":    ";".join(sorted(win["ecu"].unique()))    if not win.empty else "",
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

    intermediate_dir = Path(args.intermediate_dir)
    output_dir       = Path(args.output_dir)

    # Purge output dir before every run
    if output_dir.exists():
        shutil.rmtree(output_dir)

    signals_dir     = output_dir / "signals"
    rule_stat_dir   = output_dir / "rule_stat"
    correlation_dir = output_dir / "correlation"
    lag_dir         = output_dir / "lag"
    for d in (signals_dir, rule_stat_dir, correlation_dir, lag_dir):
        d.mkdir(parents=True)

    # 1) Load parquet + run metadata from Part A
    df, mdf_start_times = load_intermediate(intermediate_dir)

    # 2) Config
    sensor_ecu_map = load_sensor_ecu_map(args.sensor_ecu_map)
    rules_df       = load_rules(args.rules_file)
    dtc_df         = load_dtc_log(args.dtc_log)
    if not dtc_df.empty:
        dtc_df = resolve_dtc_timestamps(dtc_df, mdf_start_times)

    # 3) Sensor → ECU mapping
    sensor_ecu_df = build_sensor_ecu_table(df.columns.tolist(), sensor_ecu_map)
    sensor_ecu_df.to_csv(signals_dir / "signal_ecu_mapping_resolved.csv", index=False)

    aligned = df.copy()
    aligned.insert(0, "time_s", aligned.index.total_seconds())
    aligned.to_csv(signals_dir / "aligned_clean_signals.csv", index=False)

    # 4) Hard-limit violation audit
    sampling_seconds = df.index.to_series().diff().median().total_seconds()
    lv_df = detect_limit_violations(df, rules_df, sampling_seconds)
    lv_df.to_csv(rule_stat_dir / "limit_violations.csv", index=False)
    logging.info("Limit violations: %d  (across %d signals)",
                 len(lv_df),
                 lv_df["signal"].nunique() if not lv_df.empty else 0)

    if not dtc_df.empty:
        find_dtc_limit_violations(
            dtc_df, lv_df, sensor_ecu_df,
            args.dtc_window_before_s, args.dtc_window_after_s,
        ).to_csv(rule_stat_dir / "dtc_limit_violation_windows.csv", index=False)

    # 5) Correlation — global
    corr = compute_correlation(df)
    corr.index.name = "signal"
    corr.to_csv(correlation_dir / "correlation_matrix.csv")
    flatten_correlation_matrix(corr).to_csv(correlation_dir / "top_correlations.csv", index=False)

    if args.per_run:
        for run_id, run_df in df.groupby("run_id"):
            run_corr = compute_correlation(run_df)
            run_corr.index.name = "signal"
            run_corr.to_csv(correlation_dir / f"correlation_matrix_{run_id}.csv")
            flatten_correlation_matrix(run_corr).to_csv(
                correlation_dir / f"top_correlations_{run_id}.csv", index=False)
            logging.info("Per-run correlation written for %s", run_id)

    # 6) Lag analysis — global (FFT-based)
    if args.ref_signal:
        try:
            lag_df = cross_correlation_with_lag(df, args.ref_signal, args.max_lag)
            lag_df.to_csv(lag_dir / f"lag_vs_{args.ref_signal}.csv", index=False)
        except Exception as e:
            logging.warning("Lag analysis skipped: %s", e)

        if args.per_run:
            for run_id, run_df in df.groupby("run_id"):
                try:
                    cross_correlation_with_lag(run_df, args.ref_signal, args.max_lag).to_csv(
                        lag_dir / f"lag_vs_{args.ref_signal}_{run_id}.csv", index=False)
                except Exception as e:
                    logging.warning("Lag analysis skipped for %s: %s", run_id, e)

    logging.info("Done. Results written under %s/  [signals/ rule_stat/ correlation/ lag/]",
                 output_dir)


if __name__ == "__main__":
    main()
