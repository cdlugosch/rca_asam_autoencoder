#!/usr/bin/env python3
"""
rca_analytics.py — Part B: parquet → RCA analytics

Reads timeseries parquet files from <intermediate-dir>/timeseries/ and
run_metadata.json produced by mdf_io.py (Part A), then runs anomaly detection,
correlation, lag, and DTC analysis.

Purges <output-dir> before each run.

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
    - pandas
    - numpy
    - pyarrow  (for parquet read)
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
    p.add_argument("--intermediate-dir", required=True, help="Folder produced by mdf_io.py (Part A)")
    p.add_argument("--output-dir", required=True, help="Folder for output CSV files")
    p.add_argument("--sensor-ecu-map", default=None, help="Optional CSV: sensor,ecu")
    p.add_argument("--rules-file", default=None, help="Optional CSV: sensor,min_value,max_value,max_abs_slope")
    p.add_argument("--dtc-log", default=None, help="Optional CSV with DTC events")
    p.add_argument("--ref-signal", default=None, help="Optional reference signal for lag analysis")
    p.add_argument("--max-lag", type=int, default=30, help="Max lag in samples for lag analysis")
    p.add_argument("--rolling-window", type=int, default=20, help="Rolling window for statistical anomaly detection")
    p.add_argument("--z-threshold", type=float, default=3.0, help="Absolute z-score threshold")
    p.add_argument("--dtc-window-before-s", type=float, default=2.0, help="Seconds before DTC for RCA window")
    p.add_argument("--dtc-window-after-s", type=float, default=2.0, help="Seconds after DTC for RCA window")
    p.add_argument("--per-run", action="store_true",
                   help="Also compute correlation and lag analysis per run_id")
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
        raise ValueError(f"sensor_ecu_map missing columns: {{'sensor','ecu'}}")
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
    """
    Preferred format:  run_id, relative_time_s, dtc_code, ecu, description
    Also accepted:     timestamp, dtc_code, ecu, description  (resolved via MDF start times)
    """
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
    """
    Read all parquet files from <intermediate_dir>/timeseries/ and
    run_metadata.json. Returns (combined_df, mdf_start_times).
    """
    timeseries_dir = intermediate_dir / "timeseries"
    parquet_files = sorted(timeseries_dir.glob("*.parquet"))
    if not parquet_files:
        raise RuntimeError(f"No parquet files found in {timeseries_dir}")

    frames = [pd.read_parquet(f) for f in parquet_files]
    df = pd.concat(frames, axis=0)
    logging.info("Loaded %d runs, %d rows, %d signals", len(frames), len(df),
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
            logging.info("DTC %s: resolved to %.2fs in run %s",
                         dtc_df.loc[idx, "dtc_code"], best_rel, best_run)
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
    pairs["pair_key"] = pairs.apply(lambda r: tuple(sorted([r["signal_a"], r["signal_b"]])), axis=1)
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
        rows.append({"signal": sensor, "best_corr": float(corr_vals[best_idx]),
                     "best_lag_samples": int(lags[best_idx])})

    out = pd.DataFrame(rows)
    if not out.empty:
        out["abs_corr"] = out["best_corr"].abs()
        out = out.sort_values("abs_corr", ascending=False)
    return out


# ---------------------------------------------------------------------
# Anomaly detection — rule based
# ---------------------------------------------------------------------

def detect_rule_based_anomalies(
    df: pd.DataFrame,
    rules_df: pd.DataFrame,
    sampling_seconds: float,
) -> pd.DataFrame:
    """
    Slope computed with groupby('run_id').diff() — first sample of each run is
    NaN, preventing false positives at run boundaries.
    """
    empty = pd.DataFrame(columns=["run_id", "time_s", "signal", "anomaly_type", "score", "value"])
    if rules_df.empty:
        return empty

    records = []
    for _, rule in rules_df.iterrows():
        sensor = rule["sensor"]
        if sensor not in df.columns:
            continue
        series = df[sensor]
        mask = pd.Series(False, index=df.index)
        score = pd.Series(0.0, index=df.index)

        min_v = rule["min_value"] if pd.notna(rule["min_value"]) else None
        max_v = rule["max_value"] if pd.notna(rule["max_value"]) else None
        max_slope = rule["max_abs_slope"] if pd.notna(rule["max_abs_slope"]) else None

        if min_v is not None:
            low = series < min_v
            mask |= low
            score.loc[low] += (min_v - series.loc[low]).abs()
        if max_v is not None:
            high = series > max_v
            mask |= high
            score.loc[high] += (series.loc[high] - max_v).abs()
        if max_slope is not None:
            slope = df.groupby("run_id")[sensor].diff() / sampling_seconds
            slope_mask = slope.abs() > max_slope
            mask |= slope_mask
            score.loc[slope_mask] += slope.loc[slope_mask].abs() - max_slope

        hits = df.loc[mask, ["run_id"]].copy()
        if not hits.empty:
            hits["time_s"] = hits.index.total_seconds()
            hits["signal"] = sensor
            hits["anomaly_type"] = "rule"
            hits["score"] = score.loc[mask].fillna(0.0).values
            hits["value"] = series.loc[mask].values
            records.append(hits)

    return pd.concat(records, ignore_index=True) if records else empty


# ---------------------------------------------------------------------
# Anomaly detection — statistical
# ---------------------------------------------------------------------

def detect_statistical_anomalies(
    df: pd.DataFrame,
    rolling_window: int,
    z_threshold: float,
) -> pd.DataFrame:
    """
    Rolling z-score per run_id — window never crosses a run boundary.
    """
    empty = pd.DataFrame(columns=["run_id", "time_s", "signal", "anomaly_type", "score", "value"])
    records = []
    numeric_cols = [c for c in df.columns if c != "run_id"]
    min_periods = max(5, rolling_window // 2)

    for sensor in numeric_cols:
        series = df[sensor]
        rolling_mean = df.groupby("run_id")[sensor].transform(
            lambda x: x.rolling(rolling_window, min_periods=min_periods).mean()
        )
        rolling_std = df.groupby("run_id")[sensor].transform(
            lambda x: x.rolling(rolling_window, min_periods=min_periods).std()
        )
        z = (series - rolling_mean) / rolling_std.replace(0, np.nan)
        mask = z.abs() > z_threshold

        hits = df.loc[mask, ["run_id"]].copy()
        if not hits.empty:
            hits["time_s"] = hits.index.total_seconds()
            hits["signal"] = sensor
            hits["anomaly_type"] = "stat"
            hits["score"] = z.loc[mask].abs().fillna(0.0).values
            hits["value"] = series.loc[mask].values
            records.append(hits)

    return pd.concat(records, ignore_index=True) if records else empty


# ---------------------------------------------------------------------
# Score normalisation
# ---------------------------------------------------------------------

def normalize_anomaly_scores(anomaly_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """
    Rule scores (physical units) divided by signal std → dimensionless.
    Stat scores are already z-scores → kept as-is.
    """
    anomaly_df = anomaly_df.copy()
    if anomaly_df.empty:
        anomaly_df["score_normalized"] = pd.Series(dtype=float)
        return anomaly_df

    numeric_cols = [c for c in df.columns if c != "run_id"]
    signal_stds = df[numeric_cols].std().replace(0, np.nan)
    anomaly_df["score_normalized"] = np.nan

    rule_mask = anomaly_df["anomaly_type"] == "rule"
    stat_mask = anomaly_df["anomaly_type"] == "stat"

    if rule_mask.any():
        rule_rows = anomaly_df.loc[rule_mask].copy()
        rule_rows["_std"] = rule_rows["signal"].map(signal_stds).fillna(1.0)
        anomaly_df.loc[rule_mask, "score_normalized"] = (rule_rows["score"] / rule_rows["_std"]).values
    if stat_mask.any():
        anomaly_df.loc[stat_mask, "score_normalized"] = anomaly_df.loc[stat_mask, "score"]

    return anomaly_df


# ---------------------------------------------------------------------
# Anomaly aggregation
# ---------------------------------------------------------------------

def aggregate_signal_anomalies(anomaly_df: pd.DataFrame) -> pd.DataFrame:
    if anomaly_df.empty:
        return pd.DataFrame(columns=["signal", "anomaly_count", "score_sum", "score_normalized_sum"])
    score_col = "score_normalized" if "score_normalized" in anomaly_df.columns else "score"
    return (
        anomaly_df.groupby("signal")
        .agg(anomaly_count=("signal", "size"), score_sum=("score", "sum"),
             score_normalized_sum=(score_col, "sum"))
        .reset_index()
        .sort_values(["score_normalized_sum", "anomaly_count"], ascending=[False, False])
    )


def aggregate_ecu_relevance(anomaly_df: pd.DataFrame, sensor_ecu_df: pd.DataFrame) -> pd.DataFrame:
    if anomaly_df.empty:
        return pd.DataFrame(columns=["ecu", "anomaly_count", "score_sum", "score_normalized_sum"])
    score_col = "score_normalized" if "score_normalized" in anomaly_df.columns else "score"
    sensor_to_ecu = dict(zip(sensor_ecu_df["signal"], sensor_ecu_df["ecu"]))
    tmp = anomaly_df.copy()
    tmp["ecu"] = tmp["signal"].map(sensor_to_ecu).fillna("UNKNOWN")
    return (
        tmp.groupby("ecu")
        .agg(anomaly_count=("ecu", "size"), score_sum=("score", "sum"),
             score_normalized_sum=(score_col, "sum"))
        .reset_index()
        .sort_values(["score_normalized_sum", "anomaly_count"], ascending=[False, False])
    )


# ---------------------------------------------------------------------
# DTC linking
# ---------------------------------------------------------------------

def extract_dtc_windows(
    dtc_df: pd.DataFrame,
    anomaly_df: pd.DataFrame,
    sensor_ecu_df: pd.DataFrame,
    before_s: float,
    after_s: float,
) -> pd.DataFrame:
    empty_cols = ["run_id", "relative_time_s", "dtc_code", "dtc_ecu", "description",
                  "top_signal", "top_signal_score", "top_ecu", "top_ecu_score", "matched_anomaly_count"]
    if dtc_df.empty:
        return pd.DataFrame(columns=empty_cols)
    if "relative_time_s" not in dtc_df.columns or dtc_df["relative_time_s"].isna().all():
        return pd.DataFrame(columns=empty_cols)

    score_col = "score_normalized" if "score_normalized" in anomaly_df.columns else "score"
    sensor_to_ecu = dict(zip(sensor_ecu_df["signal"], sensor_ecu_df["ecu"]))
    rows = []

    for _, dtc in dtc_df.iterrows():
        run_id = dtc.get("run_id", "")
        rel_t = dtc.get("relative_time_s", np.nan)
        if pd.isna(rel_t):
            continue
        mask = (anomaly_df["time_s"] >= rel_t - before_s) & (anomaly_df["time_s"] <= rel_t + after_s)
        if run_id:
            mask &= anomaly_df["run_id"] == run_id
        win = anomaly_df.loc[mask].copy()
        win["ecu"] = win["signal"].map(sensor_to_ecu).fillna("UNKNOWN")
        signal_scores = win.groupby("signal")[score_col].sum().sort_values(ascending=False)
        ecu_scores = win.groupby("ecu")[score_col].sum().sort_values(ascending=False)
        rows.append({
            "run_id": run_id,
            "relative_time_s": rel_t,
            "dtc_code": dtc.get("dtc_code", ""),
            "dtc_ecu": dtc.get("ecu", ""),
            "description": dtc.get("description", ""),
            "top_signal": signal_scores.index[0] if not signal_scores.empty else "",
            "top_signal_score": float(signal_scores.iloc[0]) if not signal_scores.empty else 0.0,
            "top_ecu": ecu_scores.index[0] if not ecu_scores.empty else "",
            "top_ecu_score": float(ecu_scores.iloc[0]) if not ecu_scores.empty else 0.0,
            "matched_anomaly_count": int(len(win)),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

    intermediate_dir = Path(args.intermediate_dir)
    output_dir = Path(args.output_dir)

    # Purge output dir before every run
    if output_dir.exists():
        shutil.rmtree(output_dir)

    # Create output subdirectory layout
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
    rules_df = load_rules(args.rules_file)
    dtc_df = load_dtc_log(args.dtc_log)

    if not dtc_df.empty:
        dtc_df = resolve_dtc_timestamps(dtc_df, mdf_start_times)

    # 3) Sensor → ECU mapping
    sensor_ecu_df = build_sensor_ecu_table(df.columns.tolist(), sensor_ecu_map)
    sensor_ecu_df.to_csv(signals_dir / "signal_ecu_mapping_resolved.csv", index=False)

    # Aligned signals (human-readable CSV of what was loaded from parquet)
    aligned = df.copy()
    aligned.insert(0, "time_s", aligned.index.total_seconds())
    aligned.to_csv(signals_dir / "aligned_clean_signals.csv", index=False)

    # 4) Anomaly detection — infer sampling from data
    sampling_seconds = df.index.to_series().diff().median().total_seconds()

    rule_anom = detect_rule_based_anomalies(df, rules_df, sampling_seconds)
    stat_anom = detect_statistical_anomalies(df, args.rolling_window, args.z_threshold)
    all_anom = pd.concat([rule_anom, stat_anom], ignore_index=True)
    all_anom = normalize_anomaly_scores(all_anom, df)
    all_anom.to_csv(rule_stat_dir / "anomalies_long.csv", index=False)

    aggregate_signal_anomalies(all_anom).to_csv(rule_stat_dir / "signal_anomaly_summary.csv", index=False)
    aggregate_ecu_relevance(all_anom, sensor_ecu_df).to_csv(rule_stat_dir / "ecu_relevance_ranking.csv", index=False)

    # 5) Correlation — global
    corr = compute_correlation(df)
    corr.index.name = "signal"
    corr.to_csv(correlation_dir / "correlation_matrix.csv")
    flatten_correlation_matrix(corr).to_csv(correlation_dir / "top_correlations.csv", index=False)

    # 5b) Correlation — per run
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

    # 7) DTC linking
    if not dtc_df.empty:
        extract_dtc_windows(dtc_df, all_anom, sensor_ecu_df,
                            args.dtc_window_before_s, args.dtc_window_after_s
                            ).to_csv(rule_stat_dir / "dtc_root_cause_candidates.csv", index=False)

    logging.info("Done. Results written under %s/  [signals/ rule_stat/ correlation/ lag/]", output_dir)


if __name__ == "__main__":
    main()
