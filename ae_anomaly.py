#!/usr/bin/env python3
"""
ae_anomaly.py — Windowed autoencoder anomaly detection

Trains a windowed dense autoencoder (sklearn MLPRegressor used as X→X
reconstructor) on reference (normal) runs, then computes per-signal
reconstruction error across all runs.

Output written under <output-dir>/autoencoder/:
    ae_signal_errors.csv      — per-timestep per-signal MSE reconstruction error
    ae_anomaly_summary.csv    — signal ranking by AE anomaly score
    ae_ecu_relevance.csv      — ECU ranking by AE anomaly score
    ae_dtc_candidates.csv     — DTC window linking using AE scores
    ae_model_info.json        — architecture & training metadata

Prerequisites:
    intermediate/        produced by mdf_io.py

Example:
    python ae_anomaly.py \\
      --intermediate-dir ./intermediate \\
      --output-dir ./output \\
      --sensor-ecu-map config/sensor_ecu_map.csv \\
      --dtc-log config/dtc_log.csv

Dependencies:
    - numpy, pandas, pyarrow, scikit-learn
"""

import argparse
import json
import logging

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from config_loaders import (
    build_sensor_ecu_table,
    load_dtc_log,
    load_sensor_ecu_map,
)


# ─────────────────────────────────────────────────────────────────────────────
# Arguments
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Windowed autoencoder anomaly detection")
    p.add_argument("--intermediate-dir", required=True)
    p.add_argument("--output-dir",       required=True)
    p.add_argument("--sensor-ecu-map",   default=None)
    p.add_argument("--dtc-log",          default=None)
    p.add_argument("--normal-runs",      nargs="*", default=None,
                   help="Run IDs to use as training data.  Auto-detected if omitted "
                        "(any run_id containing 'normal', 'ref', or 'baseline').")
    p.add_argument("--window-size",      type=int,   default=20,
                   help="Sliding window width in samples (default: 20 = 2 s @ 100 ms)")
    p.add_argument("--hidden-layers",    nargs="+",  type=int, default=[64, 16, 64],
                   help="Hidden layer sizes for the MLP autoencoder (default: 64 16 64)")
    p.add_argument("--max-iter",         type=int,   default=500)
    p.add_argument("--ae-threshold-pct", type=float, default=95.0,
                   help="Percentile of training-set reconstruction error used as "
                        "anomaly threshold per signal (default: 95)")
    p.add_argument("--dtc-window-before-s", type=float, default=2.0)
    p.add_argument("--dtc-window-after-s",  type=float, default=2.0)
    p.add_argument("--save-model", default=None,
                   help="Save trained model bundle to this path (e.g. model.joblib)")
    p.add_argument("--load-model", default=None,
                   help="Load a saved model bundle; skips training entirely")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_intermediate(intermediate_dir: Path) -> pd.DataFrame:
    timeseries_dir = intermediate_dir / "timeseries"
    parquet_files = sorted(timeseries_dir.glob("*.parquet"))
    if not parquet_files:
        raise RuntimeError(f"No parquet files in {timeseries_dir}")
    frames = [pd.read_parquet(f) for f in parquet_files]
    df = pd.concat(frames, axis=0)
    logging.info("Loaded %d runs, %d rows, %d signals",
                 len(frames), len(df),
                 len([c for c in df.columns if c != "run_id"]))
    return df


def identify_training_runs(df: pd.DataFrame, normal_runs: Optional[List[str]]) -> List[str]:
    all_runs = df["run_id"].unique().tolist()
    if normal_runs:
        missing = [r for r in normal_runs if r not in all_runs]
        if missing:
            raise ValueError(f"Specified normal runs not found in data: {missing}")
        return normal_runs
    # Auto-detect
    keywords = ("normal", "ref", "baseline")
    detected = [r for r in all_runs if any(kw in r.lower() for kw in keywords)]
    if not detected:
        raise RuntimeError(
            "Could not auto-detect training runs.  "
            "Pass --normal-runs <run_id> [run_id ...] explicitly."
        )
    logging.info("Auto-detected training runs: %s", detected)
    return detected


# ─────────────────────────────────────────────────────────────────────────────
# Window utilities
# ─────────────────────────────────────────────────────────────────────────────

def make_windows(
    arr: np.ndarray,
    W: int,
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Slide a window of width W over a (T, n_signals) array.

    Returns
    -------
    windows : (n_windows, W * n_signals)  flattened windows
    starts  : (n_windows,)  start indices in the original array
    """
    T = len(arr)
    if T < W:
        return np.empty((0, W * arr.shape[1])), np.empty(0, dtype=int)
    starts = np.arange(0, T - W + 1, stride)
    windows = np.stack([arr[i: i + W].ravel() for i in starts])
    return windows, starts


def scatter_add_window_errors(
    errors_3d: np.ndarray,
    starts: np.ndarray,
    T: int,
    W: int,
    n_signals: int,
) -> np.ndarray:
    """
    Distribute per-window per-position per-signal errors back onto the
    original time axis and return the mean over overlapping windows.

    Parameters
    ----------
    errors_3d : (n_windows, W, n_signals)  squared reconstruction errors
    starts    : (n_windows,)
    T         : original time length
    W, n_signals

    Returns
    -------
    per_sample : (T, n_signals)  mean squared error per timestep per signal
    """
    acc = np.zeros((T, n_signals), dtype=np.float64)
    cnt = np.zeros(T, dtype=np.float64)
    for w_idx, t_start in enumerate(starts):
        acc[t_start: t_start + W] += errors_3d[w_idx]
        cnt[t_start: t_start + W] += 1.0
    cnt = np.maximum(cnt, 1.0)
    return acc / cnt[:, None]


# ─────────────────────────────────────────────────────────────────────────────
# Autoencoder: train + infer
# ─────────────────────────────────────────────────────────────────────────────

def train_autoencoder(
    df: pd.DataFrame,
    training_run_ids: List[str],
    signal_cols: List[str],
    W: int,
    hidden_layers: Tuple[int, ...],
    max_iter: int,
) -> Tuple[MLPRegressor, StandardScaler]:
    """
    Fit a StandardScaler on training runs, then train an MLPRegressor as
    an autoencoder (X → X) on sliding windows from those runs.
    """
    train_df = df[df["run_id"].isin(training_run_ids)]
    all_windows = []

    for run_id, run_df in train_df.groupby("run_id"):
        arr = run_df[signal_cols].values.astype(float)
        windows, _ = make_windows(arr, W, stride=1)
        if len(windows):
            all_windows.append(windows)
        logging.info("  Training windows from %s: %d", run_id, len(windows))

    if not all_windows:
        raise RuntimeError("No training windows could be constructed.")

    X_raw = np.vstack(all_windows)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    ae = MLPRegressor(
        hidden_layer_sizes=tuple(hidden_layers),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        max_iter=max_iter,
        random_state=42,
        n_iter_no_change=20,
        verbose=False,
    )
    logging.info("Training autoencoder  arch=%s  windows=%d  features=%d",
                 hidden_layers, len(X_scaled), X_scaled.shape[1])
    ae.fit(X_scaled, X_scaled)
    logging.info("Training done — loss: %.6f  iters: %d", ae.loss_, ae.n_iter_)
    return ae, scaler


def compute_run_errors(
    run_df: pd.DataFrame,
    signal_cols: List[str],
    ae: MLPRegressor,
    scaler: StandardScaler,
    W: int,
) -> np.ndarray:
    """
    Returns per_sample error array of shape (T, n_signals).
    Timesteps not covered by any window (first/last W//2 at boundaries)
    receive the nearest valid window's error.
    """
    arr = run_df[signal_cols].values.astype(float)
    T, n_signals = arr.shape

    if T < W:
        # Run is shorter than the window size — no windows can be formed.
        # Returning zeros silently excludes this run from anomaly detection;
        # its signals will show zero error and never be flagged as anomalous.
        logging.warning(
            "Run skipped in anomaly detection: only %d samples, need at least %d (window size). "
            "All signals will have zero reconstruction error for this run.",
            T, W,
        )
        return np.zeros((T, n_signals))

    windows_raw, starts = make_windows(arr, W, stride=1)
    windows_scaled = scaler.transform(windows_raw)
    preds_scaled = ae.predict(windows_scaled)

    errors_3d = ((preds_scaled - windows_scaled) ** 2).reshape(-1, W, n_signals)
    return scatter_add_window_errors(errors_3d, starts, T, W, n_signals)


def compute_all_errors(
    df: pd.DataFrame,
    signal_cols: List[str],
    ae: MLPRegressor,
    scaler: StandardScaler,
    W: int,
) -> pd.DataFrame:
    """
    Compute per-timestep per-signal MSE for every run.

    Returns long-format DataFrame:
        run_id | time_s | signal | ae_error
    """
    records = []
    for run_id, run_df in df.groupby("run_id"):
        per_sample = compute_run_errors(run_df, signal_cols, ae, scaler, W)
        time_vals  = run_df.index.total_seconds().values

        tmp = pd.DataFrame(per_sample, columns=signal_cols)
        tmp["run_id"] = run_id
        tmp["time_s"] = time_vals
        records.append(tmp.melt(id_vars=["run_id", "time_s"],
                                var_name="signal", value_name="ae_error"))

    return pd.concat(records, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Thresholding
# ─────────────────────────────────────────────────────────────────────────────

def compute_thresholds(
    error_df: pd.DataFrame,
    training_run_ids: List[str],
    percentile: float,
) -> Dict[str, float]:
    """Per-signal reconstruction error threshold based on training runs."""
    train_errors = error_df[error_df["run_id"].isin(training_run_ids)]
    thresholds = (
        train_errors.groupby("signal")["ae_error"]
        .quantile(percentile / 100.0)
        .to_dict()
    )
    for sig, thr in thresholds.items():
        logging.info("  Threshold  %-20s  %.6f  (p%.0f)", sig, thr, percentile)
    return thresholds


def flag_anomalies(
    error_df: pd.DataFrame,
    thresholds: Dict[str, float],
) -> pd.DataFrame:
    """
    Add columns:
        is_anomaly       — bool
        score_normalized — excess above threshold (0 for non-anomalies)
    """
    df = error_df.copy()
    df["threshold"] = df["signal"].map(thresholds)
    df["is_anomaly"] = df["ae_error"] > df["threshold"]
    df["score_normalized"] = (df["ae_error"] - df["threshold"]).clip(lower=0.0)
    return df.drop(columns=["threshold"])


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_ae_signal_summary(ae_long: pd.DataFrame) -> pd.DataFrame:
    anom = ae_long[ae_long["is_anomaly"]]
    if anom.empty:
        return pd.DataFrame(columns=["signal", "anomaly_count", "ae_error_sum", "score_normalized_sum"])
    return (
        anom.groupby("signal")
        .agg(anomaly_count=("signal", "size"),
             ae_error_sum=("ae_error", "sum"),
             score_normalized_sum=("score_normalized", "sum"))
        .reset_index()
        .sort_values(["score_normalized_sum", "anomaly_count"], ascending=[False, False])
    )


def aggregate_ae_ecu_relevance(
    ae_long: pd.DataFrame,
    sensor_ecu_df: pd.DataFrame,
) -> pd.DataFrame:
    anom = ae_long[ae_long["is_anomaly"]]
    if anom.empty:
        return pd.DataFrame(columns=["ecu", "anomaly_count", "ae_error_sum", "score_normalized_sum"])
    sensor_to_ecu = dict(zip(sensor_ecu_df["signal"], sensor_ecu_df["ecu"]))
    tmp = anom.copy()
    tmp["ecu"] = tmp["signal"].map(sensor_to_ecu).fillna("UNKNOWN")
    return (
        tmp.groupby("ecu")
        .agg(anomaly_count=("ecu", "size"),
             ae_error_sum=("ae_error", "sum"),
             score_normalized_sum=("score_normalized", "sum"))
        .reset_index()
        .sort_values(["score_normalized_sum", "anomaly_count"], ascending=[False, False])
    )


# ─────────────────────────────────────────────────────────────────────────────
# DTC linking (AE scores)
# ─────────────────────────────────────────────────────────────────────────────

def extract_ae_dtc_windows(
    dtc_df: pd.DataFrame,
    ae_long: pd.DataFrame,
    sensor_ecu_df: pd.DataFrame,
    before_s: float,
    after_s: float,
) -> pd.DataFrame:
    empty_cols = ["run_id", "relative_time_s", "dtc_code", "dtc_ecu", "description",
                  "top_signal", "top_signal_score", "top_ecu", "top_ecu_score",
                  "matched_anomaly_count"]
    if dtc_df.empty:
        return pd.DataFrame(columns=empty_cols)
    if "relative_time_s" not in dtc_df.columns or dtc_df["relative_time_s"].isna().all():
        return pd.DataFrame(columns=empty_cols)

    sensor_to_ecu = dict(zip(sensor_ecu_df["signal"], sensor_ecu_df["ecu"]))
    anom = ae_long[ae_long["is_anomaly"]].copy()
    anom["ecu"] = anom["signal"].map(sensor_to_ecu).fillna("UNKNOWN")
    rows = []

    for _, dtc in dtc_df.iterrows():
        run_id = dtc.get("run_id", "")
        rel_t  = dtc.get("relative_time_s", np.nan)
        if pd.isna(rel_t):
            continue
        mask = (anom["time_s"] >= rel_t - before_s) & (anom["time_s"] <= rel_t + after_s)
        if run_id:
            mask &= anom["run_id"] == run_id
        win = anom.loc[mask]
        signal_scores = win.groupby("signal")["score_normalized"].sum().sort_values(ascending=False)
        ecu_scores    = win.groupby("ecu")["score_normalized"].sum().sort_values(ascending=False)
        rows.append({
            "run_id":               run_id,
            "relative_time_s":      rel_t,
            "dtc_code":             dtc.get("dtc_code", ""),
            "dtc_ecu":              dtc.get("ecu", ""),
            "description":          dtc.get("description", ""),
            "top_signal":           signal_scores.index[0]   if not signal_scores.empty else "",
            "top_signal_score":     float(signal_scores.iloc[0]) if not signal_scores.empty else 0.0,
            "top_ecu":              ecu_scores.index[0]      if not ecu_scores.empty else "",
            "top_ecu_score":        float(ecu_scores.iloc[0])  if not ecu_scores.empty else 0.0,
            "matched_anomaly_count": int(len(win)),
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Model persistence
# ─────────────────────────────────────────────────────────────────────────────

def save_model_bundle(
    path: Path,
    ae: MLPRegressor,
    scaler: StandardScaler,
    thresholds: Dict[str, float],
    signal_cols: List[str],
    window_size: int,
    threshold_pct: float,
) -> None:
    joblib.dump({
        "ae": ae,
        "scaler": scaler,
        "thresholds": thresholds,
        "signal_cols": signal_cols,
        "window_size": window_size,
        "threshold_percentile": threshold_pct,
    }, path)
    logging.info("Model bundle saved to %s", path)


def load_model_bundle(path: Path) -> dict:
    bundle = joblib.load(path)
    logging.info("Model bundle loaded from %s", path)
    return bundle


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

    intermediate_dir = Path(args.intermediate_dir)
    output_dir       = Path(args.output_dir)
    ae_dir = output_dir / "autoencoder"
    ae_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────────
    df = load_intermediate(intermediate_dir)

    sensor_ecu_map = load_sensor_ecu_map(args.sensor_ecu_map)
    dtc_df         = load_dtc_log(args.dtc_log)

    # ── Train or load model ──────────────────────────────────────────────────
    if args.load_model:
        bundle       = load_model_bundle(Path(args.load_model))
        ae           = bundle["ae"]
        scaler       = bundle["scaler"]
        thresholds   = bundle["thresholds"]
        signal_cols  = bundle["signal_cols"]
        window_size  = bundle["window_size"]
        threshold_pct = bundle["threshold_percentile"]
        training_run_ids = []
        missing = [s for s in signal_cols if s not in df.columns]
        if missing:
            raise ValueError(f"Model expects signals not present in data: {missing}")
        logging.info("Inference runs: %s", df["run_id"].unique().tolist())
    else:
        signal_cols  = [c for c in df.columns if c != "run_id"]
        window_size  = args.window_size
        threshold_pct = args.ae_threshold_pct
        training_run_ids = identify_training_runs(df, args.normal_runs)
        logging.info("Training runs : %s", training_run_ids)
        logging.info("Inference runs: %s", df["run_id"].unique().tolist())
        ae, scaler = train_autoencoder(
            df, training_run_ids, signal_cols,
            W=window_size,
            hidden_layers=args.hidden_layers,
            max_iter=args.max_iter,
        )
        error_df   = compute_all_errors(df, signal_cols, ae, scaler, window_size)
        thresholds = compute_thresholds(error_df, training_run_ids, threshold_pct)
        if args.save_model:
            save_model_bundle(
                Path(args.save_model), ae, scaler, thresholds,
                signal_cols, window_size, threshold_pct,
            )

    sensor_ecu_df = build_sensor_ecu_table(signal_cols, sensor_ecu_map)

    # ── Compute reconstruction errors ────────────────────────────────────────
    logging.info("Computing per-signal reconstruction errors …")
    error_df = compute_all_errors(df, signal_cols, ae, scaler, window_size)

    # ── Flag anomalies ───────────────────────────────────────────────────────
    ae_long    = flag_anomalies(error_df, thresholds)
    total_anom = ae_long["is_anomaly"].sum()
    logging.info("AE anomaly events: %d  (threshold percentile=%.0f)",
                 total_anom, threshold_pct)

    # ── Aggregate ────────────────────────────────────────────────────────────
    ae_signal_summary = aggregate_ae_signal_summary(ae_long)
    ae_ecu_relevance  = aggregate_ae_ecu_relevance(ae_long, sensor_ecu_df)

    # ── DTC linking ──────────────────────────────────────────────────────────
    ae_dtc = pd.DataFrame()
    if not dtc_df.empty and "relative_time_s" in dtc_df.columns:
        ae_dtc = extract_ae_dtc_windows(
            dtc_df, ae_long, sensor_ecu_df,
            args.dtc_window_before_s, args.dtc_window_after_s,
        )

    # ── Write autoencoder outputs ─────────────────────────────────────────────
    ae_long[["run_id", "time_s", "signal", "ae_error",
             "is_anomaly", "score_normalized"]].to_csv(
        ae_dir / "ae_signal_errors.csv", index=False)
    ae_signal_summary.to_csv(ae_dir / "ae_anomaly_summary.csv", index=False)
    ae_ecu_relevance.to_csv(ae_dir / "ae_ecu_relevance.csv", index=False)
    if not ae_dtc.empty:
        ae_dtc.to_csv(ae_dir / "ae_dtc_candidates.csv", index=False)

    model_info = {
        "backend":             "sklearn.MLPRegressor",
        "hidden_layer_sizes":  list(ae.hidden_layer_sizes),
        "input_features":      window_size * len(signal_cols),
        "window_size_samples": window_size,
        "signals":             signal_cols,
        "training_runs":       training_run_ids,
        "training_windows":    int(sum(
            max(0, run_df.shape[0] - window_size + 1)
            for _, run_df in df[df["run_id"].isin(training_run_ids)].groupby("run_id")
        )) if training_run_ids else 0,
        "actual_iters":        int(ae.n_iter_),
        "final_loss":          float(ae.loss_),
        "threshold_percentile": threshold_pct,
        "thresholds":          {k: float(v) for k, v in thresholds.items()},
        "loaded_from":         str(args.load_model) if args.load_model else None,
    }
    with open(ae_dir / "ae_model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)
    logging.info("Autoencoder outputs written to %s", ae_dir)

    logging.info("Done.")


if __name__ == "__main__":
    main()
