"""
plot_wind_power_predictions.py

Loads a wind power model checkpoint produced by train_wind_power_model.py
and a dataset (same CSV / xlsx-folder format the training script accepts),
runs inference, and plots Actual vs Predicted pow_out:
  - left panel:  a time-ordered line plot (last N points)
  - right panel: a scatter of actual vs predicted with a y=x reference line

Usage:
    python plot_wind_power_predictions.py --model wind_power_model.pt --data combined_df.csv
    python plot_wind_power_predictions.py --model wind_power_model.pt \
        --data "dataset/energy generation/wind energy generation dataset/slightly polluted/" \
        --max_points 800
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------
# Must match train_wind_power_model.py
# --------------------------------------------------------------------------
FEATURE_COLS = ["wind_speed", "temp", "prs", "hum%"] +  ["pow_out_lag", "wind_speed_sq", "wind_speed_cub"]
TARGET_COL = "pow_out"
ALIASES = {"hum": "hum%"}


def _norm(s):
    return " ".join(str(s).split()).lower()


_CANONICAL_LOOKUP = {_norm(c): c for c in FEATURE_COLS + [TARGET_COL]}
for alias, canon in ALIASES.items():
    _CANONICAL_LOOKUP[_norm(alias)] = canon


def clean_columns(df):
    rename_map = {}
    for col in df.columns:
        key = _norm(col)
        rename_map[col] = _CANONICAL_LOOKUP.get(key, " ".join(str(col).split()))
    return df.rename(columns=rename_map)


class PowerMLP(nn.Module):
    """v2: BatchNorm after each Linear (helps optimization + regularizes a
    bit on its own), dropout trimmed down since weight decay now shares
    the regularization load."""

    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x)


# --------------------------------------------------------------------------
# Data loading (kept deliberately simple/raw -- no outlier stripping here,
# so the plot reflects how the model performs on the data as given)
# --------------------------------------------------------------------------
def load_series(data_path):
    """Returns a list of (name, DataFrame) with FEATURE_COLS + TARGET_COL,
    numeric, NaN rows dropped."""
    series = []

    if os.path.isfile(data_path) and data_path.lower().endswith(".csv"):
        df = clean_columns(pd.read_csv(data_path))
        series.append((os.path.basename(data_path), df))

    elif os.path.isdir(data_path):
        paths = sorted(glob.glob(os.path.join(data_path, "*.xlsx")))
        if not paths:
            raise FileNotFoundError(f"No .xlsx files found in {data_path}")
        for p in paths:
            xl = pd.ExcelFile(p)
            for sheet in xl.sheet_names:
                df = clean_columns(xl.parse(sheet))
                series.append((f"{os.path.basename(p)}::{sheet}", df))
    else:
        raise FileNotFoundError(
            f"--data must be a .csv file or a directory of .xlsx files: {data_path}"
        )

    cleaned = []
    for name, df in series:
        missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
        if missing:
            print(f"[skip] {name} -- missing columns {missing}")
            continue
        df = df[FEATURE_COLS + [TARGET_COL]].copy()
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna().reset_index(drop=True)
        if len(df) == 0:
            print(f"[skip] {name} -- 0 usable rows")
            continue
        cleaned.append((name, df))

    if not cleaned:
        raise RuntimeError("No usable series found in --data")
    return cleaned


def build_pairs(series, horizon):
    """Build (X, y) pairs of features_t -> pow_out_{t+horizon}, never
    crossing series/file/sheet boundaries -- matches the training script."""
    Xs, ys = [], []
    for name, df in series:
        if len(df) <= horizon:
            print(f"[skip] {name} -- too short for horizon={horizon}")
            continue
        X = df[FEATURE_COLS].values[:-horizon]
        y = df[TARGET_COL].values[horizon:]
        Xs.append(X)
        ys.append(y)
    if not Xs:
        raise RuntimeError("No series long enough for the requested horizon.")
    X = np.concatenate(Xs, axis=0).astype(np.float32)
    y = np.concatenate(ys, axis=0).astype(np.float32).reshape(-1, 1)
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="wind_power_model.pt",
                     help="Path to the .pt checkpoint saved by train_wind_power_model.py")
    ap.add_argument("--data", type=str, required=True,
                     help="Path to combined_df.csv OR a folder of .xlsx wind files")
    ap.add_argument("--horizon", type=int, default=None,
                     help="Override forecast horizon; defaults to the value stored in the checkpoint")
    ap.add_argument("--max_points", type=int, default=500,
                     help="Max points shown on the time-series panel (most recent N)")
    ap.add_argument("--out", type=str, default="wind_power_predictions.png",
                     help="Path to save the output plot image")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.model, weights_only=False, map_location=device)

    horizon = args.horizon if args.horizon is not None else ckpt.get("horizon", 1)

    x_mean, x_std = ckpt["x_mean"], ckpt["x_std"]
    y_mean, y_std = ckpt["y_mean"], ckpt["y_std"]

    model = PowerMLP(in_dim=len(FEATURE_COLS)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    series = load_series(args.data)
    X, y_true = build_pairs(series, horizon)

    X_s = (X - x_mean) / x_std
    with torch.no_grad():
        pred_s = model(torch.from_numpy(X_s.astype(np.float32)).to(device)).cpu().numpy()
    y_pred = pred_s * y_std + y_mean

    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    # ---- Metrics ----
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    print(f"Points: {len(y_true)} | MAE: {mae:.3f} | RMSE: {rmse:.3f} | R^2: {r2:.4f}")

    # ---- Plot ----
    n_show = min(args.max_points, len(y_true))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(y_true[-n_show:], label="Actual", linewidth=1.5)
    ax.plot(y_pred[-n_show:], label="Predicted", linewidth=1.2, alpha=0.8)
    ax.set_title(f"Actual vs Predicted pow_out (last {n_show} points, horizon={horizon})")
    ax.set_xlabel("Time step")
    ax.set_ylabel("pow_out")
    ax.legend()

    ax2 = axes[1]
    ax2.scatter(y_true, y_pred, s=8, alpha=0.4)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax2.plot(lims, lims, "r--", linewidth=1, label="Perfect prediction")
    ax2.set_title(f"Actual vs Predicted (all {len(y_true)} points)\nR^2={r2:.3f}")
    ax2.set_xlabel("Actual pow_out")
    ax2.set_ylabel("Predicted pow_out")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved plot to {args.out}")
    plt.show()


if __name__ == "__main__":
    main()