"""
plot_solar_predictions.py

Loads the trained solar power model (solar_power_model.pt, produced by
train_solar_power_model.py) plus the original .xlsx data, then:

    1. Plots the FULL actual Power (MW) time series for every file/sheet.
    2. Feeds the weather/irradiance features (+ Power lag features) through
       the model to get predictions and overlays them on the actual curve.
    3. Also draws one big concatenated "all series back to back" plot
       with vertical dashed lines marking where each file/sheet starts.

This script mirrors train_solar_power_model.py's data cleaning and feature
engineering exactly (irradiance features, corrupted-row filtering, Power
lag features), so what you see plotted matches what the model was
actually trained on.

Usage:
    python plot_solar_predictions.py \
        --data_dir "dataset/energy generation/solar energy genration dataset/polluted dataset/" \
        --model solar_power_model.pt \
        --out_dir plots
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
# Sanity ranges — must match train_solar_power_model.py so cleaning behaves
# identically at plot time and at train time.
# --------------------------------------------------------------------------
VALID_RANGES = {
    "Total solar irradiance (W/m2)": (0, 1500),
    "Direct normal irradiance (W/m2)": (0, 1500),
    "Global horizontal irradiance (W/m2)": (0, 1500),
    "Air temperature (°C)": (-40, 60),
    "Atmosphere (hpa)": (800, 1100),
    "Relative humidity (%)": (0, 100),
    "Power (MW)": (-1, 1000),
}


# --------------------------------------------------------------------------
# Model definition (must match train_solar_power_model.py's PowerMLP shape)
# --------------------------------------------------------------------------
class PowerMLP(nn.Module):
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x)


def build_model_from_checkpoint(ckpt, device):
    """Infer in_dim/hidden directly from the saved weight shapes instead of
    hardcoding them, so this script stays correct even if training-time
    hyperparameters change."""
    state_dict = ckpt["model_state_dict"]
    first_w = state_dict["net.0.weight"]  # shape: [hidden, in_dim]
    hidden, in_dim = first_w.shape
    model = PowerMLP(in_dim=in_dim, hidden=hidden).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


# --------------------------------------------------------------------------
# Column matching helpers (same normalization logic as the training script)
# --------------------------------------------------------------------------
def _norm(s: str) -> str:
    s = " ".join(str(s).split())
    s = s.replace("horicontal", "horizontal")
    return s.lower()


def clean_columns(df: pd.DataFrame, canonical_names) -> pd.DataFrame:
    lookup = {_norm(c): c for c in canonical_names}
    rename_map = {}
    for col in df.columns:
        key = _norm(col)
        rename_map[col] = lookup.get(key, " ".join(str(col).split()))
    return df.rename(columns=rename_map)


def drop_corrupted_rows(df: pd.DataFrame, source_label: str) -> pd.DataFrame:
    """Drop physically-impossible rows, identical logic to the training
    script (e.g. the -3270 degC air-temperature rows in the polluted
    dataset)."""
    before = len(df)
    mask = pd.Series(True, index=df.index)
    for col, (lo, hi) in VALID_RANGES.items():
        if col in df.columns:
            mask &= df[col].between(lo, hi)
    df = df[mask].reset_index(drop=True)
    removed = before - len(df)
    if removed:
        pct = 100.0 * removed / before
        print(f"[clean] {source_label} -- dropped {removed}/{before} rows "
              f"({pct:.1f}%) outside sanity ranges")
    return df


def add_lag_features(df: pd.DataFrame, target_col: str, lag_steps) -> pd.DataFrame:
    """Add lagged Power columns, identical logic to the training script."""
    df = df.copy()
    for lag in lag_steps:
        df[f"power_lag_{lag}"] = df[target_col].shift(lag)
    return df


def load_series(data_dir, base_feature_cols, target_col):
    """Returns a list of dicts: {name, df} — one per usable file/sheet.
    df contains only the RAW weather/irradiance columns + target; lag
    features are added later, per series, after corrupted-row cleaning."""
    paths = sorted(glob.glob(os.path.join(data_dir, "*.xlsx")))
    if not paths:
        raise FileNotFoundError(f"No .xlsx files found in {data_dir}")

    canonical = list(base_feature_cols) + [target_col]
    series = []
    for p in paths:
        xl = pd.ExcelFile(p)
        for sheet in xl.sheet_names:
            df = xl.parse(sheet)
            df = clean_columns(df, canonical)

            missing = [c for c in canonical if c not in df.columns]
            if missing:
                print(f"[skip] {p} :: {sheet} -- missing columns {missing}")
                continue

            df = df[canonical].copy()
            for c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna().reset_index(drop=True)

            name = f"{os.path.basename(p)} :: {sheet}"
            df = drop_corrupted_rows(df, name)

            if len(df) == 0:
                print(f"[skip] {name} -- 0 usable rows after cleaning")
                continue

            series.append({"name": name, "df": df})
            print(f"[ok]   {name} -> {len(df)} clean rows")

    if not series:
        raise RuntimeError("No usable sheets found across all files.")
    return series


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                         help="Folder containing the .xlsx solar files")
    parser.add_argument("--model", type=str, default="solar_power_model.pt",
                         help="Path to the trained checkpoint")
    parser.add_argument("--out_dir", type=str, default="plots",
                         help="Where to save PNG plots")
    parser.add_argument("--show", action="store_true",
                         help="Also open interactive matplotlib windows")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Load checkpoint ----
    ckpt = torch.load(args.model, weights_only=False, map_location=device)
    feature_cols = ckpt["feature_cols"]        # full list, incl. power_lag_* cols
    target_col = ckpt["target_col"]
    horizon = ckpt["horizon"]
    lags = ckpt.get("lags", [])                 # backward-compatible default
    x_mean, x_std = ckpt["x_mean"], ckpt["x_std"]
    y_mean, y_std = ckpt["y_mean"], ckpt["y_std"]

    base_feature_cols = [c for c in feature_cols if not c.startswith("power_lag_")]
    lag_offset = max(lags) if lags else 0

    model = build_model_from_checkpoint(ckpt, device)
    print(f"Loaded model: features={feature_cols}, target={target_col}, "
          f"horizon={horizon}, lags={lags}")

    # ---- Load data ----
    series = load_series(args.data_dir, base_feature_cols, target_col)

    # ---- Predict per series ----
    concat_actual = []
    concat_predicted = []  # NaN-padded to line up index-for-index with actual
    boundaries = []  # index into concat_actual where each series starts
    cursor = 0

    for s in series:
        df = s["df"]
        M = len(df)
        y_true = df[target_col].values.astype(np.float32)  # length M

        # Build lag features, then drop the first `lag_offset` rows that
        # have no lag history yet (same as training). Remaining rows
        # correspond to original positions [lag_offset, M-1].
        df_lagged = add_lag_features(df, target_col, lags)
        df_lagged = df_lagged.iloc[lag_offset:].reset_index(drop=True)

        y_pred_aligned = np.full(M, np.nan, dtype=np.float32)

        if len(df_lagged) > 0:
            X = df_lagged[feature_cols].values.astype(np.float32)
            X_scaled = (X - x_mean) / x_std
            with torch.no_grad():
                xb = torch.from_numpy(X_scaled.astype(np.float32)).to(device)
                y_pred_scaled = model(xb).cpu().numpy()
            y_pred = (y_pred_scaled * y_std + y_mean).reshape(-1)

            # Row i of df_lagged used features at original position
            # (lag_offset + i) to forecast the value at
            # (lag_offset + i + horizon).
            valid_count = M - lag_offset - horizon
            if valid_count > 0:
                start = lag_offset + horizon
                y_pred_aligned[start:start + valid_count] = y_pred[:valid_count]

        # ---- Per-series plot: actual vs predicted ----
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(y_true, label="Actual Power (MW)", color="tab:blue", linewidth=1)
        ax.plot(y_pred_aligned, label=f"Predicted (t+{horizon})", color="tab:orange",
                linewidth=1, alpha=0.8)
        ax.set_title(s["name"])
        ax.set_xlabel("Time step")
        ax.set_ylabel("Power (MW)")
        ax.legend()
        fig.tight_layout()
        safe_name = s["name"].replace("/", "_").replace(" ", "_").replace(":", "")
        out_path = os.path.join(args.out_dir, f"{safe_name}.png")
        fig.savefig(out_path, dpi=150)
        print(f"Saved {out_path}")
        if not args.show:
            plt.close(fig)

        boundaries.append(cursor)
        concat_actual.append(y_true)
        concat_predicted.append(y_pred_aligned)
        cursor += len(y_true)

    # ---- Combined "all series back to back" plot ----
    all_actual = np.concatenate(concat_actual)
    all_predicted = np.concatenate(concat_predicted)

    fig, ax = plt.subplots(figsize=(18, 5))
    ax.plot(all_actual, label="Actual Power (MW)", color="tab:blue", linewidth=0.8)
    ax.plot(all_predicted, label=f"Predicted (t+{horizon})", color="tab:orange",
            linewidth=0.8, alpha=0.8)
    for b in boundaries[1:]:
        ax.axvline(b, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)
    ax.set_title("All files/sheets concatenated — actual vs predicted Power (MW)")
    ax.set_xlabel("Time step (concatenated across files)")
    ax.set_ylabel("Power (MW)")
    ax.legend()
    fig.tight_layout()
    combined_path = os.path.join(args.out_dir, "all_series_combined.png")
    fig.savefig(combined_path, dpi=150)
    print(f"Saved {combined_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()