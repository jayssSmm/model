import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# --------------------------------------------------------------------------
# Sanity ranges -- must match train_solar_power_model.py so cleaning behaves
# identically at eval time and at train time.
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

# Pulls plant capacity out of filenames like 'solar_6_mx35.xlsx' -> 35.0 MW.
# VERIFY this against your actual known plant capacities -- it's inferred
# from the naming pattern in your files, not from any ground-truth table.
CAPACITY_RE = re.compile(r"mx(\d+(?:\.\d+)?)", re.IGNORECASE)


def extract_capacity_mw(filename: str):
    m = CAPACITY_RE.search(os.path.basename(filename))
    return float(m.group(1)) if m else None


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
    state_dict = ckpt["model_state_dict"]
    first_w = state_dict["net.0.weight"]  # [hidden, in_dim]
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
    df = df.copy()
    for lag in lag_steps:
        df[f"power_lag_{lag}"] = df[target_col].shift(lag)
    return df


def load_series(data_dir, base_feature_cols, target_col):
    """Returns a list of dicts: {name, df, capacity_mw}."""
    paths = sorted(glob.glob(os.path.join(data_dir, "*.xlsx")))
    if not paths:
        raise FileNotFoundError(f"No .xlsx files found in {data_dir}")

    canonical = list(base_feature_cols) + [target_col]
    series = []
    for p in paths:
        capacity = extract_capacity_mw(p)
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

            series.append({"name": name, "df": df, "capacity_mw": capacity})
            cap_str = f"{capacity} MW" if capacity is not None else "capacity UNKNOWN"
            print(f"[ok]   {name} -> {len(df)} clean rows ({cap_str})")

    if not series:
        raise RuntimeError("No usable sheets found across all files.")
    return series


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def compute_metrics(actual, predicted, capacity_mw):
    """All metrics computed on whatever rows are passed in -- caller decides
    whether that's all-hours, daytime-only, etc.

    NMAE / NRMSE are normalized by plant capacity (a constant, never-zero
    denominator), not by the actual value -- so they don't blow up at
    night/low-irradiance the way plain MAPE does. This is the standard way
    solar forecast error is reported for exactly that reason.
    """
    err = predicted - actual
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mbe = float(np.mean(err))  # signed: + = over-predicting, - = under-predicting

    out = {"n": len(actual), "mae_mw": mae, "rmse_mw": rmse, "mbe_mw": mbe}

    if capacity_mw:
        out["nmae_pct"] = 100.0 * mae / capacity_mw
        out["nrmse_pct"] = 100.0 * rmse / capacity_mw
        out["nmbe_pct"] = 100.0 * mbe / capacity_mw

        # Secondary, informational only: classic MAPE restricted to rows
        # where actual power is a meaningful fraction of capacity, so a
        # handful of near-zero rows can't dominate it. Still less robust
        # than NMAE above -- report it, don't optimize against it.
        sig_mask = actual > 0.05 * capacity_mw
        if sig_mask.sum() > 0:
            out["mape_daytime_pct"] = float(
                np.mean(np.abs(err[sig_mask]) / actual[sig_mask]) * 100
            )
            out["n_daytime"] = int(sig_mask.sum())

    return out


def print_metrics(name, m):
    print(f"{name}")
    print(f"  n rows scored     : {m['n']}")
    print(f"  MAE               : {m['mae_mw']:.3f} MW")
    print(f"  RMSE              : {m['rmse_mw']:.3f} MW")
    direction = "over" if m["mbe_mw"] > 0 else "under"
    print(f"  Bias (MBE)        : {m['mbe_mw']:+.3f} MW  ({direction}-predicting)")
    if "nmae_pct" in m:
        print(f"  NMAE (of capacity): {m['nmae_pct']:.2f}%   <-- primary accuracy number")
        print(f"  NRMSE(of capacity): {m['nrmse_pct']:.2f}%")
        print(f"  NMBE (of capacity): {m['nmbe_pct']:+.2f}%")
    else:
        print("  (capacity unknown -- fix CAPACITY_RE / filename to get normalized %)")
    if "mape_daytime_pct" in m:
        print(f"  MAPE (daytime only, informational): {m['mape_daytime_pct']:.2f}%  (n={m['n_daytime']})")
    print("-" * 50)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="solar_power_model.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ckpt = torch.load(args.model, weights_only=False, map_location=device)
    feature_cols = ckpt["feature_cols"]
    target_col = ckpt["target_col"]
    horizon = ckpt["horizon"]
    lags = ckpt.get("lags", [])
    x_mean, x_std = ckpt["x_mean"], ckpt["x_std"]
    y_mean, y_std = ckpt["y_mean"], ckpt["y_std"]

    base_feature_cols = [c for c in feature_cols if not c.startswith("power_lag_")]
    lag_offset = max(lags) if lags else 0

    model = build_model_from_checkpoint(ckpt, device)
    print(f"Loaded model: features={feature_cols}, target={target_col}, "
          f"horizon={horizon}, lags={lags}")

    series = load_series(args.data_dir, base_feature_cols, target_col)

    all_actual, all_pred = [], []

    for s in series:
        df = s["df"]
        capacity = s["capacity_mw"]
        M = len(df)

        y_true = df[target_col].values.astype(np.float32)

        df_lagged = add_lag_features(df, target_col, lags)
        df_lagged = df_lagged.iloc[lag_offset:].reset_index(drop=True)

        y_pred_aligned = np.full(M, np.nan, dtype=np.float32)

        if len(df_lagged) > 0:
            X = df_lagged[feature_cols].values.astype(np.float32)
            X_scaled = (X - x_mean) / x_std
            with torch.no_grad():
                xb = torch.from_numpy(X_scaled).to(device)
                y_pred_scaled = model(xb).cpu().numpy()
            y_pred = (y_pred_scaled * y_std + y_mean).reshape(-1)

            valid_count = M - lag_offset - horizon
            if valid_count > 0:
                start = lag_offset + horizon
                y_pred_aligned[start:start + valid_count] = y_pred[:valid_count]

        mask = ~np.isnan(y_pred_aligned)
        actual = y_true[mask]
        predicted = y_pred_aligned[mask]

        if len(actual) == 0:
            print(f"{s['name']} -- no scoreable rows, skipping")
            continue

        m = compute_metrics(actual, predicted, capacity)
        print_metrics(s["name"], m)

        all_actual.append(actual)
        all_pred.append(predicted)

    if all_actual:
        actual_all = np.concatenate(all_actual)
        pred_all = np.concatenate(all_pred)
        print("=" * 50)
        print("OVERALL (all series pooled, raw MW -- naturally dominated by")
        print("the largest-capacity plant; treat as a rough sanity check,")
        print("not the headline number)")
        m_all = compute_metrics(actual_all, pred_all, capacity_mw=None)
        print_metrics("pooled", m_all)


if __name__ == "__main__":
    main()