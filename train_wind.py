"""
train_wind_power_model.py

Trains a PyTorch neural network (on CUDA if available) to predict FUTURE
pow_out (wind power output) from weather features:

    - wind_speed
    - temp
    - prs   (pressure)
    - hum%  (relative humidity  -- also accepts a plain "hum" column name)

"Future" power means the model is given readings at time t and predicts
pow_out `HORIZON` steps ahead (t + HORIZON). Change --horizon to control
how far ahead you forecast (in units of rows in your data).

Works with EITHER:
  (a) a single combined CSV, e.g. combined_df.csv
        python train_wind_power_model.py --data combined_df.csv
  (b) a folder of .xlsx files (each sheet = one time series)
        python train_wind_power_model.py --data "dataset/energy generation/wind energy generation dataset/slightly polluted/"

If --data points at a .csv it's loaded as one series. If it's a
directory, every .xlsx file's every sheet is loaded as its own series
(so predictions never leak across file/sheet boundaries when building
the future-shifted target). Sheets/files missing the required columns
(e.g. the weather-only file with no pow_out) are skipped automatically.

DATA CLEANING (this dataset is labeled "slightly polluted"):
    1. Generic sanity ranges for wind_speed / temp / prs / hum%.
    2. Per-file rated-capacity bound for pow_out, parsed from filenames
       like "wind_3_mx66.xlsx" -> capacity 66. Falls back to a wide
       default bound if no "mxNN" pattern is found (e.g. for a CSV).
    3. Cut-in-speed physics check: near-zero wind_speed should mean
       near-zero pow_out. Rows violating this (e.g. wind_speed=0 with
       pow_out=15) are dropped as sensor/logging faults.
    4. Stuck-sensor detection: long runs of exactly-repeated feature
       rows (frozen instrument) are dropped -- they carry no usable
       signal and only bias the model. Controlled by
       --stuck_run_threshold / --no_drop_stuck.

The script will:
    1. Load the data (CSV or xlsx folder).
    2. Clean column names (whitespace/case-insensitive matching, "hum"
       treated as an alias of "hum%").
    3. Apply the cleaning steps above.
    4. Build (features_t -> pow_out_{t+HORIZON}) pairs.
    5. Time-ordered train/val split (not random) to avoid leakage.
    6. Scale features and target (scalers fit on train only).
    7. Train an MLP regressor on GPU (CUDA) if available.
    8. Save the trained model + scalers.
"""

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
FEATURE_COLS = ["wind_speed", "temp", "prs", "hum%"]
TARGET_COL = "pow_out"

# Alternate spellings -> canonical name
ALIASES = {
    "hum": "hum%",
}

# Generic sanity ranges (checked on every row regardless of file).
# pow_out's upper bound is handled separately (per-file capacity, see below).
VALID_RANGES = {
    "wind_speed": (0, 60),      # m/s; hurricane-force and above is not real turbine data
    "temp": (-60, 55),          # deg C, generous global range
    "prs": (800, 1100),         # hPa
    "hum%": (0, 100),
    "pow_out": (-2, None),      # small negative allowed (parasitic/idle load noise);
                                 # upper bound filled in per-file from rated capacity
}

# Physically implausible: turbines don't produce meaningful power below
# their cut-in wind speed.
CUT_IN_WIND_SPEED = 0.5          # m/s, below this wind is "calm"
CUT_IN_POWER_SLACK = 2.0         # pow_out above this (in calm wind) is flagged

# Default upper bound for pow_out when no per-file capacity can be parsed
# from the filename (e.g. a plain CSV).
DEFAULT_MAX_POWER = 1e6          # effectively "no upper bound" fallback
CAPACITY_TOLERANCE = 1.05        # allow 5% over nameplate capacity (sensor noise)

DEFAULT_STUCK_RUN_THRESHOLD = 200  # consecutive identical feature rows -> "stuck sensor"

HORIZON = 1          # how many rows ahead to predict (future step)
BATCH_SIZE = 256
EPOCHS = 50
LR = 1e-3
VAL_SPLIT = 0.15
SEED = 42


# --------------------------------------------------------------------------
# Column matching helpers
# --------------------------------------------------------------------------
def _norm(s: str) -> str:
    return " ".join(str(s).split()).lower()


_CANONICAL_LOOKUP = {_norm(c): c for c in FEATURE_COLS + [TARGET_COL]}
for alias, canon in ALIASES.items():
    _CANONICAL_LOOKUP[_norm(alias)] = canon


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for col in df.columns:
        key = _norm(col)
        rename_map[col] = _CANONICAL_LOOKUP.get(key, " ".join(str(col).split()))
    return df.rename(columns=rename_map)


# --------------------------------------------------------------------------
# Cleaning helpers
# --------------------------------------------------------------------------
def parse_capacity(source_name: str):
    """Parse a rated-capacity hint like 'mx66' out of a filename. Returns
    None if no such pattern is found (e.g. for a CSV)."""
    m = re.search(r"mx(\d+(?:\.\d+)?)", source_name, flags=re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


def drop_out_of_range_rows(df: pd.DataFrame, source_name: str, capacity) -> pd.DataFrame:
    before = len(df)
    mask = pd.Series(True, index=df.index)
    for col, (lo, hi) in VALID_RANGES.items():
        if col not in df.columns:
            continue
        if lo is not None:
            mask &= df[col] >= lo
        if hi is not None:
            mask &= df[col] <= hi

    # Per-file capacity bound on pow_out.
    max_power = (capacity * CAPACITY_TOLERANCE) if capacity is not None else DEFAULT_MAX_POWER
    mask &= df[TARGET_COL] <= max_power

    df = df[mask].reset_index(drop=True)
    removed = before - len(df)
    if removed:
        pct = 100.0 * removed / before
        cap_note = f"capacity={capacity}" if capacity is not None else "capacity=unknown (no bound applied)"
        print(f"[clean] {source_name} -- dropped {removed}/{before} rows "
              f"({pct:.1f}%) outside sanity ranges [{cap_note}]")
    return df


def drop_cut_in_violations(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Drop rows where wind_speed is ~0 but pow_out is well above zero --
    physically impossible (turbines need wind above cut-in speed to
    generate meaningful power)."""
    before = len(df)
    violation = (df["wind_speed"] < CUT_IN_WIND_SPEED) & (df[TARGET_COL] > CUT_IN_POWER_SLACK)
    df = df[~violation].reset_index(drop=True)
    removed = before - len(df)
    if removed:
        pct = 100.0 * removed / before
        print(f"[clean] {source_name} -- dropped {removed}/{before} rows "
              f"({pct:.1f}%) violating cut-in-speed physics "
              f"(wind_speed<{CUT_IN_WIND_SPEED} but pow_out>{CUT_IN_POWER_SLACK})")
    return df


def drop_stuck_sensor_runs(df: pd.DataFrame, source_name: str, threshold: int) -> pd.DataFrame:
    """Detect and drop long runs where ALL feature columns are exactly
    repeated for `threshold`+ consecutive rows -- a frozen/stuck sensor
    contributes zero information and only biases training."""
    if threshold <= 0 or len(df) == 0:
        return df

    feat = df[FEATURE_COLS]
    same_as_prev = (feat == feat.shift(1)).all(axis=1)
    # Assign a run id that increments every time the row differs from the previous one.
    run_id = (~same_as_prev).cumsum()
    run_sizes = run_id.map(run_id.value_counts())
    stuck_mask = same_as_prev & (run_sizes >= threshold)
    # Also mark the first row of a long stuck run (same_as_prev is False for it).
    stuck_mask |= (~same_as_prev) & (run_sizes >= threshold)

    before = len(df)
    df = df[~stuck_mask].reset_index(drop=True)
    removed = before - len(df)
    if removed:
        pct = 100.0 * removed / before
        print(f"[clean] {source_name} -- dropped {removed}/{before} rows "
              f"({pct:.1f}%) belonging to stuck-sensor runs (>= {threshold} identical rows)")
    return df


def _finalize(df: pd.DataFrame, source_name: str, stuck_run_threshold: int, drop_stuck: bool):
    df = clean_columns(df)
    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        print(f"[skip] {source_name} -- missing columns {missing}")
        return None
    df = df[FEATURE_COLS + [TARGET_COL]].copy()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    if len(df) == 0:
        print(f"[skip] {source_name} -- 0 usable rows after basic cleaning")
        return None

    capacity = parse_capacity(source_name)
    df = drop_out_of_range_rows(df, source_name, capacity)
    df = drop_cut_in_violations(df, source_name)
    if drop_stuck:
        df = drop_stuck_sensor_runs(df, source_name, stuck_run_threshold)

    if len(df) == 0:
        print(f"[skip] {source_name} -- 0 usable rows after full cleaning")
        return None

    print(f"[ok]   {source_name} -> {len(df)} clean rows")
    return df


def load_all_series(data_path: str, stuck_run_threshold: int, drop_stuck: bool) -> list:
    """Returns a list of DataFrames, one per independent time series."""
    frames = []

    if os.path.isfile(data_path) and data_path.lower().endswith(".csv"):
        df = pd.read_csv(data_path)
        cleaned = _finalize(df, data_path, stuck_run_threshold, drop_stuck)
        if cleaned is not None:
            frames.append(cleaned)

    elif os.path.isdir(data_path):
        paths = sorted(glob.glob(os.path.join(data_path, "*.xlsx")))
        if not paths:
            raise FileNotFoundError(f"No .xlsx files found in {data_path}")
        for p in paths:
            xl = pd.ExcelFile(p)
            for sheet in xl.sheet_names:
                raw = xl.parse(sheet)
                cleaned = _finalize(raw, f"{p} :: {sheet}", stuck_run_threshold, drop_stuck)
                if cleaned is not None:
                    frames.append(cleaned)
    else:
        raise FileNotFoundError(f"--data must be a .csv file or a directory of .xlsx files: {data_path}")

    if not frames:
        raise RuntimeError("No usable data found.")
    return frames


def time_ordered_split(frames, horizon: int, val_split: float):
    """Take the last val_split fraction of EACH series (in time order) as
    validation, instead of randomly shuffling rows -- keeps val honest
    and avoids leaking near-duplicate adjacent timesteps into train."""
    Xtr_list, ytr_list, Xva_list, yva_list = [], [], [], []
    for df in frames:
        if len(df) <= horizon:
            continue
        X = df[FEATURE_COLS].values[: -horizon]
        y = df[TARGET_COL].values[horizon:]
        n = len(X)
        cut = int(n * (1 - val_split))
        cut = max(1, min(cut, n - 1)) if n > 1 else n

        Xtr_list.append(X[:cut]); ytr_list.append(y[:cut])
        Xva_list.append(X[cut:]); yva_list.append(y[cut:])

    if not Xtr_list:
        raise RuntimeError("No series long enough for the requested horizon.")

    Xtr = np.concatenate(Xtr_list, axis=0).astype(np.float32)
    ytr = np.concatenate(ytr_list, axis=0).astype(np.float32).reshape(-1, 1)
    Xva = np.concatenate(Xva_list, axis=0).astype(np.float32)
    yva = np.concatenate(yva_list, axis=0).astype(np.float32).reshape(-1, 1)
    return Xtr, ytr, Xva, yva


# --------------------------------------------------------------------------
# Dataset / Model
# --------------------------------------------------------------------------
class PowerDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


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


class StandardScalerTorch:
    """Tiny numpy-based standard scaler (avoids sklearn dependency)."""

    def fit(self, X):
        self.mean_ = X.mean(axis=0, keepdims=True)
        self.std_ = X.std(axis=0, keepdims=True)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def inverse_transform(self, X):
        return X * self.std_ + self.mean_


# --------------------------------------------------------------------------
# Train / eval loops
# --------------------------------------------------------------------------
def train(model, train_loader, val_loader, device, epochs, lr):
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss += criterion(pred, yb).item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch:3d}/{epochs} | train loss {train_loss:.5f} | val loss {val_loss:.5f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True,
                         help="Path to combined_df.csv OR a folder of .xlsx wind files")
    parser.add_argument("--horizon", type=int, default=HORIZON,
                         help="Rows ahead to forecast pow_out")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--stuck_run_threshold", type=int, default=DEFAULT_STUCK_RUN_THRESHOLD,
                         help="Consecutive identical feature rows to treat as a stuck sensor")
    parser.add_argument("--no_drop_stuck", action="store_true",
                         help="Disable stuck-sensor-run dropping")
    parser.add_argument("--out", type=str, default="wind_power_model.pt")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ---- Load & build data ----
    frames = load_all_series(args.data, args.stuck_run_threshold, not args.no_drop_stuck)

    Xtr, ytr, Xva, yva = time_ordered_split(frames, args.horizon, VAL_SPLIT)
    print(f"Train pairs: {len(Xtr)} | Val pairs: {len(Xva)}")

    x_scaler = StandardScalerTorch().fit(Xtr)   # fit scalers on TRAIN only
    y_scaler = StandardScalerTorch().fit(ytr)

    Xtr_s = x_scaler.transform(Xtr).astype(np.float32)
    ytr_s = y_scaler.transform(ytr).astype(np.float32)
    Xva_s = x_scaler.transform(Xva).astype(np.float32)
    yva_s = y_scaler.transform(yva).astype(np.float32)

    train_ds = PowerDataset(Xtr_s, ytr_s)
    val_ds = PowerDataset(Xva_s, yva_s)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=2, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=2, pin_memory=(device.type == "cuda"))

    # ---- Train ----
    model = PowerMLP(in_dim=len(FEATURE_COLS)).to(device)
    model, best_val = train(model, train_loader, val_loader, device, args.epochs, args.lr)
    print(f"Best val loss (scaled space): {best_val:.5f}")

    # ---- Save model + scalers ----
    torch.save({
        "model_state_dict": model.state_dict(),
        "x_mean": x_scaler.mean_, "x_std": x_scaler.std_,
        "y_mean": y_scaler.mean_, "y_std": y_scaler.std_,
        "feature_cols": FEATURE_COLS,
        "target_col": TARGET_COL,
        "horizon": args.horizon,
    }, args.out)
    print(f"Saved trained model to {args.out}")


if __name__ == "__main__":
    main()