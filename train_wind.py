"""
train_wind_power_model.py  (v2 -- fixes underfitting)

Trains a PyTorch neural network (on CUDA if available) to predict FUTURE
pow_out (wind power output) from:
    - wind_speed, temp, prs, hum% (raw weather)
    - pow_out_lag   -- pow_out AT TIME t (autoregressive input; this is
                        known info, not leakage -- the target is t+HORIZON)
    - wind_speed_sq, wind_speed_cub -- engineered v^2 / v^3 terms, since
                        real turbine power curves are roughly ~v^3

WHY THIS CHANGED FROM v1:
    v1 predicted future power from weather alone and plateaued fast
    (train/val loss barely moved after ~10 epochs, R^2 ~0.75). That's a
    classic underfitting signature caused by omitting the single most
    predictive input: current power output is strongly autocorrelated
    with near-future power output, especially at small horizons. Adding
    it back in, plus explicit v^2/v^3 terms so the model doesn't have to
    rediscover the cubic power curve from scratch, should raise R^2
    substantially. Weight decay + BatchNorm were also added since val
    loss was ticking up slightly while train loss kept falling (mild
    overfitting on top of the underfitting).

Works with EITHER:
  (a) a single combined CSV, e.g. combined_df.csv
        python train_wind_power_model.py --data combined_df.csv
  (b) a folder of .xlsx files (each sheet = one time series)
        python train_wind_power_model.py --data "dataset/energy generation/wind energy generation dataset/slightly polluted/"

DATA CLEANING (unchanged from v1, this dataset is labeled "slightly polluted"):
    1. Generic sanity ranges for wind_speed / temp / prs / hum%.
    2. Per-file rated-capacity bound for pow_out, parsed from filenames
       like "wind_3_mx66.xlsx" -> capacity 66.
    3. Cut-in-speed physics check: near-zero wind_speed should mean
       near-zero pow_out.
    4. Stuck-sensor detection: long runs of exactly-repeated feature
       rows (frozen instrument) are dropped.
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
BASE_FEATURE_COLS = ["wind_speed", "temp", "prs", "hum%"]  # raw weather, used for cleaning/validation
TARGET_COL = "pow_out"

# Final input feature order fed to the network. Saved into the checkpoint
# so the plotting/inference script always builds features the same way.
AUGMENTED_FEATURE_NAMES = BASE_FEATURE_COLS + ["pow_out_lag", "wind_speed_sq", "wind_speed_cub"]

ALIASES = {
    "hum": "hum%",
}

VALID_RANGES = {
    "wind_speed": (0, 60),
    "temp": (-60, 55),
    "prs": (800, 1100),
    "hum%": (0, 100),
    "pow_out": (-2, None),
}

CUT_IN_WIND_SPEED = 0.5
CUT_IN_POWER_SLACK = 2.0
DEFAULT_MAX_POWER = 1e6
CAPACITY_TOLERANCE = 1.05

DEFAULT_STUCK_RUN_THRESHOLD = 200

HORIZON = 1
BATCH_SIZE = 256
EPOCHS = 60
LR = 1e-3
WEIGHT_DECAY = 1e-4
VAL_SPLIT = 0.15
SEED = 42


# --------------------------------------------------------------------------
# Column matching helpers
# --------------------------------------------------------------------------
def _norm(s: str) -> str:
    return " ".join(str(s).split()).lower()


_CANONICAL_LOOKUP = {_norm(c): c for c in BASE_FEATURE_COLS + [TARGET_COL]}
for alias, canon in ALIASES.items():
    _CANONICAL_LOOKUP[_norm(alias)] = canon


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for col in df.columns:
        key = _norm(col)
        rename_map[col] = _CANONICAL_LOOKUP.get(key, " ".join(str(col).split()))
    return df.rename(columns=rename_map)


# --------------------------------------------------------------------------
# Cleaning helpers (unchanged from v1)
# --------------------------------------------------------------------------
def parse_capacity(source_name: str):
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
    if threshold <= 0 or len(df) == 0:
        return df

    feat = df[BASE_FEATURE_COLS]
    same_as_prev = (feat == feat.shift(1)).all(axis=1)
    run_id = (~same_as_prev).cumsum()
    run_sizes = run_id.map(run_id.value_counts())
    stuck_mask = same_as_prev & (run_sizes >= threshold)
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
    missing = [c for c in BASE_FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        print(f"[skip] {source_name} -- missing columns {missing}")
        return None
    df = df[BASE_FEATURE_COLS + [TARGET_COL]].copy()
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


# --------------------------------------------------------------------------
# Feature engineering (NEW)
# --------------------------------------------------------------------------
def build_augmented_features(df: pd.DataFrame) -> np.ndarray:
    """Given a cleaned df with BASE_FEATURE_COLS + TARGET_COL (one row per
    timestep, in time order), return an (n, len(AUGMENTED_FEATURE_NAMES))
    array: raw weather + current pow_out (lag-0, autoregressive input) +
    wind_speed^2 / wind_speed^3 physics terms."""
    base = df[BASE_FEATURE_COLS].values.astype(np.float32)
    pow_lag = df[TARGET_COL].values.astype(np.float32).reshape(-1, 1)
    ws = base[:, 0:1]
    ws_sq = ws ** 2
    ws_cub = ws ** 3
    return np.concatenate([base, pow_lag, ws_sq, ws_cub], axis=1)


def time_ordered_split(frames, horizon: int, val_split: float):
    """Take the last val_split fraction of EACH series (in time order) as
    validation. Features at time t (including pow_out_t) predict the
    target pow_out_{t+horizon}."""
    Xtr_list, ytr_list, Xva_list, yva_list = [], [], [], []
    for df in frames:
        if len(df) <= horizon:
            continue
        X_full = build_augmented_features(df)
        X = X_full[:-horizon]
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


class StandardScalerTorch:
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
def train(model, train_loader, val_loader, device, epochs, lr, weight_decay):
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8
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
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--stuck_run_threshold", type=int, default=DEFAULT_STUCK_RUN_THRESHOLD)
    parser.add_argument("--no_drop_stuck", action="store_true")
    parser.add_argument("--out", type=str, default="wind_power_model.pt")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    frames = load_all_series(args.data, args.stuck_run_threshold, not args.no_drop_stuck)

    Xtr, ytr, Xva, yva = time_ordered_split(frames, args.horizon, VAL_SPLIT)
    print(f"Train pairs: {len(Xtr)} | Val pairs: {len(Xva)} | Input dims: {Xtr.shape[1]}")

    x_scaler = StandardScalerTorch().fit(Xtr)
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

    model = PowerMLP(in_dim=len(AUGMENTED_FEATURE_NAMES)).to(device)
    model, best_val = train(model, train_loader, val_loader, device, args.epochs, args.lr, args.weight_decay)
    print(f"Best val loss (scaled space): {best_val:.5f}")

    torch.save({
        "model_state_dict": model.state_dict(),
        "x_mean": x_scaler.mean_, "x_std": x_scaler.std_,
        "y_mean": y_scaler.mean_, "y_std": y_scaler.std_,
        "feature_cols": AUGMENTED_FEATURE_NAMES,   # what the model actually consumes
        "base_feature_cols": BASE_FEATURE_COLS,    # raw weather columns needed from data
        "target_col": TARGET_COL,
        "horizon": args.horizon,
    }, args.out)
    print(f"Saved trained model to {args.out}")


if __name__ == "__main__":
    main()