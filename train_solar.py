import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
FEATURE_COLS = [
    "Total solar irradiance (W/m2)",
    "Direct normal irradiance (W/m2)",
    "Global horizontal irradiance (W/m2)",
    "Air temperature (°C)",
    "Atmosphere (hpa)",
    "Relative humidity (%)",
]
TARGET_COL = "Power (MW)"

# Physically-plausible ranges used to drop corrupted/sensor-error rows.
# Rows with any column outside its range are dropped entirely (not clipped),
# since clipping would inject fake boundary values into a "polluted" dataset.
VALID_RANGES = {
    "Total solar irradiance (W/m2)": (0, 1500),
    "Direct normal irradiance (W/m2)": (0, 1500),
    "Global horizontal irradiance (W/m2)": (0, 1500),
    "Air temperature (°C)": (-40, 60),
    "Atmosphere (hpa)": (800, 1100),
    "Relative humidity (%)": (0, 100),
    # Small negative Power is common sensor noise near zero-output periods;
    # allow a little slack but reject anything large/negative (impossible).
    "Power (MW)": (-1, 1000),
}

# How many previous Power values to include as extra input features
# (autoregressive / momentum signal). Set to [] to disable.
LAG_STEPS = [1, 2, 3]

HORIZON = 1          # how many rows ahead to predict (future step)
BATCH_SIZE = 256
EPOCHS = 50
LR = 1e-3
VAL_SPLIT = 0.15      # last 15% of each series (in time order) used for val
SEED = 42


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def _norm(s: str) -> str:
    """Lowercased, single-spaced key used only for matching columns."""
    s = " ".join(str(s).split())
    s = s.replace("horicontal", "horizontal")  # fix known typo variant
    return s.lower()


# Build a lookup from normalized name -> canonical name we want to use
_CANONICAL_LOOKUP = {_norm(c): c for c in FEATURE_COLS + [TARGET_COL]}


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to their canonical form (matching FEATURE_COLS /
    TARGET_COL) regardless of extra whitespace, case, or the
    horicontal/horizontal typo. Columns with no canonical match are
    left untouched."""
    rename_map = {}
    for col in df.columns:
        key = _norm(col)
        rename_map[col] = _CANONICAL_LOOKUP.get(key, " ".join(str(col).split()))
    return df.rename(columns=rename_map)


def drop_corrupted_rows(df: pd.DataFrame, source_label: str) -> pd.DataFrame:
    """Drop rows where any feature/target is outside a physically
    plausible range (e.g. -3270 degC air temperature). Prints how many
    rows were removed per file/sheet so pollution can be audited."""
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


def load_all_files(data_dir: str) -> list:
    paths = sorted(glob.glob(os.path.join(data_dir, "*.xlsx")))
    if not paths:
        raise FileNotFoundError(f"No .xlsx files found in {data_dir}")

    frames = []
    for p in paths:
        xl = pd.ExcelFile(p)
        for sheet in xl.sheet_names:
            df = xl.parse(sheet)
            df = clean_columns(df)

            missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
            if missing:
                print(f"[skip] {p} :: {sheet} -- missing columns {missing}")
                continue

            df = df[FEATURE_COLS + [TARGET_COL]].copy()
            for c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna().reset_index(drop=True)

            source_label = f"{os.path.basename(p)}::{sheet}"
            df = drop_corrupted_rows(df, source_label)
            if df.empty:
                print(f"[skip] {source_label} -- nothing left after cleaning")
                continue

            df["__source__"] = source_label
            frames.append(df)
            print(f"[ok]   {source_label} -> {len(df)} clean rows")

    if not frames:
        raise RuntimeError("No usable sheets found across all files.")
    return frames  # list of per-series DataFrames (kept separate on purpose)


def add_lag_features(df: pd.DataFrame, lag_steps) -> pd.DataFrame:
    """Add lagged Power columns computed within this series only."""
    df = df.copy()
    for lag in lag_steps:
        df[f"power_lag_{lag}"] = df[TARGET_COL].shift(lag)
    return df


def build_supervised_pairs(frames, horizon: int, lag_steps):
    """Shift target forward by `horizon` within each series separately
    so we never leak across file/sheet boundaries. Also builds lag
    features of Power (also computed within-series only)."""
    lag_cols = [f"power_lag_{lag}" for lag in lag_steps]
    all_feature_cols = FEATURE_COLS + lag_cols

    X_list, y_list = [], []
    for df in frames:
        df = add_lag_features(df, lag_steps)
        df = df.dropna().reset_index(drop=True)  # drop rows with no lag history yet

        if len(df) <= horizon:
            continue
        X = df[all_feature_cols].values[: -horizon]
        y = df[TARGET_COL].values[horizon:]
        X_list.append(X)
        y_list.append(y)

    if not X_list:
        raise RuntimeError("No supervised pairs could be built -- check horizon/lag settings vs series lengths.")

    X = np.concatenate(X_list, axis=0).astype(np.float32)
    y = np.concatenate(y_list, axis=0).astype(np.float32).reshape(-1, 1)
    return X, y, all_feature_cols


def time_ordered_split(frames, horizon, lag_steps, val_split):
    """Build train/val sets by taking the LAST val_split fraction of each
    series in time order, instead of randomly shuffling rows. This avoids
    leaking near-duplicate, highly-correlated timesteps between train and
    val once lag features are involved."""
    lag_cols = [f"power_lag_{lag}" for lag in lag_steps]
    all_feature_cols = FEATURE_COLS + lag_cols

    Xtr_list, ytr_list, Xva_list, yva_list = [], [], [], []
    for df in frames:
        df = add_lag_features(df, lag_steps)
        df = df.dropna().reset_index(drop=True)
        if len(df) <= horizon:
            continue

        X = df[all_feature_cols].values[: -horizon]
        y = df[TARGET_COL].values[horizon:]
        n = len(X)
        cut = int(n * (1 - val_split))
        cut = max(1, min(cut, n - 1)) if n > 1 else n  # keep at least 1 row each side when possible

        Xtr_list.append(X[:cut]); ytr_list.append(y[:cut])
        Xva_list.append(X[cut:]); yva_list.append(y[cut:])

    Xtr = np.concatenate(Xtr_list, axis=0).astype(np.float32)
    ytr = np.concatenate(ytr_list, axis=0).astype(np.float32).reshape(-1, 1)
    Xva = np.concatenate(Xva_list, axis=0).astype(np.float32)
    yva = np.concatenate(yva_list, axis=0).astype(np.float32).reshape(-1, 1)
    return Xtr, ytr, Xva, yva, all_feature_cols


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
    # SmoothL1 (Huber) is less prone to the "safe average" collapse on
    # noisy/outlier-heavy targets than plain MSE, while still being a
    # sound regression loss.
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
    parser.add_argument("--data_dir", type=str, required=True,
                         help="Folder containing the .xlsx solar files")
    parser.add_argument("--horizon", type=int, default=HORIZON,
                         help="Rows ahead to forecast Power (MW)")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--lags", type=int, nargs="*", default=LAG_STEPS,
                         help="Lag steps of Power to use as extra features, e.g. --lags 1 2 3")
    parser.add_argument("--out", type=str, default="solar_power_model.pt")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ---- Load & build data ----
    frames = load_all_files(args.data_dir)

    Xtr, ytr, Xva, yva, all_feature_cols = time_ordered_split(
        frames, args.horizon, args.lags, VAL_SPLIT
    )
    print(f"Train pairs: {len(Xtr)} | Val pairs: {len(Xva)} | "
          f"features: {all_feature_cols}")

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
    model = PowerMLP(in_dim=len(all_feature_cols)).to(device)
    model, best_val = train(model, train_loader, val_loader, device, args.epochs, args.lr)
    print(f"Best val loss (scaled space): {best_val:.5f}")

    # ---- Save model + scalers ----
    torch.save({
        "model_state_dict": model.state_dict(),
        "x_mean": x_scaler.mean_, "x_std": x_scaler.std_,
        "y_mean": y_scaler.mean_, "y_std": y_scaler.std_,
        "feature_cols": all_feature_cols,
        "target_col": TARGET_COL,
        "horizon": args.horizon,
        "lags": args.lags,
    }, args.out)
    print(f"Saved trained model to {args.out}")


if __name__ == "__main__":
    main()