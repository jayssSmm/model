import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------
# Must match train_wind_power_model.py
# --------------------------------------------------------------------------

BASE_FEATURE_COLS = ["wind_speed", "temp", "prs", "hum%"]
TARGET_COL = "pow_out"

AUGMENTED_FEATURE_NAMES = BASE_FEATURE_COLS + [
    "pow_out_lag",
    "wind_speed_sq",
    "wind_speed_cub",
]

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


# --------------------------------------------------------------------------
# Column helpers
# --------------------------------------------------------------------------

def _norm(s: str) -> str:
    return " ".join(str(s).split()).lower()


_CANONICAL_LOOKUP = {
    _norm(c): c
    for c in BASE_FEATURE_COLS + [TARGET_COL]
}

for alias, canon in ALIASES.items():
    _CANONICAL_LOOKUP[_norm(alias)] = canon


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}

    for col in df.columns:
        key = _norm(col)
        rename_map[col] = _CANONICAL_LOOKUP.get(
            key,
            " ".join(str(col).split())
        )

    return df.rename(columns=rename_map)


# --------------------------------------------------------------------------
# Same cleaning as training
# --------------------------------------------------------------------------

def parse_capacity(source_name: str):
    m = re.search(
        r"mx(\d+(?:\.\d+)?)",
        source_name,
        flags=re.IGNORECASE
    )

    if m:
        return float(m.group(1))

    return None


def drop_out_of_range_rows(
    df: pd.DataFrame,
    source_name: str,
    capacity
) -> pd.DataFrame:

    before = len(df)

    mask = pd.Series(True, index=df.index)

    for col, (lo, hi) in VALID_RANGES.items():

        if col not in df.columns:
            continue

        if lo is not None:
            mask &= df[col] >= lo

        if hi is not None:
            mask &= df[col] <= hi

    max_power = (
        capacity * CAPACITY_TOLERANCE
        if capacity is not None
        else DEFAULT_MAX_POWER
    )

    mask &= df[TARGET_COL] <= max_power

    df = df[mask].reset_index(drop=True)

    removed = before - len(df)

    if removed:
        pct = 100.0 * removed / before

        cap_note = (
            f"capacity={capacity}"
            if capacity is not None
            else "capacity=unknown"
        )

        print(
            f"[clean] {source_name} -- dropped "
            f"{removed}/{before} rows ({pct:.1f}%) "
            f"outside sanity ranges [{cap_note}]"
        )

    return df


def drop_cut_in_violations(
    df: pd.DataFrame,
    source_name: str
) -> pd.DataFrame:

    before = len(df)

    violation = (
        (df["wind_speed"] < CUT_IN_WIND_SPEED)
        & (df[TARGET_COL] > CUT_IN_POWER_SLACK)
    )

    df = df[~violation].reset_index(drop=True)

    removed = before - len(df)

    if removed:
        pct = 100.0 * removed / before

        print(
            f"[clean] {source_name} -- dropped "
            f"{removed}/{before} rows ({pct:.1f}%) "
            f"violating cut-in-speed physics"
        )

    return df


def drop_stuck_sensor_runs(
    df: pd.DataFrame,
    source_name: str,
    threshold: int
) -> pd.DataFrame:

    if threshold <= 0 or len(df) == 0:
        return df

    feat = df[BASE_FEATURE_COLS]

    same_as_prev = (
        feat == feat.shift(1)
    ).all(axis=1)

    run_id = (~same_as_prev).cumsum()

    run_sizes = run_id.map(
        run_id.value_counts()
    )

    stuck_mask = (
        same_as_prev
        & (run_sizes >= threshold)
    )

    stuck_mask |= (
        (~same_as_prev)
        & (run_sizes >= threshold)
    )

    before = len(df)

    df = df[~stuck_mask].reset_index(drop=True)

    removed = before - len(df)

    if removed:
        pct = 100.0 * removed / before

        print(
            f"[clean] {source_name} -- dropped "
            f"{removed}/{before} rows ({pct:.1f}%) "
            f"belonging to stuck-sensor runs"
        )

    return df


def finalize(
    df: pd.DataFrame,
    source_name: str,
    stuck_run_threshold: int,
    drop_stuck: bool
):

    df = clean_columns(df)

    missing = [
        c
        for c in BASE_FEATURE_COLS + [TARGET_COL]
        if c not in df.columns
    ]

    if missing:
        print(
            f"[skip] {source_name} -- missing columns {missing}"
        )
        return None

    df = df[
        BASE_FEATURE_COLS + [TARGET_COL]
    ].copy()

    for c in df.columns:
        df[c] = pd.to_numeric(
            df[c],
            errors="coerce"
        )

    df = df.dropna().reset_index(drop=True)

    if len(df) == 0:
        print(
            f"[skip] {source_name} -- "
            f"0 usable rows after basic cleaning"
        )
        return None

    capacity = parse_capacity(source_name)

    df = drop_out_of_range_rows(
        df,
        source_name,
        capacity
    )

    df = drop_cut_in_violations(
        df,
        source_name
    )

    if drop_stuck:
        df = drop_stuck_sensor_runs(
            df,
            source_name,
            stuck_run_threshold
        )

    if len(df) == 0:
        print(
            f"[skip] {source_name} -- "
            f"0 usable rows after full cleaning"
        )
        return None

    print(
        f"[ok]   {source_name} -> "
        f"{len(df)} clean rows"
    )

    return df


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def load_series(
    data_path,
    stuck_run_threshold=DEFAULT_STUCK_RUN_THRESHOLD,
    drop_stuck=True
):

    series = []

    if (
        os.path.isfile(data_path)
        and data_path.lower().endswith(".csv")
    ):

        df = pd.read_csv(data_path)

        cleaned = finalize(
            df,
            data_path,
            stuck_run_threshold,
            drop_stuck
        )

        if cleaned is not None:
            series.append(
                (os.path.basename(data_path), cleaned)
            )

    elif os.path.isdir(data_path):

        paths = sorted(
            glob.glob(
                os.path.join(data_path, "*.xlsx")
            )
        )

        if not paths:
            raise FileNotFoundError(
                f"No .xlsx files found in {data_path}"
            )

        for p in paths:

            xl = pd.ExcelFile(p)

            for sheet in xl.sheet_names:

                raw = xl.parse(sheet)

                cleaned = finalize(
                    raw,
                    f"{os.path.basename(p)}::{sheet}",
                    stuck_run_threshold,
                    drop_stuck
                )

                if cleaned is not None:
                    series.append(
                        (
                            f"{os.path.basename(p)}::{sheet}",
                            cleaned
                        )
                    )

    else:

        raise FileNotFoundError(
            "--data must be a .csv file or "
            "a directory of .xlsx files"
        )

    if not series:
        raise RuntimeError(
            "No usable series found in --data"
        )

    return series


# --------------------------------------------------------------------------
# EXACT feature engineering from training script
# --------------------------------------------------------------------------

def build_augmented_features(df):

    base = df[
        BASE_FEATURE_COLS
    ].values.astype(np.float32)

    # pow_out at time t
    pow_lag = (
        df[TARGET_COL]
        .values
        .astype(np.float32)
        .reshape(-1, 1)
    )

    # wind speed
    ws = base[:, 0:1]

    # v^2
    ws_sq = ws ** 2

    # v^3
    ws_cub = ws ** 3

    return np.concatenate(
        [
            base,
            pow_lag,
            ws_sq,
            ws_cub
        ],
        axis=1
    )


# --------------------------------------------------------------------------
# Build exactly the same t -> t+horizon pairs as training
# --------------------------------------------------------------------------

def build_pairs(series, horizon):

    Xs = []
    ys = []

    for name, df in series:

        if len(df) <= horizon:
            print(
                f"[skip] {name} -- "
                f"too short for horizon={horizon}"
            )
            continue

        X_full = build_augmented_features(df)

        # Features at t
        X = X_full[:-horizon]

        # Target at t + horizon
        y = df[TARGET_COL].values[horizon:]

        Xs.append(X)
        ys.append(y)

    if not Xs:
        raise RuntimeError(
            "No series long enough for requested horizon."
        )

    X = np.concatenate(
        Xs,
        axis=0
    ).astype(np.float32)

    y = np.concatenate(
        ys,
        axis=0
    ).astype(np.float32).reshape(-1, 1)

    return X, y


# --------------------------------------------------------------------------
# EXACT model architecture from training
# --------------------------------------------------------------------------

class PowerMLP(nn.Module):

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
# Main
# --------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="wind_power_model.pt"
    )

    parser.add_argument(
        "--data",
        type=str,
        required=True
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=None
    )

    parser.add_argument(
        "--max_points",
        type=int,
        default=500
    )

    parser.add_argument(
        "--stuck_run_threshold",
        type=int,
        default=DEFAULT_STUCK_RUN_THRESHOLD
    )

    parser.add_argument(
        "--no_drop_stuck",
        action="store_true"
    )

    parser.add_argument(
        "--out",
        type=str,
        default="wind_power_predictions.png"
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------------------

    ckpt = torch.load(
        args.model,
        weights_only=False,
        map_location=device
    )

    horizon = (
        args.horizon
        if args.horizon is not None
        else ckpt.get("horizon", 1)
    )

    # Use the exact scaler values saved during training
    x_mean = ckpt["x_mean"]
    x_std = ckpt["x_std"]

    y_mean = ckpt["y_mean"]
    y_std = ckpt["y_std"]

    # ------------------------------------------------------------------
    # Verify feature information stored in checkpoint
    # ------------------------------------------------------------------

    checkpoint_features = ckpt.get(
        "feature_cols",
        AUGMENTED_FEATURE_NAMES
    )

    print(
        "Model features:",
        checkpoint_features
    )

    print(
        "Input dimensions:",
        len(checkpoint_features)
    )

    if checkpoint_features != AUGMENTED_FEATURE_NAMES:
        raise ValueError(
            "Checkpoint feature order does not match "
            "the expected training feature order.\n"
            f"Checkpoint: {checkpoint_features}\n"
            f"Expected:   {AUGMENTED_FEATURE_NAMES}"
        )

    # ------------------------------------------------------------------
    # Build exact model
    # ------------------------------------------------------------------

    model = PowerMLP(
        in_dim=len(checkpoint_features)
    ).to(device)

    model.load_state_dict(
        ckpt["model_state_dict"]
    )

    model.eval()

    print("Model loaded successfully.")

    # ------------------------------------------------------------------
    # Load and clean data
    # ------------------------------------------------------------------

    series = load_series(
        args.data,
        args.stuck_run_threshold,
        not args.no_drop_stuck
    )

    # ------------------------------------------------------------------
    # Build the same 7-feature inputs used during training
    # ------------------------------------------------------------------

    X, y_true = build_pairs(
        series,
        horizon
    )

    print(
        f"Prediction pairs: {len(X)}"
    )

    # ------------------------------------------------------------------
    # Apply training scaler
    # ------------------------------------------------------------------

    X_s = (
        X - x_mean
    ) / x_std

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    with torch.no_grad():

        pred_s = model(
            torch.from_numpy(
                X_s.astype(np.float32)
            ).to(device)
        ).cpu().numpy()

    # Convert prediction back to MW / original pow_out units
    y_pred = (
        pred_s * y_std
    ) + y_mean

    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    mae = np.mean(
        np.abs(y_true - y_pred)
    )

    rmse = np.sqrt(
        np.mean(
            (y_true - y_pred) ** 2
        )
    )

    ss_res = np.sum(
        (y_true - y_pred) ** 2
    )

    ss_tot = np.sum(
        (y_true - np.mean(y_true)) ** 2
    )

    r2 = (
        1 - ss_res / ss_tot
        if ss_tot > 0
        else float("nan")
    )

    print()
    print("========== RESULTS ==========")
    print(f"Points : {len(y_true)}")
    print(f"MAE    : {mae:.3f}")
    print(f"RMSE   : {rmse:.3f}")
    print(f"R^2    : {r2:.4f}")
    print("==============================")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    n_show = min(
        args.max_points,
        len(y_true)
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5)
    )

    # --------------------------------------------------------------
    # Left: time ordered
    # --------------------------------------------------------------

    ax = axes[0]

    ax.plot(
        y_true[-n_show:],
        label="Actual",
        linewidth=1.5
    )

    ax.plot(
        y_pred[-n_show:],
        label="Predicted",
        linewidth=1.2,
        alpha=0.8
    )

    ax.set_title(
        f"Actual vs Predicted pow_out "
        f"(last {n_show} points, horizon={horizon})"
    )

    ax.set_xlabel("Time step")
    ax.set_ylabel("pow_out")

    ax.legend()

    # --------------------------------------------------------------
    # Right: scatter
    # --------------------------------------------------------------

    ax2 = axes[1]

    ax2.scatter(
        y_true,
        y_pred,
        s=8,
        alpha=0.4
    )

    lims = [
        min(
            y_true.min(),
            y_pred.min()
        ),
        max(
            y_true.max(),
            y_pred.max()
        )
    ]

    ax2.plot(
        lims,
        lims,
        "r--",
        linewidth=1,
        label="Perfect prediction"
    )

    ax2.set_title(
        f"Actual vs Predicted "
        f"(all {len(y_true)} points)\n"
        f"R²={r2:.3f}"
    )

    ax2.set_xlabel(
        "Actual pow_out"
    )

    ax2.set_ylabel(
        "Predicted pow_out"
    )

    ax2.legend()

    # ------------------------------------------------------------------

    fig.tight_layout()

    fig.savefig(
        args.out,
        dpi=150
    )

    print(
        f"Saved plot to {args.out}"
    )

    plt.show()


if __name__ == "__main__":
    main()