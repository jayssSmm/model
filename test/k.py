import torch
from pathlib import Path
from train_solar import build_model_from_checkpoint

device = "cuda"

model = Path("/home/samman_2007/sih/models/solar_power_model.pt")


def main():
    global model
    ckpt = torch.load(model, weights_only=False, map_location=device)
    print(type(ckpt))
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

main()