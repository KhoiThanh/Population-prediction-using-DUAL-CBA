from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from tensorflow import keras

from model import BahdanauAttention, rolling_forecast


def parse_year_columns(df: pd.DataFrame) -> List[Tuple[int, str]]:
    pairs: List[Tuple[int, str]] = []
    for c in df.columns:
        m = re.fullmatch(r"(\d{4})_VN", str(c))
        if m:
            pairs.append((int(m.group(1)), c))
    pairs.sort(key=lambda x: x[0])
    return pairs


def main():
    ap = argparse.ArgumentParser(
        description="Forecast Vietnam provinces using model trained on Australia data."
    )
    ap.add_argument(
        "--input_csv",
        type=str,
        default=r"d:\cothexoa\Vietnam_Final_63_Provinces.csv",
        help="Path to Vietnam province CSV.",
    )
    ap.add_argument(
        "--model_path",
        type=str,
        default="artifacts/dual_cba_best.keras",
        help="Path to trained Keras model.",
    )
    ap.add_argument(
        "--scaler_path",
        type=str,
        default="artifacts/global_minmax_scaler.joblib",
        help="Path to global scaler fitted on Australia data.",
    )
    ap.add_argument(
        "--window_size",
        type=int,
        default=11,
        help="Input window size (must match trained model).",
    )
    ap.add_argument(
        "--horizon",
        type=int,
        default=10,
        help="Forecast horizon in years.",
    )
    ap.add_argument(
        "--output_csv",
        type=str,
        default="artifacts/vietnam_transfer_forecast_10y.csv",
        help="Output CSV path.",
    )
    args = ap.parse_args()

    input_path = Path(args.input_csv)
    model_path = Path(args.model_path)
    scaler_path = Path(args.scaler_path)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")

    df = pd.read_csv(input_path)
    yc = parse_year_columns(df)
    if len(yc) < args.window_size:
        raise ValueError(
            f"Not enough yearly columns in input. Need at least {args.window_size}, got {len(yc)}."
        )

    years = [y for y, _ in yc]
    cols = [c for _, c in yc]
    last_year = years[-1]
    pred_years = [last_year + i for i in range(1, args.horizon + 1)]

    scaler = joblib.load(scaler_path)
    model = keras.models.load_model(
        model_path,
        compile=False,
        custom_objects={"BahdanauAttention": BahdanauAttention},
    )
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mae")

    out_rows = []
    values_mat = df[cols].to_numpy(dtype=np.float32)
    area_ids = df["Area_ID"].to_numpy()
    area_names = df["Area_Name"].astype(str).to_numpy()

    for i in range(len(df)):
        area_id = area_ids[i]
        area_name = area_names[i]
        values = values_mat[i]

        if not np.isfinite(values).all():
            continue

        # Transfer scaling: use Australia global scaler to preserve the trained space.
        scaled = scaler.transform(values.reshape(-1, 1)).reshape(-1).astype(np.float32)
        seed = scaled[-args.window_size :]
        yhat_scaled = rolling_forecast(model, seed, horizon=args.horizon).numpy()
        yhat_inv = scaler.inverse_transform(yhat_scaled.reshape(-1, 1)).reshape(-1)

        record = {
            "Area_ID": area_id,
            "Area_Name": area_name,
            "source_last_year": last_year,
        }
        for i, py in enumerate(pred_years):
            record[f"pred_{py}"] = float(yhat_inv[i])
        out_rows.append(record)

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("Done.")
    print(f"Input: {input_path}")
    print(f"Rows forecasted: {len(out_df)}")
    print(f"Output: {output_path.resolve()}")
    print(
        "Note: This is transfer inference (Australia-trained model -> Vietnam data). "
        "Use results for reference; recalibration/fine-tuning is recommended."
    )


if __name__ == "__main__":
    main()

