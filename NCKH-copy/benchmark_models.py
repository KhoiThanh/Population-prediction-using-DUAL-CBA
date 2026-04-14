from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

try:
    from statsmodels.tsa.arima.model import ARIMA
except Exception:  # pragma: no cover - handled at runtime
    ARIMA = None

from data_processing import (
    TEST_START,
    TRAIN_END,
    YEAR_END,
    YEAR_START,
    build_splits,
    fit_global_minmax,
    inverse_transform_series,
    load_erp_csv,
    year_columns,
)
from model import rolling_forecast


ART_DIR = Path("artifacts")


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = np.maximum(np.abs(y_true), eps)
    return np.abs(y_pred - y_true) / denom * 100.0


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), eps)
    return 200.0 * np.abs(y_pred - y_true) / denom


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    e = np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean(e * e)))


def build_rnn_baseline(window_size: int, kind: str = "lstm", units: int = 96) -> keras.Model:
    inp = keras.Input(shape=(window_size, 1), name="window")
    if kind == "lstm":
        x = layers.LSTM(units, name="lstm")(inp)
    elif kind == "gru":
        x = layers.GRU(units, name="gru")(inp)
    else:
        raise ValueError(f"Unsupported kind: {kind}")

    x = layers.Dense(64, activation="relu", name="dense_1")(x)
    out = layers.Dense(1, activation="linear", name="y")(x)
    model = keras.Model(inputs=inp, outputs=out, name=f"{kind.upper()}_baseline")
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mae")
    return model


def train_rnn_model(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    batch_size: int,
) -> None:
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-6),
    ]
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=callbacks,
    )


def eval_predictions(
    model_name: str,
    scaler,
    area_ids: np.ndarray,
    gt_scaled: np.ndarray,
    preds_scaled: np.ndarray,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    rows: List[Dict[str, float]] = []

    for i, area_id in enumerate(area_ids):
        gt_inv = inverse_transform_series(scaler, gt_scaled[i])
        pr_inv = inverse_transform_series(scaler, preds_scaled[i])

        m = mape(gt_inv, pr_inv)
        s = smape(gt_inv, pr_inv)
        a = mae(gt_inv, pr_inv)
        r = rmse(gt_inv, pr_inv)

        rows.append(
            {
                "model": model_name,
                "SA2_MAINCODE_2011": int(area_id),
                "MAPE_mean": float(np.mean(m)),
                "MedAPE": float(np.median(m)),
                "MAPE_max": float(np.max(m)),
                "MAE_mean": float(np.mean(a)),
                "RMSE": float(r),
                "sMAPE_mean": float(np.mean(s)),
            }
        )

    detail_df = pd.DataFrame(rows)
    filtered = detail_df[detail_df["MAPE_mean"] <= 100.0]

    summary = {
        "model": model_name,
        "n_areas": int(len(detail_df)),
        "n_areas_filtered": int(len(filtered)),
        "mape_mean_filtered": float(filtered["MAPE_mean"].mean()) if len(filtered) else float("nan"),
        "medape_mean_filtered": float(filtered["MedAPE"].mean()) if len(filtered) else float("nan"),
        "mae_mean_filtered": float(filtered["MAE_mean"].mean()) if len(filtered) else float("nan"),
        "rmse_mean_filtered": float(filtered["RMSE"].mean()) if len(filtered) else float("nan"),
        "smape_mean_filtered": float(filtered["sMAPE_mean"].mean()) if len(filtered) else float("nan"),
    }
    return detail_df, summary


def build_area_series_map(df: pd.DataFrame) -> Dict[int, np.ndarray]:
    cols = year_columns(YEAR_START, YEAR_END)
    area_ids = df["SA2_MAINCODE_2011"].to_numpy(dtype=int)
    mat = df[cols].to_numpy(dtype=np.float32)
    mapping: Dict[int, np.ndarray] = {}
    for i, area_id in enumerate(area_ids):
        mapping[int(area_id)] = mat[i]
    return mapping


def run_arima_baseline(
    scaler,
    area_ids: np.ndarray,
    gt_scaled: np.ndarray,
    area_series_map: Dict[int, np.ndarray],
    arima_order: Tuple[int, int, int],
    horizon: int,
) -> np.ndarray:
    if ARIMA is None:
        raise ImportError(
            "statsmodels is not available. Install with: pip install statsmodels>=0.14"
        )

    years = np.arange(YEAR_START, YEAR_END + 1, dtype=int)
    idx_train_end = int(np.where(years == TRAIN_END)[0][0])

    preds_scaled = []
    for area_id in area_ids:
        series = area_series_map[int(area_id)]
        train_raw = series[: idx_train_end + 1].astype(np.float64)

        try:
            model = ARIMA(train_raw, order=arima_order)
            fit = model.fit()
            pred_raw = np.asarray(fit.forecast(steps=horizon), dtype=np.float64)
            if pred_raw.shape[0] != horizon:
                pred_raw = np.full(shape=(horizon,), fill_value=float(train_raw[-1]), dtype=np.float64)
        except Exception:
            pred_raw = np.full(shape=(horizon,), fill_value=float(train_raw[-1]), dtype=np.float64)

        pred_scaled = scaler.transform(pred_raw.reshape(-1, 1)).reshape(-1).astype(np.float32)
        preds_scaled.append(pred_scaled)

    preds_scaled = np.array(preds_scaled, dtype=np.float32)
    if preds_scaled.shape != gt_scaled.shape:
        raise RuntimeError(
            f"ARIMA prediction shape mismatch. Got {preds_scaled.shape}, expected {gt_scaled.shape}."
        )
    return preds_scaled


def run_rnn_rolling(model: keras.Model, seeds_scaled: np.ndarray, horizon: int) -> np.ndarray:
    preds_scaled = []
    for seed in seeds_scaled:
        yhat = rolling_forecast(model, seed, horizon=horizon).numpy()
        preds_scaled.append(yhat)
    return np.array(preds_scaled, dtype=np.float32)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Benchmark ARIMA, LSTM, GRU and export comparison CSV.")
    ap.add_argument("--input_csv", type=str, default="Australia_SmallAreas.csv")
    ap.add_argument("--window_size", type=int, default=11)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--val_last_k", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lstm_units", type=int, default=96)
    ap.add_argument("--gru_units", type=int, default=96)
    ap.add_argument("--arima_order", type=str, default="1,1,0")
    ap.add_argument("--max_areas", type=int, default=0, help="For quick test only. 0 means all areas.")
    ap.add_argument(
        "--summary_csv",
        type=str,
        default=str(ART_DIR / "benchmark_model_comparison.csv"),
    )
    ap.add_argument(
        "--detail_csv",
        type=str,
        default=str(ART_DIR / "benchmark_per_area_metrics.csv"),
    )
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)
    np.random.seed(args.seed)

    summary_path = Path(args.summary_csv)
    detail_path = Path(args.detail_csv)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    detail_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        arima_order = tuple(int(x.strip()) for x in args.arima_order.split(","))
        if len(arima_order) != 3:
            raise ValueError
    except Exception as exc:
        raise ValueError("--arima_order must be in format p,d,q (e.g., 1,1,0)") from exc

    print("Loading data and building shared splits...")
    df = load_erp_csv(args.input_csv)
    scaler = fit_global_minmax(df)
    splits = build_splits(
        df,
        scaler,
        window_size=args.window_size,
        forecast_horizon=args.horizon,
        val_last_k_windows_per_area=args.val_last_k,
    )

    area_ids = splits.area_ids
    seeds = splits.window_ending_2006_scaled
    gt_scaled = splits.gt_2007_2016_scaled

    if args.max_areas and args.max_areas > 0:
        k = int(args.max_areas)
        area_ids = area_ids[:k]
        seeds = seeds[:k]
        gt_scaled = gt_scaled[:k]

    area_series_map = build_area_series_map(df)

    all_detail: List[pd.DataFrame] = []
    all_summary: List[Dict[str, float]] = []

    print("Running ARIMA baseline...")
    t0 = time.time()
    arima_preds = run_arima_baseline(
        scaler=scaler,
        area_ids=area_ids,
        gt_scaled=gt_scaled,
        area_series_map=area_series_map,
        arima_order=arima_order,
        horizon=args.horizon,
    )
    arima_detail, arima_summary = eval_predictions("ARIMA", scaler, area_ids, gt_scaled, arima_preds)
    arima_summary["runtime_sec"] = float(time.time() - t0)
    all_detail.append(arima_detail)
    all_summary.append(arima_summary)

    print("Training LSTM baseline...")
    t0 = time.time()
    lstm_model = build_rnn_baseline(window_size=args.window_size, kind="lstm", units=args.lstm_units)
    train_rnn_model(
        lstm_model,
        splits.X_train,
        splits.y_train,
        splits.X_val,
        splits.y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    lstm_preds = run_rnn_rolling(lstm_model, seeds, horizon=args.horizon)
    lstm_detail, lstm_summary = eval_predictions("LSTM", scaler, area_ids, gt_scaled, lstm_preds)
    lstm_summary["runtime_sec"] = float(time.time() - t0)
    all_detail.append(lstm_detail)
    all_summary.append(lstm_summary)

    print("Training GRU baseline...")
    t0 = time.time()
    gru_model = build_rnn_baseline(window_size=args.window_size, kind="gru", units=args.gru_units)
    train_rnn_model(
        gru_model,
        splits.X_train,
        splits.y_train,
        splits.X_val,
        splits.y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    gru_preds = run_rnn_rolling(gru_model, seeds, horizon=args.horizon)
    gru_detail, gru_summary = eval_predictions("GRU", scaler, area_ids, gt_scaled, gru_preds)
    gru_summary["runtime_sec"] = float(time.time() - t0)
    all_detail.append(gru_detail)
    all_summary.append(gru_summary)

    dual_eval_path = ART_DIR / "eval_report.json"
    if dual_eval_path.exists():
        try:
            dual = json.loads(dual_eval_path.read_text(encoding="utf-8"))
            all_summary.append(
                {
                    "model": "Dual-CBA",
                    "n_areas": int(dual.get("n_areas", 0)),
                    "n_areas_filtered": int(dual.get("n_areas_filtered", 0)),
                    "mape_mean_filtered": float(dual.get("mape_mean_filtered", np.nan)),
                    "medape_mean_filtered": float(dual.get("medape_mean_filtered", np.nan)),
                    "mae_mean_filtered": float("nan"),
                    "rmse_mean_filtered": float("nan"),
                    "smape_mean_filtered": float("nan"),
                    "runtime_sec": float("nan"),
                }
            )
        except Exception:
            pass

    detail_df = pd.concat(all_detail, axis=0, ignore_index=True)
    summary_df = pd.DataFrame(all_summary)

    summary_df = summary_df[
        [
            "model",
            "mape_mean_filtered",
            "medape_mean_filtered",
            "mae_mean_filtered",
            "rmse_mean_filtered",
            "smape_mean_filtered",
            "n_areas",
            "n_areas_filtered",
            "runtime_sec",
        ]
    ]

    detail_df.to_csv(detail_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("Done benchmark.")
    print(f"Summary CSV: {summary_path.resolve()}")
    print(f"Detail CSV: {detail_path.resolve()}")
    print(summary_df.sort_values("mape_mean_filtered"))


if __name__ == "__main__":
    main()
