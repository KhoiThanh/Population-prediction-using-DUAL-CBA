from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

import keras_tuner as kt
from tensorflow import keras

from data_processing import (
    YEAR_END,
    YEAR_START,
    build_splits,
    fit_global_minmax,
    inverse_transform_series,
    load_erp_csv,
)
from model import build_dual_cba, rolling_forecast


ART_DIR = Path("artifacts")


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = np.maximum(np.abs(y_true), eps)
    return np.abs(y_pred - y_true) / denom * 100.0


def medape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.median(mape(y_true, y_pred)))


@dataclass(frozen=True)
class EvalReport:
    n_areas: int
    n_areas_filtered: int
    medape_mean_filtered: float
    mape_mean_filtered: float


class DualCBAHyperModel(kt.HyperModel):
    def __init__(self, window_size: int):
        super().__init__()
        self.window_size = int(window_size)

    def build(self, hp: kt.HyperParameters) -> keras.Model:
        lstm_units = hp.Int("lstm_units", min_value=32, max_value=192, step=32)
        attn_units = hp.Int("attn_units", min_value=16, max_value=128, step=16)
        dense_1 = hp.Int("dense_units_1", min_value=128, max_value=512, step=64)
        dense_2 = hp.Int("dense_units_2", min_value=128, max_value=512, step=64)
        lr = hp.Float("lr", min_value=1e-4, max_value=3e-3, sampling="log")

        model = build_dual_cba(
            window_size=self.window_size,
            lstm_units=int(lstm_units),
            attn_units=int(attn_units),
            dense_units_1=int(dense_1),
            dense_units_2=int(dense_2),
        )
        model.compile(optimizer=keras.optimizers.Adam(lr), loss="mae")
        return model


def tune_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    window_size: int,
    max_trials: int = 30,
) -> Tuple[keras.Model, Dict]:
    hp_model = DualCBAHyperModel(window_size=window_size)
    tuner = kt.BayesianOptimization(
        hypermodel=hp_model,
        objective=kt.Objective("val_loss", direction="min"),
        max_trials=max_trials,
        overwrite=True,
        directory=str(ART_DIR / "tuner"),
        project_name="dual_cba_bayesopt",
    )

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
    ]

    tuner.search(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=80,
        batch_size=64,
        verbose=1,
        callbacks=callbacks,
    )

    best_hp = tuner.get_best_hyperparameters(1)[0]
    best_model = tuner.hypermodel.build(best_hp)
    return best_model, best_hp.values


def train_best(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    max_epochs: int = 200,
) -> keras.callbacks.History:
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=15,
            min_lr=1e-6,
        ),
    ]
    return model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=max_epochs,
        batch_size=64,
        verbose=1,
        callbacks=callbacks,
    )


def eval_rolling_all_areas(
    model: keras.Model,
    scaler,
    area_ids: np.ndarray,
    seed_windows_scaled: np.ndarray,
    gt_scaled: np.ndarray,
) -> Tuple[pd.DataFrame, EvalReport]:
    preds_scaled = []
    for seed in seed_windows_scaled:
        yhat = rolling_forecast(model, seed, horizon=gt_scaled.shape[1]).numpy()
        preds_scaled.append(yhat)
    preds_scaled = np.array(preds_scaled, dtype=np.float32)

    rows = []
    for i, area_id in enumerate(area_ids):
        gt_inv = inverse_transform_series(scaler, gt_scaled[i])
        pr_inv = inverse_transform_series(scaler, preds_scaled[i])
        m = mape(gt_inv, pr_inv)
        rows.append(
            {
                "SA2_MAINCODE_2011": int(area_id),
                "MAPE_mean": float(np.mean(m)),
                "MedAPE": float(np.median(m)),
                "MAPE_max": float(np.max(m)),
            }
        )

    dfm = pd.DataFrame(rows)
    mask = dfm["MAPE_mean"] <= 100.0
    df_f = dfm[mask]
    report = EvalReport(
        n_areas=int(len(dfm)),
        n_areas_filtered=int(len(df_f)),
        medape_mean_filtered=float(df_f["MedAPE"].mean()) if len(df_f) else float("nan"),
        mape_mean_filtered=float(df_f["MAPE_mean"].mean()) if len(df_f) else float("nan"),
    )
    return dfm, report


def main():
    tf.keras.utils.set_random_seed(42)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    ART_DIR.mkdir(exist_ok=True, parents=True)

    df = load_erp_csv("Australia_SmallAreas.csv")
    scaler = fit_global_minmax(df)
    splits = build_splits(df, scaler, window_size=11, forecast_horizon=10, val_last_k_windows_per_area=3)

    print(f"Train samples: {len(splits.X_train)} | Val samples: {len(splits.X_val)}")
    print(f"Eval areas: {len(splits.area_ids)}")

    best_model, best_hp = tune_model(
        splits.X_train, splits.y_train, splits.X_val, splits.y_val, window_size=11, max_trials=30
    )

    print("Best hyperparameters:", best_hp)
    train_best(best_model, splits.X_train, splits.y_train, splits.X_val, splits.y_val, max_epochs=200)

    df_metrics, report = eval_rolling_all_areas(
        best_model,
        scaler,
        splits.area_ids,
        splits.window_ending_2006_scaled,
        splits.gt_2007_2016_scaled,
    )

    # Persist artifacts
    best_model.save(ART_DIR / "dual_cba_best.keras")
    joblib.dump(scaler, ART_DIR / "global_minmax_scaler.joblib")
    df_metrics.to_csv(ART_DIR / "rolling_eval_metrics_by_area.csv", index=False)
    (ART_DIR / "best_hyperparameters.json").write_text(json.dumps(best_hp, indent=2), encoding="utf-8")
    (ART_DIR / "eval_report.json").write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")

    print("Saved artifacts to:", ART_DIR.resolve())
    print("Eval report:", report)


if __name__ == "__main__":
    main()

