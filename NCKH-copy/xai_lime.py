from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from tensorflow import keras


def lag_feature_names_for_window_1996_2006() -> List[str]:
    """
    11 lags mapped to years 1996..2006, named so that:
      - year 2005 -> lag_10_2005
      - year 2006 -> lag_11_2006
    """
    years = list(range(1996, 2007))
    names = []
    for i, y in enumerate(years, start=1):
        if y == 2005:
            names.append("lag_10_2005")
        elif y == 2006:
            names.append("lag_11_2006")
        else:
            names.append(f"lag_{i}_{y}")
    return names


def build_lime_training_data_for_2007(
    df: pd.DataFrame,
    scaler,
) -> np.ndarray:
    """
    Build a matrix X where each row is a scaled 11-lag window (1996..2006) for an area.
    This is used as LIME's training_data to preserve global distribution across areas.
    """
    years = list(range(1991, 2017))
    cols = [str(y) for y in years]
    mat = df[cols].to_numpy(dtype=np.float32)
    X_rows = []
    for i in range(len(df)):
        y_raw = mat[i]
        if not np.isfinite(y_raw).all():
            continue
        y_scaled = scaler.transform(y_raw.reshape(-1, 1)).reshape(-1).astype(np.float32)
        # 1996..2006 indices are 5..15 in 1991..2016 indexing
        w = y_scaled[5:16]
        if len(w) == 11:
            X_rows.append(w)
    return np.array(X_rows, dtype=np.float32)


def lime_predict_fn_for_model(model: keras.Model):
    def _predict(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32).reshape(-1, 11, 1)
        yhat = model.predict(X, verbose=0).reshape(-1, 1)
        return yhat

    return _predict


@dataclass(frozen=True)
class LimeResult:
    local_weights: pd.DataFrame  # columns: feature, weight
    global_abs_weights: pd.DataFrame  # columns: feature, mean_abs_weight


def explain_area_window(
    model: keras.Model,
    area_window_1996_2006_scaled: np.ndarray,
    lime_training_matrix: np.ndarray,
    num_features: int = 11,
    global_samples: int = 250,
) -> LimeResult:
    feature_names = lag_feature_names_for_window_1996_2006()
    explainer = LimeTabularExplainer(
        training_data=lime_training_matrix,
        feature_names=feature_names,
        mode="regression",
        discretize_continuous=True,
        random_state=42,
    )

    instance = np.asarray(area_window_1996_2006_scaled, dtype=np.float32).reshape(-1)
    exp = explainer.explain_instance(
        data_row=instance,
        predict_fn=lime_predict_fn_for_model(model),
        num_features=min(num_features, len(feature_names)),
    )

    # Local weights from feature indices (stable for both discretized/non-discretized outputs)
    w_local: Dict[str, float] = {
        feature_names[idx]: float(w) for idx, w in exp.as_map().get(1, [])
    }
    df_local = (
        pd.DataFrame({"feature": list(w_local.keys()), "weight": list(w_local.values())})
        .assign(abs_weight=lambda d: d["weight"].abs())
        .sort_values("abs_weight", ascending=False)
        .drop(columns=["abs_weight"])
        .reset_index(drop=True)
    )

    # Global weights: sample many instances and average |weight|
    rng = np.random.default_rng(42)
    n = len(lime_training_matrix)
    take = int(min(global_samples, n))
    idx = rng.choice(n, size=take, replace=False) if take > 0 else np.array([], dtype=int)

    agg: Dict[str, List[float]] = {fn: [] for fn in feature_names}
    for j in idx:
        inst = lime_training_matrix[j]
        exp_j = explainer.explain_instance(
            data_row=inst,
            predict_fn=lime_predict_fn_for_model(model),
            num_features=len(feature_names),
        )
        wj: Dict[str, float] = {
            feature_names[idx]: float(w) for idx, w in exp_j.as_map().get(1, [])
        }
        for fn in feature_names:
            agg[fn].append(abs(wj.get(fn, 0.0)))

    df_global = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "mean_abs_weight": [float(np.mean(agg[fn])) if agg[fn] else 0.0 for fn in feature_names],
            }
        )
        .sort_values("mean_abs_weight", ascending=False)
        .reset_index(drop=True)
    )

    return LimeResult(local_weights=df_local, global_abs_weights=df_global)

