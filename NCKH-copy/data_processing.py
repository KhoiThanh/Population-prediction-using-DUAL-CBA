from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


YEAR_START = 1991
YEAR_END = 2016

TRAIN_END = 2006
TEST_START = 2007


@dataclass(frozen=True)
class DatasetSplits:
    # Scaled arrays for Keras: X shape (N, window, 1); y shape (N, 1)
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    # Per-area evaluation seeds for rolling forecast:
    # - window_ending_2006_scaled: (window,)
    # - gt_2007_2016_scaled: (10,)
    area_ids: np.ndarray
    window_ending_2006_scaled: np.ndarray
    gt_2007_2016_scaled: np.ndarray


def year_columns(start: int = YEAR_START, end: int = YEAR_END) -> List[str]:
    return [str(y) for y in range(start, end + 1)]


def load_erp_csv(path: str = "Australia_SmallAreas.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df["SA2_MAINCODE_2011"] = df["SA2_MAINCODE_2011"].astype(int)
    df["STATE_TERRITORY_NAME"] = df["STATE_TERRITORY_NAME"].astype(str)
    df["SA2_NAME_2011"] = df["SA2_NAME_2011"].astype(str)
    return df


def fit_global_minmax(df: pd.DataFrame) -> MinMaxScaler:
    cols = year_columns()
    mat = df[cols].to_numpy(dtype=np.float32).reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    scaler.fit(mat)
    return scaler


def sliding_windows_1step(
    series_scaled: np.ndarray,
    window_size: int,
    start_idx: int,
    end_idx: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create (X,y) windows for one-step prediction within [start_idx, end_idx] inclusive
    for target index. X uses the previous `window_size` values.
    """
    xs, ys = [], []
    for t in range(start_idx, end_idx + 1):
        if t - window_size < 0:
            continue
        xs.append(series_scaled[t - window_size : t])
        ys.append(series_scaled[t])
    if not xs:
        return np.empty((0, window_size, 1), dtype=np.float32), np.empty((0, 1), dtype=np.float32)
    X = np.array(xs, dtype=np.float32)[..., None]
    y = np.array(ys, dtype=np.float32).reshape(-1, 1)
    return X, y


def build_splits(
    df: pd.DataFrame,
    scaler: MinMaxScaler,
    window_size: int = 11,
    forecast_horizon: int = 10,
    val_last_k_windows_per_area: int = 3,
) -> DatasetSplits:
    years = np.arange(YEAR_START, YEAR_END + 1, dtype=int)
    year_to_idx = {y: i for i, y in enumerate(years)}

    idx_train_end = year_to_idx[TRAIN_END]  # index of 2006 in full series
    idx_test_start = year_to_idx[TEST_START]  # index of 2007

    X_train_all, y_train_all = [], []
    X_val_all, y_val_all = [], []

    area_ids: List[int] = []
    seed_windows: List[np.ndarray] = []
    gt_blocks: List[np.ndarray] = []

    # IMPORTANT: don't use getattr(row, "1991") because numeric columns are not valid attributes
    year_cols = year_columns()
    mat = df[year_cols].to_numpy(dtype=np.float32)  # (n_areas, 26)
    area_codes = df["SA2_MAINCODE_2011"].to_numpy(dtype=int)

    for i in range(len(df)):
        area_id = int(area_codes[i])
        y_raw = mat[i]
        if not np.isfinite(y_raw).all():
            continue
        if len(y_raw) < (window_size + 1 + forecast_horizon):
            continue

        y_scaled = scaler.transform(y_raw.reshape(-1, 1)).reshape(-1).astype(np.float32)

        # --- Train/Val windows: one-step targets within 1991..2006 ---
        # Targets are years [YEAR_START+window_size .. TRAIN_END]
        t_start = window_size  # first target index with enough history
        t_end = idx_train_end  # target at 2006

        X_area, y_area = sliding_windows_1step(
            series_scaled=y_scaled, window_size=window_size, start_idx=t_start, end_idx=t_end
        )
        if len(X_area) <= val_last_k_windows_per_area:
            continue

        # Last k windows reserved for validation for this area
        k = val_last_k_windows_per_area
        X_tr, y_tr = X_area[:-k], y_area[:-k]
        X_va, y_va = X_area[-k:], y_area[-k:]

        X_train_all.append(X_tr)
        y_train_all.append(y_tr)
        X_val_all.append(X_va)
        y_val_all.append(y_va)

        # --- Rolling evaluation seed (window ending 2006 -> predict 2007..2016) ---
        # window covers 1996..2006 (11 points)
        seed = y_scaled[idx_train_end - window_size + 1 : idx_train_end + 1]
        gt = y_scaled[idx_test_start : idx_test_start + forecast_horizon]
        if len(seed) != window_size or len(gt) != forecast_horizon:
            continue
        area_ids.append(area_id)
        seed_windows.append(seed)
        gt_blocks.append(gt)

    if not X_train_all:
        raise ValueError("No eligible areas found after filtering for window/validation rules.")

    X_train = np.concatenate(X_train_all, axis=0)
    y_train = np.concatenate(y_train_all, axis=0)
    X_val = np.concatenate(X_val_all, axis=0)
    y_val = np.concatenate(y_val_all, axis=0)

    return DatasetSplits(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        area_ids=np.array(area_ids, dtype=int),
        window_ending_2006_scaled=np.array(seed_windows, dtype=np.float32),
        gt_2007_2016_scaled=np.array(gt_blocks, dtype=np.float32),
    )


def inverse_transform_series(scaler: MinMaxScaler, x_scaled: np.ndarray) -> np.ndarray:
    a = np.asarray(x_scaled, dtype=np.float32).reshape(-1, 1)
    return scaler.inverse_transform(a).reshape(-1)

