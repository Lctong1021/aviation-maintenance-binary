from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from maintenance_binary.data import BenchmarkDataBundle


def pad_or_truncate_flight(
    arr: np.ndarray,
    mins: np.ndarray,
    maxs: np.ndarray,
    max_length: int,
) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    arr = arr[-max_length:, :]
    denom = np.where((maxs - mins) == 0, 1.0, (maxs - mins))
    arr = (arr - mins) / denom
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    tensor = np.zeros((arr.shape[1], max_length), dtype=np.float32)
    tensor[:, : arr.shape[0]] = arr.T
    return tensor


def build_sequence_tensor(
    header_df: pd.DataFrame,
    flight_arrays: Dict[int, np.ndarray],
    mins: np.ndarray,
    maxs: np.ndarray,
    max_length: int,
    desc: str | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_samples = len(header_df)
    n_channels = mins.shape[0]
    X = np.zeros((n_samples, n_channels, max_length), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.int32)
    ids = np.zeros(n_samples, dtype=np.int32)

    iterator = header_df.iterrows()
    if desc:
        iterator = tqdm(iterator, total=n_samples, desc=desc, leave=False)

    for row_idx, (master_index, row) in enumerate(iterator):
        X[row_idx] = pad_or_truncate_flight(
            flight_arrays[int(master_index)],
            mins=mins,
            maxs=maxs,
            max_length=max_length,
        )
        y[row_idx] = int(row["before_after"])
        ids[row_idx] = int(master_index)

    return X, y, ids
