"""从变长航班时序中提取手工统计特征"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def scale_flight(arr: np.ndarray, mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    """对单条航班时序按通道归一化，并安全处理无效值"""
    arr = np.asarray(arr, dtype=np.float32)
    denom = np.where((maxs - mins) == 0, 1.0, (maxs - mins))
    scaled = (arr - mins) / denom
    return np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)


def _channel_slope(arr: np.ndarray) -> np.ndarray:
    """使用首尾点近似计算每个通道的整体变化趋势"""
    if arr.shape[0] <= 1:
        return np.zeros(arr.shape[1], dtype=np.float32)
    return (arr[-1] - arr[0]) / float(arr.shape[0] - 1)


def extract_flight_features(arr: np.ndarray, mins: np.ndarray, maxs: np.ndarray) -> Dict[str, float]:
    """将单条多变量航班时序转换为扁平化统计特征字典"""
    arr = scale_flight(arr, mins, maxs)
    arr = np.asarray(arr, dtype=np.float32)

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D flight array, got shape {arr.shape}")

    q25 = np.quantile(arr, 0.25, axis=0)
    q50 = np.quantile(arr, 0.50, axis=0)
    q75 = np.quantile(arr, 0.75, axis=0)

    stats = {
        "mean": arr.mean(axis=0),
        "std": arr.std(axis=0),
        "min": arr.min(axis=0),
        "max": arr.max(axis=0),
        "q25": q25,
        "median": q50,
        "q75": q75,
        "start": arr[0],
        "end": arr[-1],
        "delta": arr[-1] - arr[0],
        "slope": _channel_slope(arr),
        "energy": np.mean(arr * arr, axis=0),
        "iqr": q75 - q25,
    }

    feature_dict: Dict[str, float] = {"flight_length": float(arr.shape[0])}

    #取23个通道的全局特征
    for stat_name, values in stats.items():
        for channel_idx, value in enumerate(values):
            feature_dict[f"ch{channel_idx:02d}_{stat_name}"] = float(value)

    return feature_dict


def build_feature_table(
    header_df: pd.DataFrame,
    flight_arrays: Dict[int, np.ndarray],
    mins: np.ndarray,
    maxs: np.ndarray,
    desc: str | None = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """为某个折的数据构建表格特征矩阵及其对应标签"""
    rows: List[Dict[str, float]] = []
    labels: List[int] = []
    indices: List[int] = []

    iterator = header_df.iterrows()
    if desc:
        iterator = tqdm(iterator, total=len(header_df), desc=desc, leave=False)

    #构建训练集数据
    for master_index, row in iterator:
        arr = flight_arrays[int(master_index)]
        feature_row = extract_flight_features(arr, mins=mins, maxs=maxs)
        feature_row["master_index"] = int(master_index)
        rows.append(feature_row)
        labels.append(int(row["before_after"]))
        indices.append(int(master_index))

    #X为统计特征，y是整段flights对应标签
    X = pd.DataFrame(rows).set_index("master_index").sort_index()
    y = pd.Series(labels, index=indices, name="before_after").sort_index()
    return X, y
