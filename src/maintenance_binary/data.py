from __future__ import annotations

import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import gdown
import numpy as np
import pandas as pd
from compress_pickle import load
from gdown.exceptions import FileURLRetrievalError

from maintenance_binary.constants import BENCHMARK_NAME, BENCHMARK_URL, MAX_CHANNELS


@dataclass
class BenchmarkDataBundle:
    flight_header: pd.DataFrame
    flight_arrays: Dict[int, np.ndarray]
    stats: pd.DataFrame
    mins: np.ndarray
    maxs: np.ndarray


def ensure_benchmark_downloaded(data_root: Path) -> Path:
    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    extract_dir = data_root / BENCHMARK_NAME
    header_path = extract_dir / "flight_header.csv"
    data_path = extract_dir / "flight_data.pkl"
    stats_path = extract_dir / "stats.csv"

    if header_path.exists() and data_path.exists() and stats_path.exists():
        return extract_dir

    archive_path = data_root / f"{BENCHMARK_NAME}.tar.gz"
    if not archive_path.exists():
        try:
            gdown.download(BENCHMARK_URL, str(archive_path), quiet=False)
        except FileURLRetrievalError as exc:
            raise RuntimeError(
                "Automatic download of the NGAFID benchmark subset failed. "
                f"Please manually download the official '{BENCHMARK_NAME}.tar.gz' archive from:\n"
                f"{BENCHMARK_URL}\n"
                f"and place it at '{archive_path}', or extract it to '{extract_dir}'."
            ) from exc

    with tarfile.open(archive_path) as tar:
        tar.extractall(data_root)

    return extract_dir


def load_benchmark_dataset(data_root: Path) -> BenchmarkDataBundle:
    extract_dir = ensure_benchmark_downloaded(data_root)

    header = pd.read_csv(extract_dir / "flight_header.csv", index_col="Master Index")
    stats = pd.read_csv(extract_dir / "stats.csv")
    flight_arrays = load(extract_dir / "flight_data.pkl")

    maxs = stats.iloc[0, 1 : MAX_CHANNELS + 1].to_numpy(dtype=np.float32)
    mins = stats.iloc[1, 1 : MAX_CHANNELS + 1].to_numpy(dtype=np.float32)

    return BenchmarkDataBundle(
        flight_header=header,
        flight_arrays=flight_arrays,
        stats=stats,
        mins=mins,
        maxs=maxs,
    )


def get_fold_split(header_df: pd.DataFrame, fold: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    test_df = header_df.loc[header_df["fold"] == fold].copy()
    train_df = header_df.loc[header_df["fold"] != fold].copy()
    return train_df, test_df
