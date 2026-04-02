"""用于检查 benchmark 子集，并导出可复用的数据概览文件"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from maintenance_binary.constants import DEFAULT_DATA_ROOT  # noqa: E402
from maintenance_binary.data import load_benchmark_dataset  # noqa: E402


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "data_overview"


def parse_args() -> argparse.Namespace:
    """解析数据检查脚本所需的命令行参数"""
    parser = argparse.ArgumentParser(description="Inspect the NGAFID 2days benchmark subset and export overview files.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Directory for raw benchmark data.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where overview files will be saved.",
    )
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=10,
        help="Number of rows to keep for preview tables.",
    )
    return parser.parse_args()


def build_length_summary(lengths: np.ndarray) -> dict[str, float]:
    """计算航班序列长度的描述性统计信息"""
    return {
        "count": int(lengths.size),
        "mean": float(np.mean(lengths)),
        "std": float(np.std(lengths)),
        "min": int(np.min(lengths)),
        "q25": float(np.quantile(lengths, 0.25)),
        "median": float(np.quantile(lengths, 0.50)),
        "q75": float(np.quantile(lengths, 0.75)),
        "q90": float(np.quantile(lengths, 0.90)),
        "max": int(np.max(lengths)),
    }


def save_table(df: pd.DataFrame, path: Path) -> None:
    """将表格保存为 CSV，并在需要时创建父目录"""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True)


def main() -> None:
    """加载数据集、生成概览统计，并将结果写入磁盘"""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Inspect] Loading dataset from {args.data_root}", flush=True)
    bundle = load_benchmark_dataset(args.data_root)

    header = bundle.flight_header.copy()
    lengths = pd.Series(
        {int(master_index): int(arr.shape[0]) for master_index, arr in bundle.flight_arrays.items()},
        name="flight_length",
    ).sort_index()
    channels = next(iter(bundle.flight_arrays.values())).shape[1]

    header_with_length = header.copy()
    if "flight_length" in header_with_length.columns:
        header_with_length["computed_flight_length"] = lengths
        length_column = "computed_flight_length"
    else:
        header_with_length["flight_length"] = lengths
        length_column = "flight_length"
    label_counts = header["before_after"].value_counts().sort_index().rename("count").to_frame()
    fold_counts = header["fold"].value_counts().sort_index().rename("count").to_frame()
    fold_label_counts = pd.crosstab(header["fold"], header["before_after"])
    fold_label_counts.index.name = "fold"
    fold_label_counts.columns = [f"before_after_{int(col)}" for col in fold_label_counts.columns]

    length_summary = build_length_summary(lengths.to_numpy())
    class_length_summary = (
        header_with_length.groupby("before_after")[length_column]
        .agg(["count", "mean", "std", "min", "median", "max"])
        .round(3)
    )
    fold_length_summary = (
        header_with_length.groupby("fold")[length_column]
        .agg(["count", "mean", "std", "min", "median", "max"])
        .round(3)
    )

    preview_index = int(header.index[0])
    preview_array = np.asarray(bundle.flight_arrays[preview_index], dtype=np.float32)
    preview_columns = [f"ch_{idx:02d}" for idx in range(preview_array.shape[1])]
    preview_df = pd.DataFrame(preview_array[: args.preview_rows], columns=preview_columns)
    preview_df.index.name = "timestep"

    summary = {
        "num_flights": int(len(header)),
        "num_time_series_records": int(len(bundle.flight_arrays)),
        "num_channels": int(channels),
        "header_columns": list(header.columns),
        "label_counts": {str(int(idx)): int(value) for idx, value in label_counts["count"].items()},
        "fold_counts": {str(int(idx)): int(value) for idx, value in fold_counts["count"].items()},
        "length_summary": length_summary,
        "preview_master_index": preview_index,
        "preview_shape": [int(preview_array.shape[0]), int(preview_array.shape[1])],
    }

    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    save_table(label_counts, args.output_dir / "label_counts.csv")
    save_table(fold_counts, args.output_dir / "fold_counts.csv")
    save_table(fold_label_counts, args.output_dir / "fold_label_counts.csv")
    save_table(class_length_summary, args.output_dir / "class_length_summary.csv")
    save_table(fold_length_summary, args.output_dir / "fold_length_summary.csv")
    save_table(header.head(args.preview_rows), args.output_dir / "header_preview.csv")
    save_table(bundle.stats, args.output_dir / "stats_preview.csv")
    save_table(preview_df, args.output_dir / "sample_flight_preview.csv")

    report_lines = [
        "# Dataset Overview",
        "",
        f"- Number of flights: `{summary['num_flights']}`",
        f"- Number of channels: `{summary['num_channels']}`",
        f"- Label counts: `{summary['label_counts']}`",
        f"- Fold counts: `{summary['fold_counts']}`",
        f"- Flight length summary: `{length_summary}`",
        f"- Preview flight index: `{preview_index}`",
        f"- Preview flight shape: `{tuple(summary['preview_shape'])}`",
        "",
        "## Header Columns",
        "",
        ", ".join(summary["header_columns"]),
        "",
        "## Fold x Label Counts",
        "",
        fold_label_counts.to_string(),
        "",
        "## Class-wise Flight Length Summary",
        "",
        class_length_summary.to_string(),
        "",
        "## Fold-wise Flight Length Summary",
        "",
        fold_length_summary.to_string(),
    ]

    (args.output_dir / "dataset_overview.md").write_text("\n".join(report_lines), encoding="utf-8")

    print("\n[Inspect] Basic summary", flush=True)
    print(f"Flights: {summary['num_flights']}", flush=True)
    print(f"Channels: {summary['num_channels']}", flush=True)
    print(f"Label counts: {summary['label_counts']}", flush=True)
    print(f"Fold counts: {summary['fold_counts']}", flush=True)
    print(
        "Length summary: "
        f"mean={length_summary['mean']:.2f}, median={length_summary['median']:.2f}, "
        f"q75={length_summary['q75']:.2f}, q90={length_summary['q90']:.2f}",
        flush=True,
    )

    print("\n[Inspect] Fold x label counts", flush=True)
    print(fold_label_counts.to_string(), flush=True)

    print("\n[Inspect] Sample flight preview", flush=True)
    print(preview_df.head(min(5, len(preview_df))).to_string(), flush=True)

    print(f"\n[Inspect] Saved overview files to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
