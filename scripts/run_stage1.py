"""Stage 1 统计特征 baseline 的命令行入口"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from maintenance_binary.constants import DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_ROOT  # noqa: E402
from maintenance_binary.train_baseline import run_stage1  # noqa: E402


def parse_args() -> argparse.Namespace:
    """解析 Stage 1 实验所需的命令行参数"""
    parser = argparse.ArgumentParser(description="Run stage 1 baseline for NGAFID maintenance binary detection.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Directory for raw benchmark data.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Directory for stage 1 outputs.")
    return parser.parse_args()


def main() -> None:
    """执行 Stage 1 流程并输出最终评估指标"""
    args = parse_args()
    result = run_stage1(data_root=args.data_root, output_dir=args.output_dir)

    print("\nPer-fold metrics")
    print(result["fold_metrics"].to_string(index=False))

    print("\nSummary")
    for metric_name, metric_values in result["summary"].items():
        print(f"{metric_name}: {metric_values['mean']:.4f} ± {metric_values['std']:.4f}")


if __name__ == "__main__":
    main()
