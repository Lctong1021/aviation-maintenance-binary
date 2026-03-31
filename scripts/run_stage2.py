from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from maintenance_binary.constants import DEFAULT_DATA_ROOT, PROJECT_ROOT  # noqa: E402
from maintenance_binary.train_minirocket import run_stage2  # noqa: E402


DEFAULT_STAGE2_OUTPUT = PROJECT_ROOT / "artifacts" / "stage2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stage 2 MiniRocket model for NGAFID maintenance binary detection.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Directory for raw benchmark data.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_STAGE2_OUTPUT, help="Directory for stage 2 outputs.")
    parser.add_argument("--max-length", type=int, default=1024, help="Maximum number of timesteps kept for each flight.")
    parser.add_argument("--num-kernels", type=int, default=10000, help="Number of MiniRocket kernels.")
    parser.add_argument("--n-jobs", type=int, default=1, help="Number of CPU jobs used by MiniRocket.")
    parser.add_argument(
        "--folds",
        type=int,
        nargs="*",
        default=None,
        help="Optional subset of folds to run, for example: --folds 0 1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_stage2(
        data_root=args.data_root,
        output_dir=args.output_dir,
        max_length=args.max_length,
        num_kernels=args.num_kernels,
        n_jobs=args.n_jobs,
        folds=args.folds,
    )

    print("\nPer-fold metrics")
    print(result["fold_metrics"].to_string(index=False))

    print("\nSummary")
    for metric_name, metric_values in result["summary"].items():
        print(f"{metric_name}: {metric_values['mean']:.4f} ± {metric_values['std']:.4f}")


if __name__ == "__main__":
    main()
