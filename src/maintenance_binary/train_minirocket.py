from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sktime.classification.kernel_based import RocketClassifier

from maintenance_binary.constants import RANDOM_SEED
from maintenance_binary.data import load_benchmark_dataset
from maintenance_binary.metrics import compute_binary_metrics
from maintenance_binary.reports import build_summary, save_experiment_outputs
from maintenance_binary.tensor_data import build_sequence_tensor


@dataclass
class FoldResult:
    fold: int
    accuracy: float
    f1: float
    precision: float
    recall: float
    roc_auc: float


def build_minirocket_classifier(num_kernels: int, n_jobs: int) -> RocketClassifier:
    return RocketClassifier(
        num_kernels=num_kernels,
        rocket_transform="minirocket",
        use_multivariate="auto",
        n_jobs=n_jobs,
        random_state=RANDOM_SEED,
    )


def run_stage2(
    data_root: Path,
    output_dir: Path,
    max_length: int = 1024,
    num_kernels: int = 10000,
    n_jobs: int = 1,
    folds: Sequence[int] | None = None,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Stage 2] Loading dataset from {data_root}", flush=True)
    bundle = load_benchmark_dataset(data_root)
    print(
        f"[Stage 2] Loaded {len(bundle.flight_header)} flights with {len(bundle.flight_arrays)} time-series records",
        flush=True,
    )
    print(
        f"[Stage 2] MiniRocket settings: max_length={max_length}, num_kernels={num_kernels}, n_jobs={n_jobs}",
        flush=True,
    )
    print("[Stage 2] Building full sequence tensor once for all flights", flush=True)
    full_header = bundle.flight_header.sort_index()
    X_all, y_all, ids_all = build_sequence_tensor(
        full_header,
        bundle.flight_arrays,
        bundle.mins,
        bundle.maxs,
        max_length=max_length,
        desc="All flights to tensors",
    )
    fold_values = full_header["fold"].to_numpy(dtype=np.int32)

    fold_results: List[FoldResult] = []
    prediction_frames: List[pd.DataFrame] = []
    fold_iterable = list(folds) if folds is not None else list(range(5))

    for fold in fold_iterable:
        print(f"\n[Fold {fold}] Preparing split", flush=True)
        train_mask = fold_values != fold
        test_mask = fold_values == fold
        X_train, y_train, train_ids = X_all[train_mask], y_all[train_mask], ids_all[train_mask]
        X_test, y_test, test_ids = X_all[test_mask], y_all[test_mask], ids_all[test_mask]
        print(f"[Fold {fold}] Train size: {len(y_train)}, Test size: {len(y_test)}", flush=True)

        print(f"[Fold {fold}] Training MiniRocket classifier", flush=True)
        classifier = build_minirocket_classifier(num_kernels=num_kernels, n_jobs=n_jobs)
        classifier.fit(X_train, y_train)

        print(f"[Fold {fold}] Evaluating", flush=True)
        y_pred = classifier.predict(X_test)
        y_prob = classifier.predict_proba(X_test)[:, 1]
        metrics = compute_binary_metrics(y_test, y_pred, y_prob)
        print(
            f"[Fold {fold}] "
            f"accuracy={metrics['accuracy']:.4f}, "
            f"f1={metrics['f1']:.4f}, "
            f"roc_auc={metrics['roc_auc']:.4f}",
            flush=True,
        )

        fold_results.append(FoldResult(fold=fold, **metrics))
        prediction_frames.append(
            pd.DataFrame(
                {
                    "master_index": test_ids,
                    "fold": fold,
                    "y_true": y_test,
                    "y_pred": y_pred,
                    "y_prob": y_prob,
                }
            )
        )

    metrics_df = pd.DataFrame([asdict(result) for result in fold_results])
    summary = build_summary(metrics_df)

    print(f"\n[Stage 2] Saving results to {output_dir}", flush=True)
    save_experiment_outputs(
        output_dir=output_dir,
        metrics_df=metrics_df,
        predictions_df=pd.concat(prediction_frames, ignore_index=True),
        summary=summary,
        report_title="Stage 2 Results",
        report_filename="stage2_report.md",
    )

    return {"fold_metrics": metrics_df, "summary": summary}
