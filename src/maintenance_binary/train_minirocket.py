from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sktime.transformations.panel.rocket import MiniRocketMultivariate

from maintenance_binary.constants import RANDOM_SEED
from maintenance_binary.data import get_fold_split, load_benchmark_dataset
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


def build_minirocket_transformer(num_kernels: int, n_jobs: int) -> MiniRocketMultivariate:
    return MiniRocketMultivariate(
        num_kernels=num_kernels,
        n_jobs=n_jobs,
        random_state=RANDOM_SEED,
    )


def build_minirocket_classifier() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=False)),
            (
                "classifier",
                RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), class_weight="balanced"),
            ),
        ]
    )


def run_stage2(
    data_root: Path,
    output_dir: Path,
    max_length: int = 4096,
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

    fold_results: List[FoldResult] = []
    prediction_frames: List[pd.DataFrame] = []
    fold_iterable = list(folds) if folds is not None else list(range(5))

    for fold in fold_iterable:
        print(f"\n[Fold {fold}] Preparing split", flush=True)
        train_df, test_df = get_fold_split(bundle.flight_header, fold)
        print(f"[Fold {fold}] Train size: {len(train_df)}, Test size: {len(test_df)}", flush=True)
        X_train, y_train, train_ids = build_sequence_tensor(
            train_df,
            bundle.flight_arrays,
            bundle.mins,
            bundle.maxs,
            max_length=max_length,
            desc=f"Fold {fold} train tensors",
        )
        X_test, y_test, test_ids = build_sequence_tensor(
            test_df,
            bundle.flight_arrays,
            bundle.mins,
            bundle.maxs,
            max_length=max_length,
            desc=f"Fold {fold} test tensors",
        )

        print(f"[Fold {fold}] Fitting MiniRocket transformer", flush=True)
        transformer = build_minirocket_transformer(num_kernels=num_kernels, n_jobs=n_jobs)
        transformer.fit(X_train)
        print(f"[Fold {fold}] Transforming train/test sequences", flush=True)
        X_train_feat = transformer.transform(X_train)
        X_test_feat = transformer.transform(X_test)

        print(f"[Fold {fold}] Training ridge classifier on MiniRocket features", flush=True)
        classifier = build_minirocket_classifier()
        classifier.fit(X_train_feat, y_train)

        print(f"[Fold {fold}] Evaluating", flush=True)
        y_pred = classifier.predict(X_test_feat)
        y_score = classifier.decision_function(X_test_feat)
        metrics = compute_binary_metrics(y_test, y_pred, y_score)
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
                    "y_score": y_score,
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
