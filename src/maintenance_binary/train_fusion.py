from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from maintenance_binary.constants import RANDOM_SEED
from maintenance_binary.data import get_fold_split, load_benchmark_dataset
from maintenance_binary.features import build_feature_table
from maintenance_binary.metrics import compute_binary_metrics
from maintenance_binary.reports import build_summary, save_experiment_outputs
from maintenance_binary.tensor_data import build_sequence_tensor
from maintenance_binary.train_minirocket import build_minirocket_transformer


@dataclass
class FoldResult:
    fold: int
    accuracy: float
    f1: float
    precision: float
    recall: float
    roc_auc: float


def build_fusion_classifier() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=10000,
                    solver="saga",
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def to_numpy_2d(features: object) -> np.ndarray:
    if isinstance(features, pd.DataFrame):
        return features.to_numpy(dtype=np.float32)
    if isinstance(features, pd.Series):
        return features.to_frame().to_numpy(dtype=np.float32)
    return np.asarray(features, dtype=np.float32)


def run_stage3(
    data_root: Path,
    output_dir: Path,
    max_length: int = 6144,
    num_kernels: int = 10000,
    n_jobs: int = 1,
    folds: Sequence[int] | None = None,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Stage 3] Loading dataset from {data_root}", flush=True)
    bundle = load_benchmark_dataset(data_root)
    print(
        f"[Stage 3] Loaded {len(bundle.flight_header)} flights with {len(bundle.flight_arrays)} time-series records",
        flush=True,
    )
    print(
        f"[Stage 3] Fusion settings: max_length={max_length}, num_kernels={num_kernels}, n_jobs={n_jobs}",
        flush=True,
    )

    fold_results: List[FoldResult] = []
    prediction_frames: List[pd.DataFrame] = []
    fold_iterable = list(folds) if folds is not None else list(range(5))

    for fold in fold_iterable:
        print(f"\n[Fold {fold}] Preparing split", flush=True)
        train_df, test_df = get_fold_split(bundle.flight_header, fold)
        print(f"[Fold {fold}] Train size: {len(train_df)}, Test size: {len(test_df)}", flush=True)

        print(f"[Fold {fold}] Building statistical features", flush=True)
        X_train_stats, y_train = build_feature_table(
            train_df,
            bundle.flight_arrays,
            bundle.mins,
            bundle.maxs,
            desc=f"Fold {fold} train stats",
        )
        X_test_stats, y_test = build_feature_table(
            test_df,
            bundle.flight_arrays,
            bundle.mins,
            bundle.maxs,
            desc=f"Fold {fold} test stats",
        )

        print(f"[Fold {fold}] Building sequence tensors", flush=True)
        X_train_seq, _, train_ids = build_sequence_tensor(
            train_df,
            bundle.flight_arrays,
            bundle.mins,
            bundle.maxs,
            max_length=max_length,
            desc=f"Fold {fold} train tensors",
        )
        X_test_seq, _, test_ids = build_sequence_tensor(
            test_df,
            bundle.flight_arrays,
            bundle.mins,
            bundle.maxs,
            max_length=max_length,
            desc=f"Fold {fold} test tensors",
        )

        print(f"[Fold {fold}] Fitting MiniRocket transformer", flush=True)
        transformer = build_minirocket_transformer(num_kernels=num_kernels, n_jobs=n_jobs)
        transformer.fit(X_train_seq)
        print(f"[Fold {fold}] Transforming sequences", flush=True)
        X_train_mr = to_numpy_2d(transformer.transform(X_train_seq))
        X_test_mr = to_numpy_2d(transformer.transform(X_test_seq))

        X_train_all = np.concatenate([X_train_stats.to_numpy(dtype=np.float32), X_train_mr], axis=1)
        X_test_all = np.concatenate([X_test_stats.to_numpy(dtype=np.float32), X_test_mr], axis=1)

        print(f"[Fold {fold}] Training fusion classifier", flush=True)
        classifier = build_fusion_classifier()
        classifier.fit(X_train_all, y_train)

        print(f"[Fold {fold}] Evaluating", flush=True)
        y_pred = classifier.predict(X_test_all)
        y_score = classifier.decision_function(X_test_all)
        metrics = compute_binary_metrics(y_test.to_numpy(), y_pred, y_score)
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
                    "y_true": y_test.to_numpy(),
                    "y_pred": y_pred,
                    "y_score": y_score,
                }
            )
        )

    metrics_df = pd.DataFrame([asdict(result) for result in fold_results])
    summary = build_summary(metrics_df)

    print(f"\n[Stage 3] Saving results to {output_dir}", flush=True)
    save_experiment_outputs(
        output_dir=output_dir,
        metrics_df=metrics_df,
        predictions_df=pd.concat(prediction_frames, ignore_index=True),
        summary=summary,
        report_title="Stage 3 Results",
        report_filename="stage3_report.md",
    )

    return {"fold_metrics": metrics_df, "summary": summary}
