"""训练并评估 Stage 1 的统计特征 baseline"""

from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

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


@dataclass
class FoldResult:
    """保存单个交叉验证折的评估指标"""

    fold: int
    accuracy: float
    f1: float
    precision: float
    recall: float
    roc_auc: float


def build_baseline_pipeline() -> Pipeline:
    """构建包含预处理和逻辑回归的 Stage 1 baseline 流水线"""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def run_stage1(data_root: Path, output_dir: Path) -> Dict[str, object]:
    """执行 Stage 1 的五折交叉验证并保存结果"""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Stage 1] Loading dataset from {data_root}", flush=True)
    bundle = load_benchmark_dataset(data_root)
    print(
        f"[Stage 1] Loaded {len(bundle.flight_header)} flights with {len(bundle.flight_arrays)} time-series records",
        flush=True,
    )
    fold_results: List[FoldResult] = []
    prediction_frames: List[pd.DataFrame] = []

    for fold in range(5):
        print(f"\n[Fold {fold}] Preparing split", flush=True)
        train_df, test_df = get_fold_split(bundle.flight_header, fold)
        print(f"[Fold {fold}] Train size: {len(train_df)}, Test size: {len(test_df)}", flush=True)
        X_train, y_train = build_feature_table(
            train_df,
            bundle.flight_arrays,
            bundle.mins,
            bundle.maxs,
            desc=f"Fold {fold} train features",
        )
        X_test, y_test = build_feature_table(
            test_df,
            bundle.flight_arrays,
            bundle.mins,
            bundle.maxs,
            desc=f"Fold {fold} test features",
        )

        print(f"[Fold {fold}] Training logistic regression baseline", flush=True)
        pipeline = build_baseline_pipeline()
        pipeline.fit(X_train, y_train)

        print(f"[Fold {fold}] Evaluating", flush=True)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        metrics = compute_binary_metrics(y_test.to_numpy(), y_pred, y_prob)
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
                    "master_index": X_test.index.to_numpy(),
                    "fold": fold,
                    "y_true": y_test.to_numpy(),
                    "y_pred": y_pred,
                    "y_prob": y_prob,
                }
            )
        )

    metrics_df = pd.DataFrame([asdict(result) for result in fold_results])
    summary = build_summary(metrics_df)

    print(f"\n[Stage 1] Saving results to {output_dir}", flush=True)
    save_experiment_outputs(
        output_dir=output_dir,
        metrics_df=metrics_df,
        predictions_df=pd.concat(prediction_frames, ignore_index=True),
        summary=summary,
        report_title="Stage 1 Results",
        report_filename="stage1_report.md",
    )

    return {"fold_metrics": metrics_df, "summary": summary}
