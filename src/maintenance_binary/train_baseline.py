from __future__ import annotations

import json
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


@dataclass
class FoldResult:
    fold: int
    accuracy: float
    f1: float
    precision: float
    recall: float
    roc_auc: float


def build_baseline_pipeline() -> Pipeline:
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
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_benchmark_dataset(data_root)
    fold_results: List[FoldResult] = []
    prediction_frames: List[pd.DataFrame] = []

    for fold in range(5):
        train_df, test_df = get_fold_split(bundle.flight_header, fold)
        X_train, y_train = build_feature_table(train_df, bundle.flight_arrays, bundle.mins, bundle.maxs)
        X_test, y_test = build_feature_table(test_df, bundle.flight_arrays, bundle.mins, bundle.maxs)

        pipeline = build_baseline_pipeline()
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        metrics = compute_binary_metrics(y_test.to_numpy(), y_pred, y_prob)

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
    summary = {
        metric: {
            "mean": float(metrics_df[metric].mean()),
            "std": float(metrics_df[metric].std(ddof=1)),
        }
        for metric in ["accuracy", "f1", "precision", "recall", "roc_auc"]
    }

    metrics_df.to_csv(output_dir / "fold_metrics.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(output_dir / "predictions.csv", index=False)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)

    return {"fold_metrics": metrics_df, "summary": summary}
