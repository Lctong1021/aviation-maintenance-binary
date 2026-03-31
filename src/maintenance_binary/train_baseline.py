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


def write_stage1_report(output_dir: Path, metrics_df: pd.DataFrame, summary: Dict[str, Dict[str, float]]) -> None:
    report_lines = [
        "# Stage 1 Results",
        "",
        "## 5-Fold Metrics",
        "",
        metrics_df.to_markdown(index=False),
        "",
        "## Summary",
        "",
    ]

    for metric_name in ["accuracy", "f1", "precision", "recall", "roc_auc"]:
        metric_values = summary[metric_name]
        report_lines.append(
            f"- {metric_name}: {metric_values['mean']:.4f} ± {metric_values['std']:.4f}"
        )

    report_lines.append("")
    (output_dir / "stage1_report.md").write_text("\n".join(report_lines), encoding="utf-8")


def run_stage1(data_root: Path, output_dir: Path) -> Dict[str, object]:
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
    summary = {
        metric: {
            "mean": float(metrics_df[metric].mean()),
            "std": float(metrics_df[metric].std(ddof=1)),
        }
        for metric in ["accuracy", "f1", "precision", "recall", "roc_auc"]
    }

    print(f"\n[Stage 1] Saving results to {output_dir}", flush=True)
    metrics_df.to_csv(output_dir / "fold_metrics.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(output_dir / "predictions.csv", index=False)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)
    write_stage1_report(output_dir, metrics_df, summary)

    return {"fold_metrics": metrics_df, "summary": summary}
