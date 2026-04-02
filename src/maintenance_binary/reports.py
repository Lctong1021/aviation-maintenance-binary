"""保存折级指标、汇总结果、预测输出与简要 markdown 报告"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd


def build_summary(metrics_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """将各折指标汇总为均值和标准差"""
    return {
        metric: {
            "mean": float(metrics_df[metric].mean()),
            "std": float(metrics_df[metric].std(ddof=1)),
        }
        for metric in ["accuracy", "f1", "precision", "recall", "roc_auc"]
    }


def write_experiment_report(
    output_dir: Path,
    metrics_df: pd.DataFrame,
    summary: Dict[str, Dict[str, float]],
    title: str,
    report_filename: str,
) -> None:
    """为某个实验阶段写出简要的 markdown 结果报告"""
    report_lines = [
        f"# {title}",
        "",
        "## 5-Fold Metrics",
        "",
        "```text",
        metrics_df.to_string(index=False),
        "```",
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
    (output_dir / report_filename).write_text("\n".join(report_lines), encoding="utf-8")


def save_experiment_outputs(
    output_dir: Path,
    metrics_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    summary: Dict[str, Dict[str, float]],
    report_title: str,
    report_filename: str,
) -> None:
    """保存所有实验阶段共用的标准输出文件"""
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_dir / "fold_metrics.csv", index=False)
    predictions_df.to_csv(output_dir / "predictions.csv", index=False)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)
    write_experiment_report(
        output_dir=output_dir,
        metrics_df=metrics_df,
        summary=summary,
        title=report_title,
        report_filename=report_filename,
    )
