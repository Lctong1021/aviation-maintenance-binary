# Aviation Maintenance Binary Detection

基于 NGAFID Section 3.2 benchmark 子集的维护事件二分类任务：区分维护前航班与维护后航班。

当前阶段实现：

- `2days` benchmark 子集加载
- 统计特征 baseline
- `Logistic Regression` 5-fold cross validation
- 每折结果与均值、标准差输出

## Project Structure

```text
.
├── configs/
├── data/
├── docs/
├── scripts/
├── src/
│   └── maintenance_binary/
└── tests/
```

## Environment

```bash
conda create -n ngbin python=3.11
conda activate ngbin
pip install -r requirements.txt
```

如果环境已经创建过，后续只需要：

```bash
conda activate ngbin
```

## Stage 1 Run

```bash
python scripts/run_stage1.py
```

常用参数：

```bash
python scripts/run_stage1.py --data-root data/raw --output-dir artifacts/stage1
```

运行结束后会生成：

- `artifacts/stage1/fold_metrics.csv`
- `artifacts/stage1/summary.json`
- `artifacts/stage1/predictions.csv`
- `artifacts/stage1/stage1_report.md`

## Stage 1 Results

当前 `Logistic Regression` baseline 的 5-fold 结果如下：

- accuracy: `0.6775 ± 0.0098`
- f1: `0.6758 ± 0.0108`
- precision: `0.6653 ± 0.0195`
- recall: `0.6870 ± 0.0136`
- roc_auc: `0.7327 ± 0.0132`

详细每折结果见：

- `artifacts/stage1/fold_metrics.csv`
- `artifacts/stage1/stage1_report.md`

## Dataset

本项目默认下载官方仓库 `hyang0129/NGAFIDDATASET` 中 benchmark 用到的 `2days` 子集。  
下载链接来自官方实现中的 Google Drive 文件地址。

如果项目根目录下已经有：

```text
./2days/
├── flight_data.pkl
├── flight_header.csv
└── stats.csv
```

脚本会优先直接使用这份本地数据。

如果 Google Drive 限流导致自动下载失败，可以手工下载后放到：

```text
data/raw/2days.tar.gz
```

或者直接解压成：

```text
data/raw/2days/
├── flight_data.pkl
├── flight_header.csv
└── stats.csv
```

## Metrics

第一阶段默认报告：

- Accuracy
- F1
- ROC-AUC

其中 `F1` 和 `ROC-AUC` 会比单独的准确率更稳，适合后续分析类别不平衡问题。

## Notes

- 不硬编码绝对路径
- 使用数据自带 `fold` 划分
- 默认随机种子为 `42`
