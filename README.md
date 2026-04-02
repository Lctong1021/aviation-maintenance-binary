# Aviation Maintenance Binary Detection

基于 NGAFID Section 3.2 benchmark 子集的维护事件二分类任务：区分维护前航班与维护后航班。

当前阶段实现：

- `2days` benchmark 子集加载
- 数据集结构检查与统计输出
- 统计特征 baseline
- `Logistic Regression` 5-fold cross validation
- `MiniRocket` 5-fold cross validation
- `MiniRocket` 不同分类头对比
- `MiniRocket + statistical features` fusion
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

## Dataset

本项目默认下载官方仓库 `hyang0129/NGAFIDDATASET` 中 benchmark 用到的 `2days` 子集。  
下载链接来自官方实现中的 Google Drive 文件地址。


./2days/
├── flight_data.pkl
├── flight_header.csv
└── stats.csv
```



## Dataset Inspection

```bash
python scripts/inspect_dataset.py
```

该脚本会输出：

- 数据规模与标签分布
- `fold` 划分情况
- 航班长度统计
- 单条航班样本预览

结果保存在：

- `artifacts/data_overview/summary.json`
- `artifacts/data_overview/dataset_overview.md`
- `artifacts/data_overview/fold_label_counts.csv`
- `artifacts/data_overview/sample_flight_preview.csv`

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

- accuracy: `0.6780 ± 0.0091`
- f1: `0.6765 ± 0.0095`
- precision: `0.6656 ± 0.0190`
- recall: `0.6881 ± 0.0133`
- roc_auc: `0.7327 ± 0.0132`

详细每折结果见：

- `artifacts/stage1/fold_metrics.csv`
- `artifacts/stage1/stage1_report.md`

## Stage 2 Run

```bash
python scripts/run_stage2.py
```

常用参数：

```bash
python scripts/run_stage2.py --max-length 2048 --num-kernels 10000 --n-jobs 1
```

切换不同分类头时可以直接改参数：

```bash
python scripts/run_stage2.py --max-length 2048 --num-kernels 10000 --classifier ridge --output-dir artifacts/stage2_2048_ridge
python scripts/run_stage2.py --max-length 2048 --num-kernels 10000 --classifier logistic --output-dir artifacts/stage2_2048_logistic
python scripts/run_stage2.py --max-length 2048 --num-kernels 10000 --classifier linear_svc --output-dir artifacts/stage2_2048_linearsvc
```

如果机器内存足够，也可以尝试更长的序列：

```bash
python scripts/run_stage2.py --max-length 6144 --num-kernels 10000 --classifier ridge --output-dir artifacts/stage2_6144
```

运行结束后会生成：

- `<output-dir>/fold_metrics.csv`
- `<output-dir>/summary.json`
- `<output-dir>/predictions.csv`
- `<output-dir>/stage2_report.md`

## Stage 2 Results

### Length Comparison

| max_length | accuracy | f1 | roc_auc |
|---|---:|---:|---:|
| `1024` | `0.6758 ± 0.0113` | `0.6668 ± 0.0115` | `0.6756 ± 0.0112` |
| `2048` | `0.7049 ± 0.0100` | `0.6979 ± 0.0124` | `0.7710 ± 0.0075` |
| `4096` | `0.7296 ± 0.0147` | `0.7257 ± 0.0111` | `0.7984 ± 0.0142` |
| `6144` | `0.7360 ± 0.0092` | `0.7349 ± 0.0067` | `0.8051 ± 0.0109` |
| `9000` | `0.7312 ± 0.0122` | `0.7286 ± 0.0089` | `0.7974 ± 0.0142` |

当前最佳单模型结果来自 `MiniRocket(6144)`：

- accuracy: `0.7360 ± 0.0092`
- f1: `0.7349 ± 0.0067`
- precision: `0.7226 ± 0.0066`
- recall: `0.7480 ± 0.0196`
- roc_auc: `0.8051 ± 0.0109`

### Classifier Head Comparison (`max_length=2048`)

| classifier | accuracy | f1 | roc_auc |
|---|---:|---:|---:|
| `RidgeClassifierCV` | `0.7049 ± 0.0100` | `0.6979 ± 0.0124` | `0.7710 ± 0.0075` |
| `LogisticRegression` | `0.6999 ± 0.0151` | `0.6922 ± 0.0167` | `0.7644 ± 0.0148` |
| `LinearSVC` | `0.4933 ± 0.0161` | `0.5255 ± 0.2940` | `0.5500 ± 0.0215` |

对应结果目录：

- `artifacts/stage2_2048_ridge/`
- `artifacts/stage2_2048_logistic/`
- `artifacts/stage2_2048_linearsvc/`

## Stage 3 Run

```bash
python scripts/run_stage3.py
```

常用参数：

```bash
python scripts/run_stage3.py --max-length 6144 --num-kernels 10000 --n-jobs 1
```

运行结束后会生成：

- `artifacts/stage3/fold_metrics.csv`
- `artifacts/stage3/summary.json`
- `artifacts/stage3/predictions.csv`
- `artifacts/stage3/stage3_report.md`

## Stage 3 Results

Stage 3 使用 `MiniRocket + statistical features` 融合方案，在 `6144` 长度下的 5-fold 结果如下：

- accuracy: `0.7251 ± 0.0098`
- f1: `0.7221 ± 0.0082`
- precision: `0.7145 ± 0.0068`
- recall: `0.7302 ± 0.0191`
- roc_auc: `0.7943 ± 0.0096`

该结果明显优于 Stage 1 baseline，但没有超过 Stage 2 的最佳 `MiniRocket(6144)`，说明在当前实现下，简单的统计特征拼接并没有带来额外收益。

## Overall Summary

| stage | method | accuracy | f1 | roc_auc |
|---|---|---:|---:|---:|
| Stage 1 | Statistical features + Logistic Regression | `0.6780 ± 0.0091` | `0.6765 ± 0.0095` | `0.7327 ± 0.0132` |
| Stage 2 | MiniRocket (`6144`) + RidgeClassifierCV | `0.7360 ± 0.0092` | `0.7349 ± 0.0067` | `0.8051 ± 0.0109` |
| Stage 3 | MiniRocket (`6144`) + statistical feature fusion | `0.7251 ± 0.0098` | `0.7221 ± 0.0082` | `0.7943 ± 0.0096` |



## Metrics

当前默认报告：

- Accuracy
- F1
- ROC-AUC

其中 `F1` 和 `ROC-AUC` 会比单独的准确率更稳，适合后续分析类别不平衡问题。

## Notes

- 使用数据自带 `fold` 划分
- 默认随机种子为 `42`
- 维护标签含义：`before_after = 1` 表示维护前航班，`before_after = 0` 表示维护后航班
