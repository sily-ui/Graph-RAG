# 评估汇总

## Overall 指标对比（4 baseline × 2/3/4 跳）

| Baseline | PathError↓ | Hallu(整体)↓ | Hallu(逐跳)↓ | Recall↑ | Precision↑ | TemporalAcc↑ | Provenance↑ |
|---|---|---|---|---|---|---|---|
| B4_Full_GraphRAG | 0.490 | 0.038 | 0.038 | 0.632 | 0.557 | 0.626 | 0.851 |

## 按跳数细分

### 2 跳
| Baseline | PathError↓ | Hallu(整体)↓ | Hallu(逐跳)↓ | Recall↑ | Precision↑ | TemporalAcc↑ | Provenance↑ |
|---|---|---|---|---|---|---|---|
| B4_Full_GraphRAG | 0.470 | 0.022 | 0.019 | 0.790 | 0.621 | 0.629 | 0.771 |

### 3 跳
| Baseline | PathError↓ | Hallu(整体)↓ | Hallu(逐跳)↓ | Recall↑ | Precision↑ | TemporalAcc↑ | Provenance↑ |
|---|---|---|---|---|---|---|---|
| B4_Full_GraphRAG | 0.320 | 0.014 | 0.017 | 0.727 | 0.714 | 0.780 | 0.795 |

### 4 跳
| Baseline | PathError↓ | Hallu(整体)↓ | Hallu(逐跳)↓ | Recall↑ | Precision↑ | TemporalAcc↑ | Provenance↑ |
|---|---|---|---|---|---|---|---|
| B4_Full_GraphRAG | 0.680 | 0.078 | 0.079 | 0.380 | 0.337 | 0.470 | 0.988 |