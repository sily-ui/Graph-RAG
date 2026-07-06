# Azure Public Dataset V2 下载说明

本目录用于存放 Azure V2 真实数据集，**不入库 git**（已在 .gitignore 排除）。

## 下载步骤

### 1. 下载链接清单

```bash
# 下载 Azure V2 的文件 URL 清单（198 个分片）
curl -L -o azure_v2_links.txt \
  https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/AzurePublicDatasetV2.txt

# 查看前几个链接
head -n 3 azure_v2_links.txt
```

### 2. 下载分片（推荐 1-3 个，每个约 1.2GB）

```bash
# 下载前 2 个分片
head -n 2 azure_v2_links.txt | xargs -n 1 -P 2 curl -L -O
```

下载后的文件名形如 `vmtablev2_000000000000.csv`、`vmtablev2_000000000001.csv`。

### 3. 数据格式

Azure V2 是**长格式** CSV（无表头，20 列）：

| 列号 | 字段 | 说明 |
|---|---|---|
| 1 | subscription_id | 订阅 ID |
| 6 | vm_id | VM ID |
| 7 | vm_created | VM 创建时间戳（秒） |
| 8 | vm_deleted | VM 删除时间戳（秒，0=未删除） |
| 11 | vm_category | VM 类别 |
| 13 | vcore_bucket | vCPU 桶 |
| 14 | memory_gb_bucket | 内存桶 |
| 15 | timestamp | 5 分钟时间戳 |
| 18 | cpu_avg_5min | 该 5 分钟内平均 CPU |

### 4. 验证下载

```bash
# 查看文件大小
ls -lh vmtablev2_*.csv

# 查看前 3 行
head -n 3 vmtablev2_000000000000.csv
```

### 5. 用本项目的 loader 加载

```bash
cd ~/Graph-RAG
python scripts/bootstrap_graph.py \
  --csv dataset/vmtablev2_000000000000.csv \
  --vms 50 \
  --dry-run
```

`--dry-run` 先验证加载流程，无误后去掉 `--dry-run` 真实写入 Neo4j。

## 数据量建议

| 用途 | 文件数 | 大小 | 预计 VM 数 |
|---|---|---|---|
| 开发调试 | 1 | 1.2GB | ~1.3万 |
| 考核验证 | 2-3 | 2.4-3.6GB | ~3-4万 |
| 全量分析 | 198 | 235GB | 269万 |

针对考核场景，**下载 1-2 个分片即可**。
