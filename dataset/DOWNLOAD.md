# 数据集下载说明

本项目支持四种数据源，按与「服务器集群故障排查」主题的匹配度推荐顺序：

| 数据源 | 匹配度 | 体积 | 含故障标注 | 含集群拓扑 | 国内下载 |
|---|---|---|---|---|---|
| **SMD** | ★★★★★ | 200MB | ✅ 异常标签 | ✅ 28 台集群 | ✅ GitHub |
| **GAIA** | ★★★★★ | 2-5GB | ✅ 故障注入记录 | ✅ trace 调用链 | ✅ GitHub |
| Azure V2 | ★★★ | 1.2GB/分片 | ❌ 仅 vm_deleted | ❌ 独立 VM | ⚠️ 较慢 |
| 合成数据 | - | 0 | ✅ 自带 | ❌ | ✅ 无需下载 |

**强烈推荐**：用 SMD + GAIA 跑考核演示。Azure V2 仅作对比 baseline。

---

## 1. SMD（Server Machine Dataset）— 推荐主数据源

28 台服务器 × 38 维指标 × 5 周时序，含异常标签与维度贡献标注。来自某互联网公司真实集群。

### 下载

```bash
cd ~/Graph-RAG/dataset

# 方式 A：完整 clone（推荐，含全部 28 台机器数据）
git clone https://github.com/NetManAIOps/OmniAnomaly.git tmp_omni
mv tmp_omni/ServerMachineDataset .
rm -rf tmp_omni

# 方式 B：GitHub 加速（如果直连慢）
git clone https://ghproxy.com/https://github.com/NetManAIOps/OmniAnomaly.git tmp_omni
mv tmp_omni/ServerMachineDataset .
rm -rf tmp_omni
```

### 目录结构

```
dataset/ServerMachineDataset/
├── train/                    # 训练集时序（无标签）
│   ├── machine-1-1.txt
│   ├── machine-1-2.txt
│   └── ... (共 28 台)
├── test/                     # 测试集时序
├── test_label/               # 测试集异常标签（0/1）
└── interpretation_label/     # 每个异常窗口的贡献维度
```

每个 `.txt` 文件：每行 38 个数值（tab 分隔），1 分钟一条，无表头。

### 加载

```bash
cd ~/Graph-RAG

# dry-run 验证（前 3 台机器）
python scripts/bootstrap_graph.py \
  --source smd \
  --smd-dir dataset/ServerMachineDataset \
  --vms 3 \
  --dry-run

# 真实写入 Neo4j
python scripts/bootstrap_graph.py \
  --source smd \
  --smd-dir dataset/ServerMachineDataset \
  --vms 5 \
  --verify
```

---

## 2. GAIA（Generic AIOps Atlas）— 推荐辅助数据源

CloudWise 开源的微服务系统数据集，含 metric / trace / business / run 四类。**run 目录含故障注入记录**（memory/cpu/network/disk anomalies），是核心创新点的关键数据源。

### 下载

```bash
cd ~/Graph-RAG/dataset

# 方式 A：完整 clone
git clone https://github.com/CloudWise-OpenSource/GAIA-DataSet.git

# 方式 B：GitHub 加速
git clone https://ghproxy.com/https://github.com/CloudWise-OpenSource/GAIA-DataSet.git
```

### 目录结构

```
dataset/GAIA-DataSet/
├── MicroSS/
│   ├── metric/          # 单节点单指标时序 CSV（timestamp,value）
│   ├── trace/           # 调用链 CSV（含 trace_id/span_id/parent_id）
│   ├── business/        # 业务日志
│   └── run/             # 故障注入记录 + 系统日志（关键）
└── Companion_Data/
    ├── metric_detection/  # 含异常标签的指标（timestamp,value,label）
    ├── metric_forecast/
    └── log/
```

### 加载

```bash
cd ~/Graph-RAG

# dry-run 验证（只构建 episode 不写入）
python scripts/bootstrap_graph.py \
  --source gaia \
  --gaia-dir dataset/GAIA-DataSet \
  --dry-run

# 真实写入 Neo4j
python scripts/bootstrap_graph.py \
  --source gaia \
  --gaia-dir dataset/GAIA-DataSet \
  --verify
```

---

## 3. Azure V2（Public Dataset V2）— baseline 对比

微软 Azure 公有云 VM 时序数据，每行 = 一个 VM × 5 分钟。仅 CPU 单维度，无故障标签，作为 baseline 对比用。

### 下载

```bash
cd ~/Graph-RAG/dataset

# 用项目自带的下载脚本（含多源容错 + 断点续传）
bash download_azure_v2.sh 2
```

或手动下载：

```bash
# 直接从 Azure Blob Storage 下载（稳定公开）
curl -L -C - -o vmtablev2_000000000000.csv \
  https://azurepublicdatasetv2.blob.core.windows.net/vmtablev2/vmtablev2_000000000000.csv
```

每个分片约 1.2GB，建议下载 1-2 个分片即可。

### 加载

```bash
cd ~/Graph-RAG

python scripts/bootstrap_graph.py \
  --source azure \
  --csv dataset/vmtablev2_000000000000.csv \
  --vms 50 \
  --dry-run
```

---

## 4. 合成数据 — 开发调试用

无需下载，由 `data_ingest/synthetic_data.py` 即时生成，固定种子可复现。

```bash
cd ~/Graph-RAG

python scripts/bootstrap_graph.py --vms 30 --dry-run
```

---

## 推荐组合方案

**考核演示场景**：

1. **SMD 跑多维症状分析**（核心展示）：
   ```bash
   python scripts/bootstrap_graph.py --source smd --smd-dir dataset/ServerMachineDataset --vms 5 --verify
   ```
   展示 38 维指标异常检测 + 维度贡献归因

2. **GAIA 跑根因定位**（核心创新）：
   ```bash
   python scripts/bootstrap_graph.py --source gaia --gaia-dir dataset/GAIA-DataSet --verify
   ```
   展示 trace 调用链拓扑 + 故障注入记录 → 多跳根因推理

3. **Azure V2 跑 baseline 对比**（可选）：
   ```bash
   python scripts/bootstrap_graph.py --source azure --csv dataset/vmtablev2_000000000000.csv --vms 50 --verify
   ```
   展示与 SMD/GAIA 的对比效果

## 数据集体积汇总

| 数据集 | 下载体积 | 解压后 | 考核所需 |
|---|---|---|---|
| SMD | ~200MB | ~200MB | 全量（28 台机器） |
| GAIA | ~2GB | ~5GB | MicroSS/ 子集即可 |
| Azure V2 | 1.2GB/分片 | 1.2GB/分片 | 1-2 个分片 |

总下载量约 **2-3GB**，Linux 服务器约 5-10 分钟下载完成。
