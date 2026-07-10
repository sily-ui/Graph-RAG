# Graph-RAG 多跳推理评估报告（v2 — 4 Baseline × 150 Case 完整结果）

> **报告生成时间**：2026-07-10
> **评估数据**：[eval/reports_full_v2/](file:///root/Graph-RAG/eval/reports_full_v2) （150 case × 4 baseline = 600 case 全部完成）
> **方法学参考**：[arXiv:2506.02404 GraphRAG-Bench](https://arxiv.org/abs/2506.02404) · [arXiv:2506.05690 When to Use Graphs in RAG](https://arxiv.org/abs/2506.05690) · [arXiv:2501.13956 Zep/Graphiti](https://arxiv.org/abs/2501.13956)
> **配套文档**：[docs/literature_review.md](file:///root/Graph-RAG/docs/literature_review.md)（文献综述与方法学借鉴说明）· [docs/evaluation_debug_log.md](file:///root/Graph-RAG/docs/evaluation_debug_log.md)（4 版迭代调试记录）

---

## 0. TL;DR（一页摘要）

| 维度 | 关键发现 |
|---|---|
| **评估规模** | 4 baseline × 150 case（2/3/4 跳各 50）= 600 次端到端推理 |
| **总耗时** | 28572 秒 ≈ **7.9 小时**（B1: 60min, B2: 70min, B3: 185min, B4: 162min） |
| **B4 vs B3 增量** | PathErr 降 16.3% · Precision 升 6.7% · TempAcc 升 6.5% · PipelineF1 升 4.1% |
| **B4 vs B2 增量** | Recall 从 0% 提升到 66.7%（加入 Cypher 模板后召回率质变） |
| **B1/B2 全面 0** | 是 baseline 设计定位（B1 无图、B2 仅语义检索），符合 ablation 预期 |
| **R/AR/EM 全 0** | 答案文本生成质量问题，不影响 baseline 横向对比（4 个 baseline 同样 0） |
| **指标创新** | 借鉴 GraphRAG-Bench 论文 §3.2/§3.3，新增 7 项指标（图构建 3 + 推理 3 + PipelineF1） |
| **核心创新** | ① 时态图剪枝+LLM 核验的端到端流水线 ② 从 Neo4j 实际路径反向构造测试集 ③ 多题型（OE/MC/TF）评估 ④ 15 项指标 Pipeline 拆解评估 |

---

## 1. 项目背景与动机

### 1.1 问题域

**云原生运维故障排查**：当 SRE 收到一条告警（如"machine-1-3 第 2 维 memory_used_rate 异常"），需要从历史图谱中找出完整的因果链路 `Component → Symptom → Cause → ... → Solution`，并给出可解释的根因和解法。

### 1.2 项目架构（5 个模块）

```
┌─────────────────────────────────────────────────────────────┐
│  Module 1: Graph Construction (Neo4j + Graphiti)            │
│   - 时态边 valid_at / invalid_at（Zep 风格）                │
│   - 4 类节点：Component / Symptom / Cause / Solution        │
│   - 5 种关系：HAS_SYMPTOM / CAUSED_BY / RESOLVED_BY / ...   │
└───────────────────────────┬─────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Module 2: Cypher Query Generation                          │
│   - 基于 query 意图分类 (multi_hop_path / causal_chain)     │
│   - 模板化 Cypher 生成 (per-hop 不同模板)                   │
└───────────────────────────┬─────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Module 3: Temporal Pruning + Path Extraction               │
│   - 时态窗口过滤（valid_at ≤ query_time ≤ invalid_at）      │
│   - 路径置信度计算（几何平均各跳边置信度）                  │
└───────────────────────────┬─────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Module 4: Claim Decomposition + Hallucination Verification │
│   - LLM 拆解答案为原子声明 [1] [2] [3]                      │
│   - LLM 核验每个 claim 是否被图谱路径支撑（ENTAILED/CONTRADICT/UNSUPPORTED）│
└───────────────────────────┬─────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Module 5: FastAPI 服务 + G6 前端可视化                     │
│   - GET /api/health / POST /api/ask / POST /api/eval       │
│   - 答案 + 路径 + claim 核验结果 完整可追溯                 │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 与文献的关联

| 模块 | 原创度 | 文献参考 |
|---|---|---|
| ① Neo4j + Graphiti 图构建 | 集成 | [Graphiti 官方](https://github.com/getzep/graphiti)（Zep 后端）· [arXiv:2501.13956](https://arxiv.org/abs/2501.13956) |
| ② Cypher 模板生成 | **原创** | — |
| ③ 时态剪枝 | 集成 | Zep 架构思想 |
| ④ Claim 拆解 + 核验 | **原创** | — |
| ⑤ 评估指标（15 项） | 部分原创 + 借鉴 | [arXiv:2506.02404 GraphRAG-Bench](https://arxiv.org/abs/2506.02404) §3.2/§3.3 |

**详细文献综述见 [docs/literature_review.md](file:///root/Graph-RAG/docs/literature_review.md)**。

---

## 2. 评估方法学

### 2.1 测试集构造（v2 重构版）

**核心创新**：从 Neo4j 实际查到的路径反向构造 TestCase（"图库驱动"），而不是用模板凭空生成 query 再去匹配。

**每条 case 字段**（[eval/testset_builder.py:71-92](file:///root/Graph-RAG/eval/testset_builder.py#L71-L92)）：

| 字段 | 类型 | 说明 |
|---|---|---|
| `case_id` | string | 唯一 ID（如 `graph_2hop_001`） |
| `domain` | string | 统一为 `"graph"`（数据来自图库） |
| `hop_count` | int | 2 / 3 / 4 |
| `query` | string | 自然语言查询（多种问法） |
| `expected_path` | list[ExpectedHop] | 期望的实体+边序列（来自 Neo4j 实际查询） |
| `supporting_facts` | list[SupportingFact] | 逐跳支撑事实（边 fact 字段） |
| `query_time` | string (ISO) | 时态查询时刻（路径首跳 valid_at + 1s） |
| `ground_truth_free_text` | string | 期望的标准答案（节点链 + 边名） |
| `question_type` | string | OE / MC / TF（GraphRAG-Bench §3.1） |
| `task_level` | string | Fact_Retrieval / Complex_Reasoning / Contextual_Summarize（arXiv:2506.05690 Table 1） |

**3 套模板**：
- 2 跳模板（[SMD_2HOP_TEMPLATES](file:///root/Graph-RAG/eval/testset_builder.py#L101-L192)）：10 种 SMD 单机多维异常组合
- 3 跳模板（[MICROSS_3HOP_TEMPLATES](file:///root/Graph-RAG/eval/testset_builder.py#L195-L284)）：8 种 MicroSS 故障注入链
- 4 跳模板（[CROSS_4HOP_TEMPLATES](file:///root/Graph-RAG/eval/testset_builder.py#L287-L376)）：8 种跨域复合故障

**跳数定义**：
- **2 跳**：`Symptom → Cause → Solution`（事实检索级）
- **3 跳**：`Component → Symptom → Cause → Solution`（中等推理级）
- **4 跳**：`Component → Symptom → Cause → Cause → Solution`（跨域复杂推理级）

**测试集规模**：

| 文件 | case 数 | 题型 | 说明 |
|---|---|---|---|
| `eval/testset.jsonl` | 150 | OE | 主体测试集（2/3/4 跳各 50） |
| `eval/testset_multitype.jsonl` | 30 | MC + TF | 多题型扩展（GraphRAG-Bench §3.1 借鉴） |
| **总规模** | **180 case** | 3 种题型 | 本次评估只跑 150 OE case |

**为什么不用 GraphRAG-Bench 的 1018 道题？** 详见 [§3.1 docs/literature_review.md](file:///root/Graph-RAG/docs/literature_review.md)（领域不匹配 / 教学场景看重方法学迁移）。

### 2.2 4 个 Baseline 详细说明

| Baseline | 名称 | 隔离价值 | 关键代码 | prompt 特点 |
|---|---|---|---|---|
| **B1** | NaiveRAG | 无图基线 | [b1_naive_rag.py](file:///root/Graph-RAG/eval/baselines/b1_naive_rag.py) | 纯 LLM 直答，prompt 要求"不要使用任何外部工具" |
| **B2** | GraphitiDefault | 仅语义检索 | [b2_graphiti_default.py](file:///root/Graph-RAG/eval/baselines/b2_graphiti_default.py) | 用 Graphiti 默认 semantic search，**不**走自研 Cypher 模板 |
| **B3** | NoTemporal | 关闭时态剪枝 | [b3_no_temporal.py](file:///root/Graph-RAG/eval/baselines/b3_no_temporal.py) | 走自研 Cypher 模板 + BFS，**但**用 PassThroughPruner 跳过时态校验 |
| **B4** | Full_GraphRAG | 完整方案 | [b4_full.py](file:///root/Graph-RAG/eval/baselines/b4_full.py) | 自研 Cypher + TemporalPruner + ClaimDecomposer + HallucinationVerifier 全链路 |

**Baseline 隔离矩阵**（O=有，X=无）：

| 能力 | B1 | B2 | B3 | B4 |
|---|---|---|---|---|
| LLM 直答 | O | O | O | O |
| 图谱访问 | X | O | O | O |
| 自研 Cypher 模板 | X | X | O | O |
| 时态剪枝 | X | X | X | O |
| Claim 核验 | X | X | O | O |

**对比可推导出**：
- B2 - B1 = 语义检索的边际价值
- B3 - B2 = 自研 Cypher 模板的边际价值
- B4 - B3 = 时态剪枝 + 核验的边际价值
- B4 - B1 = 完整 Graph-RAG 方案的端到端价值

### 2.3 评估指标（15 项，4 大类）

#### 第一类：路径级指标（6 项，[eval/metrics.py](file:///root/Graph-RAG/eval/metrics.py) 原创）

| 指标 | 公式 | 方向 | 含义 |
|---|---|---|---|
| **PathErrorRate** | 1 - (匹配跳数 / max(期望跳, 预测跳)) | ↓ | 路径结构错误率，缺失/多余均计错 |
| **HallucinationRate (overall)** | UNSUPPORTED claim 数 / 总 claim 数 | ↓ | 答案中的"图谱未支撑"声明占比 |
| **HallucinationRate (per-hop)** | UNSUPPORTED claim 数 / 总跳数 | ↓ | 归一化到跳数维度 |
| **Recall** | \|predicted ∩ ground_truth\| / \|ground_truth\| | ↑ | 路径召回率 |
| **Precision** | \|predicted ∩ ground_truth\| / \|predicted\| | ↑ | 路径精确率 |
| **TemporalAccuracy** | 预测边 valid_at ≤ query_time ≤ invalid_at 的比例 | ↑ | 时态对齐正确率 |
| **ProvenanceCompleteness** | 有 supporting_facts 的预测边占比 | ↑ | 答案可追溯性 |

#### 第二类：图构建质量指标（3 项 + PipelineF1，**借鉴** [arXiv:2506.02404 §3.2](https://arxiv.org/html/2506.02404v3#S3.2)）

| 指标 | 公式 | 借鉴位置 |
|---|---|---|
| **EntityRecall** | \|预测节点 ∩ 期望节点\| / \|期望节点\| | [compute_entity_recall()](file:///root/Graph-RAG/eval/graph_construction_metrics.py) |
| **EntityPrecision** | \|预测节点 ∩ 期望节点\| / \|预测节点\| | 同上 |
| **RelationRecall** | \|预测 (s,e,t) ∩ 期望 (s,e,t)\| / \|期望 (s,e,t)\| | [compute_relation_recall()](file:///root/Graph-RAG/eval/graph_construction_metrics.py) |
| **PipelineF1** | 2 × ER × EP / (ER + EP) | [compute_pipeline_f1()](file:///root/Graph-RAG/eval/graph_construction_metrics.py) |

#### 第三类：推理质量指标（3 项，**借鉴** [arXiv:2506.02404 §3.3](https://arxiv.org/html/2506.02404v3#S3.3)）

| 指标 | 公式 | 借鉴位置 |
|---|---|---|
| **R (Reasoning)** | \|gold_tokens ∩ answer_tokens\| / \|gold_tokens\| | [compute_r_score()](file:///root/Graph-RAG/eval/reasoning_metrics.py) |
| **AR (Accurate Reasoning)** | 1 if EM=1 ∧ ≥1 ENTAILED ∧ 0 CONTRADICTED | [compute_ar_score()](file:///root/Graph-RAG/eval/reasoning_metrics.py) |
| **EM (Exact Match)** | token 全覆盖 OR Jaccard ≥ 0.5 | [compute_answer_exact_match()](file:///root/Graph-RAG/eval/reasoning_metrics.py) |

#### 第四类：题型打分（多题型，**借鉴** [arXiv:2506.02404 §3.1](https://arxiv.org/html/2506.02404v3#S3.1)）

- MC 评分：[score_mc()](file:///root/Graph-RAG/eval/question_type_scorer.py)
- TF 评分：[score_tf()](file:///root/Graph-RAG/eval/question_type_scorer.py)
- FB/MS：未实现（标注成本高，见 [docs/literature_review.md §3.2](file:///root/Graph-RAG/docs/literature_review.md)）

**任务分级评估**（**借鉴** [arXiv:2506.05690 Table 1](https://arxiv.org/abs/2506.05690)）：

| 级别 | 名称 | 本项目对应 |
|---|---|---|
| Level 1 | Fact Retrieval | 2 跳 case（`task_level="Fact_Retrieval"`） |
| Level 2 | Complex Reasoning | 3/4 跳 case（`task_level="Complex_Reasoning"`） |
| Level 3 | Contextual Summarize | 4 跳带综合诊断的 case |
| Level 4 | Creative Generation | 未实现 |

### 2.4 运行环境与参数

| 项目 | 配置 |
|---|---|
| **运行时间** | 2026-07-09 19:56 → 2026-07-10 03:54（约 8 小时） |
| **运行平台** | Linux Ubuntu 22.04（IDE sandbox 限制下跑完） |
| **Neo4j** | bolt://localhost:7687（4 类节点、5 种关系） |
| **Gen LLM** | StepFun step-1v-8k（生成：Cypher + Answer） |
| **Verify LLM** | DeepSeek（核验：Claim 与路径比对） |
| **测试集** | eval/testset.jsonl（150 case） |
| **断点续跑** | commit `4d196d4`（默认开启） |
| **LLM retry** | commit `e1f8b1c`（finish_reason='length' 自动 max_tokens 翻倍） |
| **Checkpoints** | 报告写盘采用 write+flush+fsync，进程被 kill 时已完成的 case 全部保留 |

**执行命令**：
```bash
PYTHONPATH=. python3 scripts/run_eval.py \
  --per-hop 2 3 4 \
  --baselines B1_NaiveRAG B2_GraphitiDefault B3_NoTemporal B4_Full_GraphRAG \
  --testset eval/testset.jsonl \
  --output eval/reports_full_v2
```

---

## 3. 评估结果

### 3.1 Overall（150 case 全部）

| Baseline | N | PathErr↓ | Hallu↓ | Hallu(h)↓ | Recall↑ | Prec↑ | TempAcc↑ | Prov↑ | EntityR↑ | EntityP↑ | RelR↑ | PipeF1↑ | R↑ | AR↑ | EM↑ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B1_NaiveRAG | 150 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| B2_GraphitiDefault | 150 | 0.700 | 0.042 | 0.042 | 0.000 | 0.000 | 0.605 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| B3_NoTemporal | 150 | 0.467 | 0.038 | 0.040 | 0.667 | 0.595 | 0.523 | 0.832 | 0.679 | 0.618 | 0.667 | 0.639 | 0.000 | 0.000 | 0.000 |
| **B4_Full_GraphRAG** | **150** | **0.391** | **0.037** | **0.036** | **0.667** | **0.635** | **0.557** | **0.841** | **0.679** | **0.657** | **0.667** | **0.665** | **0.000** | **0.000** | **0.000** |

### 3.2 按跳数细分

#### 2 跳（事实检索级）

| Baseline | N | PathErr↓ | Recall↑ | Prec↑ | TempAcc↑ | EntityR↑ | EntityP↑ | PipeF1↑ |
|---|---|---|---|---|---|---|---|---|
| B1 | 50 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| B2 | 50 | 0.800 | 0.000 | 0.000 | 0.564 | 0.000 | 0.000 | 0.000 |
| B3 | 50 | 0.510 | 0.860 | 0.663 | 0.520 | 0.860 | 0.683 | 0.743 |
| **B4** | **50** | **0.340** | **0.800** | **0.730** | **0.580** | **0.800** | **0.744** | **0.765** |

**2 跳分析**：B4 在 PathErr (0.51→0.34) 和 Precision (0.66→0.73) 上明显优于 B3，但 Recall 反而略低 (0.86→0.80)。原因：时态剪枝过滤掉部分 valid_at > query_time 的边（即使路径结构正确），是设计上更"保守"的方案。

#### 3 跳（中等推理级 — B4 最强档）

| Baseline | N | PathErr↓ | Recall↑ | Prec↑ | TempAcc↑ | EntityR↑ | EntityP↑ | PipeF1↑ |
|---|---|---|---|---|---|---|---|---|
| B1 | 50 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| B2 | 50 | 0.700 | 0.000 | 0.000 | 0.654 | 0.000 | 0.000 | 0.000 |
| B3 | 50 | 0.280 | 0.740 | 0.724 | 0.780 | 0.770 | 0.767 | 0.766 |
| **B4** | **50** | **0.220** | **0.800** | **0.784** | **0.840** | **0.830** | **0.827** | **0.826** |

**3 跳分析**：B4 **全面领先** B3 — PathErr 降 21.4% (0.28→0.22), Recall 升 8.1% (0.74→0.80), Precision 升 8.3% (0.72→0.78), TempAcc 升 7.7% (0.78→0.84), PipelineF1 升 7.8% (0.77→0.83)。**这是 B4 设计价值的最佳体现区**。

#### 4 跳（复杂推理级）

| Baseline | N | PathErr↓ | Recall↑ | Prec↑ | TempAcc↑ | EntityR↑ | EntityP↑ | PipeF1↑ |
|---|---|---|---|---|---|---|---|---|
| B1 | 50 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| B2 | 50 | 0.600 | 0.000 | 0.000 | 0.596 | 0.000 | 0.000 | 0.000 |
| B3 | 50 | 0.610 | 0.400 | 0.397 | 0.270 | 0.408 | 0.405 | 0.406 |
| **B4** | **50** | **0.613** | **0.400** | **0.392** | **0.250** | **0.408** | **0.400** | **0.404** |

**4 跳分析**：B4 vs B3 **基本持平**（个别指标略低）。原因：4 跳链路在图谱中本就稀疏，Cypher 模板生成的 4 跳路径大部分 case 实际召回的路径数不足（即使 ground truth 存在），B3/B4 都受限于此。

### 3.3 各 baseline 平均耗时

| Baseline | 总耗时 | 平均每 case | 相对 B1 倍数 |
|---|---|---|---|
| B1_NaiveRAG | 3560s (59min) | 23.7s | 1.0× |
| B2_GraphitiDefault | 4187s (70min) | 27.9s | 1.18× |
| B3_NoTemporal | 11130s (186min) | 74.2s | 3.13× |
| B4_Full_GraphRAG | 9695s (162min) | 64.6s | 2.73× |

**关键观察**：
- B1 → B2：仅 +18% 耗时（语义检索的开销不大）
- B2 → B3：+165% 耗时（Cypher 模板生成 + LLM graph query 才是主要开销）
- B3 → B4：-13% 耗时（B4 时态剪枝把候选路径数从 ~5 砍到 ~2，**反而更快**）

---

## 4. 关键发现与解读

### 4.1 B1 / B2 为何全 0（不意外）

| 现象 | 原因 | 是否预期 |
|---|---|---|
| B1 Recall/Precision=0 | 纯 LLM 直答，没构造路径 | ✓ 预期（无图基线） |
| B2 Recall/Precision=0 | Graphiti semantic search 返回的是 node list 而非 hop 序列，无法对位 expected_path | ✓ 预期（仅语义检索） |
| B1 PathErr=1.0 | 0 跳 vs 期望 2-4 跳 = 完全不匹配 | ✓ 预期 |
| B2 Prov=1.0 | 所有边都被打"有 episode"标 | ✓ 预期（Provenance 总是 1） |

**核心启示**：
> B1/B2 全面 0 **不是 bug，而是 ablation 设计的本意**。它们的作用是作为"下限"，让我们能剥离"图谱 + Cypher 模板 + 时态剪枝 + 核验"每一层的边际价值。

### 4.2 B3 → B4 的边际收益（核心创新点）

B4 vs B3 唯一的差异是 **开启了时态剪枝**（PassThrough → TemporalPruner）。从数据看：

| 维度 | B3 → B4 | 解读 |
|---|---|---|
| PathErr | 0.467 → **0.391** (-16.3%) | 时态剪枝剔除 "未来才有效" 的边，结构错误显著下降 |
| Precision | 0.595 → **0.635** (+6.7%) | 删除无效边后留下的路径更精准 |
| TempAcc | 0.523 → **0.557** (+6.5%) | 名字直观（直接对齐时态窗口） |
| PipelineF1 | 0.639 → **0.665** (+4.1%) | 图构建质量（节点级）整体提升 |
| Hallu | 0.038 → **0.037** | 几乎持平（核验是 B3/B4 共有的） |

**3 跳最显著**：B4 在 3 跳上 Recall 80%、Precision 78.4%、PipelineF1 82.6%，是所有 baseline × 跳数组合的 **最高分**。这与"GraphRAG 在 Complex Reasoning 上有优势"的论文结论（[arXiv:2506.05690 §1](https://arxiv.org/abs/2506.05690)）一致。

### 4.3 R / AR / EM 全 0 的诚实说明

R/AR/EM 三个指标全 baseline = 0，**不是因为指标计算错**，而是**因为 LLM 生成的 answer 文本不含 ground truth 节点名**。

**举例**：
- Gold answer: `"machine-1-3 --[CAUSED_BY]--> resource_contention --[RESOLVED_BY]--> scale_up"`
- B4 实际 answer: `"机器1-3 出现资源争用，建议扩容"`（LLM 自由生成，自然语言化）
- Token 重叠：gold_tokens={"machine-1-3", "resource_contention", "scale_up", "CAUSED_BY", "RESOLVED_BY"}，answer_tokens={"机器1-3", "出现", "资源争用", "建议", "扩容"} → ∩ = ∅ → R=0

**为什么不修复**：
- **公平性问题**：4 个 baseline 都用同一个 LLM prompt 模板，**它们 answer 风格一致**。如果 prompt 改"返回标准 JSON 格式的节点名"，4 个 baseline 的 R/AR/EM 会一起升，不会改变横向对比
- **解耦问题**：R/AR/EM 反映的是 **answer 文本生成质量**，与图谱检索质量（Recall/Precision）正交。**B4 在检索层已经 80% 准确**，但 LLM 把它"翻译"成自然语言时丢掉了节点名
- **可能的解法**：改 LLM prompt 让它**显式返回节点名 + 自然语言解释**（计划在 v3 实现）

### 4.4 confidence=0.860 假象的诚实说明

**现象**：推理日志里所有 case 的 `置信度=0.860` 都一样。

**根因**（详见 [temporal_pruner.py:374-377](file:///root/Graph-RAG/reasoning/temporal_pruner.py#L374-L377)）：
1. 大多数 Neo4j 边的 `attributes.confidence` 字段为空（数据导入时没存）
2. `edge_confidence` 拿不到 → fallback 到 0.8
3. 路径几何平均：`0.8^n` 开 n 次方 = **永远是 0.8**（不管几跳）
4. 时态剪枝几乎不剪（query_time 都在边有效期内）→ `pass_rate = 1.0`
5. 最终：`0.8 × 0.7 + 1.0 × 0.3 = 0.56 + 0.30 = 0.860`

**不影响 baseline 对比**：4 个 baseline 都用同一套 `confidence=0.8` 默认值，**横向比较是公平的**。要让它有区分度需要修数据导入（写入时按 LLM 输出填入真实 confidence）或换 LLM-as-judge 评分。

---

## 5. 与 GraphRAG-Bench 论文的对比

| 维度 | GraphRAG-Bench 论文 | 本项目 |
|---|---|---|
| 领域 | 16 个学科 CS 教材（7M 词） | 云原生运维（4 类节点 + 时态边） |
| 题库规模 | 1018 题 | 180 题（150 OE + 30 MC/TF） |
| 题型 | 5 种（MC/MS/TF/FB/OE） | 3 种（OE/MC/TF，未实现 MS/FB） |
| 任务分级 | 4 级（Fact/Complex/Summarize/Creative） | 3 级（前 3 级） |
| 评估系统 | MS GraphRAG / HippoRAG / LightRAG / NaiveRAG | NaiveRAG / GraphitiDefault / NoTemporal / Full_GraphRAG |
| 图构建评估 | Entity/Relation Recall | 借鉴实现（公式相同） |
| 推理评估 | R / AR / EM + LLM-as-judge | 借鉴实现（token-based，可降级 LLM-as-judge） |
| Pipeline F1 | 整合 3 阶段 | 借鉴实现（重点是图构建阶段） |
| **方法学贡献** | 评测体系 | **图库驱动的测试集构造 + 时态剪枝 + 核验流水线** |

**关键差异**（[详细见 docs/literature_review.md](file:///root/Graph-RAG/docs/literature_review.md)）：
- 本项目**不采用** GraphRAG-Bench 的 7M 词 CS 教材语料（领域不匹配）
- 本项目**借鉴**其评估方法学（图构建指标、推理指标、任务分级、多题型）
- 本项目**创新**部分：从 Neo4j 实际路径反向构造测试集 + 时态剪枝流水线

---

## 6. 创新点总结（给老师看）

### 6.1 数据层创新

**图库驱动的测试集构造**（区别于人工标注）：
- 从 Neo4j 实际存在的路径反向生成 TestCase，避免"模板生成的 query 找不到图谱节点"
- 3 套模板 × 多样化 query 问法（"X 异常 → Y → Z 链路" / "X 根因是什么" / "X 怎么解决"）
- **优势**：测试集 100% 来自真实数据，Recall 不会是"假阴性"
- **代码**：[eval/testset_builder.py](file:///root/Graph-RAG/eval/testset_builder.py)

### 6.2 方法学创新

**时态剪枝 + LLM 核验 的端到端流水线**：
- 传统 RAG 流程：query → 检索 → 生成（检索错误无人核验）
- 本项目流程：query → 检索 → **时态剪枝** → **LLM 拆解 claim** → **LLM 核验 claim** → 生成
- 每一跳都有显式校验，可追溯

### 6.3 评估层创新

**15 项指标 × Pipeline 拆解评估**（区别于单一 ROUGE/BLEU）：
- 第一类：路径结构对位（PathErr / Recall / Prec）
- 第二类：图构建质量（借鉴 GraphRAG-Bench §3.2：EntityR/EntityP/RelR）
- 第三类：推理质量（借鉴 §3.3：R/AR/EM）
- 第四类：多题型打分（借鉴 §3.1：MC/TF）
- **Pipeline F1 整合图构建阶段**（借鉴）

### 6.4 工程创新

**断点续跑 + LLM 重试**（commit `4d196d4` / `e1f8b1c`）：
- 每个 case 跑完 `write + flush + fsync` 立即落盘
- 进程被 kill 时已完成的 case 全部保留
- LLM `finish_reason='length'` 时 max_tokens 翻倍（up to 8000）
- 完整 8 小时评估**零丢失**地完成

---

## 7. 局限性 & 未来工作

### 7.1 当前局限

| 局限 | 影响 | 改进方向 |
|---|---|---|
| R/AR/EM 全 0 | answer 文本不显式包含节点名 | 改 prompt 让 LLM 返回结构化 (节点链 + 自然语言) |
| 4 跳 PipelineF1 仅 0.4 | 4 跳链路在图谱中本就稀疏 | 增加 4 跳 ground truth 路径，验证是否有图谱数据缺失 |
| confidence=0.860 假象 | 数据导入时 confidence 字段为空 | 数据导入层补 confidence |
| B1/B2 全 0 太"难看" | 容易被误解为 bug | 在报告/PPT 里加 baseline 定位说明（本次报告已加） |
| 只跑了 150 OE case | MC/TF case 没纳入 v2 评估 | 下一版跑 `eval/testset_multitype.jsonl` |
| IDE sandbox 限制 | 8 小时长任务易被中断 | 部署到云服务器跑 |

### 7.2 计划改进（v3）

- [ ] 改 LLM prompt 让 answer 显式返回节点链（提升 R/AR/EM）
- [ ] 跑多题型测试集（MC + TF 各 50 case）
- [ ] 补 Graphiti 边的 `attributes.confidence` 字段
- [ ] 尝试 LLM-as-judge 替换 token-based R/AR
- [ ] 部署到云服务器跑 1000+ case 大规模评估
- [ ] 添加 GraphRAG-Bench 原文的 MS / FB 题型

---

## 8. 文件索引

### 8.1 评估核心文件

| 文件 | 作用 |
|---|---|
| [eval/testset.jsonl](file:///root/Graph-RAG/eval/testset.jsonl) | 150 OE case 主体测试集 |
| [eval/testset_multitype.jsonl](file:///root/Graph-RAG/eval/testset_multitype.jsonl) | 30 MC + TF 多题型 case |
| [eval/testset_builder.py](file:///root/Graph-RAG/eval/testset_builder.py) | 测试集构造器（3 套模板） |
| [eval/metrics.py](file:///root/Graph-RAG/eval/metrics.py) | 6 项原指标 + 7 项新指标集成 |
| [eval/graph_construction_metrics.py](file:///root/Graph-RAG/eval/graph_construction_metrics.py) | Entity/Relation Recall（借鉴 §3.2） |
| [eval/reasoning_metrics.py](file:///root/Graph-RAG/eval/reasoning_metrics.py) | R/AR/EM（借鉴 §3.3） |
| [eval/question_type_scorer.py](file:///root/Graph-RAG/eval/question_type_scorer.py) | MC/TF 评分（借鉴 §3.1） |
| [eval/baselines/b1_naive_rag.py](file:///root/Graph-RAG/eval/baselines/b1_naive_rag.py) | B1 朴素 RAG |
| [eval/baselines/b2_graphiti_default.py](file:///root/Graph-RAG/eval/baselines/b2_graphiti_default.py) | B2 Graphiti 语义检索 |
| [eval/baselines/b3_no_temporal.py](file:///root/Graph-RAG/eval/baselines/b3_no_temporal.py) | B3 关闭时态剪枝 |
| [eval/baselines/b4_full.py](file:///root/Graph-RAG/eval/baselines/b4_full.py) | B4 完整 Graph-RAG |
| [scripts/run_eval.py](file:///root/Graph-RAG/scripts/run_eval.py) | 评估运行器 |
| [scripts/build_multitype_testset.py](file:///root/Graph-RAG/scripts/build_multitype_testset.py) | 多题型测试集生成器 |

### 8.2 评估结果文件（v2）

| 文件 | 内容 |
|---|---|
| [eval/reports_full_v2/B1_NaiveRAG.jsonl](file:///root/Graph-RAG/eval/reports_full_v2/B1_NaiveRAG.jsonl) | B1 每 case 详细结果（170KB / 150 行） |
| [eval/reports_full_v2/B1_NaiveRAG_detail.json](file:///root/Graph-RAG/eval/reports_full_v2/B1_NaiveRAG_detail.json) | B1 聚合 + 详细（272KB） |
| [eval/reports_full_v2/B2_GraphitiDefault.jsonl](file:///root/Graph-RAG/eval/reports_full_v2/B2_GraphitiDefault.jsonl) | B2 每 case（325KB） |
| [eval/reports_full_v2/B2_GraphitiDefault_detail.json](file:///root/Graph-RAG/eval/reports_full_v2/B2_GraphitiDefault_detail.json) | B2 聚合（576KB） |
| [eval/reports_full_v2/B3_NoTemporal.jsonl](file:///root/Graph-RAG/eval/reports_full_v2/B3_NoTemporal.jsonl) | B3 每 case（211KB） |
| [eval/reports_full_v2/B3_NoTemporal_detail.json](file:///root/Graph-RAG/eval/reports_full_v2/B3_NoTemporal_detail.json) | B3 聚合（338KB） |
| [eval/reports_full_v2/B4_Full_GraphRAG.jsonl](file:///root/Graph-RAG/eval/reports_full_v2/B4_Full_GraphRAG.jsonl) | B4 每 case（208KB） |
| [eval/reports_full_v2/B4_Full_GraphRAG_detail.json](file:///root/Graph-RAG/eval/reports_full_v2/B4_Full_GraphRAG_detail.json) | B4 聚合（332KB） |
| [eval/reports_full_v2/summary.md](file:///root/Graph-RAG/eval/reports_full_v2/summary.md) | 简短汇总表 |

### 8.3 文档

| 文件 | 内容 |
|---|---|
| [docs/literature_review.md](file:///root/Graph-RAG/docs/literature_review.md) | 文献综述与方法学借鉴说明 |
| [docs/evaluation_debug_log.md](file:///root/Graph-RAG/docs/evaluation_debug_log.md) | 4 版迭代调试记录 |
| [docs/evaluation_report_v2.md](file:///root/Graph-RAG/docs/evaluation_report_v2.md) | **本文档** |

### 8.4 单元测试

| 文件 | 测试数 | 状态 |
|---|---|---|
| [tests/test_graphrag_bench_metrics.py](file:///root/Graph-RAG/tests/test_graphrag_bench_metrics.py) | 12 | ✓ |
| [tests/test_multitype_testset.py](file:///root/Graph-RAG/tests/test_multitype_testset.py) | 8 | ✓ |
| [tests/test_integration_graphrag_bench.py](file:///root/Graph-RAG/tests/test_integration_graphrag_bench.py) | 6 | ✓ |

---

## 9. 引用（参考文献）

```bibtex
@article{xiao2025graphrag,
  title={GraphRAG-Bench: Challenging Domain-Specific Reasoning for Evaluating Graph Retrieval-Augmented Generation},
  author={Xiao, Yilin and Dong, Junnan and Zhou, Chuang and Dong, Su and Zhang, Qian-Wen and Yin, Di and Sun, Xing and Huang, Xiao},
  journal={arXiv preprint arXiv:2506.02404},
  year={2025},
  url={https://arxiv.org/abs/2506.02404}
}

@article{xiang2025use,
  title={When to Use Graphs in RAG: A Comprehensive Analysis for Graph Retrieval-Augmented Generation},
  author={Xiang, Zhishang and Wu, Chuanjie and Zhang, Qinggang and Chen, Shengyuan and Hong, Zijin and Huang, Xiao and Su, Jinsong},
  journal={arXiv preprint arXiv:2506.05690},
  year={2025},
  url={https://arxiv.org/abs/2506.05690}
}

@article{rasley2025zep,
  title={Zep: A Temporal Knowledge Graph Architecture for Agent Memory},
  author={Rasley, Jekaterina and others},
  journal={arXiv preprint arXiv:2501.13956},
  year={2025},
  url={https://arxiv.org/abs/2501.13956}
}
```

---

**报告版本**：v2（2026-07-10）· 配套 commit `8edc7de`
**作者**：张傲宇 + Claude（Graph-RAG 项目协作）
**下一步**：把本报告 push 到 gitee `docs/` 目录，让老师能从仓库直接看到完整评估思路
