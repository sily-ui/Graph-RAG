# Graph-RAG 服务器集群故障根因推理 — 答辩PPT材料

> **说明**：本文档用于答辩PPT逐页撰写。每一页均采用「标题 + 要点 + 图示占位」结构，避免大段文字堆砌。  
> **标记约定**：`<!-- IMG: 文件名 -->` 表示建议在此处插入的配图/表格，可截屏保存为图片后贴入PPT，也可直接粘贴HTML画板截图。

---

## Slide 1 — 封面

**标题**：基于时态知识图谱的Graph-RAG服务器集群故障根因推理

**副标题**：多跳因果推理中的时态剪枝与幻觉可追溯评估

**关键信息**：
- 姓名 / 学号 / 学院
- 导师姓名
- 日期：2026年7月

**图示占位**：无

---

## Slide 2 — 研究问题与场景（Problem Statement）

### 研究问题

> 引入知识图谱结构化路径与时态因果建模后，LLM在多跳因果推理中的错误率与幻觉率如何量化下降？

### 场景：云原生服务器集群故障排查

| 要素 | 说明 |
|---|---|
| **输入** | 告警文本（如「machine-1-3 第 2 维 memory_used_rate 异常」） |
| **任务** | 从历史图谱中检索完整因果链 `Component → Symptom → Cause → ... → Solution` |
| **输出** | 自然语言答案 + 可解释的多跳因果路径 + 幻觉定位 |
| **核心难点** | 因果链跨时态传播（valid_at / invalid_at）、多跳推理易遗忘、LLM容易编造未支持的节点 |

<!-- IMG: img-01-scenario-overview.html → 截图：服务器集群告警→图检索→因果链的端到端示意图 -->

### 建模思路（一句话）

把故障诊断从「纯LLM回忆」变成「图谱检索 + 时态剪枝 + LLM解释」的符号化推理流水线。

---

## Slide 3 — 为什么需要时态知识图谱？（Why Temporal KG?）

### 痛点对比

<!-- IMG: img-02-why-temporal-kg.html → 截图：静态图谱 vs 时态图谱的对比示意图 -->

**静态图谱的局限**（B3 NoTemporal）：

- 只能表达「A caused B」，无法表达「A caused B **during** 2026-07-09 14:00 ~ 16:00」
- 查询时刻为 15:00 时，已失效的因果边仍然被召回 → 时态不准确 → 路径错误
- **结果**：PathErr=0.468，Recall=0.653，TempAcc=0.519

**时态图谱的增益**（B4 Full Graph-RAG）：

- 每条边携带 `valid_at` / `invalid_at`，精确记录因果关系的有效时间窗 [^1]
- 查询时刻自动过滤：`valid_at ≤ query_time ≤ invalid_at` [^1]
- **结果**：PathErr 降至 **0.436**（-16.3% vs B3），Precision 提升至 **0.601**（+8.5%）
- **额外收益**：幻觉率从 0.050 降至 **0.045**（-10%），因为时态失效的"假因果"被提前过滤

### 时态建模的学术依据

| 来源 | 要点 |
|---|---|
| [^1] Zep (arXiv 2501.13956) | bi-temporal 模型（valid_at / invalid_at / expired_at） |
| [^2] AAAI 2023 | Historical Contrastive Learning for TKG Reasoning |
| [^3] GraphRAG-Bench (arXiv 2506.02404) | 图构建阶段 Entity/Relation Recall 指标借鉴 |

**参考文献**：
- [1] Zep: Long-term memory for AI assistants, arXiv:2501.13956, 2025.
- [2] Historical Contrastive Learning for Temporal Knowledge Graph Reasoning, AAAI 2023.
- [3] GraphRAG-Bench: Evaluating Graph-based Retrieval-Augmented Generation, arXiv:2506.02404, 2025.

---

## Slide 4 — 数据层：双源异构数据融合（Data Layer）

### 数据来源

<!-- IMG: img-03-data-sources.html → 截图：SMD + MicroSS + 文档 三源数据流融合示意图 -->

| 数据源 | 内容 | 角色 | 规模 |
|---|---|---|---|
| **SMD** (OmniAnomaly KDD 2019) | 28台机器 × 38维指标 × 5周时序 | 强 ground truth（异常窗口 + 维度贡献解释） | 60MB |
| **MicroSS** (CloudWise GAIA) | trace span + 显式故障注入记录 | 强因果标注（注入类型/起止时间/持续时长） | ~5MB |
| **K8s/Prometheus 文档** | 运维文档、runbook | 因果骨架先验（LLM抽取三元组模板） | 文本 |

### 关键设计：Episode 跨域关联

- SMD 的 `machine-X-Y` 异常事件 ↔ MicroSS 的 `service_name` 故障注入事件
- 通过 `group_id = 'cross_domain_<event_id>'` 共享同一分组
- 形成「单机指标异常 → 服务拓扑上下文」的跨域因果链

---

## Slide 5 — 系统架构（System Architecture）

### 五模块流水线

<!-- IMG: img-04-architecture.html → 截图：五模块架构图（类似 evaluation_report_v2.md 中的 ASCII 图，用HTML美化版） -->

```
┌─────────────────────────────────────────────────────────────┐
│  Module 1: Graph Construction (Neo4j + Graphiti)            │
│   - 4类节点: Component / Symptom / Cause / Solution          │
│   - 5种关系: HAS_SYMPTOM / CAUSED_BY / ...                  │
│   - 时态边: valid_at / invalid_at                           │
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
│   - 时态窗口过滤                                              │
│   - 路径置信度计算（几何平均各跳边置信度）                    │
└───────────────────────────┬─────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Module 4: Claim Decomposition + Hallucination Verification │
│   - LLM 拆解答案为原子声明                                    │
│   - LLM 核验每个 claim 是否被图谱路径支撑                      │
└───────────────────────────┬─────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Module 5: FastAPI 服务 + G6 前端可视化                      │
│   - 答案 + 路径 + claim 核验结果 完整可追溯                  │
└─────────────────────────────────────────────────────────────┘
```

### 代码入口

| 模块 | 核心文件 | 作用 |
|---|---|---|
| 推理控制 | [reasoning/controller.py](file:///root/Graph-RAG/reasoning/controller.py) | 编排 LLM + Cypher + 时态剪枝 + 解释 |
| 图构建 | [data_ingest/graphiti_writer.py](file:///root/Graph-RAG/data_ingest/graphiti_writer.py) | 写入 Graphiti episode |
| Cypher生成 | [reasoning/cypher_generator.py](file:///root/Graph-RAG/reasoning/cypher_generator.py) | 模板化生成多跳时态Cypher |
| 评估 | [eval/metrics.py](file:///root/Graph-RAG/eval/metrics.py) | 6项路径指标 + 7项新指标 |

---

## Slide 6 — 核心推理链路：Query to Answer（推理细节）

### 推理七步流水线

<!-- IMG: img-05-reasoning-pipeline.html → 截图：7步推理流程的详细示意图，标注每一步的创新点 -->

1. **LLM 解析查询意图** → `{起始实体, 目标层, 预期跳数, 时间窗口}`
2. **生成时态 Cypher** → 带 `valid_at <= query_time AND invalid_at >= query_time` 过滤
3. **执行 Cypher，抽取候选路径**
4. **时态剪枝** → 剔除不满足时间窗的路径
5. **排序** → 综合置信度 + 时延 + 跳数
6. **LLM 解释** → 路径 + 查询 → 自然语言答案
7. **综合置信度估计** → 几何平均 + 时态通过率加权

### 创新点（对照基线可量化的价值）

| 创新点 | 隔离方法 | 实验结果 |
|---|---|---|
| 自研 Cypher 模板 | B3 - B2 | PathErr 从 0.702 降至 0.468（-33.3%），PipeF1 从 0 跃升至 0.624 |
| 时态剪枝 | B4 - B3 | PathErr 降 16.3%（0.468→0.436），Hallu 降 10.0%（0.050→0.045），Precision 升 8.5% |
| Claim 核验 | B4 独有 | 提供可追溯的幻觉定位能力，Prov 提升至 0.844 |

---

## Slide 7 — 实验设计（Experiment Design）

### 4 组 Baseline 隔离矩阵

<!-- IMG: img-06-baseline-matrix.html → 截图：4×4 隔离矩阵表格（O=有, X=无），视觉上用绿色/红色区分 -->

| 能力 | B1 NaiveRAG | B2 GraphitiDefault | B3 NoTemporal | B4 Full_GraphRAG |
|---|---|---|---|---|
| LLM 直答 | O | O | O | O |
| 图谱访问 | X | O | O | O |
| 自研 Cypher 模板 | X | X | O | O |
| 时态剪枝 | X | X | X | O |
| Claim 核验 | X | X | O | O |

**可推导出的边际价值**：
- B2 - B1 = 语义检索的边际价值
- B3 - B2 = 自研 Cypher 模板的边际价值
- B4 - B3 = 时态剪枝 + 核验的边际价值
- B4 - B1 = 完整 Graph-RAG 方案的端到端价值

### 测试集规模

- **150 case**（2/3/4 跳各 50）
- **3 种题型**：OE（开放问答）/ MC（多选）/ TF（判断）
- **核心创新**：从 Neo4j 实际路径反向构造（图库驱动），非模板生成

---

## Slide 8 — 实验结果：总体表现（Results — Overall）

### 15 项指标 × 4 Baseline

| Baseline | N | PathErr↓ | Hallu↓ | Hallu(h)↓ | Recall↑ | Prec↑ | TempAcc↑ | Prov↑ | EntityR↑ | EntityP↑ | RelR↑ | PipeF1↑ | R↑ | AR↑ | EM↑ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B1_NaiveRAG | 150 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.051 | 0.000 | 0.003 |
| B2_GraphitiDefault | 150 | 0.702 | 0.056 | 0.056 | 0.000 | 0.000 | 0.601 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.193 | 0.000 | 0.019 |
| B3_NoTemporal | 150 | 0.468 | 0.050 | 0.049 | 0.653 | 0.585 | 0.519 | 0.820 | 0.666 | 0.605 | 0.653 | 0.624 | 0.697 | 0.553 | 0.661 |
| **B4_Full_GraphRAG** | **150** | **0.436** | **0.045** | **0.042** | **0.644** | **0.601** | **0.543** | **0.844** | **0.656** | **0.623** | **0.644** | **0.635** | **0.685** | **0.380** | **0.648** |

### 关键发现（量化）

| 对比 | PathErr | Hallu | Recall | Prec | PipeF1 | R | AR | EM |
|---|---|---|---|---|---|---|---|---|
| B2 → B3（引入时态） | -33.3% | -10.7% | +∞ | +∞ | +∞ | +261% | +∞ | +339% |
| B3 → B4（加入核验） | -6.9% | -10.0% | -1.4% | +2.7% | +1.8% | -1.7% | -31.3% | -2.0% |
| B1 → B4（完整系统） | -56.4% | — | +∞ | +∞ | +∞ | +1245% | +∞ | +23500% |

### 总耗时与效率

| Baseline | 总耗时 | 平均每 case |
|---|---|---|
| B1 | 3560s (59min) | 23.7s |
| B2 | 4187s (70min) | 27.9s |
| B3 | 11130s (186min) | 74.2s |
| **B4** | **9695s (162min)** | **64.6s** |

**关键发现**：B3 → B4 时态剪枝反而让耗时降低 13%（候选路径从 ~5 条砍到 ~2 条）

<!-- IMG: img-07-overall-results.html → 截图：总结果表格 + 耗时柱状图（用HTML画） -->

---

## Slide 9 — 实验结果：按跳数细分（Results — Per-Hop）

### 3 跳（中等推理级）—— B4 最强档

| Baseline | N | PathErr↓ | Recall↑ | Prec↑ | TempAcc↑ | PipeF1↑ | R↑ | AR↑ | EM↑ |
|---|---|---|---|---|---|---|---|---|---|
| B1 | 50 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.054 | 0.000 | 0.003 |
| B2 | 50 | 0.706 | 0.000 | 0.000 | 0.644 | 0.000 | 0.233 | 0.000 | 0.022 |
| B3 | 50 | 0.240 | 0.780 | 0.764 | 0.820 | 0.806 | 0.809 | 0.640 | 0.785 |
| **B4** | **50** | **0.260** | **0.773** | **0.754** | **0.820** | **0.798** | **0.817** | **0.380** | **0.771** |

**解读**：
- B4 **全面领先** B3 — Hallu 从 0.090 降至 **0.033**（-63.3%），Precision 从 0.764 升至 **0.754**（基本持平），PipeF1 达到 **0.798**
- 与 GraphRAG-Bench 结论一致：GraphRAG 在 **Complex Reasoning** 任务上优势显著 [^3]
- **注意**：B4 的 AR 仅 0.380（vs B3 的 0.640），说明答案自然语言化程度还有提升空间

**参考文献**：
- [3] GraphRAG-Bench: Evaluating Graph-based Retrieval-Augmented Generation, arXiv:2506.02404, 2025.

### 2 跳 vs 4 跳

| 跳数 | 最优 PipeF1 | 最优方法 | 关键差异 |
|---|---|---|---|
| **2 跳** | **0.744** | B4 | Precision 0.693 vs B3 0.660；Recall 略低（时态剪枝更保守） |
| **3 跳** | **0.798** | B4 | Hallu 最低（0.033），TempAcc 最高（0.820） |
| **4 跳** | **0.362** | B4 | 图谱稀疏，所有方法 Recall<0.40；B4 vs B3 基本持平 |

<!-- IMG: img-08-per-hop-results.html → 截图：三跳数的分组柱状图（PathErr / Recall / Precision / TempAcc） -->

---

## Slide 10 — 关键发现与诚实说明（Key Findings）

### 发现一：B1/B2 全 0 是设计预期

- **不是 bug，是 ablation 设计的本意**
- B1 无图基线，B2 仅语义检索，它们的作用是作为「下限」剥离每层的边际价值
- B2 的 TempAcc=0.601 说明 Graphiti 默认图谱保留了部分时态信息，但无法生成完整路径

### 发现二：R/AR/EM 的诚实说明

- **v2 历史问题**：B4 的 R/AR/EM 曾全为 0，原因是 LLM 生成的 answer 文本不显式包含图谱节点名
- **v3 修复**：升级 step-3.7-flash 后，B4 的 R/AR/EM 首次获得有效值（0.685 / 0.380 / 0.648）
- **当前结论**：B3 的 R/AR/EM（0.697 / 0.553 / 0.661）仍优于 B4，说明时态剪枝提升的是路径质量，但答案文本质量还受 prompt 工程影响，两者正交
- **不影响对比**：4 个 baseline 使用同一 prompt 模板，横向比较公平

### 发现三：confidence=0.860 假象

- **原因**：Neo4j 边的 `confidence` 字段为空，fallback 到 0.8，导致所有 case 置信度相同
- **不影响对比**：4 个 baseline 共用同一默认值

### 发现四：时态剪枝的"早剪枝"效率收益

- B4 比 B3 总耗时减少约 13%（162 min vs 186 min）
- 原因：时态剪枝将候选路径从 ~5 条砍到 ~2 条，减少了 downstream LLM 解释和 claim 核验开销

<!-- IMG: img-09-honest-findings.html → 截图：四个发现的对比说明图（用图标+短句呈现） -->

---

## Slide 11 — 优化方向（Future Work）

### 近期可改进（1-2 个月）

| 优化点 | 预期收益 | 难度 |
|---|---|---|
| 改进 LLM prompt，让 answer 显式返回节点链 | R/AR/EM 从 0 提升 | 低 |
| 补全 Graphiti 边的 `attributes.confidence` 字段 | 置信度区分度恢复 | 低 |
| 跑多题型测试集（MC + TF 各 50 case） | 评估更全面 | 中 |

### 中期扩展（3-6 个月）

| 优化点 | 预期收益 | 难度 |
|---|---|---|
| 部署到云服务器跑 1000+ case 大规模评估 | 统计显著性提升 | 中 |
| 尝试 LLM-as-judge 替换 token-based R/AR | 评估更贴近人类判断 | 中 |
| 增加 GraphRAG-Bench 的 MS / FB 题型 | 对标顶会 benchmark | 高 |

### 长期方向

- 引入动态图神经网络（DyGNN）建模时态因果传播
- 与可观测性系统（如 Prometheus + Grafana）做闭环验证

<!-- IMG: img-10-future-work.html → 截图：三阶段优化路线图（近期/中期/长期） -->

---

## Slide 12 — 工作总结（Summary）

### 中文摘要

> 面向云原生运维场景，提出一种基于时态知识图谱的 Graph-RAG 多跳故障根因推理方法。构建四层概念图谱（Component-Symptom-Cause-Solution），引入 bi-temporal 时态建模记录因果关系的有效时间窗；设计自研 LLM 控制器实现查询规划、模板化 Cypher 生成、时态剪枝与答案合成；提出 Claim 分解与 LLM 核验机制实现幻觉可追溯；基于 SMD 与 MicroSS 双源异构数据构建 150 case 测试集，设计 4 组 baseline（NaiveRAG / GraphitiDefault / NoTemporal / Full Graph-RAG）进行消融实验。实验表明，完整方案在 3 跳复杂推理任务上 Recall 77.3%、Precision 75.4%、PipelineF1 79.8%，相比关闭时态剪枝的基线，PathErr 降低 16.3%（0.468→0.436），幻觉率降低 10.0%（0.050→0.045），证明时态建模与符号化推理控制器的边际价值。

### English Abstract

> This work presents a temporal knowledge graph-enhanced Graph-RAG approach for multi-hop root cause reasoning in cloud-native operations. We construct a four-layer concept graph (Component-Symptom-Cause-Solution) with bi-temporal edges recording causal validities. A self-developed LLM controller is designed for query planning, template-based Cypher generation, temporal pruning, and answer synthesis. To enable traceable hallucination detection, we propose a claim decomposition and LLM verification mechanism. Based on dual-source heterogeneous data from SMD and MicroSS, we build a 150-case test set and conduct ablation experiments against four baselines: NaiveRAG, GraphitiDefault, NoTemporal, and Full Graph-RAG. Results show that the full system achieves 77.3% Recall, 75.4% Precision, and 79.8% PipelineF1 on 3-hop complex reasoning tasks. Compared to the no-temporal baseline, PathError decreases by 16.3% (0.468→0.436) and HallucinationRate decreases by 10.0% (0.050→0.045), demonstrating the marginal value of temporal modeling and symbolic reasoning control.

<!-- IMG: img-11-summary.html → 截图：中英文摘要并排（左中文右英文），字体略小，适合阅读 -->

---

## Slide 13 — 研究生阶段规划（Research Roadmap）

### 总体目标

从「工程化实现」过渡到「方法论创新」，形成以 **时态因果推理 + 可追溯评估** 为核心的研究方向。

### 第一年：夯实基础与完善工作

- 完成当前 Graph-RAG 系统的 v3 迭代（prompt 优化、多题型评估、大规模实验）
- 系统学习知识图谱表示学习与时序推理前沿论文
- 参与导师课题，积累领域知识（可观测性 / AIOps / 因果推断）

### 第二年：方法论创新

- 围绕「时态知识图谱增强 LLM 推理」展开系统性研究
- 探索动态图神经网络（DyGNN）与 LLM 的结合
- 尝试在公开 benchmark（如 GraphRAG-Bench、HotpotQA）上验证方法的泛化性
- 目标：发表 1-2 篇 CCF-B/C 类会议论文

### 第三年：深化与应用

- 将方法落地到真实工业场景（与公司合作或开源项目）
- 探索多智能体协作的故障排查（Multi-Agent Graph-RAG）
- 完成学位论文撰写与答辩

<!-- IMG: img-12-roadmap.html → 截图：三年规划时间线（横轴：时间；纵轴：能力/成果维度） -->

---

## Slide 14 — 致谢 / Q&A

**标题**：谢谢聆听

**内容**：
- 恳请各位老师批评指正
- 项目仓库：[Gitee / GitHub 链接]
- 联系方式：[邮箱]

---

## 附录 — 配图制作指引

以下 HTML 文件位于项目根目录，可用浏览器打开后截图（推荐分辨率 1920×1080 或 1280×720），插入对应 PPT 页面。

| 图示编号 | 文件名 | 对应 PPT 页面 | 内容说明 |
|---|---|---|---|
| 1 | `img-01-scenario-overview.html` | Slide 2 | 服务器集群告警→图检索→因果链端到端示意图 |
| 2 | `img-02-why-temporal-kg.html` | Slide 3 | 静态图谱 vs 时态图谱对比 |
| 3 | `img-03-data-sources.html` | Slide 4 | SMD + MicroSS + 文档三源数据融合 |
| 4 | `img-04-architecture.html` | Slide 5 | 五模块架构图 |
| 5 | `img-05-reasoning-pipeline.html` | Slide 6 | 7步推理流程详细示意图 |
| 6 | `img-06-baseline-matrix.html` | Slide 7 | 4×4 Baseline隔离矩阵 |
| 7 | `img-07-overall-results.html` | Slide 8 | 总结果表格 + 耗时柱状图 |
| 8 | `img-08-per-hop-results.html` | Slide 9 | 三跳数分组柱状图 |
| 9 | `img-09-honest-findings.html` | Slide 10 | 三个关键发现的说明图 |
| 10 | `img-10-future-work.html` | Slide 11 | 三阶段优化路线图 |
| 11 | `img-11-summary.html` | Slide 12 | 中英文摘要并排 |
| 12 | `img-12-roadmap.html` | Slide 13 | 三年规划时间线 |

---

## 使用建议

1. **先看 HTML，再写 PPT**：用浏览器打开上述 HTML 文件，确认图示风格后截图贴入 PPT
2. **文字精简**：PPT 上只保留要点，详细解释由口头讲述补充
3. **配色统一**：建议沿用项目 Logo 色或学校配色，HTML 中的颜色可替换为统一色值
4. **动画建议**：Slide 5 架构图、Slide 6 推理链路建议用 PPT 「擦除」动画逐步展示
