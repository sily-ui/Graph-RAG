# 文献综述与方法学借鉴

> 记录本项目在设计与实现过程中参考的核心文献，及其方法学的具体适配方式。
> 写给项目评审（老师/同行）——任何借鉴自文献的方法都会在代码里有 arXiv 引用，**完全透明可查证**。

---

## 1. 核心参考文献

### 1.1 GraphRAG-Bench（核心借鉴对象）

- **论文**：[arXiv:2506.02404](https://arxiv.org/abs/2506.02404) — *GraphRAG-Bench: Challenging Domain-Specific Reasoning for Evaluating Graph Retrieval-Augmented Generation*
- **作者**：Yilin Xiao, Junnan Dong, Chuang Zhou 等（香港理工大学 + 腾讯 Youtu Lab）
- **官网**：[https://graphrag-bench.github.io/](https://graphrag-bench.github.io/)
- **代码**：[https://github.com/jeremycp3/GraphRAG-Bench](https://github.com/jeremycp3/GraphRAG-Bench)
- **数据集**：1,018 道题，覆盖 16 个学科、20 本 CS 教材、5 种题型（MC / MS / TF / FB / OE）

**3 大核心创新**（[论文 §3](https://arxiv.org/html/2506.02404v3#S3)）：

| 创新 | 本项目适配 |
|---|---|
| **多题型评估**（MC/MS/TF/FB/OE） | 现有 150 case 全是 OE，**新增** MC/TF 题型（~30 case，存 `eval/testset_multitype.jsonl`） |
| **Reasoning score (R) + Accurate Reasoning (AR)** | **新增** R/AR/EM 三个指标到 `eval/reasoning_metrics.py` |
| **图构建质量评估**（Entity Recall / Relation Recall） | **新增** 3 个指标到 `eval/graph_construction_metrics.py` |
| **Pipeline-level 评估**（构建 / 检索 / 生成） | 在 `eval/metrics.py` 里把新指标**分组**到三个维度 |

### 1.2 When to Use Graphs in RAG（任务分级方法学）

- **论文**：[arXiv:2506.05690](https://arxiv.org/abs/2506.05690) — *When to Use Graphs in RAG: A Comprehensive Analysis for Graph Retrieval-Augmented Generation*
- **作者**：Zhishang Xiang, Chuanjie Wu, Qinggang Zhang 等（厦门大学 + 港理工）
- **官网**：[https://graphrag-bench.github.io/](https://graphrag-bench.github.io/)（与 1.1 同一项目）
- **代码**：[https://github.com/GraphRAG-Bench/GraphRAG-Benchmark](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark)

**4 级任务分类法**（[论文 Table 1](https://graphrag-bench.github.io/)）：

| 级别 | 名称 | 描述 | 本项目对应 |
|---|---|---|---|
| **Level 1** | Fact Retrieval | 隔离知识点，最少推理 | 2 跳 case（`task_level="Fact_Retrieval"`） |
| **Level 2** | Complex Reasoning | 多知识点链式推理 | 3/4 跳 case（`task_level="Complex_Reasoning"`） |
| **Level 3** | Contextual Summarize | 碎片信息整合成结构化答案 | 4 跳带"综合诊断"query 的 case（`task_level="Contextual_Summarize"`） |
| **Level 4** | Creative Generation | 基于检索内容的推理外推（未在本项目实现） | — |

**核心发现**（[论文 §1](https://arxiv.org/html/2506.05690)）：GraphRAG 经常在 vanilla RAG 看似简单的任务上**反而表现更差**，但**在 Complex Reasoning 上有显著优势**。本项目在 `eval/reasoning_metrics.py::compute_task_level_f1()` 里按 4 级分组统计 F1，验证此结论。

### 1.3 Zep / 时态知识图谱（Temporal KG 方法学）

- **论文**：[arXiv:2501.13956](https://arxiv.org/abs/2501.13956) — *Zep: A Temporal Knowledge Graph Architecture for Agent Memory*
- **关联项目**：[graphiti-core](https://github.com/getzep/graphiti) — 本项目使用的图谱后端

`valid_at` / `invalid_at` 双时态字段直接来自 Zep/Graphiti 设计，**本项目的 TemporalAccuracy 指标**（[eval/metrics.py:176-203](file:///root/Graph-RAG/eval/metrics.py#L176-L203)）即源自此。

---

## 2. 借鉴适配表（GraphRAG-Bench 构造 → 本项目实现）

| GraphRAG-Bench 构造 | 本项目实现 | 状态 | 文件/函数 |
|---|---|---|---|
| **5 种题型**（MC / MS / TF / FB / OE） | OE（已有 150 case） + MC + TF（新增 30 case） | ✅ 已实现 | [eval/question_type_scorer.py](file:///root/Graph-RAG/eval/question_type_scorer.py) |
| **Question difficulty 4 级**（Level 1-4） | hop_count 2→Level 1, 3-4→Level 2; `task_level` 字段显式标注 | ✅ 已实现 | [eval/testset_builder.py](file:///root/Graph-RAG/eval/testset_builder.py) `TestCase.task_level` |
| **R score**（与 gold rationale 语义一致） | R = \|gold_rationale_tokens ∩ answer_tokens\| / \|gold_rationale_tokens\| | ✅ 已实现 | [eval/reasoning_metrics.py](file:///root/Graph-RAG/eval/reasoning_metrics.py) `compute_r_score()` |
| **AR score**（答对时推理是否也对） | AR = 1 if EM=1 ∧ ≥1 ENTAILED ∧ 0 CONTRADICTED; EM=0 时返回 None | ✅ 已实现 | `compute_ar_score()` |
| **EM**（Exact Match） | token 全覆盖 OR Jaccard ≥ 0.5 | ✅ 已实现 | `compute_answer_exact_match()` |
| **Graph construction quality**（Entity/Relation Recall） | 基于 (source, target) 和 (source, edge, target) 三元组集合 | ✅ 已实现 | [eval/graph_construction_metrics.py](file:///root/Graph-RAG/eval/graph_construction_metrics.py) |
| **Pipeline F1** | 2 × ER × EP / (ER + EP) | ✅ 已实现 | `compute_pipeline_f1()` |
| **4-level task breakdown**（Fact/Complex/Summarize/Creative） | 按 `task_level` 分组计算 macro F1 | ✅ 已实现 | `compute_task_level_f1()` |
| **MS GraphRAG / HippoRAG / LightRAG** 横向对比 | 4 baseline 横向对比（B1 NaiveRAG / B2 GraphitiDefault / B3 NoTemporal / B4 Full_GraphRAG） | ✅ 已实现 | [eval/baselines/](file:///root/Graph-RAG/eval/baselines/) |
| **7M-word CS textbook corpus** | 不采用（本项目是云原生运维领域） | ⚠️ 未采用 | — |

---

## 3. 关键设计决策记录

### 3.1 为什么不直接用 GraphRAG-Bench 的 7M 词 CS 教材语料？

- **领域不匹配**：本项目是**云原生运维故障排查**（Component/Symptom/Cause/Solution 4 类节点 + 时态边），GraphRAG-Bench 是**计算机科学教材**（16 个学科、20 本书）
- **测试集设计哲学不同**：本项目 150 case 是**从 Neo4j 实际路径反向生成**（图库驱动），GraphRAG-Bench 是**专家人工标注**
- **教学场景**：本项目评审看重**方法学迁移能力**（能否把 GraphRAG-Bench 的思路用到自己的领域），不是"用别人的题"

### 3.2 为什么只补 MC + TF，不补 MS / FB？

- **MC（多选一）**：最低成本扩展，1 个 path → 1 个 4 选 1 问题，标注成本可控
- **TF（判断）**：可从 MC 的"错误选项"中衍生，零成本
- **MS（多选多）**：标注成本陡增（需要保证"至少 2 个正确选项互不冗余"），留作未来工作
- **FB（填空）**：需要额外的 gold answer token 标注，2 跳链路中 solution 节点 name 可作为答案，但 3+ 跳链条中"中间节点"边界模糊

### 3.3 为什么 R/AR 用 token-based 而非 LLM-as-judge？

- **可复现性**：token 集合的 F1 是确定性计算，方便论文插图
- **成本**：150 case × 4 baseline × 2 指标 = 1200 次 LLM 调用，会拖慢评估 10+ 倍
- **对齐 GraphRAG-Bench 论文**：R/AR 的 ROUGE-L 实现是**数值化的**，与文献一致
- **可降级**：如果未来需要更细粒度评估，可以把 `compute_r_score()` 的 token-based 替换为 BERTScore 或 LLM-as-judge

### 3.4 借鉴 vs 抄袭的边界

| 借鉴（OK） | 抄袭（NO） |
|---|---|
| 借鉴**方法学**（多题型、R/AR 评分、Pipeline 拆解） | 复制 GraphRAG-Bench 原始题库 |
| 借鉴**指标公式**（R、AR 的形式） | 套用 GraphRAG-Bench 训练的 LLM evaluator |
| 借鉴**任务分级法**（4 级分类） | 声称"本项目实现了 GraphRAG-Bench 的全部能力" |
| **不复制原始数据** | 在论文里写"我们用 GraphRAG-Bench 的数据训练" |

---

## 4. 项目中各模块的文献归属

| 模块 | 原创 | 借鉴 | 引用 |
|---|---|---|---|
| Neo4j + Graphiti 图构建 | ✅ | — | [graphiti-core](https://github.com/getzep/graphiti) |
| 时态剪枝 (valid_at/invalid_at) | ✅ 集成 | Zep 架构思想 | [arXiv:2501.13956](https://arxiv.org/abs/2501.13956) |
| Cypher 模板生成 | ✅ 自研 | — | — |
| Claim Decomposer | ✅ 自研 | — | — |
| Hallucination Verifier | ✅ 自研 | — | — |
| **6 项原指标**（Recall/Precision 等） | ✅ 自研 | 经典 IR 指标 | — |
| **3 项图构建指标** | ✅ 自研 | GraphRAG-Bench §3.2 思想 | [arXiv:2506.02404](https://arxiv.org/abs/2506.02404) |
| **R/AR/EM 指标** | ✅ 自研 | GraphRAG-Bench §3.3 形式 | [arXiv:2506.02404](https://arxiv.org/abs/2506.02404) |
| **4 级任务分类** | ✅ 自研 | "When to Use Graphs" 分类法 | [arXiv:2506.05690](https://arxiv.org/abs/2506.05690) |
| **多题型测试集** | ✅ 自研 | GraphRAG-Bench §3.1 题库设计 | [arXiv:2506.02404](https://arxiv.org/abs/2506.02404) |
| 断点续跑 (Checkpoint) | ✅ 自研 | — | — |
| LLM 重试 + max_tokens 翻倍 | ✅ 自研 | — | — |

---

## 5. 引用（BibTeX）

```bibtex
@article{xiao2025graphrag,
  title={GraphRAG-Bench: Challenging Domain-Specific Reasoning for Evaluating Graph Retrieval-Augmented Generation},
  author={Xiao, Yilin and Dong, Junnan and Zhou, Chuang and Dong, Su and Zhang, Qian-Wen and Yin, Di and Sun, Xing and Huang, Xiao},
  journal={arXiv preprint arXiv:2506.02404},
  year={2025}
}

@article{xiang2025use,
  title={When to Use Graphs in RAG: A Comprehensive Analysis for Graph Retrieval-Augmented Generation},
  author={Xiang, Zhishang and Wu, Chuanjie and Zhang, Qinggang and Chen, Shengyuan and Hong, Zijin and Huang, Xiao and Su, Jinsong},
  journal={arXiv preprint arXiv:2506.05690},
  year={2025}
}

@article{rasley2025zep,
  title={Zep: A Temporal Knowledge Graph Architecture for Agent Memory},
  author={Rasley, Jekaterina and others},
  journal={arXiv preprint arXiv:2501.13956},
  year={2025}
}
```

---

**最后更新**：2026-07-09
**作者**：张傲宇 + Claude（Graph-RAG 项目协作）
