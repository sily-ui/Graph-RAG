# Graph-RAG 服务器集群故障推理考核实现方案

## Summary

面向推免考核，构建一个基于时态知识图谱的 Graph-RAG 故障根因推理问答系统，场景选定为服务器集群故障排查。核心研究问题是：**"引入知识图谱结构化路径与时态因果建模后，LLM 在多跳因果推理中的错误率与幻觉率如何量化下降"**。

系统采用 Graphiti 开源版构建时态因果图谱层（记录故障时序），自研 LLM 控制器做查询生成与多跳路径抽取（不外包核心推理），基于 Azure Public Dataset V2 + Google clusterdata-2019 公开数据集与 K8s/Prometheus 运维文档构建因果骨架，交付形态为后端推理系统 + 严谨量化实验报告（2/3/4 跳对照实验）+ G6 因果路径动态可视化。

本方案将用户长处（AntV G6 可视化、工程化能力）最大化转化为研究论证工具，通过逐跳动画、时态查询对比、幻觉定位可视化三个维度让 G6 服务于实验论证而非装饰。

## Current State Analysis

### 项目现状
- 目标目录 `d:\Python练习\Graph-RAG` 当前为空目录，为全新项目。
- 用户已确定三大核心决策（见 Assumptions & Decisions）。
- 用户技术背景：大三推免生，工程化能力强（计算机设计大赛经验），擅长 AntV G6 关系图可视化，技术栈 React/TypeScript。

### 考核本质认知
导师考核的是研究者视角而非产品视角，核心评分维度：
1. **方法理解深度**：理解"为什么符号推理能压制 LLM 幻觉"，而非仅会用工具
2. **实验严谨性**：任务反复强调"量化展示"，这是研究者核心能力
3. **技术选型合理性**：选型有依据、有对照、有消融

与计算机设计大赛的关键差异：60% 精力放实验设计与量化评估，30% 放核心算法实现，10% 放可视化呈现。

### 已识别的关键技术点（基于 Web 研究）
- **Graphiti 开源版**：`pip install graphiti-core`，后端用 Neo4j Community Server（Windows 原生安装，无需 Docker）。核心 API 为 `Graphiti.add_episode()`（时态锚点 `reference_time`）与 `search()`（支持 `bfs_max_depth` 控制跳数）。时态建模通过 `EntityEdge` 的 `valid_at`/`invalid_at`/`expired_at` 实现 bi-temporal 模型。
- **Azure Public Dataset V2**：2019 年 Azure 某区域 30 天数据，2,695,548 VM，1.9B 条 5 分钟 CPU 读数，235GB，可从 GitHub Releases 直接下载。含 `timestamp_deleted` + CPU 时序，可抽取故障/驱逐事件。
- **Google clusterdata-2019**：8 个 Borg cell，约 2.4 TiB，需 BigQuery 访问。含 alloc set 父子关系，适合补充因果骨架，作为辅助数据源。
- **幻觉可追溯评估**：借鉴 MuSiQue/HotpotQA 的 supporting facts 逐跳标注思想，将答案分解为原子陈述，每条标注依赖跳数，逐跳核验 provenance。

## Proposed Changes

### 模块 1: 知识图谱 Schema 定义

**文件**: `graph_schema/nodes.py`, `graph_schema/edges.py`, `graph_schema/constraints.py`

**What**: 用 Graphiti 的 prescribed ontology 机制（Pydantic 子类化 `EntityNode` 与 `EntityEdge`）定义四层概念图谱。

**Why**: 考核要求"清晰涵盖组件-症状-因果-解法的概念等级"。四层分层 + 边合法性约束是图谱语义正确性的前提，也是 2/3/4 跳路径模板可构造的基础。用 Pydantic validator 在 `add_episode` 前校验非法边（如 Solution→Symptom），保证图谱结构正确性。

**How - 节点四层**:
- 第1层 Component（组件）：vm/pod/container/node/service/deployment/namespace，含 cluster_id、SKU 规格
- 第2层 Symptom（症状）：metric_anomaly/event/log_pattern，含 severity、metric_name、threshold、first_observed_at
- 第3层 Cause（因果）：resource_contention/misconfiguration/dependency_failure/network_partition/hardware_fault/priority_inversion/noisy_neighbor，含 confidence、is_root
- 第4层 Solution（解法）：restart_pod/scale_up/drain_node/rollback_deployment/increase_limit/runbook_procedure，含 runbook_ref、estimated_mttr

**How - 边三类（带时态）**:
- `HAS_SYMPTOM`：Component→Symptom，valid_at=症状开始，invalid_at=症状结束
- `CAUSED_BY`/`TRIGGERED_BY`/`PROPAGATED_TO`：Symptom/Cause→Cause，带 `lag_seconds`（因果时延）与 `evidence_episodes`（provenance）
- `RESOLVED_BY`/`MITIGATED_BY`：Cause/Symptom→Solution，带 `effectiveness`

**多跳路径模板**:
- 2 跳：Symptom→Cause(root)→Solution
- 3 跳：Symptom→Cause→Cause(root)→Solution
- 4 跳：Component→Symptom→Cause→Cause(root)→Solution

**可借鉴文献**:
- Zep: A Temporal Knowledge Graph Architecture for Agent Memory (arXiv 2501.13956) — bi-temporal 模型设计依据
- Temporal Knowledge Graph Reasoning with Historical Contrastive Learning (AAAI 2023) — 时序因果建模方法参考

---

### 模块 2: 数据接入与图谱构建管线

**文件**: `data_ingest/azure_trace_loader.py`, `data_ingest/google_trace_loader.py`, `data_ingest/fault_event_extractor.py`, `data_ingest/doc_skeleton_seeder.py`, `data_ingest/episode_builder.py`, `data_ingest/graphiti_writer.py`

**What**: 从公开数据集抽取故障事件 + 从运维文档抽取因果骨架，打包成 Graphiti episode 写入时态图谱。

**Why**: 数据真实性决定图谱可信度。Azure V2 提供真实 CPU 时序与 VM 删除事件，K8s/Prometheus 文档提供因果先验，两者结合既保证领域真实又控制工程量。抽样策略（1 区域 × 7 天 × 故障率 top VM 子集）避免 235GB 全量加载。

**How**:
- `azure_trace_loader.py`：流式读取 Azure V2 的 198 个文件，按 5 分钟粒度对每 VM 计算 CPU 时序，滑动窗口 + IQR/3-sigma 检测异常点，抽取 CPU spike（超 p95 阈值持续 N 分钟）与 VM 删除（=故障/驱逐）事件
- `google_trace_loader.py`：BigQuery 客户端抽样，取 alloc set 父子关系补充因果骨架
- `doc_skeleton_seeder.py`：解析 K8s 官方文档（Pod 生命周期、驱逐策略、OOMKilled）、Prometheus alerting rules、runbook，用 LLM 抽取因果三元组模板作为骨架种子
- `episode_builder.py`：把故障事件 + 相关 trace 片段 + 文档因果片段打包成 episode，`reference_time` 设为故障发生时刻（时态锚点正确性的关键）
- `graphiti_writer.py`：调用 `graphiti.add_episode(name, episode_body, source=EpisodeType.json, reference_time=event.ts_start, group_id=cluster_id)` 批量写入

**可借鉴文献**:
- AzurePublicDataset V2 (https://github.com/Azure/AzurePublicDataset/blob/master/AzurePublicDatasetV2.md) — 主数据源
- Google clusterdata-2019 (https://github.com/google/cluster-data/blob/master/ClusterData2019.md) — 辅助数据源

---

### 模块 3: 自研 LLM 控制器（核心研究贡献）

**文件**: `reasoning/query_planner.py`, `reasoning/cypher_generator.py`, `reasoning/subgraph_retriever.py`, `reasoning/causal_chain_ranker.py`, `reasoning/answer_synthesizer.py`, `reasoning/claim_decomposer.py`, `reasoning/hallucination_verifier.py`

**What**: 自研从自然语言查询到多跳因果路径抽取再到答案合成与幻觉核验的完整推理链路。Graphiti 仅提供存储与混合检索，推理全部自研。

**Why**: 这是考核核心，也是避免"外包核心推理"陷阱的关键。B2（Graphiti-default）作为对照 baseline 证明自研控制器的价值。时态感知的 Cypher 生成是创新点，用 `valid_at`/`invalid_at` 做时态剪枝，确保只返回查询时刻仍为真的因果链。

**How - 各子模块**:
- `query_planner.py`：LLM 把自然语言拆为 {起始实体、目标层、预期跳数、时间窗口}
- `cypher_generator.py`：生成带时态过滤的参数化 Cypher 多跳查询，核心是 `valid_at <= query_time AND (invalid_at IS NULL OR invalid_at >= query_time)` 时态剪枝 + `relationships(path)` 上的 lag_seconds 累加排序
- `subgraph_retriever.py`：双路召回——(a) Graphiti 内置 `COMBINED_HYBRID_SEARCH_CROSS_ENCODER` + `bfs_max_depth`；(b) 自研裸 Cypher 精确多跳（通过 `graphiti.driver.execute_query()`）。合并去重保留 provenance
- `causal_chain_ranker.py`：LLM 对候选路径打分，综合时态一致性、置信度、provenance 数量、文档先验，输出 Top-K 因果链
- `answer_synthesizer.py`：LLM 基于因果链生成自然语言答案
- `claim_decomposer.py`：把答案拆成原子陈述 `AtomicClaim`，每条标注 `hop_index` 与 `supporting_episodes`
- `hallucination_verifier.py`：对每个 claim 查询其 `supporting_episodes` 是否真存在且 episode 内容确实蕴含该 claim（用独立 LLM/NLI 模型判定，与生成 LLM 解耦），无支持则标记为对应跳数的幻觉

**可借鉴文献**:
- Think-on-Graph (ToG): Deep and Responsible Reasoning of Large Language Model on Knowledge Graph (ICLR 2024) — LLM 作为 KG 推理控制器范式
- From Local to Global: A Graph RAG Approach to Query-Focused Summarization (Microsoft, arXiv 2404.16130) — GraphRAG 方法论参考，虽发 arxiv 但影响力大且开源
- GraphEval: A Knowledge-Graph Based LLM Hallucination Evaluation Framework (arXiv 2407.10793) — 基于知识图谱的幻觉检测框架，claim 分解与核验思路
- KnowHalu: Hallucination Detection via Multi-Form Knowledge — 多形式知识核验的融合检测机制

---

### 模块 4: 实验评估系统

**文件**: `eval/testset_builder.py`, `eval/ground_truth_annotator.py`, `eval/metrics.py`, `eval/baselines/naive_rag.py`, `eval/baselines/graphiti_default.py`, `eval/baselines/no_temporal.py`, `eval/runner.py`, `eval/case_study.py`, `eval/reports/`

**What**: 独立于推理的量化评估系统，构造 2/3/4 跳测试集，定义精确指标，设置四组对照 baseline，产出实验报告与 case study。

**Why**: 这是考核的真正核心（"量化展示"是任务原文最显眼的要求）。四组 baseline 让每个设计决策都有可量化的贡献归因。逐跳幻觉率定义是"可追溯"要求的直接落点。

**How - 测试集构造**:
- 每跳数 100 条共 300 条，按因果类型（资源争抢/配置/依赖/网络）均衡分布
- 2 跳：Azure V2 单因故障（CPU spike→noisy neighbor→迁移 VM）
- 3 跳：K8s 典型链（OOMKilled→memory limit 低→配置错误→改 limit）
- 4 跳：复合故障（VM→延迟突增→上游依赖→网络分区→重调度）
- 每条标注：query、expected_path（实体+边+valid_at）、supporting_facts_per_hop

**How - 四组 Baseline**:
| Baseline | 检索 | 推理 | 时态 | 用途 |
|---|---|---|---|---|
| B1 Naive RAG | 纯向量检索 | LLM 直答 | 无 | 隔离"图谱"价值 |
| B2 Graphiti-default | Graphiti 内置 search+cross_encoder | Graphiti 内置 LLM | 有但不剪枝 | 隔离"自研控制器"价值 |
| B3 Graph-RAG-noTemporal | 自研 BFS | 自研控制器 | 无（消融） | 隔离"时态建模"价值 |
| B4 Full Graph-RAG | 自研 Cypher+BFS+时态剪枝 | 自研+provenance 核验 | 有 | 完整方法 |

**How - 六项评估指标（精确定义）**:
1. **路径错误率 PathErrorRate**：逐跳对齐，缺失/多余均计错误，按 2/3/4 跳分别报告
2. **幻觉率 HallucinationRate**：可追溯核心创新。`supported(c)=1 if ∃ episode 蕴含 c else 0`；整体幻觉率 = 无支持陈述数/总陈述数；逐跳幻觉率 = 第 k 跳无支持陈述数/第 k 跳陈述数
3. **覆盖率 Recall**：|predicted ∩ ground_truth| / |ground_truth|
4. **精确率 Precision**：|predicted ∩ ground_truth| / |predicted|
5. **时态准确率 TemporalAccuracy**：预测路径所有边 valid_at ≤ query_time ≤ invalid_at 的比例
6. **provenance 完备率**：有 episode 的预测边占比

**How - Case Study**：选 5 个代表性 case（2 跳单因、3 跳配置链、4 跳复合、时态对比、幻觉定位），每个产出四套 baseline 预测路径对比 + 逐跳 claim 表 + 幻觉标记 + provenance 溯源链，导出给 G6 演示。

**How - 评估独立性**：生成 LLM 与核验 LLM 必须不同实例/provider（如生成 GPT-4o，核验 Claude 或本地 NLI），避免自评偏置；ground truth 标注与系统开发双盲。

**可借鉴文献**:
- MuSiQue: Multihop Questions via Single-hop Question Composition (TACL) — 多跳问答 supporting facts 逐跳标注思想
- HotpotQA: Learning to Explain Multi-hop Questions (EMNLP 2018) — 多跳推理评估与 supporting facts 范式
- A Survey on Hallucination in LLM — 幻觉评估基准与检测基准的区分

---

### 模块 5: API 后端

**文件**: `api/main.py`, `api/routes/query.py`, `api/routes/graph.py`, `api/routes/eval.py`, `api/schemas.py`

**What**: FastAPI 提供推理查询、子图导出（给 G6）、评估触发三个接口。

**Why**: 后端系统是交付形态要求，同时为 G6 可视化提供数据源。流式返回评估进度体现工程严谨。

**How**:
- `POST /query` → `{answer, causal_path, claims, hallucination_report}`
- `GET /graph?case_id=` → 导出 G6 GraphData（节点+边+时态+provenance）
- `POST /eval` → 触发实验，流式返回进度

---

### 模块 6: G6 可视化前端

**文件**: `frontend/` 目录（Vue3 + @antv/g6@5 + TypeScript），含 `CausalGraph.vue`, `TemporalSlider.vue`, `HopHighlighter.vue`, `ProvenancePanel.vue`, `HallucinationBadge.vue` 等组件

**What**: 把逐跳推理过程、时态查询差异、幻觉定位三个研究贡献全部可视化，让 G6 成为论证工具而非装饰。

**Why**: 最大化用户 G6 长处。可视化不只做静态展示，而是把实验指标（错误率、幻觉率、时态准确率）变成可交互论证，直接服务答辩演示。

**How - 六大可视化设计**:
1. **四层分层布局**：dagre 或自定义 layered 布局，按 layer 字段强制四层水平带状，颜色编码（component 蓝/symptom 红/cause 橙/solution 绿），节点形状区分（矩形/菱形/椭圆双线边框根因/圆角矩形）
2. **时态因果路径动态渲染**：边 fact 作 label，hover 显示 valid_at~invalid_at 时间窗，lag_seconds 用边粗细编码
3. **逐跳高亮动画**（打动导师的亮点）：播放按钮触发，从症状节点出发按 hop 顺序逐个点亮，每 hop 0.8s 同步展示该跳 claim/supporting episode/是否幻觉，用 `graph.focusElement()` + `setItemState('highlight')` 实现
4. **时间轴滑块**：拖动改变 query_time，前端过滤 valid_at ≤ t AND invalid_at ≥ t 的边，实时展示"同一查询不同时刻因果链差异"——时态图谱区别于静态图谱的可视化论证
5. **Provenance 溯源侧栏**：点击节点/边展示其 episodes（原始 trace 片段或文档原文）
6. **幻觉标记徽标**：幻觉 claim 所在节点/边加红色脉冲光晕，hover 展示判定依据——实验指标的可视化呈现

---

### 模块 7: 工程化基础设施

**文件**: `pyproject.toml`, `.env.example`, `scripts/setup_neo4j.ps1`, `scripts/bootstrap_graph.py`, `scripts/run_eval.py`, `scripts/export_case_study.py`, `tests/`

**What**: uv 依赖管理、Neo4j 原生安装脚本、bootstrap/eval 脚本、单元测试。全程无 Docker。

**Why**: 用户未学过 Docker，采用 Windows 原生方案降低部署门槛。Neo4j Community Server 在 Windows 下只需 JDK + 解压即用，比 Docker Desktop 更省内存（Neo4j 占 1-2GB vs Docker Desktop 常驻 4-6GB）。一键脚本保障研究可复现性。

**How - Neo4j 原生安装步骤**（写入 `scripts/setup_neo4j.ps1` 一键执行）:
1. 检测/安装 JDK 17：从 Oracle 官网下载，`winget install Microsoft.OpenJDK.17` 或手动安装
2. 下载 Neo4j Community Server Windows zip：从 https://neo4j.com/download-center/#community 获取
3. 解压到项目同级 `neo4j/` 目录（路径不含中文）
4. 配置环境变量 `NEO4J_HOME`，`bin` 加入 PATH
5. 管理员运行 `neo4j.bat console` 启动，浏览器访问 `http://localhost:7474`，默认密码 `neo4j` 首次登录修改
6. Graphiti 连接配置：`uri="bolt://localhost:7687"`, `user="neo4j"`, `password="你的密码"`

**How - 项目依赖管理**：`pyproject.toml` 用 uv（Graphiti 官方已迁移）；后端直接 `python -m uvicorn api.main:app` 跑；前端 `npm run dev` 跑；`tests/` 覆盖 schema 约束、Cypher 生成、幻觉核验三个关键单元。

**How - Graphiti 初始化代码示例**:
```python
from graphiti_core import Graphiti

graphiti = Graphiti(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="你的密码",
)
await graphiti.build_index()  # 首次运行建索引
```

## Assumptions & Decisions

### 已确定的三大决策（用户确认）
1. **Zep 融合度**：Graphiti 开源版做时态因果图谱层 + 自研 LLM 控制器做查询生成与多跳路径抽取。Graphiti 只到 `subgraph_retriever` 为止，`reasoning/` 全部自研。B2（Graphiti-default）作为对照证明自研价值。**不使用 Zep Cloud 托管 API**，避免外包核心推理。
2. **数据来源**：Azure Public Dataset V2 作主数据（可下载、有 CPU 时序与 VM 删除事件），Google clusterdata-2019 作补充（alloc set 父子关系），K8s/Prometheus 运维文档转因果骨架种子。
3. **交付形态**：后端推理系统 + 严谨实验报告 + G6 因果路径动态可视化。不做完整 Web 应用（避免工作量分散稀释实验核心）。

### 关键假设
- LLM 选型：生成用 GPT-4o（或同等级闭源 API），核验用 Claude 或本地 NLI 模型（必须与生成解耦）。评估可复现性要求记录 LLM 版本与 prompt。
- 数据规模：Azure V2 抽样 1 区域 × 7 天 × 故障率 top VM 子集（约 2-5 万 VM），保证故障事件密度足够构造 300 条测试集。不加载全量 235GB。
- 图数据库：采用 Neo4j Community Server（Windows 原生安装，无需 Docker）。安装步骤为 JDK 17 + 解压 Neo4j zip + `neo4j.bat console` 启动。Neo4j 是 Graphiti 官方推荐后端，生态成熟、Cypher 支持完整、自带 Web 管理界面（localhost:7474）。不用 FalkorDB（避免 Docker 依赖），不用已 deprecated 的 Kuzu。
- group_id 分区：每个故障场景一个 group_id，检索时传 `group_ids=[case_group]`，既隔离实验又加速查询。

### 对导师的核心说服点
1. **研究问题清晰**：不是"做故障诊断产品"，而是"量化 Graph-RAG 在多跳因果推理中的错误率与幻觉率，并证明时态建模与自研推理控制器的边际价值"。四组对照让每个设计决策都有可量化贡献归因。
2. **幻觉可追溯**：不是笼统幻觉率，而是分解到跳数，用 provenance 逐跳核验——比 Zep Analytics 聚合指标更贴近研究规范，直接回应"Analytics 指标当实验指标"陷阱。
3. **时态不是噱头**：时态准确率指标 + 时态对比 case + G6 时间轴滑块，三重证明 valid_at/invalid_at 实际价值。
4. **工程与可视化深度**：G6 逐跳动画 + provenance 溯源 + 幻觉光晕，把抽象"多跳因果链"变成可交互论证工具。

## 可借鉴的学术文献

### Graph-RAG / 知识图谱增强生成（核心方法论）
| 文献 | 发表venue | 借鉴点 |
|---|---|---|
| From Local to Global: A Graph RAG Approach to Query-Focused Summarization (Microsoft) | arXiv 2404.16130, 2024 | GraphRAG 方法论参考，开源 github.com/microsoft/graphrag。虽发 arxiv 但微软出品影响力大，是领域奠基性工作 |
| Think-on-Graph (ToG): Deep and Responsible Reasoning of LLM on KG | ICLR 2024 | LLM 作为 KG 推理控制器范式，自研控制器的理论基础 |
| Medical Graph RAG: Towards Safe Medical LLM via Graph RAG | ACL 2025 | 领域 GraphRAG 的对照实验设计参考，刷新问答准确性记录 |
| GraphRAG: A Survey | 综述 | 领域全貌与方法分类 |

### 时态知识图谱（时态建模依据）
| 文献 | 发表venue | 借鉴点 |
|---|---|---|
| Zep: A Temporal Knowledge Graph Architecture for Agent Memory | arXiv 2501.13956, 2025 | bi-temporal 模型（valid_at/invalid_at/expired_at）设计依据，Graphiti 的论文 |
| Temporal Knowledge Graph Reasoning with Historical Contrastive Learning | AAAI 2023 | 时序因果推理方法参考 |
| DaeMon: DAptivE path-MemOry Network for TKG Reasoning | 顶会 | 时序知识图谱路径记忆推理 |

### 幻觉检测与评估（评估方法依据）
| 文献 | 发表venue | 借鉴点 |
|---|---|---|
| GraphEval: A Knowledge-Graph Based LLM Hallucination Evaluation Framework | arXiv 2407.10793 | 基于知识图谱的幻觉检测，claim 分解与核验思路 |
| KnowHalu: Hallucination Detection via Multi-Form Knowledge | arXiv | 多形式知识核验的融合检测机制 |
| A Survey on Hallucination in LLM | 综述 | 幻觉评估基准与检测基准的区分 |

### 多跳推理评估（测试集构造依据）
| 文献 | 发表venue | 借鉴点 |
|---|---|---|
| MuSiQue: Multihop Questions via Single-hop Question Composition | TACL | 多跳问答 supporting facts 逐跳标注思想，幻觉逐跳核验的方法源头 |
| HotpotQA: Learning to Explain Multi-hop Questions | EMNLP 2018 | 多跳推理评估与 supporting facts 范式 |
| MINTQA: A Multi-Hop QA Benchmark | arXiv 2412.17032 | 多跳问答评估基准 |

### 数据源（无 venue，公开数据集）
| 数据集 | 来源 | 用途 |
|---|---|---|
| AzurePublicDataset V2 | github.com/Azure/AzurePublicDataset | 主数据源，VM CPU 时序与删除事件 |
| Google clusterdata-2019 | github.com/google/cluster-data | 辅助数据源，alloc set 父子关系补充因果骨架 |

## Verification Steps

1. **Schema 正确性验证**：运行 `tests/test_schema_constraints.py`，确认非法边（如 Solution→Symptom）被 Pydantic validator 拒绝，四层路径模板可构造
2. **图谱构建验证**：运行 `scripts/bootstrap_graph.py`，确认实体数 ≥ 50、关系数 ≥ 100，覆盖四层概念，时态字段（valid_at/invalid_at）正确填充
3. **Cypher 生成验证**：运行 `tests/test_cypher_generator.py`，确认 2/3/4 跳查询语法正确且时态过滤生效，`bfs_max_depth` 旋钮可控
4. **幻觉核验验证**：运行 `tests/test_hallucination_verifier.py`，确认无 supporting episode 的 claim 被正确标记，逐跳幻觉率可计算
5. **对照实验验证**：运行 `scripts/run_eval.py`，确认 B1/B2/B3/B4 四组都能跑通，六项指标 × 三跳数 × 四 baseline 表格完整，B4 在 4 跳场景幻觉率显著低于 B1/B2/B3
6. **评估独立性验证**：确认生成 LLM 与核验 LLM 不同实例/provider，ground truth 标注与系统输出双盲
7. **可视化验证**：启动前端，确认四层分层布局、逐跳高亮动画、时间轴滑块过滤、provenance 溯源、幻觉光晕五个交互均可用，case study 数据可加载演示
8. **可复现性验证**：从空目录出发，按 README 步骤（`scripts/setup_neo4j.ps1` 启动 Neo4j → `bootstrap_graph.py` 建图 → `run_eval.py` 跑评估）一键跑通全流程

## 实现依赖顺序（非时间线）

schema 定义 → 数据接入与 episode 构建 → 子图检索与 Cypher 生成 → 答案合成与 claim 分解 → 幻觉核验 → 测试集标注 → 评估运行 → G6 可视化 → case study

其中 schema 与测试集标注可并行启动，因为标注依赖的 ground truth 格式由 schema 决定。
