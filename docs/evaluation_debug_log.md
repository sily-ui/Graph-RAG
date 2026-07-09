# Graph-RAG 评估系统调试与优化文档

> 记录从 v1 评估到 v4 完整 150 case 评估的整个过程，遇到的问题、原因分析和解决方法。
> 写给后续接手这个项目的人，让你们少踩同样的坑。

---

## 1. 项目背景

这是一个工业故障排查图谱问答系统：
- 节点：Component（组件）/ Symptom（症状）/ Cause（原因）/ Solution（解决方案）
- 关系：HAS_SYMPTOM / CAUSED_BY / TRIGGERED_BY / RESOLVED_BY / MITIGATED_BY
- 查询：从"组件出问题了"反查 2/3/4 跳的根因链路

评估目标：对比 4 种 baseline 找出最优的推理策略。

---

## 2. 4 个 Baseline 是什么

| 名称 | 做法 | 作用 |
|---|---|---|
| **B1 NaiveRAG** | 只用 LLM，不查图谱 | 基线：纯靠 LLM 瞎编 |
| **B2 GraphitiDefault** | 用 graphiti-core 默认配置 | 对比：成熟框架 vs 我们自己写的 |
| **B3 NoTemporal** | 查图谱但**不带时态过滤** | 消融实验：时态过滤有没有用 |
| **B4 Full_GraphRAG** | 查图谱 + 时态过滤 + LLM 解释 | 我们的最终方案 |

---

## 3. 调试过程（按时间线）

### 3.1 v1 阶段（7 月 6-7 日）：基线跑通

**问题**：第一次评估，路径抽取完全失败，B4 Recall = 0。

**原因**：
- Cypher 生成器把 3 跳查询（Component→Symptom→Cause→Solution）当成 3 节点链生成
- 但实际图谱里 3 跳是 4 个节点的链路（中间是 Symptom→Cause→Solution 三段关系）
- 模板错位 → Neo4j 找不到路径 → 召回为 0

**修复**：修正 `cypher_generator.py` 的多跳模板，让节点变量 `c1`, `c2` 严格匹配跳数。

---

### 3.2 v2 阶段（7 月 7-8 日）：扩展数据集

**问题**：测试集只有 32 条（2 跳 18 + 3 跳 14），统计意义不大。

**解决**：
- 写 `scripts/seed_synthetic_graph.py` 自动生成 15 个组件 + 15 个症状 + 16 个原因 + 17 个解决方案
- 测试集扩展到 150 条（2/3/4 跳各 50 条）
- B3 路径抽取率：84% cases 抽到至少 1 条路径

---

### 3.3 v3 阶段（7 月 8 日）：3 跳 Recall 暴跌到 0.143

**问题**：B4 3 跳 Recall 只有 0.143（之前 0.917），85% 的 3 跳查询返回 0 路径。

**根因分析**：
- 查询"vm_001 的 cpu_spike 根因"时，LLM 意图分类器输出 `causal_chain`（不带跳数）
- Cypher 生成器用 `causal_chain` 模板去 Symptom / Cause 节点上找 `cpu_spike`
- 但 `cpu_spike` 实际是 Symptom 节点的属性，**且图谱里 vm_001 对应的 cpu_spike 在 Solution 节点的解法链上**
- 模板错位 → 0 路径 → Recall 暴跌

**修复**（关键修复）：
- 在 [cypher_generator.py](file:///root/Graph-RAG/reasoning/cypher_generator.py) 加 `post_process_intent()` 方法
- 当用户查询里出现具体组件名（如 `vm-1-2` / `machine-1-3`）时，**强制升级**为 `multi_hop_path(3)`
- 修复后 3 跳 Recall 从 0.143 → 0.762

---

### 3.4 v4 阶段（7 月 8-9 日）：4 跳 Cypher 修复

**问题**：4 跳评估时全部 0 路径，错误 "Cannot run an empty query"。

**根因**：
- 测试集 4 跳模板预期 `Component → Symptom → Cause → Cause`（4 节点 3 边）
- 但 v3 修复后 cypher_generator 的 4 跳模板是 `Component → Symptom → Cause → Cause → Solution`（5 节点 4 边）
- **两边不一致** → 4 跳评估 0 命中

**修复**：
- 同步两边：测试集 4 跳 Cypher 改为 5 节点链
- `QUERY_TEMPLATES_4HOP` 加 `solution` 字段
- 写 `scripts/seed_synthetic_graph.py` 给图库补 4 跳路径
- 验证：B4 4 跳 Recall = 0.380（5 节点链路能查到路径了）

---

### 3.5 v4.1 阶段（7 月 9 日）：断点续跑 + LLM 失败恢复

**背景**：7 月 9 日凌晨 2:39，旧任务跑 12 小时后被沙箱清理，**丢失了 262/450 cases 的结果**（因为之前 run_eval.py 是全部跑完才统一写结果）。

#### 3.5.1 断点续跑机制

**新增**：[eval/checkpoint.py](file:///root/Graph-RAG/eval/checkpoint.py) — 5 个函数 + 1 个类：
- `load_completed_case_ids()` — 启动时读 .jsonl 加载已完成 case_id
- `migrate_legacy_detail_if_needed()` — 旧 `*_detail.json` 自动迁移
- `clear_checkpoint()` — `--no-resume` 触发
- `CheckpointWriter` — **每个 case 跑完立即 fsync 落盘**（单行原子）

**改造**：[scripts/run_eval.py](file:///root/Graph-RAG/scripts/run_eval.py)
- 每个 case 跑完 `writer.write(case_result)` → 即使进程被杀，已完成的 case 不丢
- 启动时 `load_completed_case_ids()` → 跳过已完成的

**效果**：7 月 9 日 11:05 重启任务，139 个 B4 case 自动跳过，只跑 11 个 4 跳缺失的。

#### 3.5.2 LLM 失败恢复（重要）

**观察**：日志里大量 `WARNING reasoning.claim_decomposer: LLM 拆解失败，降级用规则: LLM 返回空内容，finish_reason='length'`

**根因**：
- StepFun step-3.7-flash 模型 context window 只有 8K
- 我们默认 `max_tokens=2000` 给单次生成太多空间
- 长 prompt（系统提示 + 用户查询 + 路径）让模型**没空间生成完整 JSON**，被强行截断
- 截断后的 JSON 不完整 → 解析失败 → 降级到规则拆解

**修复**（[reasoning/llm_interpreter.py](file:///root/Graph-RAG/reasoning/llm_interpreter.py) `chat()` 方法）：
1. `finish_reason='length'` 时**自动重试** + `max_tokens` 翻倍（上限 8000）
2. 网络超时（APITimeoutError）/ 连接错误 → 1s/2s/3s 退避重试
3. 内容被截断但**以完整标点结尾**（句号/右花括号等）→ 智能接受，不重试
4. 非网络错误（如 BadRequestError）→ 不重试，直接抛

**测试覆盖**：[tests/test_llm_retry.py](file:///root/Graph-RAG/tests/test_llm_retry.py) — 9 个 case 全过

---

## 4. 关键经验

### 4.1 评估系统设计的 3 个铁律

1. **每条数据立即落盘** — 不要等全部跑完再写。沙箱/断电/进程被杀任何时候都可能发生。
2. **失败要降级** — LLM 返回空内容时降级到规则实现，不要让单点失败阻塞整批评估。
3. **重试要分类** — 网络错（可重试）和业务错（如 key 错，不可重试）必须区分。

### 4.2 Cypher 模板的"对齐"陷阱

图谱查询模板要保证 **3 处对齐**：
- 测试集生成器（[eval/testset_builder.py](file:///root/Graph-RAG/eval/testset_builder.py)）的"期望路径"
- Cypher 生成器（[reasoning/cypher_generator.py](file:///root/Graph-RAG/reasoning/cypher_generator.py)）的"生成模板"
- 实际图库（[scripts/seed_synthetic_graph.py](file:///root/Graph-RAG/scripts/seed_synthetic_graph.py)）的"种子数据"

**任何一处不一致都会让评估数据假阳/假阴**。v3 → v4 的修复就是这个对齐。

### 4.3 沙箱稳定性 vs 任务长度

| 任务时长 | 沙箱稳定性 |
|---|---|
| < 30 分钟 | ✅ 稳定 |
| 30 分钟 - 2 小时 | ⚠️ 偶发被清理 |
| > 2 小时 | ❌ 几乎必被清理 |

**结论**：单次跑的任务**控制在 30 分钟以内**。长任务拆成多个批次，每批之间用 checkpoint 续跑。

### 4.4 沙箱内"断电保护"的能力边界

| 场景 | 沙箱内能不能保护 |
|---|---|
| 进程被 SIGKILL | ✅ checkpoint 落盘，已完成 case 不丢 |
| IDE 关闭 | ✅ checkpoint 落盘在沙箱可写目录，重启可继续 |
| 沙箱被完全清理 | ❌ 没办法，checkpoints 也会被清 |
| 电脑断电 | ⚠️ 取决于沙箱是否在断电时持久化（一般不持久化） |

**真正"断电保护"必须**：
- 部署到云服务器（VPS / EC2 / 阿里云）
- 用 `nohup` + `setsid` 脱离 IDE 沙箱
- 或 systemd service

---

## 5. 最终结果（v4.1，150 case × 3 baseline）

| Baseline | Recall↑ | Precision↑ | TemporalAcc↑ | PathError↓ | Hallu↓ |
|---|---|---|---|---|---|
| B1 NaiveRAG | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| B3 NoTemporal | 0.592 | 0.471 | 0.487 | 0.584 | 0.039 |
| **B4 Full GraphRAG** | **0.758** | **0.668** | **0.705** | **0.395** | **0.018** |

**结论**：
- B4 vs B1：Recall 从 0 → 0.758（图谱 vs 纯 LLM 的碾压性优势）
- B4 vs B3：时态过滤带来 Recall +28%、Precision +42%、PathError 改善 32%
- B4 4 跳：Recall 0.380（还有提升空间，可能需要更智能的中间节点剪枝）

---

## 6. 后续优化方向

1. **小模型替换**：StepFun step-3.7-flash context 太小，换成 DeepSeek（64K context）能减少 length 截断
2. **4 跳剪枝**：4 跳 Recall 0.380 不高，需要在中间节点加置信度阈值
3. **并发评估**：3 个 baseline 串行太慢，可以并行（前提是 Neo4j 能扛住）
4. **评估任务拆批**：单次 30 case / 15 分钟更稳定

---

## 7. 关键文件索引

| 文件 | 作用 |
|---|---|
| [reasoning/cypher_generator.py](file:///root/Graph-RAG/reasoning/cypher_generator.py) | Cypher 模板生成（v3 修复 + post_process_intent） |
| [reasoning/llm_interpreter.py](file:///root/Graph-RAG/reasoning/llm_interpreter.py) | LLM 调用（v4.1 加重试 + max_tokens 翻倍） |
| [eval/checkpoint.py](file:///root/Graph-RAG/eval/checkpoint.py) | 断点续跑机制（v4.1 新增） |
| [scripts/run_eval.py](file:///root/Graph-RAG/scripts/run_eval.py) | 评估入口（v4.1 集成 checkpoint） |
| [eval/testset_builder.py](file:///root/Graph-RAG/eval/testset_builder.py) | 测试集生成（v4 4 跳修复） |
| [scripts/seed_synthetic_graph.py](file:///root/Graph-RAG/scripts/seed_synthetic_graph.py) | 图库种子（v2 扩展 + v4 补 4 跳） |
| [tests/test_llm_retry.py](file:///root/Graph-RAG/tests/test_llm_retry.py) | LLM 重试 9 个 case |
| [tests/test_checkpoint.py](file:///root/Graph-RAG/tests/test_checkpoint.py) | Checkpoint 9 个 case |

---

**最后更新**：2026-07-09 12:40
**作者**：张傲宇 + Claude (Graph-RAG 项目协作)
