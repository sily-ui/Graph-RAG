"""测试集构造器 —— 从图库实际路径反向构造 2/3/4 跳评估查询。

本模块是 Graph-RAG 评估系统（模块 4）的第一阶段。

设计原则（v2，2026-07-08 重构）：
- **图库驱动**：从 Neo4j 实际查到的路径反向构造 TestCase，确保 expected_path
  和图库真实节点严格对齐。这避免了之前用模板凭空生成 query 但 entity 在图库里
  不存在导致 Recall=0 的问题。
- **query 多样化**：每条图库路径通过多种自然语言问法（"X 上 Y 异常的根因"、
  "X 的 2 跳链路"、"Y 故障怎么解决"等）扩展为多条 case，扩大测试规模。
- **时态对齐**：query_time 取自路径首跳边的 valid_at（保证 TemporalAcc 评估有意义）。

测试集字段（每条 case）：
    case_id:          唯一 ID（如 "graph_2hop_001"）
    domain:           "graph"（统一域，数据来自图库）
    hop_count:        2 | 3 | 4
    query:            自然语言查询
    expected_path:    期望的实体+边序列（直接来自图库查询结果）
    supporting_facts: 逐跳支撑事实（边 fact 字段）
    query_time:       时态查询时刻（路径首跳 valid_at + 1s，保证在边有效期内）
    ground_truth_free_text: 期望的标准答案（节点链 + 边名）
    question_type:    题型（"OE" / "MC" / "MS" / "TF" / "FB"）— GraphRAG-Bench §3.1 借鉴
    task_level:       任务分级（"Fact_Retrieval" / "Complex_Reasoning" /
                                  "Contextual_Summarize" / "Creative_Generation"）
                      —— arXiv:2506.05690 (When to use Graphs in RAG) Table 1 借鉴

跳数定义：
    2 跳：Symptom → Cause → Solution
    3 跳：Component → Symptom → Cause → Solution
    4 跳：Component → Symptom → Cause → Cause → Solution
"""
from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ============================================================
#  测试集数据模型
# ============================================================

@dataclass
class ExpectedHop:
    """单跳的 ground truth。"""
    edge_name: str                    # 边名（CAUSED_BY / RESOLVED_BY / HAS_SYMPTOM 等）
    source_name: str                  # 源实体 name
    source_label: str                 # 源 label（Symptom / Cause / Component）
    target_name: str                  # 目标实体 name
    target_label: str                 # 目标 label
    valid_at: str | None = None       # 边 valid_at（ISO 字符串，便于 JSON 序列化）
    invalid_at: str | None = None
    lag_seconds: int = 0


@dataclass
class SupportingFact:
    """单跳的支撑事实（provenance）。"""
    hop_index: int
    source: str                       # 事实来源（"smd_interpretation" / "micross_run_log" / "trace_span"）
    text: str                         # 事实描述（如 "machine-1-3 第 2 维 memory_used_rate 在 t=15849 起异常"）
    reference: str                    # 引用（如 "machine-1-3.txt line 15849"）


@dataclass
class TestCase:
    """单条测试用例。

    借鉴自 GraphRAG-Bench（arXiv:2506.02404）§3.1 多题型设计：
        - question_type: 题型（OE/MC/MS/TF/FB），默认 OE
        - task_level: 任务分级（arXiv:2506.05690 Table 1），默认 Complex_Reasoning

    两个字段都带默认值，老 JSONL 文件（无这两个字段）能正常 load。
    """
    case_id: str
    domain: str
    hop_count: int
    query: str
    expected_path: list[ExpectedHop]
    supporting_facts: list[SupportingFact]
    query_time: str                   # ISO
    ground_truth_free_text: str
    metadata: dict = field(default_factory=dict)
    # GraphRAG-Bench 借鉴字段（带默认值 → 向后兼容老 JSONL）
    question_type: str = "OE"
    task_level: str = "Complex_Reasoning"


# ============================================================
#  模板常量
# ============================================================

# 2 跳模板：SMD 单机多维异常
# 每条模板独立采一种 (metric, symptom, cause, solution) 组合，覆盖常见故障模式
SMD_2HOP_TEMPLATES = [
    {
        "metric": "cpu_user_rate",
        "metric_idx": 0,
        "symptom": "cpu_spike",
        "cause": "resource_contention",
        "solution": "scale_up",
        "lag": 300,
        "query_template": "{machine} 上 {metric} 异常的 2 跳根因链路",
    },
    {
        "metric": "cpu_system_rate",
        "metric_idx": 1,
        "symptom": "cpu_spike",
        "cause": "resource_contention",
        "solution": "scale_up",
        "lag": 300,
        "query_template": "{machine} 上 {metric} 飙升的 2 跳根因链路",
    },
    {
        "metric": "memory_used_rate",
        "metric_idx": 3,
        "symptom": "memory_pressure",
        "cause": "resource_contention",
        "solution": "scale_up",
        "lag": 0,
        "query_template": "{machine} 上 {metric} 飙升的根因链路是什么？",
    },
    {
        "metric": "memory_used_rate",
        "metric_idx": 3,
        "symptom": "memory_pressure",
        "cause": "noisy_neighbor",
        "solution": "drain_node",
        "lag": 600,
        "query_template": "{machine} 上 {metric} 异常 → noisy_neighbor → drain_node 链路",
    },
    {
        "metric": "net_in_bytes",
        "metric_idx": 6,
        "symptom": "network_saturation",
        "cause": "network_partition",
        "solution": "runbook-azure/network-partition",
        "lag": 0,
        "query_template": "{machine} 上 {metric} 异常 → 网络分区 → 解法的链路",
    },
    {
        "metric": "net_out_bytes",
        "metric_idx": 7,
        "symptom": "network_saturation",
        "cause": "network_partition",
        "solution": "runbook-azure/network-partition",
        "lag": 0,
        "query_template": "{machine} 出方向 {metric} 异常 → 网络分区 → runbook",
    },
    {
        "metric": "memory_cached",
        "metric_idx": 5,
        "symptom": "memory_pressure",
        "cause": "memory_leak",
        "solution": "restart_service",
        "lag": 120,
        "query_template": "{machine} 上 {metric} 异常 → 内存泄漏 → 重启服务的链路",
    },
    {
        "metric": "dim_11",
        "metric_idx": 11,
        "symptom": "cpu_spike",
        "cause": "resource_contention",
        "solution": "scale_up",
        "lag": 300,
        "query_template": "{machine} 上 {metric} 飙升的 2 跳根因链路",
    },
    {
        "metric": "dim_16",
        "metric_idx": 16,
        "symptom": "disk_io_saturation",
        "cause": "disk_io_contention",
        "solution": "scale_up",
        "lag": 240,
        "query_template": "{machine} 上 {metric} 异常 → 磁盘 IO 争用 → 扩容的链路",
    },
    {
        "metric": "dim_19",
        "metric_idx": 19,
        "symptom": "cpu_spike",
        "cause": "resource_contention",
        "solution": "scale_up",
        "lag": 300,
        "query_template": "{machine} 第 {idx} 维 {metric} 异常的 2 跳链路",
    },
]

# 3 跳模板：MicroSS 故障注入链
MICROSS_3HOP_TEMPLATES = [
    {
        "injection_type": "memory_anomalies",
        "service": "dbservice1",
        "symptom": "latency_ms",
        "cause_mid": "oom_killed",
        "cause_root": "misconfiguration",
        "solution": "runbook-azure/dependency-failure",
        "lag_mid": 60,
        "lag_root": 120,
        "query_template": "{service} 上 latency_ms 飙升的 3 跳根因链路（上游 OOMKilled → 误配置 → 解法）",
    },
    {
        "injection_type": "cpu_anomalies",
        "service": "redisservice1",
        "symptom": "latency_ms",
        "cause_mid": "cpu_throttling",
        "cause_root": "resource_contention",
        "solution": "scale_up",
        "lag_mid": 90,
        "lag_root": 180,
        "query_template": "{service} 延迟异常的 3 跳根因链路（CPU 节流 → 资源争用 → 扩容）",
    },
    {
        "injection_type": "network_anomalies",
        "service": "mongodb1",
        "symptom": "latency_ms",
        "cause_mid": "network_partition",
        "cause_root": "dependency_failure",
        "solution": "runbook-azure/network-partition",
        "lag_mid": 30,
        "lag_root": 60,
        "query_template": "{service} 延迟 → 网络分区 → 依赖失败的 3 跳链路",
    },
    {
        "injection_type": "memory_anomalies",
        "service": "mobservice1",
        "symptom": "latency_ms",
        "cause_mid": "oom_killed",
        "cause_root": "resource_contention",
        "solution": "scale_up",
        "lag_mid": 60,
        "lag_root": 150,
        "query_template": "{service} 内存 OOM → 资源争用 → 扩容的 3 跳链路",
    },
    {
        "injection_type": "cpu_anomalies",
        "service": "webservice1",
        "symptom": "latency_ms",
        "cause_mid": "cpu_throttling",
        "cause_root": "dependency_failure",
        "solution": "runbook-azure/dependency-failure",
        "lag_mid": 90,
        "lag_root": 120,
        "query_template": "{service} CPU 节流 → 依赖失败 → runbook 的 3 跳链路",
    },
    {
        "injection_type": "memory_anomalies",
        "service": "logservice1",
        "symptom": "latency_ms",
        "cause_mid": "oom_killed",
        "cause_root": "misconfiguration",
        "solution": "runbook-azure/dependency-failure",
        "lag_mid": 60,
        "lag_root": 120,
        "query_template": "{service} OOM → 误配置 → runbook 的 3 跳链路",
    },
    {
        "injection_type": "network_anomalies",
        "service": "redisservice2",
        "symptom": "latency_ms",
        "cause_mid": "network_partition",
        "cause_root": "resource_contention",
        "solution": "scale_up",
        "lag_mid": 30,
        "lag_root": 90,
        "query_template": "{service} 网络分区 → 资源争用 → 扩容的 3 跳链路",
    },
    {
        "injection_type": "cpu_anomalies",
        "service": "dbservice2",
        "symptom": "latency_ms",
        "cause_mid": "cpu_throttling",
        "cause_root": "misconfiguration",
        "solution": "runbook-azure/dependency-failure",
        "lag_mid": 90,
        "lag_root": 120,
        "query_template": "{service} CPU 异常 → 误配置 → runbook 的 3 跳链路",
    },
]

# 4 跳模板：跨域复合故障
CROSS_4HOP_TEMPLATES = [
    {
        "injection_type": "cpu_anomalies",
        "service": "redisservice1",
        "component_metric": "cpu_user_rate",
        "symptom_1": "cpu_spike",
        "cause_1": "resource_contention",
        "cause_2": "dependency_failure",
        "cause_root_label": "dependency_failure",
        "solution": "scale_up",
        "query_template": "{service} CPU 飙升 → 资源争用 → 依赖失败 → 扩容的 4 跳链路",
    },
    {
        "injection_type": "memory_anomalies",
        "service": "dbservice1",
        "component_metric": "memory_used_rate",
        "symptom_1": "memory_pressure",
        "cause_1": "noisy_neighbor",
        "cause_2": "resource_contention",
        "cause_root_label": "resource_contention",
        "solution": "drain_node",
        "query_template": "{service} 内存压力 → noisy_neighbor → 资源争用 → drain_node 的 4 跳链路",
    },
    {
        "injection_type": "memory_anomalies",
        "service": "mobservice1",
        "component_metric": "memory_used_rate",
        "symptom_1": "memory_pressure",
        "cause_1": "oom_killed",
        "cause_2": "dependency_failure",
        "cause_root_label": "dependency_failure",
        "solution": "runbook-azure/dependency-failure",
        "query_template": "{service} 内存 OOM → 依赖失败 → runbook 的 4 跳链路",
    },
    {
        "injection_type": "cpu_anomalies",
        "service": "webservice1",
        "component_metric": "cpu_user_rate",
        "symptom_1": "cpu_spike",
        "cause_1": "cpu_throttling",
        "cause_2": "resource_contention",
        "cause_root_label": "resource_contention",
        "solution": "scale_up",
        "query_template": "{service} CPU 节流 → 资源争用 → 扩容的 4 跳链路",
    },
    {
        "injection_type": "network_anomalies",
        "service": "mongodb1",
        "component_metric": "net_in_bytes",
        "symptom_1": "network_saturation",
        "cause_1": "network_partition",
        "cause_2": "dependency_failure",
        "cause_root_label": "dependency_failure",
        "solution": "runbook-azure/network-partition",
        "query_template": "{service} 网络分区 → 依赖失败 → runbook 的 4 跳链路",
    },
    {
        "injection_type": "memory_anomalies",
        "service": "logservice1",
        "component_metric": "memory_used_rate",
        "symptom_1": "memory_pressure",
        "cause_1": "memory_leak",
        "cause_2": "resource_contention",
        "cause_root_label": "resource_contention",
        "solution": "scale_up",
        "query_template": "{service} 内存泄漏 → 资源争用 → 扩容的 4 跳链路",
    },
    {
        "injection_type": "cpu_anomalies",
        "service": "redisservice2",
        "component_metric": "cpu_user_rate",
        "symptom_1": "cpu_spike",
        "cause_1": "cpu_throttling",
        "cause_2": "dependency_failure",
        "cause_root_label": "dependency_failure",
        "solution": "runbook-azure/dependency-failure",
        "query_template": "{service} CPU 异常 → 依赖失败 → runbook 的 4 跳链路",
    },
    {
        "injection_type": "memory_anomalies",
        "service": "dbservice2",
        "component_metric": "memory_used_rate",
        "symptom_1": "memory_pressure",
        "cause_1": "noisy_neighbor",
        "cause_2": "dependency_failure",
        "cause_root_label": "dependency_failure",
        "solution": "runbook-azure/dependency-failure",
        "query_template": "{service} noisy_neighbor → 依赖失败 → runbook 的 4 跳链路",
    },
]


# ============================================================
#  图库连接 + 路径查询
# ============================================================

def _query_graph_paths(driver, hop_count: int) -> list[dict[str, Any]]:
    """从 Neo4j 查询实际存在的路径，返回结构化路径列表。

    每条路径是一个 dict：
    {
      "nodes": [{"name":..., "label":...}, ...],   # 路径上所有节点
      "edges": [{"name":..., "valid_at":..., "invalid_at":..., "lag_seconds":..., "fact":...}, ...],
    }

    跳数定义：
        2 跳：Symptom → Cause → Solution
        3 跳：Component → Symptom → Cause → Solution
        4 跳：Component → Symptom → Cause → Cause → Solution
    """
    if hop_count == 2:
        cypher = """
        MATCH path = (s)-[r1:RELATES_TO]->(c)-[r2:RELATES_TO]->(sol)
        WHERE 'Symptom' IN labels(s) AND 'Cause' IN labels(c) AND 'Solution' IN labels(sol)
          AND r1.name IN ['CAUSED_BY', 'TRIGGERED_BY']
          AND r2.name IN ['RESOLVED_BY', 'MITIGATED_BY']
        RETURN [n IN nodes(path) | {name: n.name, label: [l IN labels(n) WHERE l IN ['Component','Symptom','Cause','Solution']][0]}] as nodes,
               [r IN relationships(path) | {name: r.name, valid_at: toString(r.valid_at), invalid_at: toString(r.invalid_at), lag_seconds: r.lag_seconds, fact: r.fact, mechanism: r.mechanism}] as edges
        """
    elif hop_count == 3:
        cypher = """
        MATCH path = (comp)-[r1:RELATES_TO]->(s)-[r2:RELATES_TO]->(c)-[r3:RELATES_TO]->(sol)
        WHERE 'Component' IN labels(comp) AND 'Symptom' IN labels(s)
          AND 'Cause' IN labels(c) AND 'Solution' IN labels(sol)
          AND r1.name IN ['HAS_SYMPTOM']
          AND r2.name IN ['CAUSED_BY', 'TRIGGERED_BY']
          AND r3.name IN ['RESOLVED_BY', 'MITIGATED_BY']
        RETURN [n IN nodes(path) | {name: n.name, label: [l IN labels(n) WHERE l IN ['Component','Symptom','Cause','Solution']][0]}] as nodes,
               [r IN relationships(path) | {name: r.name, valid_at: toString(r.valid_at), invalid_at: toString(r.invalid_at), lag_seconds: r.lag_seconds, fact: r.fact, mechanism: r.mechanism}] as edges
        """
    elif hop_count == 4:
        # 4 跳：Component → Symptom → Cause → Cause → Solution
        # 与 cypher_generator._generate_multi_hop_path 中 4 跳模式完全对齐
        # （4 跳 = 4 条边，5 个节点）
        cypher = """
        MATCH path = (comp)-[r1:RELATES_TO]->(s)-[r2:RELATES_TO]->(c1)-[r3:RELATES_TO]->(c2)-[r4:RELATES_TO]->(sol)
        WHERE 'Component' IN labels(comp) AND 'Symptom' IN labels(s)
          AND 'Cause' IN labels(c1) AND 'Cause' IN labels(c2)
          AND 'Solution' IN labels(sol)
          AND r1.name IN ['HAS_SYMPTOM']
          AND r2.name IN ['CAUSED_BY', 'TRIGGERED_BY']
          AND r3.name IN ['CAUSED_BY', 'TRIGGERED_BY', 'PROPAGATED_TO']
          AND r4.name IN ['RESOLVED_BY', 'MITIGATED_BY']
        RETURN [n IN nodes(path) | {name: n.name, label: [l IN labels(n) WHERE l IN ['Component','Symptom','Cause','Solution']][0]}] as nodes,
               [r IN relationships(path) | {name: r.name, valid_at: toString(r.valid_at), invalid_at: toString(r.invalid_at), lag_seconds: r.lag_seconds, fact: r.fact, mechanism: r.mechanism}] as edges
        """
    else:
        return []

    with driver.session() as sess:
        result = sess.run(cypher)
        return [
            {"nodes": rec["nodes"], "edges": rec["edges"]}
            for rec in result
        ]


def _parse_iso_datetime(s: str | None) -> datetime | None:
    """安全解析 ISO datetime 字符串。"""
    if not s or s == "None":
        return None
    try:
        # neo4j toString 可能带 nanoseconds，截到 microseconds
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


# ============================================================
#  query 多样化生成
# ============================================================

# 2 跳 query 模板：以 Symptom 名 + Cause 名为核心
QUERY_TEMPLATES_2HOP = [
    "{symptom} 异常的根因链路",
    "{symptom} 故障的 2 跳根因",
    "{symptom} → {cause} → {solution} 链路",
    "{symptom} 的根因和解法",
    "为什么 {symptom}？根因和解决方案",
    "{symptom} → {cause} 是什么原因",
    "诊断 {symptom} 故障",
    "{symptom} 异常怎么排查",
]

# 3 跳 query 模板：以 Component 名 + Symptom 名为核心
QUERY_TEMPLATES_3HOP = [
    "{component} 上 {symptom} 异常的 3 跳根因链路",
    "{component} 的 {symptom} 故障链路",
    "{component} → {symptom} → {cause} → {solution}",
    "{component} 故障的根因链",
    "{component} 上 {symptom} 怎么解决",
    "{component} 上 {symptom} 飙升的根因",
    "诊断 {component} 的 {symptom}",
    "{component} 故障排查",
]

# 4 跳 query 模板：Component → Symptom → Cause → Cause → Solution
QUERY_TEMPLATES_4HOP = [
    "{component} 上 {symptom} 异常的 4 跳根因链路",
    "{component} → {symptom} → {cause1} → {cause2} → {solution} 链路",
    "{component} 故障的深层根因和解法",
    "{component} 上 {symptom} 的复合故障链路",
    "{component} 故障的多跳排查",
    "{component} 故障的根因链和解法",
]


def _expand_queries(
    path: dict[str, Any],
    hop_count: int,
    n_per_path: int,
    rng: random.Random,
) -> list[str]:
    """对单条图库路径，生成 n_per_path 条多样化的 query。"""
    nodes = path["nodes"]
    if hop_count == 2:
        # nodes: [Symptom, Cause, Solution]
        symptom = nodes[0]["name"] if len(nodes) > 0 else "symptom"
        cause = nodes[1]["name"] if len(nodes) > 1 else "cause"
        solution = nodes[2]["name"] if len(nodes) > 2 else "solution"
        templates = QUERY_TEMPLATES_2HOP
        formatter = lambda t: t.format(symptom=symptom, cause=cause, solution=solution)
    elif hop_count == 3:
        # nodes: [Component, Symptom, Cause, Solution]
        component = nodes[0]["name"] if len(nodes) > 0 else "component"
        symptom = nodes[1]["name"] if len(nodes) > 1 else "symptom"
        cause = nodes[2]["name"] if len(nodes) > 2 else "cause"
        solution = nodes[3]["name"] if len(nodes) > 3 else "solution"
        templates = QUERY_TEMPLATES_3HOP
        formatter = lambda t: t.format(
            component=component, symptom=symptom, cause=cause, solution=solution
        )
    else:  # 4
        # nodes: [Component, Symptom, Cause, Cause, Solution]
        component = nodes[0]["name"] if len(nodes) > 0 else "component"
        symptom = nodes[1]["name"] if len(nodes) > 1 else "symptom"
        cause1 = nodes[2]["name"] if len(nodes) > 2 else "cause1"
        cause2 = nodes[3]["name"] if len(nodes) > 3 else "cause2"
        solution = nodes[4]["name"] if len(nodes) > 4 else "solution"
        templates = QUERY_TEMPLATES_4HOP
        formatter = lambda t: t.format(
            component=component, symptom=symptom,
            cause1=cause1, cause2=cause2, solution=solution
        )

    # 如果模板数本身够，按 n_per_path 取前 n_per_path 个；不足则循环采样
    queries = [formatter(t) for t in templates]
    if len(queries) >= n_per_path:
        return queries[:n_per_path]
    # 不足：随机采样补足
    while len(queries) < n_per_path:
        queries.append(formatter(rng.choice(templates)))
    return queries


def _build_expected_path(path: dict[str, Any]) -> list[ExpectedHop]:
    """从图库路径结构构造 ExpectedHop 列表。"""
    nodes = path["nodes"]
    edges = path["edges"]
    hops: list[ExpectedHop] = []
    for i, edge in enumerate(edges):
        if i + 1 >= len(nodes):
            break
        src = nodes[i]
        tgt = nodes[i + 1]
        hops.append(ExpectedHop(
            edge_name=edge["name"] or "RELATES_TO",
            source_name=src["name"],
            source_label=src["label"] or "Entity",
            target_name=tgt["name"],
            target_label=tgt["label"] or "Entity",
            valid_at=edge.get("valid_at"),
            invalid_at=edge.get("invalid_at"),
            lag_seconds=int(edge.get("lag_seconds") or 0),
        ))
    return hops


def _build_supporting_facts(path: dict[str, Any]) -> list[SupportingFact]:
    """从图库路径边构造 SupportingFact 列表（用 fact/mechanism 作为 provenance）。"""
    facts: list[SupportingFact] = []
    for i, edge in enumerate(path["edges"]):
        fact_text = edge.get("fact") or edge.get("mechanism") or ""
        if not fact_text:
            fact_text = f"边 {edge.get('name', 'RELATES_TO')}（lag={edge.get('lag_seconds', 0)}s）"
        facts.append(SupportingFact(
            hop_index=i,
            source="neo4j_edge",
            text=fact_text,
            reference=f"graph_edge:{edge.get('name', '')}",
        ))
    return facts


def _build_query_time(path: dict[str, Any]) -> str:
    """取首跳边的 valid_at + 1s 作为 query_time（保证落在边有效期内）。"""
    if not path["edges"]:
        return datetime.now().isoformat()
    first_edge = path["edges"][0]
    valid_at = _parse_iso_datetime(first_edge.get("valid_at"))
    if valid_at is None:
        return datetime.now().isoformat()
    return (valid_at + timedelta(seconds=1)).isoformat()


def _build_ground_truth_text(path: dict[str, Any]) -> str:
    """从路径节点+边构造自然语言 ground truth 文本。"""
    nodes = path["nodes"]
    edges = path["edges"]
    parts = [nodes[0]["name"]] if nodes else []
    for i, edge in enumerate(edges):
        if i + 1 < len(nodes):
            parts.append(f" -[{edge.get('name', 'RELATES_TO')}]-> {nodes[i+1]['name']}")
    return "".join(parts)


# ============================================================
#  图库驱动的测试集构造
# ============================================================

def _build_neo4j_driver():
    """构造 Neo4j driver（从 config 读连接信息）。"""
    from config import get_config
    from neo4j import GraphDatabase
    cfg = get_config()
    return GraphDatabase.driver(cfg.neo4j.uri, auth=(cfg.neo4j.user, cfg.neo4j.password))


def build_graph_cases(
    hop_count: int,
    n_per_path: int = 10,
    seed: int = 42,
    driver=None,
) -> list[TestCase]:
    """从图库实际路径反向构造测试用例。

    Parameters
    ----------
    hop_count : int
        2 | 3 | 4
    n_per_path : int
        每条图库路径生成多少条 case（通过 query 多样化）
    driver : neo4j.Driver | None
        None=自动从 config 构造
    """
    rng = random.Random(seed)
    own_driver = False
    if driver is None:
        driver = _build_neo4j_driver()
        own_driver = True

    try:
        paths = _query_graph_paths(driver, hop_count)
    finally:
        if own_driver:
            driver.close()

    logger.info(f"图库查到 {len(paths)} 条 {hop_count} 跳路径")

    cases: list[TestCase] = []
    case_idx = 1
    for path_idx, path in enumerate(paths):
        queries = _expand_queries(path, hop_count, n_per_path, rng)
        expected_path = _build_expected_path(path)
        supporting_facts = _build_supporting_facts(path)
        query_time = _build_query_time(path)
        gt_text = _build_ground_truth_text(path)
        # GraphRAG-Bench §3.1 + arXiv:2506.05690 Table 1 借鉴：题型 + 任务分级
        task_level = _classify_task_level(hop_count, path)

        for q in queries:
            cases.append(TestCase(
                case_id=f"graph_{hop_count}hop_{case_idx:03d}",
                domain="graph",
                hop_count=hop_count,
                query=q,
                expected_path=expected_path,
                supporting_facts=supporting_facts,
                query_time=query_time,
                ground_truth_free_text=gt_text,
                metadata={
                    "source_path_idx": path_idx,
                    "node_count": len(path["nodes"]),
                    "edge_count": len(path["edges"]),
                    "node_names": [n["name"] for n in path["nodes"]],
                },
                question_type="OE",
                task_level=task_level,
            ))
            case_idx += 1

    return cases


def _classify_task_level(hop_count: int, path: dict) -> str:
    """借鉴 arXiv:2506.05690 (When to use Graphs in RAG) Table 1 的 4 级任务分类法。

    | hop_count | 默认 task_level              | 备注                |
    |-----------|------------------------------|---------------------|
    | 2         | Fact_Retrieval               | 单跳事实查询        |
    | 3         | Complex_Reasoning            | 多跳链式推理        |
    | 4         | Complex_Reasoning / Contextual_Summarize | 看节点数 ≥ 5 → Summarize |

    Returns
    -------
    str: 4 级任务分类之一
    """
    if hop_count <= 2:
        return "Fact_Retrieval"
    if hop_count == 3:
        return "Complex_Reasoning"
    # 4 跳：节点数 ≥ 5 且含"复合"语义 → Contextual_Summarize
    if hop_count >= 4 and len(path.get("nodes", [])) >= 5:
        return "Contextual_Summarize"
    return "Complex_Reasoning"


# 保留旧函数名作为别名，便于向后兼容
def build_smd_2hop_cases(n: int = 100, seed: int = 42, driver=None) -> list[TestCase]:
    """2 跳测试集（图库驱动）。n 是目标条数，会尽量用 query 多样化达到。"""
    # 每条路径至少生成 10 条，最多 n 条
    paths = []
    if driver is None:
        d = _build_neo4j_driver()
        try:
            paths = _query_graph_paths(d, 2)
        finally:
            d.close()
    else:
        paths = _query_graph_paths(driver, 2)
    n_per_path = max(1, n // max(1, len(paths))) if paths else 0
    return build_graph_cases(2, n_per_path=n_per_path, seed=seed, driver=driver)


def build_micross_3hop_cases(n: int = 100, seed: int = 42, driver=None) -> list[TestCase]:
    """3 跳测试集（图库驱动）。"""
    paths = []
    if driver is None:
        d = _build_neo4j_driver()
        try:
            paths = _query_graph_paths(d, 3)
        finally:
            d.close()
    else:
        paths = _query_graph_paths(driver, 3)
    n_per_path = max(1, n // max(1, len(paths))) if paths else 0
    return build_graph_cases(3, n_per_path=n_per_path, seed=seed, driver=driver)


def build_cross_4hop_cases(n: int = 100, seed: int = 42, driver=None) -> list[TestCase]:
    """4 跳测试集（图库驱动）。"""
    paths = []
    if driver is None:
        d = _build_neo4j_driver()
        try:
            paths = _query_graph_paths(d, 4)
        finally:
            d.close()
    else:
        paths = _query_graph_paths(driver, 4)
    n_per_path = max(1, n // max(1, len(paths))) if paths else 0
    return build_graph_cases(4, n_per_path=n_per_path, seed=seed, driver=driver)




# ============================================================
#  统一入口
# ============================================================

def build_testset(
    n_per_hop: dict[int, int] | None = None,
    output_path: str | Path | None = None,
    driver=None,
) -> list[TestCase]:
    """构造完整测试集（图库驱动 v2）。

    Parameters
    ----------
    n_per_hop : dict[int, int] | None
        每跳数的目标用例数，默认 {2: 30, 3: 30, 4: 0}。
        实际生成数取决于图库路径数 × query 多样化倍数。
    output_path : str | Path | None
        写入 JSONL 路径，None=不写文件
    driver : neo4j.Driver | None
        复用已有 driver，None=自动构造

    Returns
    -------
    list[TestCase]
    """
    if n_per_hop is None:
        # 默认目标：图库实际只有 2/3 跳路径，4 跳数据为空
        n_per_hop = {2: 30, 3: 30, 4: 0}

    own_driver = False
    if driver is None:
        driver = _build_neo4j_driver()
        own_driver = True

    try:
        all_cases: list[TestCase] = []

        if n_per_hop.get(2, 0) > 0:
            cases = build_graph_cases(2, n_per_path=max(1, n_per_hop[2] // 5), driver=driver)
            cases = cases[: n_per_hop[2]]
            logger.info(f"2 跳: {len(cases)} 条（截取自 {n_per_hop[2]} 目标）")
            all_cases.extend(cases)

        if n_per_hop.get(3, 0) > 0:
            cases = build_graph_cases(3, n_per_path=max(1, n_per_hop[3] // 4), driver=driver)
            cases = cases[: n_per_hop[3]]
            logger.info(f"3 跳: {len(cases)} 条（截取自 {n_per_hop[3]} 目标）")
            all_cases.extend(cases)

        if n_per_hop.get(4, 0) > 0:
            cases = build_graph_cases(4, n_per_path=max(1, n_per_hop[4] // 3), driver=driver)
            cases = cases[: n_per_hop[4]]
            logger.info(f"4 跳: {len(cases)} 条（截取自 {n_per_hop[4]} 目标）")
            all_cases.extend(cases)

        logger.info(f"测试集总数: {len(all_cases)} 条")

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as f:
                for c in all_cases:
                    f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
            logger.info(f"测试集已写入: {output_path}")

        return all_cases
    finally:
        if own_driver:
            driver.close()


def load_testset(path: str | Path = "eval/testset.jsonl") -> list[dict]:
    """从 JSONL 文件加载测试集（每行一个 case dict）。"""
    cases: list[dict] = []
    path = Path(path)
    if not path.exists():
        return cases
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))
    return cases


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cases = build_testset(
        n_per_hop={2: 30, 3: 30, 4: 0},
        output_path=Path("eval/testset.jsonl"),
    )
    print(f"\n=== 测试集概览 ===")
    print(f"总数: {len(cases)}")
    from collections import Counter
    by_domain = Counter(c.domain for c in cases)
    by_hop = Counter(c.hop_count for c in cases)
    print(f"按域: {dict(by_domain)}")
    print(f"按跳数: {dict(by_hop)}")
    print(f"\n前 5 条样例:")
    for c in cases[:5]:
        print(f"  [{c.case_id}] hop={c.hop_count} | {c.query[:70]}")
        print(f"    expected_path: {[(h.source_name, h.edge_name, h.target_name) for h in c.expected_path]}")
