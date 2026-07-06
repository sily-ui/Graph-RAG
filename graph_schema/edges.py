"""知识图谱边 Schema 定义 —— 三类带时态的因果边。

本模块定义服务器集群故障图谱中的三类边：

    HAS_SYMPTOM      Component → Symptom    组件表现出症状
    CAUSED_BY / TRIGGERED_BY / PROPAGATED_TO
                     Symptom/Cause → Cause  因果链传导（多跳核心）
    RESOLVED_BY / MITIGATED_BY / PREVENTED_BY
                     Cause/Symptom → Solution  解法作用于因或症状

设计说明（基于 graphiti-core 0.29.2 源码研究）：

1. 关系类型统一为 RELATES_TO：
   Graphiti 在 Neo4j 中所有 EntityEdge 都存为 RELATES_TO 关系类型，
   区分靠 name 字段（如 name="CAUSED_BY"）。因此 Cypher 多跳查询时
   必须用关系属性过滤，不能用关系类型过滤：
       MATCH (s)-[r:RELATES_TO]->(c) WHERE r.name='CAUSED_BY'
   而不是：
       MATCH (s)-[:CAUSED_BY]->(c)   -- 这个查不到任何边！

2. 时态字段说明（bi-temporal 模型）：
   - valid_at:     事实何时变为真（业务时间，如症状开始时刻）
   - invalid_at:   事实何时失效（业务时间，如症状结束时刻；None=仍有效）
   - expired_at:   系统侧失效时间（被新事实覆盖时由 Graphiti 设置）
   - reference_time: 产生该边时的参考时刻（episode 的 reference_time）
   - created_at:   边记录创建时刻（系统时间，必填无默认！）

3. provenance 溯源：
   episodes 字段是 list[str]，存储支撑该边的 episode uuid 列表。
   通过这些 uuid 可以追溯到原始的 trace 片段或运维文档。

4. 自定义字段存放：
   与节点类似，边的扩展字段（如 lag_seconds、mechanism、effectiveness）
   存入 attributes dict，不作为 EntityEdge 子类的模型字段。
   （DB 往返后自定义字段不会自动还原到模型字段）
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ============================================================
#  边名称枚举 —— 对应 EntityEdge.name 字段，统一管理关系名
# ============================================================

class EdgeName(str, Enum):
    """
    所有可能的边名称。在 Neo4j 中都存为 RELATES_TO 关系类型，
    通过 name 属性区分具体语义。

    查询时用：
        MATCH (s)-[r:RELATES_TO]->(c)
        WHERE r.name IN ['CAUSED_BY', 'TRIGGERED_BY', 'PROPAGATED_TO']
    """

    # Component → Symptom
    HAS_SYMPTOM = "HAS_SYMPTOM"

    # 因果链传导（Symptom/Cause → Cause）
    CAUSED_BY = "CAUSED_BY"            # A 由 B 引起（直接因果）
    TRIGGERED_BY = "TRIGGERED_BY"      # A 被 B 触发（即时触发关系）
    PROPAGATED_TO = "PROPAGATED_TO"    # A 传播到 B（连锁反应）

    # 解法关系（Cause/Symptom → Solution）
    RESOLVED_BY = "RESOLVED_BY"        # A 被 B 解决（完全修复）
    MITIGATED_BY = "MITIGATED_BY"      # A 被 B 缓解（部分减轻）
    PREVENTED_BY = "PREVENTED_BY"      # A 被 B 预防（未来不再发生）


# ============================================================
#  边名称分组 —— 便于查询时按组过滤
# ============================================================

# 因果链边：用于多跳路径抽取
CAUSAL_EDGE_NAMES: list[str] = [
    EdgeName.CAUSED_BY.value,
    EdgeName.TRIGGERED_BY.value,
    EdgeName.PROPAGATED_TO.value,
]

# 解法边：用于从因/症状找到解法
SOLUTION_EDGE_NAMES: list[str] = [
    EdgeName.RESOLVED_BY.value,
    EdgeName.MITIGATED_BY.value,
    EdgeName.PREVENTED_BY.value,
]

# 全部边名称
ALL_EDGE_NAMES: list[str] = [e.value for e in EdgeName]


# ============================================================
#  自定义边类型 —— 独立 Pydantic 模型，用于 add_episode 的 edge_types
#
#  重要：字段名不能与 EntityEdge 的保留字段重名
#  （uuid, name, group_id, source_node_uuid, target_node_uuid,
#   created_at, fact, fact_embedding, episodes, expired_at,
#   valid_at, invalid_at, reference_time, attributes）
# ============================================================

class HasSymptomEdge(BaseModel):
    """HAS_SYMPTOM 边类型（Component → Symptom）。

    语义：某个组件在特定时间窗内表现出某个症状。
    时态：valid_at = 症状开始时刻，invalid_at = 症状结束时刻（None=未恢复）。
    """
    detection_method: str | None = Field(
        default=None,
        description="症状检测方法，如 'iqr'/'3-sigma'/'prometheus_alert'"
    )


class CausalEdge(BaseModel):
    """因果链边类型（Symptom/Cause → Cause）。

    语义：源节点的异常由目标节点的原因导致/触发/传播而来。
    这是多跳因果推理的核心边类型。

    lag_seconds 是本方案的关键创新字段：
    它记录因果时延（症状出现到原因发生的时间差），用于：
    1. 因果链排序：总时延最小的路径优先
    2. 时态剪枝：排除时延不合理的候选路径
    3. 对照实验：时态准确率指标的评估依据
    """
    mechanism: str = Field(
        description="自然语言因果机制说明，如 '内存不足触发 OOMKiller 杀死进程'"
    )
    lag_seconds: int = Field(
        default=0,
        ge=0,
        description="因果时延（秒）：从原因发生到症状出现的时间差"
    )
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="因果置信度 [0,1]，由 LLM 基于证据和时态一致性打分"
    )


class SolutionEdge(BaseModel):
    """解法边类型（Cause/Symptom → Solution）。

    语义：某个解法用于解决/缓解/预防某个因或症状。
    effectiveness 记录历史有效率，用于解法优先级排序。
    """
    effectiveness: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="历史有效率 [0,1]，来自运维工单统计或文档标注"
    )
    is_immediate: bool = Field(
        default=False,
        description="是否为即时解法（true=可立即执行，false=需审批或调度）"
    )


# ============================================================
#  edge_types 注册表 —— 传给 graphiti.add_episode(edge_types=...)
# ============================================================

EDGE_TYPES: dict[str, type[BaseModel]] = {
    "HAS_SYMPTOM": HasSymptomEdge,
    "CAUSED_BY": CausalEdge,
    "TRIGGERED_BY": CausalEdge,
    "PROPAGATED_TO": CausalEdge,
    "RESOLVED_BY": SolutionEdge,
    "MITIGATED_BY": SolutionEdge,
    "PREVENTED_BY": SolutionEdge,
}

# edge_type_map —— 指定哪些源类型可以连哪些目标类型（传给 add_episode）
# key: (source_label, target_label)，value: 允许的边名称列表
EDGE_TYPE_MAP: dict[tuple[str, str], list[str]] = {
    ("Component", "Symptom"): ["HAS_SYMPTOM"],
    ("Symptom", "Cause"): ["CAUSED_BY", "TRIGGERED_BY"],
    ("Cause", "Cause"): ["CAUSED_BY", "TRIGGERED_BY", "PROPAGATED_TO"],
    ("Cause", "Solution"): ["RESOLVED_BY", "MITIGATED_BY", "PREVENTED_BY"],
    ("Symptom", "Solution"): ["MITIGATED_BY", "PREVENTED_BY"],
}


# ============================================================
#  辅助函数 —— 手动构造边时把扩展字段塞进 attributes
# ============================================================

def build_causal_edge_attributes(
    mechanism: str,
    lag_seconds: int = 0,
    confidence: float = 0.8,
) -> dict[str, Any]:
    """构造因果边的 attributes dict。

    Parameters
    ----------
    mechanism : str
        自然语言因果机制说明。例如：
        "内存不足触发 OOMKiller，杀死 Pod 内主进程"
    lag_seconds : int
        因果时延（秒）。例如症状在原因发生后 30 秒才出现，则 lag_seconds=30
    confidence : float
        因果置信度 [0,1]。LLM 基于证据充分性和时态一致性打分
    """
    return {
        "mechanism": mechanism,
        "lag_seconds": lag_seconds,
        "confidence": confidence,
    }


def build_solution_edge_attributes(
    effectiveness: float = 0.8,
    is_immediate: bool = False,
) -> dict[str, Any]:
    """构造解法边的 attributes dict。"""
    return {
        "effectiveness": effectiveness,
        "is_immediate": is_immediate,
    }


def build_has_symptom_edge_attributes(
    detection_method: str | None = None,
) -> dict[str, Any]:
    """构造 HAS_SYMPTOM 边的 attributes dict。"""
    attrs: dict[str, Any] = {}
    if detection_method is not None:
        attrs["detection_method"] = detection_method
    return attrs
