"""知识图谱节点 Schema 定义 —— 四层概念模型。

本模块定义服务器集群故障排查场景的四层概念等级：
    第1层 Component（组件）—— 物理或逻辑实体，如 VM、Pod、Node、Service
    第2层 Symptom（症状）  —— 可观测的异常信号，如 CPU 飙升、延迟突增
    第3层 Cause（因果）    —— 故障根因或中间原因，如资源争抢、配置错误
    第4层 Solution（解法） —— 修复手段，如重启 Pod、扩容、回滚

设计说明（基于 graphiti-core 0.29.2 源码研究）：
- Graphiti 的 EntityNode 自身有 8 个保留字段（uuid/name/group_id/labels/
  created_at/name_embedding/summary/attributes），自定义字段不能与之重名。
- 自定义字段会被存入节点的 attributes dict，经 DB 往返后不会自动还原为
  模型字段。因此本模块采用"独立 Pydantic 模型 + attributes 存放"的设计：
  每个节点类型定义一个独立的 BaseModel（字段名避开保留字段），用于
  add_episode 的 entity_types 参数指导 LLM 抽取；手动构造时把字段值
  塞进 EntityNode.attributes。
- 节点的 layer 信息通过 labels 字段表达（如 labels=["Component"]），
  labels 会叠加在系统自动追加的 "Entity" label 之上。
- labels 命名限制：只能 [A-Za-z_][A-Za-z0-9_]*，不能含中文/空格/连字符。
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ============================================================
#  枚举定义 —— 每层的类型分类，用 Enum 保证取值可控
# ============================================================

class ComponentType(str, Enum):
    """组件类型 —— 服务器集群中的物理或逻辑实体。"""
    VM = "vm"                    # 虚拟机（Azure V2 数据集的主要单位）
    POD = "pod"                  # Kubernetes Pod
    CONTAINER = "container"      # 容器
    NODE = "node"                # 物理机/虚拟机节点
    SERVICE = "service"          # 微服务实例
    DEPLOYMENT = "deployment"    # K8s Deployment 控制器
    NAMESPACE = "namespace"      # K8s 命名空间


class SymptomType(str, Enum):
    """症状类型 —— 可观测的异常信号分类。"""
    METRIC_ANOMALY = "metric_anomaly"    # 指标异常：CPU/内存/延迟等超出阈值
    EVENT = "event"                      # 事件类：OOMKilled、Evicted、CrashLoopBackOff
    LOG_PATTERN = "log_pattern"          # 日志模式：错误日志聚集、异常堆栈


class Severity(str, Enum):
    """严重程度分级。"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class CauseType(str, Enum):
    """因果类型 —— 故障根因或中间原因的分类。

    这 7 种类型覆盖了服务器集群最常见的故障成因，每种都有明确的
    语义边界，便于对照实验中按因果类型均衡分布测试集。
    """
    RESOURCE_CONTENTION = "resource_contention"    # 资源争抢（CPU/内存/IO 竞争）
    MISCONFIGURATION = "misconfiguration"          # 配置错误（limit 过低、端口冲突）
    DEPENDENCY_FAILURE = "dependency_failure"      # 上游依赖故障（级联失败）
    NETWORK_PARTITION = "network_partition"        # 网络分区/丢包
    HARDWARE_FAULT = "hardware_fault"              # 硬件故障（磁盘坏道、内存 ECC）
    PRIORITY_INVERSION = "priority_inversion"     # 优先级反转（低优先级任务阻塞高优先级）
    NOISY_NEIGHBOR = "noisy_neighbor"              # 噪声邻居（同宿主机干扰）


class SolutionType(str, Enum):
    """解法类型 —— 修复手段分类。"""
    RESTART_POD = "restart_pod"                # 重启 Pod
    SCALE_UP = "scale_up"                      # 扩容（增加副本数）
    SCALE_DOWN = "scale_down"                  # 缩容（减少副本数释放资源）
    DRAIN_NODE = "drain_node"                  # 驱逐节点（标记不可调度+迁移）
    ROLLBACK_DEPLOYMENT = "rollback_deployment"  # 回滚 Deployment 到上一版本
    INCREASE_LIMIT = "increase_limit"          # 提高资源 limit（CPU/memory）
    TAINT_NODE = "taint_node"                  # 给节点打污点隔离
    RUNBOOK_PROCEDURE = "runbook_procedure"    # 执行运维手册流程


# ============================================================
#  层级标识 —— 统一的 layer 常量，用于 labels 和查询过滤
# ============================================================

class GraphLayer(str, Enum):
    """四层概念等级标识，对应 Neo4j 节点 label。"""
    COMPONENT = "Component"
    SYMPTOM = "Symptom"
    CAUSE = "Cause"
    SOLUTION = "Solution"


# ============================================================
#  自定义节点类型 —— 独立 Pydantic 模型，用于 add_episode 的 entity_types
#
#  重要：字段名不能与 EntityNode 的 8 个保留字段重名
#  （uuid, name, group_id, labels, created_at, name_embedding, summary, attributes）
# ============================================================

class ComponentEntity(BaseModel):
    """组件层节点类型定义。

    对应 GraphLayer.COMPONENT，表示服务器集群中的一个物理或逻辑实体。
    主数据来源：Azure Public Dataset V2 的 VM 表 + K8s 资源对象。
    """
    component_type: ComponentType = Field(
        description="组件类型，如 vm/pod/container/node/service"
    )
    cluster_id: str = Field(
        description="所属集群标识，用于隔离不同故障场景的查询范围"
    )
    sku: str | None = Field(
        default=None,
        description="VM 规格桶，如 D-series/E-series，来自 Azure V2 数据集"
    )
    vcore_bucket: str | None = Field(
        default=None,
        description="vCPU 核数桶，如 '8-16'，来自 Azure V2 的 vm表"
    )
    memory_gb_bucket: str | None = Field(
        default=None,
        description="内存 GB 桶，如 '64-128'，来自 Azure V2 的 vm表"
    )


class SymptomEntity(BaseModel):
    """症状层节点类型定义。

    对应 GraphLayer.SYMPTOM，表示一个可观测的异常信号。
    症状是故障排查的入口：用户提问通常从症状出发，沿因果链追溯根因。
    """
    symptom_type: SymptomType = Field(
        description="症状类型：metric_anomaly/event/log_pattern"
    )
    severity: Severity = Field(
        default=Severity.WARNING,
        description="严重程度：info/warning/critical"
    )
    metric_name: str | None = Field(
        default=None,
        description="异常指标名称，如 cpu_usage/memory_usage/latency_ms"
    )
    threshold: float | None = Field(
        default=None,
        description="触发阈值，如 0.95 表示 CPU 95%"
    )
    observed_value: float | None = Field(
        default=None,
        description="实际观测到的峰值，如 0.98 表示 CPU 98%"
    )
    first_observed_at: str | None = Field(
        default=None,
        description="症状首次出现的时刻（ISO8601 字符串），作为 valid_at 锚点"
    )


class CauseEntity(BaseModel):
    """因果层节点类型定义。

    对应 GraphLayer.CAUSE，表示故障的一个原因节点。
    因果链可以是单跳（症状→根因）或多跳（症状→中间因→...→根因），
    is_root=true 标记链尾的根因节点。
    """
    cause_type: CauseType = Field(
        description="因果类型，如 resource_contention/misconfiguration"
    )
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="因果置信度 [0,1]，由 LLM 基于证据打分"
    )
    is_root: bool = Field(
        default=False,
        description="是否为根因（因果链末端）。true=根因，false=中间原因"
    )


class SolutionEntity(BaseModel):
    """解法层节点类型定义。

    对应 GraphLayer.SOLUTION，表示针对某个因/症状的修复手段。
    解法节点的 runbook_ref 关联到运维文档片段，支撑 provenance 溯源。
    """
    solution_type: SolutionType = Field(
        description="解法类型，如 restart_pod/scale_up/drain_node"
    )
    runbook_ref: str | None = Field(
        default=None,
        description="关联运维文档片段 id，用于 provenance 溯源到原始文档"
    )
    estimated_mttr_min: int | None = Field(
        default=None,
        description="预估修复时间（分钟），用于解法优先级排序"
    )


# ============================================================
#  entity_types 注册表 —— 传给 graphiti.add_episode(entity_types=...)
#
#  key 是类型名（与 labels 对应），value 是 Pydantic 模型类
# ============================================================

ENTITY_TYPES: dict[str, type[BaseModel]] = {
    "Component": ComponentEntity,
    "Symptom": SymptomEntity,
    "Cause": CauseEntity,
    "Solution": SolutionEntity,
}


# ============================================================
#  辅助函数 —— 手动构造节点时把自定义字段塞进 attributes
# ============================================================

def build_component_attributes(
    component_type: ComponentType,
    cluster_id: str,
    sku: str | None = None,
    vcore_bucket: str | None = None,
    memory_gb_bucket: str | None = None,
) -> dict[str, Any]:
    """构造 Component 节点的 attributes dict。

    手动创建 EntityNode 时调用此函数，把自定义字段打包进 attributes，
    避免与 EntityNode 保留字段冲突。
    """
    attrs: dict[str, Any] = {
        "component_type": component_type.value,
        "cluster_id": cluster_id,
    }
    if sku is not None:
        attrs["sku"] = sku
    if vcore_bucket is not None:
        attrs["vcore_bucket"] = vcore_bucket
    if memory_gb_bucket is not None:
        attrs["memory_gb_bucket"] = memory_gb_bucket
    return attrs


def build_symptom_attributes(
    symptom_type: SymptomType,
    severity: Severity = Severity.WARNING,
    metric_name: str | None = None,
    threshold: float | None = None,
    observed_value: float | None = None,
    first_observed_at: str | None = None,
) -> dict[str, Any]:
    """构造 Symptom 节点的 attributes dict。"""
    attrs: dict[str, Any] = {
        "symptom_type": symptom_type.value,
        "severity": severity.value,
    }
    if metric_name is not None:
        attrs["metric_name"] = metric_name
    if threshold is not None:
        attrs["threshold"] = threshold
    if observed_value is not None:
        attrs["observed_value"] = observed_value
    if first_observed_at is not None:
        attrs["first_observed_at"] = first_observed_at
    return attrs


def build_cause_attributes(
    cause_type: CauseType,
    confidence: float = 0.8,
    is_root: bool = False,
) -> dict[str, Any]:
    """构造 Cause 节点的 attributes dict。"""
    return {
        "cause_type": cause_type.value,
        "confidence": confidence,
        "is_root": is_root,
    }


def build_solution_attributes(
    solution_type: SolutionType,
    runbook_ref: str | None = None,
    estimated_mttr_min: int | None = None,
) -> dict[str, Any]:
    """构造 Solution 节点的 attributes dict。"""
    attrs: dict[str, Any] = {
        "solution_type": solution_type.value,
    }
    if runbook_ref is not None:
        attrs["runbook_ref"] = runbook_ref
    if estimated_mttr_min is not None:
        attrs["estimated_mttr_min"] = estimated_mttr_min
    return attrs
