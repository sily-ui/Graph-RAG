"""数据接入层公共数据模型。

本模块定义数据接入管线中各阶段传递的数据结构，衔接原始数据与
Graphiti episode：

    Azure V2 / 合成数据
            │
            ▼
       VMTimeSeries ── (原始 CPU 时序)
            │
            ▼
   fault_event_extractor
            │
            ▼
       FaultEvent ── (结构化故障事件，含时态窗口)
            │
            ▼
    episode_builder
            │
            ▼
    EpisodePayload ── (打包好的 episode，待写入)
            │
            ▼
    graphiti_writer → Neo4j

文档因果骨架路径：
    K8s/Prometheus 文档
            │
            ▼
    doc_skeleton_seeder
            │
            ▼
    CausalTriple ── (因果三元组种子)
            │
            ▼
    episode_builder → EpisodePayload → graphiti_writer

设计原则：
- 所有时间字段统一用 datetime（带时区），便于时态剪枝
- 模型不带 Graphiti 依赖，纯 Pydantic，便于离线测试
- 与 graph_schema 的枚举保持一致，避免类型转换错误
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from graph_schema.nodes import (
    CauseType,
    ComponentType,
    Severity,
    SolutionType,
    SymptomType,
)


# ============================================================
#  原始时序数据
# ============================================================

class VMTimeSeries(BaseModel):
    """单个 VM 的 CPU 时序数据（来自 Azure V2 或合成数据）。

    Azure V2 数据集每 5 分钟一条 CPU 读数，共 30 天 ≈ 8640 个点。
    本模型持有单个 VM 的完整时序与元信息，供异常检测使用。

    Attributes
    ----------
    vm_id : str
        VM 唯一标识（Azure V2 的 vmid 字段）
    cluster_id : str
        所属集群标识（用于 group_id 隔离）
    cpu_readings : list[tuple[datetime, float]]
        时间戳与 CPU 利用率的二元组列表，按时间升序
    vm_created : datetime
        VM 创建时刻
    vm_deleted : datetime | None
        VM 删除时刻，None 表示 VM 在观察期内未删除
    sku : str | None
        VM 规格桶（如 D-series/E-series）
    vcore_bucket : str | None
        vCPU 核数桶（如 "8-16"）
    memory_gb_bucket : str | None
        内存 GB 桶（如 "64-128"）
    """
    vm_id: str
    cluster_id: str
    cpu_readings: list[tuple[datetime, float]] = Field(
        default_factory=list,
        description="(timestamp, cpu_utilization) 二元组列表，升序"
    )
    vm_created: datetime
    vm_deleted: datetime | None = None
    sku: str | None = None
    vcore_bucket: str | None = None
    memory_gb_bucket: str | None = None

    @property
    def is_deleted(self) -> bool:
        """VM 在观察期内是否被删除（=故障/驱逐的强信号）。"""
        return self.vm_deleted is not None

    @property
    def reading_count(self) -> int:
        return len(self.cpu_readings)


# ============================================================
#  异常检测结果
# ============================================================

class AnomalyType(str, Enum):
    """异常类型枚举。"""
    CPU_SPIKE = "cpu_spike"              # CPU 飙升（持续超阈值）
    CPU_DROP = "cpu_drop"                # CPU 骤降（可能崩溃前兆）
    VM_DELETION = "vm_deletion"          # VM 被删除/驱逐
    HIGH_VARIANCE = "high_variance"      # 高方差（噪声邻居特征）


class AnomalyPoint(BaseModel):
    """单个异常点/异常窗口的检测结果。

    由 azure_trace_loader 的异常检测算法输出，
    供 fault_event_extractor 聚合成结构化故障事件。
    """
    anomaly_type: AnomalyType
    vm_id: str
    cluster_id: str
    timestamp: datetime                  # 异常发生时刻
    end_timestamp: datetime | None = Field(
        default=None,
        description="异常结束时刻，None=单点异常"
    )
    observed_value: float = Field(
        description="实际观测值（如 CPU 利用率 0.98）"
    )
    baseline_value: float = Field(
        description="基线值（如 IQR 中位数 0.45）"
    )
    threshold: float = Field(
        description="触发阈值（如 p95 = 0.85）"
    )
    detection_method: str = Field(
        description="检测方法，如 'iqr'/'3-sigma'/'deletion_event'"
    )
    duration_seconds: int = Field(
        default=0,
        description="异常持续时长（秒），单点异常为0"
    )

    @property
    def deviation_ratio(self) -> float:
        """偏离倍数 = (observed - baseline) / baseline。"""
        if self.baseline_value == 0:
            return float("inf") if self.observed_value > 0 else 0.0
        return (self.observed_value - self.baseline_value) / self.baseline_value


# ============================================================
#  结构化故障事件
# ============================================================

class FaultEventType(str, Enum):
    """故障事件类型 —— 对应 Symptom 层的具体表现。

    每种事件类型有明确的因果链模板，便于 episode_builder
    关联到正确的 Cause/Solution 节点。
    """
    CPU_SPIKE = "cpu_spike"                      # CPU 飙升 → noisy_neighbor/resource_contention
    VM_DELETION = "vm_deletion"                  # VM 删除 → hardware_fault/misconfiguration
    OOM_KILLED = "oom_killed"                    # OOM → resource_contention/misconfiguration
    EVICTION = "eviction"                        # 驱逐 → resource_contention/noisy_neighbor
    LATENCY_SURGE = "latency_surge"              # 延迟突增 → dependency_failure/network_partition
    CRASH_LOOP = "crash_loop"                    # CrashLoopBackOff → misconfiguration/dependency_failure


class FaultEvent(BaseModel):
    """结构化故障事件 —— 数据接入层的核心产物。

    一个 FaultEvent 对应一次完整的故障观察，包含：
    - 时态信息（何时开始、何时结束）
    - 关联的组件信息（哪个 VM/Pod）
    - 症状描述（异常指标、严重程度）
    - 原始 trace 片段（用于 provenance 溯源）

    fault_event_extractor 把多个相关的 AnomalyPoint 聚合成一个 FaultEvent，
    episode_builder 再把 FaultEvent 打包成 EpisodePayload。
    """
    event_id: str = Field(description="事件唯一标识，如 'fault_<cluster>_<vm>_<ts>'")
    event_type: FaultEventType
    vm_id: str
    cluster_id: str
    timestamp_start: datetime = Field(description="故障开始时刻（valid_at 锚点）")
    timestamp_end: datetime | None = Field(
        default=None,
        description="故障结束时刻（invalid_at 锚点），None=未恢复"
    )
    severity: Severity = Severity.WARNING
    component_type: ComponentType = ComponentType.VM
    sku: str | None = None
    vcore_bucket: str | None = None
    memory_gb_bucket: str | None = None

    # 症状详情
    metric_name: str = Field(description="异常指标名，如 cpu_usage/memory_usage")
    observed_value: float = Field(description="观测峰值")
    baseline_value: float = Field(description="基线值")
    threshold: float = Field(description="触发阈值")

    # provenance 溯源
    trace_fragment: list[tuple[datetime, float]] = Field(
        default_factory=list,
        description="故障窗口内的原始 CPU trace 片段，用于 provenance 溯源"
    )
    detection_method: str = Field(description="检测方法")
    source_dataset: str = Field(
        default="azure_v2",
        description="数据来源，azure_v2/synthetic/google_cluster"
    )

    # 关联的文档因果骨架（由 episode_builder 填充）
    linked_cause_type: CauseType | None = Field(
        default=None,
        description="关联的根因类型，由因果骨架匹配得到"
    )
    linked_solution_type: SolutionType | None = Field(
        default=None,
        description="关联的解法类型"
    )

    @property
    def duration_seconds(self) -> int:
        """故障持续时长（秒）。"""
        if self.timestamp_end is None:
            return 0
        delta = self.timestamp_end - self.timestamp_start
        return int(delta.total_seconds())


# ============================================================
#  文档因果骨架
# ============================================================

class CausalTriple(BaseModel):
    """因果三元组 —— 从运维文档抽取的因果骨架种子。

    每个 CausalTriple 描述一条「症状→因→解法」的因果先验，
    作为图谱的骨架种子，真实故障事件会挂载到这些骨架上。

    来源：K8s 官方文档、Prometheus alerting rules、运维 runbook。
    """
    # 症状侧
    symptom_type: SymptomType
    symptom_keywords: list[str] = Field(
        description="症状关键词，用于匹配 FaultEvent，如 ['cpu_usage','spike','high']"
    )

    # 因果侧
    cause_type: CauseType
    cause_mechanism: str = Field(
        description="因果机制自然语言说明"
    )
    is_root: bool = Field(
        default=True,
        description="是否为根因（默认 True，中间因在多跳链中单独定义）"
    )

    # 解法侧
    solution_type: SolutionType
    solution_runbook_ref: str = Field(
        description="运维文档引用，如 'k8s-docs/pod-lifecycle#oomkilled'"
    )
    estimated_mttr_min: int = Field(
        default=10,
        description="预估修复时间（分钟）"
    )

    # 边扩展字段
    lag_seconds: int = Field(
        default=0,
        description="因果时延（秒），从原因发生到症状出现"
    )
    confidence: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="因果置信度（文档先验通常较高）"
    )
    effectiveness: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="解法历史有效率"
    )

    source_doc: str = Field(
        description="来源文档标识，如 'k8s-docs'/'prometheus-alerts'/'runbook-azure'"
    )


# ============================================================
#  Episode 载荷
# ============================================================

class EpisodePayload(BaseModel):
    """打包好的 episode 载荷 —— graphiti_writer 的输入。

    一个 EpisodePayload 对应一次 graphiti.add_episode 调用，
    包含 Graphiti 所需的全部参数：

    - name: episode 名称（唯一标识）
    - episode_body: episode 正文（自然语言或 JSON 字符串）
    - reference_time: 时态锚点（故障发生时刻）
    - group_id: 分区标识（按故障场景隔离）
    - source_description: 来源描述

    reference_time 是时态因果建模的关键：Graphiti 内部会把它
    作为 EntityEdge.valid_at 的默认值，确保时态锚点正确。
    """
    name: str = Field(description="episode 唯一名称")
    episode_body: str = Field(
        description="episode 正文，自然语言或 JSON 字符串"
    )
    reference_time: datetime = Field(
        description="时态锚点，设为故障发生时刻"
    )
    group_id: str = Field(
        description="分区标识，如 'cluster_A_fault_001'"
    )
    source_description: str = Field(
        default="",
        description="来源描述，如 'Azure V2 VM cpu_spike event'"
    )
    # 元信息（不传给 Graphiti，仅供 writer/调试使用）
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="元信息，如 event_id、cause_type、跳数等"
    )


# ============================================================
#  图谱构建统计
# ============================================================

class GraphBuildStats(BaseModel):
    """图谱构建结果统计 —— bootstrap_graph 的输出。"""
    total_episodes: int = 0
    total_fault_events: int = 0
    total_causal_triples: int = 0
    episodes_written: int = 0
    episodes_failed: int = 0
    failure_reasons: list[str] = Field(default_factory=list)
    duration_seconds: float = 0.0

    def merge(self, other: GraphBuildStats) -> GraphBuildStats:
        """合并两个统计。"""
        return GraphBuildStats(
            total_episodes=self.total_episodes + other.total_episodes,
            total_fault_events=self.total_fault_events + other.total_fault_events,
            total_causal_triples=self.total_causal_triples + other.total_causal_triples,
            episodes_written=self.episodes_written + other.episodes_written,
            episodes_failed=self.episodes_failed + other.episodes_failed,
            failure_reasons=self.failure_reasons + other.failure_reasons,
            duration_seconds=self.duration_seconds + other.duration_seconds,
        )
