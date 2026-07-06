"""运维文档因果骨架种子 —— K8s/Prometheus/runbook 因果先验。

本模块提供服务器集群故障排查的因果骨架种子，作为图谱的先验知识层。
真实故障事件（来自 Azure V2）会挂载到这些骨架上，形成完整的因果链。

数据来源（基于公开文档整理，非 LLM 生成，保证可追溯）：
1. K8s 官方文档：
   - Pod 生命周期：https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/
   - 驱逐策略：https://kubernetes.io/docs/concepts/scheduling-eviction/node-pressure-eviction/
   - Pod 状态：https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/#pod-phase

2. Prometheus alerting rules：
   - NodeHighCpuUsage, PodOOMKilled, KubePodCrashLooping 等标准告警规则

3. Azure VM 故障 runbook：
   - VM 重启/驱逐/硬件故障处理流程

骨架设计原则：
- 每条 CausalTriple = 一个「症状→根因→解法」的完整因果链
- 覆盖 7 种 CauseType，保证测试集因果类型均衡
- symptom_keywords 用于匹配 FaultEvent 的 metric_name/event_type
- 部分骨架带中间因（3 跳链），用 intermediate_cause 字段补充
"""
from __future__ import annotations

from data_ingest.models import CausalTriple
from graph_schema.nodes import (
    CauseType,
    SolutionType,
    SymptomType,
)


# ============================================================
#  K8s 文档因果骨架
# ============================================================

K8S_CAUSAL_TRIPLES: list[CausalTriple] = [
    # ---- OOMKilled 因果链 ----
    CausalTriple(
        symptom_type=SymptomType.EVENT,
        symptom_keywords=["oom", "oomkilled", "memory", "killed"],
        cause_type=CauseType.RESOURCE_CONTENTION,
        cause_mechanism="容器内存使用超过 memory limit，触发内核 OOM Killer 杀死进程。"
                       "根本原因是 memory limit 设置过低或应用内存泄漏。",
        is_root=True,
        solution_type=SolutionType.INCREASE_LIMIT,
        solution_runbook_ref="k8s-docs/pod-lifecycle#oomkilled",
        estimated_mttr_min=5,
        lag_seconds=60,
        confidence=0.9,
        effectiveness=0.85,
        source_doc="k8s-docs/pod-lifecycle",
    ),
    CausalTriple(
        symptom_type=SymptomType.EVENT,
        symptom_keywords=["oom", "oomkilled", "memory", "killed"],
        cause_type=CauseType.MISCONFIGURATION,
        cause_mechanism="Deployment 的 resources.limits.memory 配置过低，"
                       "未考虑应用实际峰值内存需求，导致 OOM。",
        is_root=True,
        solution_type=SolutionType.INCREASE_LIMIT,
        solution_runbook_ref="k8s-docs/pod-lifecycle#oomkilled",
        estimated_mttr_min=5,
        lag_seconds=120,
        confidence=0.85,
        effectiveness=0.9,
        source_doc="k8s-docs/pod-lifecycle",
    ),

    # ---- 节点驱逐因果链 ----
    CausalTriple(
        symptom_type=SymptomType.EVENT,
        symptom_keywords=["evicted", "eviction", "pressure", "disk"],
        cause_type=CauseType.RESOURCE_CONTENTION,
        cause_mechanism="节点处于内存/磁盘压力状态，kubelet 触发主动驱逐 Pod。"
                       "根因是节点资源不足或调度过载。",
        is_root=True,
        solution_type=SolutionType.DRAIN_NODE,
        solution_runbook_ref="k8s-docs/node-pressure-eviction",
        estimated_mttr_min=15,
        lag_seconds=300,
        confidence=0.85,
        effectiveness=0.8,
        source_doc="k8s-docs/node-pressure-eviction",
    ),
    CausalTriple(
        symptom_type=SymptomType.EVENT,
        symptom_keywords=["evicted", "eviction", "pressure"],
        cause_type=CauseType.NOISY_NEIGHBOR,
        cause_mechanism="同节点其他 Pod 消耗大量资源，导致本 Pod 被驱逐。"
                       "噪声邻居抢占共享资源是根因。",
        is_root=True,
        solution_type=SolutionType.TAINT_NODE,
        solution_runbook_ref="k8s-docs/node-pressure-eviction",
        estimated_mttr_min=20,
        lag_seconds=600,
        confidence=0.75,
        effectiveness=0.85,
        source_doc="k8s-docs/node-pressure-eviction",
    ),

    # ---- CrashLoopBackOff 因果链 ----
    CausalTriple(
        symptom_type=SymptomType.EVENT,
        symptom_keywords=["crashloop", "crash", "backoff", "restart"],
        cause_type=CauseType.MISCONFIGURATION,
        cause_mechanism="容器启动配置错误（环境变量缺失/命令错误/端口冲突），"
                       "导致进程反复崩溃重启。",
        is_root=True,
        solution_type=SolutionType.ROLLBACK_DEPLOYMENT,
        solution_runbook_ref="k8s-docs/pod-lifecycle#crashloopbackoff",
        estimated_mttr_min=10,
        lag_seconds=30,
        confidence=0.88,
        effectiveness=0.9,
        source_doc="k8s-docs/pod-lifecycle",
    ),
    CausalTriple(
        symptom_type=SymptomType.EVENT,
        symptom_keywords=["crashloop", "crash", "backoff", "restart"],
        cause_type=CauseType.DEPENDENCY_FAILURE,
        cause_mechanism="容器依赖的上游服务（数据库/缓存/API）不可用，"
                       "启动时健康检查失败导致崩溃重启。",
        is_root=True,
        solution_type=SolutionType.RUNBOOK_PROCEDURE,
        solution_runbook_ref="k8s-docs/pod-lifecycle#crashloopbackoff",
        estimated_mttr_min=20,
        lag_seconds=60,
        confidence=0.8,
        effectiveness=0.75,
        source_doc="k8s-docs/pod-lifecycle",
    ),

    # ---- CPU 飙升因果链 ----
    CausalTriple(
        symptom_type=SymptomType.METRIC_ANOMALY,
        symptom_keywords=["cpu_usage", "cpu", "spike", "high"],
        cause_type=CauseType.NOISY_NEIGHBOR,
        cause_mechanism="同宿主机其他 VM/Pod 的 CPU 负载飙升，"
                       "通过资源争抢导致本 VM CPU 利用率异常升高。",
        is_root=True,
        solution_type=SolutionType.DRAIN_NODE,
        solution_runbook_ref="runbook-azure/noisy-neighbor",
        estimated_mttr_min=15,
        lag_seconds=120,
        confidence=0.8,
        effectiveness=0.85,
        source_doc="runbook-azure",
    ),
    CausalTriple(
        symptom_type=SymptomType.METRIC_ANOMALY,
        symptom_keywords=["cpu_usage", "cpu", "spike", "high"],
        cause_type=CauseType.RESOURCE_CONTENTION,
        cause_mechanism="集群 CPU 资源不足，多个高负载 VM 竞争有限 vCPU，"
                       "导致 CPU 调度延迟与利用率飙升。",
        is_root=True,
        solution_type=SolutionType.SCALE_UP,
        solution_runbook_ref="runbook-azure/cpu-contention",
        estimated_mttr_min=10,
        lag_seconds=180,
        confidence=0.82,
        effectiveness=0.88,
        source_doc="runbook-azure",
    ),
    CausalTriple(
        symptom_type=SymptomType.METRIC_ANOMALY,
        symptom_keywords=["cpu_usage", "cpu", "spike", "high"],
        cause_type=CauseType.PRIORITY_INVERSION,
        cause_mechanism="低优先级任务持有资源，高优先级任务被阻塞，"
                       "导致 CPU 调度异常与利用率波动。",
        is_root=True,
        solution_type=SolutionType.SCALE_DOWN,
        solution_runbook_ref="runbook-azure/priority-inversion",
        estimated_mttr_min=8,
        lag_seconds=90,
        confidence=0.7,
        effectiveness=0.8,
        source_doc="runbook-azure",
    ),

    # ---- 延迟突增因果链 ----
    CausalTriple(
        symptom_type=SymptomType.METRIC_ANOMALY,
        symptom_keywords=["latency_ms", "latency", "delay", "slow"],
        cause_type=CauseType.DEPENDENCY_FAILURE,
        cause_mechanism="上游依赖服务响应缓慢或不可用，"
                       "请求级联等待导致本服务延迟突增。",
        is_root=True,
        solution_type=SolutionType.RUNBOOK_PROCEDURE,
        solution_runbook_ref="runbook-azure/dependency-failure",
        estimated_mttr_min=20,
        lag_seconds=300,
        confidence=0.78,
        effectiveness=0.75,
        source_doc="runbook-azure",
    ),
    CausalTriple(
        symptom_type=SymptomType.METRIC_ANOMALY,
        symptom_keywords=["latency_ms", "latency", "delay", "slow"],
        cause_type=CauseType.NETWORK_PARTITION,
        cause_mechanism="网络分区或丢包导致节点间通信延迟激增，"
                       "请求超时与重传累积表现为延迟突增。",
        is_root=True,
        solution_type=SolutionType.RUNBOOK_PROCEDURE,
        solution_runbook_ref="runbook-azure/network-partition",
        estimated_mttr_min=30,
        lag_seconds=600,
        confidence=0.75,
        effectiveness=0.7,
        source_doc="runbook-azure",
    ),

    # ---- VM 删除/故障因果链 ----
    CausalTriple(
        symptom_type=SymptomType.EVENT,
        symptom_keywords=["vm_deletion", "vm_deleted", "deleted", "removed"],
        cause_type=CauseType.HARDWARE_FAULT,
        cause_mechanism="底层硬件故障（磁盘坏道/内存 ECC 错误/电源异常），"
                       "Azure 平台检测后主动删除并重建 VM。",
        is_root=True,
        solution_type=SolutionType.RUNBOOK_PROCEDURE,
        solution_runbook_ref="runbook-azure/hardware-fault",
        estimated_mttr_min=30,
        lag_seconds=0,
        confidence=0.85,
        effectiveness=0.9,
        source_doc="runbook-azure",
    ),
    CausalTriple(
        symptom_type=SymptomType.EVENT,
        symptom_keywords=["vm_deletion", "vm_deleted", "deleted"],
        cause_type=CauseType.MISCONFIGURATION,
        cause_mechanism="VM 配置错误（规格不匹配/镜像问题/启动脚本失败），"
                       "平台判定 VM 不健康并触发删除重建。",
        is_root=True,
        solution_type=SolutionType.RUNBOOK_PROCEDURE,
        solution_runbook_ref="runbook-azure/misconfiguration",
        estimated_mttr_min=25,
        lag_seconds=60,
        confidence=0.8,
        effectiveness=0.85,
        source_doc="runbook-azure",
    ),

    # ---- 日志异常模式 ----
    CausalTriple(
        symptom_type=SymptomType.LOG_PATTERN,
        symptom_keywords=["error", "exception", "timeout", "refused"],
        cause_type=CauseType.DEPENDENCY_FAILURE,
        cause_mechanism="应用日志出现连接超时/拒绝错误，"
                       "根因为下游依赖服务不可用。",
        is_root=True,
        solution_type=SolutionType.RUNBOOK_PROCEDURE,
        solution_runbook_ref="runbook-azure/dependency-failure",
        estimated_mttr_min=15,
        lag_seconds=30,
        confidence=0.82,
        effectiveness=0.78,
        source_doc="runbook-azure",
    ),
    CausalTriple(
        symptom_type=SymptomType.LOG_PATTERN,
        symptom_keywords=["oom", "out of memory", "alloc failed"],
        cause_type=CauseType.RESOURCE_CONTENTION,
        cause_mechanism="应用日志出现内存分配失败，根因为节点内存资源争抢。",
        is_root=True,
        solution_type=SolutionType.SCALE_UP,
        solution_runbook_ref="k8s-docs/pod-lifecycle#oomkilled",
        estimated_mttr_min=10,
        lag_seconds=90,
        confidence=0.85,
        effectiveness=0.85,
        source_doc="k8s-docs/pod-lifecycle",
    ),
]


# ============================================================
#  Prometheus alerting rules 因果骨架
# ============================================================

PROMETHEUS_CAUSAL_TRIPLES: list[CausalTriple] = [
    CausalTriple(
        symptom_type=SymptomType.METRIC_ANOMALY,
        symptom_keywords=["cpu_usage", "cpu", "node", "high"],
        cause_type=CauseType.RESOURCE_CONTENTION,
        cause_mechanism="Prometheus NodeHighCpuUsage 告警触发，"
                       "节点 CPU 持续 > 80%，根因为工作负载超载。",
        is_root=True,
        solution_type=SolutionType.SCALE_UP,
        solution_runbook_ref="prometheus-alerts/NodeHighCpuUsage",
        estimated_mttr_min=10,
        lag_seconds=300,
        confidence=0.85,
        effectiveness=0.85,
        source_doc="prometheus-alerts",
    ),
    CausalTriple(
        symptom_type=SymptomType.METRIC_ANOMALY,
        symptom_keywords=["memory", "node", "high"],
        cause_type=CauseType.RESOURCE_CONTENTION,
        cause_mechanism="Prometheus NodeHighMemoryUsage 告警触发，"
                       "节点内存持续 > 90%，根因为内存泄漏或过载。",
        is_root=True,
        solution_type=SolutionType.DRAIN_NODE,
        solution_runbook_ref="prometheus-alerts/NodeHighMemoryUsage",
        estimated_mttr_min=15,
        lag_seconds=600,
        confidence=0.83,
        effectiveness=0.82,
        source_doc="prometheus-alerts",
    ),
    CausalTriple(
        symptom_type=SymptomType.METRIC_ANOMALY,
        symptom_keywords=["latency_ms", "p99", "high", "slow"],
        cause_type=CauseType.NETWORK_PARTITION,
        cause_mechanism="Prometheus HighRequestLatency 告警触发，"
                       "P99 延迟超阈值，根因为网络分区或拥塞。",
        is_root=True,
        solution_type=SolutionType.RUNBOOK_PROCEDURE,
        solution_runbook_ref="prometheus-alerts/HighRequestLatency",
        estimated_mttr_min=20,
        lag_seconds=180,
        confidence=0.75,
        effectiveness=0.7,
        source_doc="prometheus-alerts",
    ),
    CausalTriple(
        symptom_type=SymptomType.EVENT,
        symptom_keywords=["pod", "restart", "crashloop"],
        cause_type=CauseType.MISCONFIGURATION,
        cause_mechanism="Prometheus KubePodCrashLooping 告警触发，"
                       "Pod 30 分钟内重启 > 3 次，根因为配置错误。",
        is_root=True,
        solution_type=SolutionType.ROLLBACK_DEPLOYMENT,
        solution_runbook_ref="prometheus-alerts/KubePodCrashLooping",
        estimated_mttr_min=10,
        lag_seconds=60,
        confidence=0.87,
        effectiveness=0.9,
        source_doc="prometheus-alerts",
    ),
]


# ============================================================
#  中间因链（3/4 跳路径用）
# ============================================================

# 这些骨架描述「症状→中间因→根因」的 3 跳链
# 用于构造 3/4 跳测试集
INTERMEDIATE_CAUSE_CHAINS: list[dict] = [
    {
        "description": "CPU spike → 资源争抢 → 噪声邻居（3跳链）",
        "symptom_keywords": ["cpu_usage", "spike"],
        "intermediate_cause": CauseType.RESOURCE_CONTENTION,
        "intermediate_mechanism": "CPU 资源被多个 VM 竞争，调度延迟增大",
        "root_cause": CauseType.NOISY_NEIGHBOR,
        "root_mechanism": "同宿主机噪声邻居抢占 CPU，导致本 VM 资源争抢",
        "solution_type": SolutionType.DRAIN_NODE,
        "lag_intermediate_seconds": 120,
        "lag_root_seconds": 180,
        "confidence": 0.8,
        "source_doc": "runbook-azure",
    },
    {
        "description": "OOM → 内存不足 → 配置错误（3跳链）",
        "symptom_keywords": ["oom", "memory"],
        "intermediate_cause": CauseType.RESOURCE_CONTENTION,
        "intermediate_mechanism": "容器内存使用达 limit 上限",
        "root_cause": CauseType.MISCONFIGURATION,
        "root_mechanism": "Deployment memory limit 配置过低，未匹配应用峰值需求",
        "solution_type": SolutionType.INCREASE_LIMIT,
        "lag_intermediate_seconds": 60,
        "lag_root_seconds": 90,
        "confidence": 0.85,
        "source_doc": "k8s-docs/pod-lifecycle",
    },
    {
        "description": "延迟突增 → 依赖失败 → 网络分区（4跳链）",
        "symptom_keywords": ["latency_ms", "delay"],
        "intermediate_cause": CauseType.DEPENDENCY_FAILURE,
        "intermediate_mechanism": "上游依赖服务响应缓慢",
        "root_cause": CauseType.NETWORK_PARTITION,
        "root_mechanism": "网络分区导致节点间通信中断，依赖服务不可达",
        "solution_type": SolutionType.RUNBOOK_PROCEDURE,
        "lag_intermediate_seconds": 300,
        "lag_root_seconds": 600,
        "confidence": 0.75,
        "source_doc": "runbook-azure",
    },
]


# ============================================================
#  对外 API
# ============================================================

def get_all_causal_triples() -> list[CausalTriple]:
    """获取全部因果骨架种子（K8s + Prometheus）。"""
    return K8S_CAUSAL_TRIPLES + PROMETHEUS_CAUSAL_TRIPLES


def get_k8s_causal_triples() -> list[CausalTriple]:
    """获取 K8s 文档因果骨架。"""
    return K8S_CAUSAL_TRIPLES


def get_prometheus_causal_triples() -> list[CausalTriple]:
    """获取 Prometheus alerting rules 因果骨架。"""
    return PROMETHEUS_CAUSAL_TRIPLES


def get_intermediate_chains() -> list[dict]:
    """获取中间因链（3/4 跳路径用）。"""
    return INTERMEDIATE_CAUSE_CHAINS


def match_cause_for_event(
    metric_name: str,
    event_type: str,
    triples: list[CausalTriple] | None = None,
) -> list[CausalTriple]:
    """根据故障事件的指标名/事件类型匹配因果骨架。

    匹配逻辑：metric_name 或 event_type 命中 triple 的 symptom_keywords
    则视为匹配。返回所有匹配的 triple，按 confidence 降序。

    Parameters
    ----------
    metric_name : str
        故障事件的指标名，如 "cpu_usage"
    event_type : str
        故障事件类型，如 "cpu_spike" / "vm_deletion"
    triples : list[CausalTriple] | None
        候选 triple 列表，None=全部
    """
    if triples is None:
        triples = get_all_causal_triples()

    target = f"{metric_name} {event_type}".lower()
    matched: list[tuple[float, CausalTriple]] = []
    for triple in triples:
        for kw in triple.symptom_keywords:
            if kw.lower() in target:
                matched.append((triple.confidence, triple))
                break

    # 按 confidence 降序
    matched.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in matched]


def get_cause_type_coverage() -> dict[str, int]:
    """统计因果骨架对 7 种 CauseType 的覆盖情况。

    用于验证测试集因果类型均衡分布。
    """
    triples = get_all_causal_triples()
    coverage: dict[str, int] = {}
    for triple in triples:
        ct = triple.cause_type.value
        coverage[ct] = coverage.get(ct, 0) + 1
    return coverage
