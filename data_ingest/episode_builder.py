"""Episode 构建器 —— 把故障事件与因果骨架打包成 Graphiti episode。

本模块是数据接入管线的核心环节：
    FaultEvent + CausalTriple → EpisodePayload → graphiti_writer

关键设计：
1. reference_time = 故障发生时刻（timestamp_start）
   这是时态因果建模的核心——Graphiti 用它作为 EntityEdge.valid_at 默认值。
2. episode_body 用结构化 JSON 字符串，包含：
   - 故障描述（自然语言，供 LLM 抽取实体）
   - 因果链（症状→因→解法，供 LLM 建边）
   - trace 片段（provenance 溯源）
3. group_id 按故障场景隔离，格式：{cluster_id}_{event_id}
4. 一次故障事件构建一个 episode，保证时态锚点精确

Graphiti add_episode 参数说明：
    name: episode 唯一名称
    episode_body: 正文（文本或 JSON）
    source: EpisodeType.json / .text / .message
    reference_time: 时态锚点
    group_id: 分区标识
    entity_types: 自定义节点类型（来自 ENTITY_TYPES）
    edge_types: 自定义边类型（来自 EDGE_TYPES）

episode_body 设计为半结构化文本，包含自然语言描述 + JSON 元数据，
这样 Graphiti 的 LLM 既能理解语义抽取实体，又能读取结构化字段建边。
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from data_ingest.doc_skeleton_seeder import (
    get_all_causal_triples,
    match_cause_for_event,
)
from data_ingest.models import (
    CausalTriple,
    EpisodePayload,
    FaultEvent,
    FaultEventType,
)
from graph_schema.nodes import (
    CauseType,
    SolutionType,
    SymptomType,
)


# ============================================================
#  episode_body 生成
# ============================================================

def _format_trace_fragment(trace: list[tuple[datetime, float]], max_points: int = 20) -> str:
    """格式化 trace 片段为可读字符串（截断避免过长）。"""
    if not trace:
        return "(无 trace 数据)"
    if len(trace) > max_points:
        # 取前 5 + 中间 + 后 5
        head = trace[:5]
        tail = trace[-5:]
        mid_idx = len(trace) // 2
        mid = trace[mid_idx:mid_idx + 2]
        parts = head + [None] + mid + [None] + tail
        lines = []
        for p in parts:
            if p is None:
                lines.append("    ... (省略)")
            else:
                lines.append(f"    {p[0].isoformat()} cpu={p[1]:.3f}")
        return "\n".join(lines)
    return "\n".join(f"    {ts.isoformat()} cpu={cpu:.3f}" for ts, cpu in trace)


def _severity_to_text(severity: str) -> str:
    """严重程度转中文描述。"""
    return {
        "info": "信息级",
        "warning": "警告级",
        "critical": "严重级",
    }.get(severity, severity)


def build_fault_episode_body(
    event: FaultEvent,
    matched_triple: CausalTriple | None,
) -> str:
    """构建故障 episode 的正文（半结构化文本 + JSON 元数据）。

    正文包含三部分：
    1. 自然语言故障描述（供 Graphiti LLM 抽取实体与关系）
    2. 因果链说明（症状→因→解法）
    3. provenance 溯源信息（trace 片段 + 文档引用）

    Parameters
    ----------
    event : FaultEvent
        故障事件
    matched_triple : CausalTriple | None
        匹配到的因果骨架（可能为 None，表示无文档先验）
    """
    # 自然语言描述
    severity_text = _severity_to_text(event.severity.value)
    desc_lines = [
        f"【故障事件】{event.event_id}",
        f"时间: {event.timestamp_start.isoformat()} ~ "
        f"{event.timestamp_end.isoformat() if event.timestamp_end else '未恢复'}",
        f"严重程度: {severity_text}",
        f"组件: {event.component_type.value} (id={event.vm_id}, cluster={event.cluster_id})",
        f"症状: {event.event_type.value} - {event.metric_name}",
        f"  观测值: {event.observed_value:.3f}, 基线: {event.baseline_value:.3f}, "
        f"阈值: {event.threshold:.3f}",
        f"检测方法: {event.detection_method}",
        f"数据来源: {event.source_dataset}",
    ]

    if event.sku:
        desc_lines.append(f"VM 规格: {event.sku}")
    if event.vcore_bucket:
        desc_lines.append(f"vCPU 桶: {event.vcore_bucket}")
    if event.memory_gb_bucket:
        desc_lines.append(f"内存桶: {event.memory_gb_bucket}")

    # 因果链说明
    cause_lines = ["", "【因果分析】"]
    if matched_triple is not None:
        cause_lines.extend([
            f"症状类型: {matched_triple.symptom_type.value}",
            f"根因类型: {matched_triple.cause_type.value}",
            f"因果机制: {matched_triple.cause_mechanism}",
            f"因果置信度: {matched_triple.confidence}",
            f"因果时延: {matched_triple.lag_seconds} 秒",
            f"解法类型: {matched_triple.solution_type.value}",
            f"运维文档: {matched_triple.solution_runbook_ref}",
            f"预估修复时间: {matched_triple.estimated_mttr_min} 分钟",
            f"解法有效率: {matched_triple.effectiveness}",
            f"文档来源: {matched_triple.source_doc}",
        ])
    else:
        cause_lines.append("未匹配到文档因果骨架（需 LLM 推理）")

    # provenance 溯源
    provenance_lines = ["", "【Provenance 溯源】"]
    if event.trace_fragment:
        provenance_lines.append("原始 CPU trace 片段:")
        provenance_lines.append(_format_trace_fragment(event.trace_fragment))
    else:
        provenance_lines.append("(无 trace 片段)")

    # JSON 元数据（供程序化解析）
    metadata: dict[str, Any] = {
        "event_id": event.event_id,
        "event_type": event.event_type.value,
        "vm_id": event.vm_id,
        "cluster_id": event.cluster_id,
        "timestamp_start": event.timestamp_start.isoformat(),
        "timestamp_end": event.timestamp_end.isoformat() if event.timestamp_end else None,
        "severity": event.severity.value,
        "component_type": event.component_type.value,
        "metric_name": event.metric_name,
        "observed_value": event.observed_value,
        "baseline_value": event.baseline_value,
        "threshold": event.threshold,
        "detection_method": event.detection_method,
        "source_dataset": event.source_dataset,
    }
    if event.sku:
        metadata["sku"] = event.sku
    if event.vcore_bucket:
        metadata["vcore_bucket"] = event.vcore_bucket
    if event.memory_gb_bucket:
        metadata["memory_gb_bucket"] = event.memory_gb_bucket

    if matched_triple is not None:
        metadata["cause"] = {
            "cause_type": matched_triple.cause_type.value,
            "is_root": matched_triple.is_root,
            "mechanism": matched_triple.cause_mechanism,
            "confidence": matched_triple.confidence,
            "lag_seconds": matched_triple.lag_seconds,
            "source_doc": matched_triple.source_doc,
        }
        metadata["solution"] = {
            "solution_type": matched_triple.solution_type.value,
            "runbook_ref": matched_triple.solution_runbook_ref,
            "estimated_mttr_min": matched_triple.estimated_mttr_min,
            "effectiveness": matched_triple.effectiveness,
        }

    metadata_lines = ["", "【结构化元数据 JSON】", "```json", json.dumps(metadata, ensure_ascii=False, indent=2), "```"]

    return "\n".join(desc_lines + cause_lines + provenance_lines + metadata_lines)


def build_skeleton_episode_body(triple: CausalTriple) -> str:
    """构建因果骨架 episode 的正文（用于先验知识种子）。

    骨架 episode 不绑定具体故障事件，作为领域知识写入图谱，
    供后续故障事件挂载。
    """
    desc_lines = [
        f"【因果骨架种子】{triple.source_doc}/{triple.cause_type.value}",
        f"症状类型: {triple.symptom_type.value}",
        f"症状关键词: {', '.join(triple.symptom_keywords)}",
        f"根因类型: {triple.cause_type.value} (is_root={triple.is_root})",
        f"因果机制: {triple.cause_mechanism}",
        f"解法类型: {triple.solution_type.value}",
        f"运维文档: {triple.solution_runbook_ref}",
        f"预估修复时间: {triple.estimated_mttr_min} 分钟",
        f"因果置信度: {triple.confidence}",
        f"因果时延: {triple.lag_seconds} 秒",
        f"解法有效率: {triple.effectiveness}",
        f"文档来源: {triple.source_doc}",
    ]

    metadata = {
        "type": "causal_skeleton",
        "symptom_type": triple.symptom_type.value,
        "symptom_keywords": triple.symptom_keywords,
        "cause_type": triple.cause_type.value,
        "is_root": triple.is_root,
        "mechanism": triple.cause_mechanism,
        "solution_type": triple.solution_type.value,
        "runbook_ref": triple.solution_runbook_ref,
        "estimated_mttr_min": triple.estimated_mttr_min,
        "confidence": triple.confidence,
        "lag_seconds": triple.lag_seconds,
        "effectiveness": triple.effectiveness,
        "source_doc": triple.source_doc,
    }

    metadata_lines = ["", "【结构化元数据 JSON】", "```json", json.dumps(metadata, ensure_ascii=False, indent=2), "```"]

    return "\n".join(desc_lines + metadata_lines)


# ============================================================
#  Episode 构建器
# ============================================================

class EpisodeBuilder:
    """Episode 构建器 —— 把 FaultEvent + CausalTriple 打包成 EpisodePayload。

    使用示例：
        builder = EpisodeBuilder()
        episodes = builder.build_from_fault_events(events)
        for ep in episodes:
            await writer.write(ep)
    """

    def __init__(
        self,
        causal_triples: list[CausalTriple] | None = None,
        skeleton_reference_time: datetime | None = None,
    ):
        """
        Parameters
        ----------
        causal_triples : list[CausalTriple] | None
            因果骨架列表，None=加载全部默认骨架
        skeleton_reference_time : datetime | None
            骨架 episode 的时态锚点，None=用当前时间
        """
        self.causal_triples = causal_triples if causal_triples is not None else get_all_causal_triples()
        self.skeleton_reference_time = skeleton_reference_time or datetime.now()

    def build_from_fault_event(self, event: FaultEvent) -> EpisodePayload:
        """把单个故障事件构建成 episode。

        reference_time = event.timestamp_start（时态锚点 = 故障发生时刻）
        group_id = {cluster_id}_{event_id}（按故障场景隔离）
        """
        # 匹配因果骨架
        matched = match_cause_for_event(
            metric_name=event.metric_name,
            event_type=event.event_type.value,
            triples=self.causal_triples,
        )
        matched_triple = matched[0] if matched else None

        # 回填到 event（便于后续使用）
        if matched_triple is not None:
            event.linked_cause_type = matched_triple.cause_type
            event.linked_solution_type = matched_triple.solution_type

        # 构建 episode_body
        body = build_fault_episode_body(event, matched_triple)

        # 构建 metadata
        metadata: dict[str, Any] = {
            "event_id": event.event_id,
            "event_type": event.event_type.value,
            "vm_id": event.vm_id,
            "cluster_id": event.cluster_id,
            "severity": event.severity.value,
            "hop_count": 2,  # 默认 2 跳（症状→根因→解法）
        }
        if matched_triple is not None:
            metadata["cause_type"] = matched_triple.cause_type.value
            metadata["solution_type"] = matched_triple.solution_type.value
            metadata["confidence"] = matched_triple.confidence
            metadata["source_doc"] = matched_triple.source_doc

        return EpisodePayload(
            name=f"episode_{event.event_id}",
            episode_body=body,
            reference_time=event.timestamp_start,
            group_id=f"{event.cluster_id}_{event.event_id}",
            source_description=f"Fault event from {event.source_dataset}: {event.event_type.value} on {event.vm_id}",
            metadata=metadata,
        )

    def build_from_fault_events(self, events: list[FaultEvent]) -> list[EpisodePayload]:
        """批量构建故障事件 episode。"""
        return [self.build_from_fault_event(e) for e in events]

    def build_skeleton_episodes(self) -> list[EpisodePayload]:
        """构建因果骨架 episode（先验知识种子）。

        骨架 episode 不绑定具体故障，用统一的 group_id 与较早的 reference_time，
        确保它们在图谱中先于故障事件存在，供故障事件挂载。
        """
        episodes: list[EpisodePayload] = []
        for i, triple in enumerate(self.causal_triples):
            body = build_skeleton_episode_body(triple)
            episodes.append(EpisodePayload(
                name=f"skeleton_{triple.source_doc}_{triple.cause_type.value}_{i}",
                episode_body=body,
                reference_time=self.skeleton_reference_time,
                group_id="skeleton_prior_knowledge",
                source_description=f"Causal skeleton from {triple.source_doc}",
                metadata={
                    "type": "causal_skeleton",
                    "cause_type": triple.cause_type.value,
                    "symptom_type": triple.symptom_type.value,
                    "solution_type": triple.solution_type.value,
                    "source_doc": triple.source_doc,
                    "hop_count": 2,
                },
            ))
        return episodes


# ============================================================
#  便捷函数
# ============================================================

def build_episodes(
    events: list[FaultEvent],
    include_skeletons: bool = True,
) -> list[EpisodePayload]:
    """便捷函数：从故障事件构建全部 episode。

    Parameters
    ----------
    events : list[FaultEvent]
        故障事件列表
    include_skeletons : bool
        是否包含因果骨架 episode
    """
    builder = EpisodeBuilder()
    episodes = builder.build_from_fault_events(events)
    if include_skeletons:
        episodes = builder.build_skeleton_episodes() + episodes
    return episodes
