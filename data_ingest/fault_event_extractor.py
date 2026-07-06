"""故障事件抽取器 —— 把原始异常点聚合成结构化 FaultEvent。

azure_trace_loader 输出的 AnomalyPoint 是「单点/单窗口」的异常观察，
但一次故障通常表现为多个关联信号（CPU spike + VM 删除 + 高方差）。
本模块负责：

1. 聚合同一 VM 在时间窗内相关的 AnomalyPoint 为一个 FaultEvent
2. 关联 CPU spike 与 VM 删除（删除紧跟 spike = 复合故障）
3. 根据偏离倍数划分 severity
4. 提取故障窗口内的原始 trace 片段（provenance 溯源用）
5. 生成唯一 event_id

聚合策略：
- 同一 VM 的 CPU spike 间隔 < merge_window_seconds 合并为一个事件
- VM 删除若发生在 spike 后 link_window_seconds 内，合并为复合事件
- HIGH_VARIANCE 异常单独成事件（噪声邻居特征）
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from data_ingest.models import (
    AnomalyPoint,
    AnomalyType,
    FaultEvent,
    FaultEventType,
    VMTimeSeries,
)
from graph_schema.nodes import ComponentType, Severity


# ============================================================
#  聚合参数
# ============================================================

# 同一 VM 的两个 spike 间隔小于此值则合并（秒）
DEFAULT_MERGE_WINDOW_SECONDS = 600  # 10 分钟

# VM 删除与 spike 的时间窗（删除在 spike 后此范围内视为关联）
DEFAULT_LINK_WINDOW_SECONDS = 1800  # 30 分钟

# trace 片段提取的上下文窗口（故障前后各扩展的点数）
DEFAULT_CONTEXT_PADDING_POINTS = 6  # 前 30 分钟 + 后 30 分钟


# ============================================================
#  严重程度分级
# ============================================================

def classify_severity(
    deviation_ratio: float,
    anomaly_type: AnomalyType,
    is_vm_deleted: bool = False,
) -> Severity:
    """根据偏离倍数与异常类型划分严重程度。

    分级规则：
    - VM 删除 = critical（强故障信号）
    - deviation >= 0.5 (高于基线 50%) = critical
    - deviation >= 0.2 = warning
    - 其余 = info

    Parameters
    ----------
    deviation_ratio : float
        偏离倍数 (observed - baseline) / baseline
    anomaly_type : AnomalyType
        异常类型
    is_vm_deleted : bool
        是否关联了 VM 删除
    """
    if anomaly_type == AnomalyType.VM_DELETION or is_vm_deleted:
        return Severity.CRITICAL
    if deviation_ratio >= 0.5:
        return Severity.CRITICAL
    if deviation_ratio >= 0.2:
        return Severity.WARNING
    return Severity.INFO


# ============================================================
#  event_id 生成
# ============================================================

def make_event_id(cluster_id: str, vm_id: str, timestamp: datetime, event_type: str) -> str:
    """生成故障事件唯一 ID。

    格式：fault_<cluster>_<vm>_<event_type>_<ts>
    时间戳用 ISO 紧凑格式确保唯一性。
    """
    ts_str = timestamp.strftime("%Y%m%dT%H%M%S")
    # vm_id 可能很长，取后 8 位
    vm_suffix = vm_id[-8:] if len(vm_id) > 8 else vm_id
    return f"fault_{cluster_id}_{vm_suffix}_{event_type}_{ts_str}"


# ============================================================
#  trace 片段提取
# ============================================================

def extract_trace_fragment(
    ts: VMTimeSeries,
    window_start: datetime,
    window_end: datetime | None,
    padding_points: int = DEFAULT_CONTEXT_PADDING_POINTS,
) -> list[tuple[datetime, float]]:
    """提取故障窗口内的 CPU trace 片段，供 provenance 溯源。

    在故障窗口前后各扩展 padding_points 个点，保留上下文。

    Parameters
    ----------
    ts : VMTimeSeries
        VM 时序数据
    window_start : datetime
        故障开始时刻
    window_end : datetime | None
        故障结束时刻，None=取到末尾
    padding_points : int
        前后扩展的点数
    """
    if not ts.cpu_readings:
        return []

    # 找到窗口起点索引
    start_idx = 0
    for i, (dt, _) in enumerate(ts.cpu_readings):
        if dt >= window_start:
            start_idx = i
            break

    # 找到窗口终点索引
    end_idx = len(ts.cpu_readings) - 1
    if window_end is not None:
        for i, (dt, _) in enumerate(ts.cpu_readings):
            if dt > window_end:
                end_idx = i - 1
                break

    # 扩展上下文
    padded_start = max(0, start_idx - padding_points)
    padded_end = min(len(ts.cpu_readings) - 1, end_idx + padding_points)

    return ts.cpu_readings[padded_start:padded_end + 1]


# ============================================================
#  故障事件抽取器
# ============================================================

class FaultEventExtractor:
    """故障事件抽取器 —— 把 AnomalyPoint 聚合成 FaultEvent。

    使用示例：
        extractor = FaultEventExtractor()
        for ts, anomalies in loader.load_and_detect(csv_path, max_vms=100):
            events = extractor.extract(ts, anomalies)
            for ev in events:
                print(ev.event_id, ev.event_type, ev.severity)

    聚合逻辑：
    1. 按 anomaly_type 分组
    2. CPU spike 按时间窗合并（间隔 < merge_window 的合并）
    3. VM 删除若在 spike 后 link_window 内，合并为复合事件
    4. HIGH_VARIANCE 单独成事件
    """

    def __init__(
        self,
        merge_window_seconds: int = DEFAULT_MERGE_WINDOW_SECONDS,
        link_window_seconds: int = DEFAULT_LINK_WINDOW_SECONDS,
        context_padding_points: int = DEFAULT_CONTEXT_PADDING_POINTS,
    ):
        self.merge_window_seconds = merge_window_seconds
        self.link_window_seconds = link_window_seconds
        self.context_padding_points = context_padding_points

    def extract(
        self,
        ts: VMTimeSeries,
        anomalies: list[AnomalyPoint],
    ) -> list[FaultEvent]:
        """从异常点列表抽取结构化故障事件。

        Parameters
        ----------
        ts : VMTimeSeries
            VM 时序数据（用于提取 trace 片段与元信息）
        anomalies : list[AnomalyPoint]
            该 VM 的所有异常点
        """
        if not anomalies:
            return []

        # 按 VM 分组（防御性，正常情况所有 anomaly 都属于同一 VM）
        vm_anomalies = [a for a in anomalies if a.vm_id == ts.vm_id]
        if not vm_anomalies:
            return []

        # 按类型分组
        spikes = [a for a in vm_anomalies if a.anomaly_type == AnomalyType.CPU_SPIKE]
        deletions = [a for a in vm_anomalies if a.anomaly_type == AnomalyType.VM_DELETION]
        variances = [a for a in vm_anomalies if a.anomaly_type == AnomalyType.HIGH_VARIANCE]

        # 合并相邻的 spike
        merged_spikes = self._merge_adjacent_spikes(spikes)

        # 找出删除事件（VM 最多一个删除事件）
        deletion = deletions[0] if deletions else None

        events: list[FaultEvent] = []

        # 处理 spike 事件（可能关联删除）
        for spike in merged_spikes:
            linked_deletion = self._find_linked_deletion(spike, deletion)
            event = self._build_spike_event(ts, spike, linked_deletion)
            events.append(event)

        # 如果有删除但没关联到任何 spike，单独生成一个删除事件
        if deletion is not None:
            already_linked = any(
                e.event_type == FaultEventType.VM_DELETION
                or (e.timestamp_end == deletion.timestamp)
                for e in events
            )
            if not already_linked:
                events.append(self._build_deletion_event(ts, deletion))

        # 处理高方差事件
        for var in variances:
            events.append(self._build_variance_event(ts, var))

        # 按 timestamp_start 排序
        events.sort(key=lambda e: e.timestamp_start)
        return events

    def _merge_adjacent_spikes(self, spikes: list[AnomalyPoint]) -> list[AnomalyPoint]:
        """合并相邻的 CPU spike（间隔 < merge_window 的合并成一个）。"""
        if not spikes:
            return []
        spikes_sorted = sorted(spikes, key=lambda a: a.timestamp)
        merged: list[AnomalyPoint] = [spikes_sorted[0]]

        for current in spikes_sorted[1:]:
            last = merged[-1]
            # 如果当前 spike 与上一个的 end_timestamp 间隔小于 merge_window，合并
            last_end = last.end_timestamp or last.timestamp
            gap = (current.timestamp - last_end).total_seconds()
            if gap <= self.merge_window_seconds:
                # 合并：扩展窗口，取更大峰值
                merged[-1] = AnomalyPoint(
                    anomaly_type=AnomalyType.CPU_SPIKE,
                    vm_id=last.vm_id,
                    cluster_id=last.cluster_id,
                    timestamp=last.timestamp,
                    end_timestamp=current.end_timestamp or current.timestamp,
                    observed_value=max(last.observed_value, current.observed_value),
                    baseline_value=last.baseline_value,
                    threshold=last.threshold,
                    detection_method=last.detection_method,
                    duration_seconds=int((
                        (current.end_timestamp or current.timestamp) - last.timestamp
                    ).total_seconds()),
                )
            else:
                merged.append(current)

        return merged

    def _find_linked_deletion(
        self,
        spike: AnomalyPoint,
        deletion: AnomalyPoint | None,
    ) -> AnomalyPoint | None:
        """检查 VM 删除是否在 spike 后的 link_window 内。

        删除在 spike 结束后 link_window 秒内发生 → 视为关联（复合故障）。
        """
        if deletion is None:
            return None
        spike_end = spike.end_timestamp or spike.timestamp
        gap = (deletion.timestamp - spike_end).total_seconds()
        # 删除在 spike 之后 0 ~ link_window 秒内
        if 0 <= gap <= self.link_window_seconds:
            return deletion
        # 删除在 spike 之前（反向因果，spike 是删除前兆）
        if -self.link_window_seconds <= gap < 0:
            return deletion
        return None

    def _build_spike_event(
        self,
        ts: VMTimeSeries,
        spike: AnomalyPoint,
        linked_deletion: AnomalyPoint | None,
    ) -> FaultEvent:
        """构建 CPU spike 故障事件（可能关联 VM 删除）。"""
        is_deleted = linked_deletion is not None
        severity = classify_severity(
            deviation_ratio=spike.deviation_ratio,
            anomaly_type=spike.anomaly_type,
            is_vm_deleted=is_deleted,
        )

        # 事件结束时间：如有关联删除，用删除时刻；否则用 spike 结束
        timestamp_end = None
        if linked_deletion is not None:
            timestamp_end = linked_deletion.timestamp
        elif spike.end_timestamp is not None:
            timestamp_end = spike.end_timestamp

        # 事件类型：有删除 = VM_DELETION，否则 CPU_SPIKE
        event_type = FaultEventType.VM_DELETION if is_deleted else FaultEventType.CPU_SPIKE

        # 提取 trace 片段
        trace_fragment = extract_trace_fragment(
            ts=ts,
            window_start=spike.timestamp,
            window_end=timestamp_end,
            padding_points=self.context_padding_points,
        )

        event_id = make_event_id(
            cluster_id=ts.cluster_id,
            vm_id=ts.vm_id,
            timestamp=spike.timestamp,
            event_type=event_type.value,
        )

        return FaultEvent(
            event_id=event_id,
            event_type=event_type,
            vm_id=ts.vm_id,
            cluster_id=ts.cluster_id,
            timestamp_start=spike.timestamp,
            timestamp_end=timestamp_end,
            severity=severity,
            component_type=ComponentType.VM,
            sku=ts.sku,
            vcore_bucket=ts.vcore_bucket,
            memory_gb_bucket=ts.memory_gb_bucket,
            metric_name="cpu_usage",
            observed_value=spike.observed_value,
            baseline_value=spike.baseline_value,
            threshold=spike.threshold,
            trace_fragment=trace_fragment,
            detection_method=spike.detection_method,
            source_dataset="azure_v2",
        )

    def _build_deletion_event(
        self,
        ts: VMTimeSeries,
        deletion: AnomalyPoint,
    ) -> FaultEvent:
        """构建单独的 VM 删除事件（无关联 spike）。"""
        trace_fragment = extract_trace_fragment(
            ts=ts,
            window_start=deletion.timestamp - timedelta(minutes=30),
            window_end=deletion.timestamp,
            padding_points=self.context_padding_points,
        )

        event_id = make_event_id(
            cluster_id=ts.cluster_id,
            vm_id=ts.vm_id,
            timestamp=deletion.timestamp,
            event_type="vm_deletion",
        )

        return FaultEvent(
            event_id=event_id,
            event_type=FaultEventType.VM_DELETION,
            vm_id=ts.vm_id,
            cluster_id=ts.cluster_id,
            timestamp_start=deletion.timestamp,
            timestamp_end=None,
            severity=Severity.CRITICAL,
            component_type=ComponentType.VM,
            sku=ts.sku,
            vcore_bucket=ts.vcore_bucket,
            memory_gb_bucket=ts.memory_gb_bucket,
            metric_name="vm_deleted",
            observed_value=deletion.observed_value,
            baseline_value=deletion.baseline_value,
            threshold=deletion.threshold,
            trace_fragment=trace_fragment,
            detection_method=deletion.detection_method,
            source_dataset="azure_v2",
        )

    def _build_variance_event(
        self,
        ts: VMTimeSeries,
        var: AnomalyPoint,
    ) -> FaultEvent:
        """构建高方差事件（噪声邻居特征）。"""
        trace_fragment = extract_trace_fragment(
            ts=ts,
            window_start=var.timestamp,
            window_end=var.end_timestamp,
            padding_points=self.context_padding_points,
        )

        event_id = make_event_id(
            cluster_id=ts.cluster_id,
            vm_id=ts.vm_id,
            timestamp=var.timestamp,
            event_type="cpu_spike",  # 高方差表现为 CPU 抖动
        )

        return FaultEvent(
            event_id=event_id,
            event_type=FaultEventType.CPU_SPIKE,
            vm_id=ts.vm_id,
            cluster_id=ts.cluster_id,
            timestamp_start=var.timestamp,
            timestamp_end=var.end_timestamp,
            severity=classify_severity(
                deviation_ratio=var.deviation_ratio,
                anomaly_type=var.anomaly_type,
            ),
            component_type=ComponentType.VM,
            sku=ts.sku,
            vcore_bucket=ts.vcore_bucket,
            memory_gb_bucket=ts.memory_gb_bucket,
            metric_name="cpu_variance",
            observed_value=var.observed_value,
            baseline_value=var.baseline_value,
            threshold=var.threshold,
            trace_fragment=trace_fragment,
            detection_method=var.detection_method,
            source_dataset="azure_v2",
        )


# ============================================================
#  便捷函数
# ============================================================

def extract_fault_events(
    ts: VMTimeSeries,
    anomalies: list[AnomalyPoint],
    merge_window_seconds: int = DEFAULT_MERGE_WINDOW_SECONDS,
    link_window_seconds: int = DEFAULT_LINK_WINDOW_SECONDS,
) -> list[FaultEvent]:
    """便捷函数：从单个 VM 的异常点抽取故障事件。"""
    extractor = FaultEventExtractor(
        merge_window_seconds=merge_window_seconds,
        link_window_seconds=link_window_seconds,
    )
    return extractor.extract(ts, anomalies)
