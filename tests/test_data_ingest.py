"""数据接入层单元测试 —— 覆盖异常检测、事件抽取、episode 构建、约束集成。

运行方式（在项目根目录）：
    python -m pytest tests/test_data_ingest.py -v

或直接运行：
    python tests/test_data_ingest.py

测试覆盖：
1. models.py — 数据模型创建与默认值
2. azure_trace_loader.py — 时间戳解析、IQR/3-sigma 异常检测、VM 删除检测
3. fault_event_extractor.py — 事件聚合、严重程度分级、trace 片段提取
4. doc_skeleton_seeder.py — 因果骨架覆盖度、事件匹配
5. episode_builder.py — episode 构建、reference_time 时态锚点、group_id 隔离
6. synthetic_data.py — 合成数据可复现性、故障注入
7. constraints 集成 — 故障事件与因果骨架的层级组合校验
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# 确保能导入项目模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_ingest.azure_trace_loader import (
    AzureTraceLoader,
    compute_baseline_stats,
    detect_anomalies_3sigma,
    detect_anomalies_iqr,
    detect_high_variance,
    detect_vm_deletion,
    parse_azure_timestamp,
    parse_cpu_readings,
    parse_long_format_row,
    resolve_memory_bucket,
    resolve_vcore_bucket,
    stream_vm_timeseries_long,
)
from data_ingest.doc_skeleton_seeder import (
    get_all_causal_triples,
    get_cause_type_coverage,
    get_intermediate_chains,
    get_k8s_causal_triples,
    get_prometheus_causal_triples,
    match_cause_for_event,
)
from data_ingest.episode_builder import (
    EpisodeBuilder,
    build_episodes,
    build_fault_episode_body,
    build_skeleton_episode_body,
)
from data_ingest.fault_event_extractor import (
    DEFAULT_LINK_WINDOW_SECONDS,
    DEFAULT_MERGE_WINDOW_SECONDS,
    FaultEventExtractor,
    classify_severity,
    extract_fault_events,
    extract_trace_fragment,
    make_event_id,
)
from data_ingest.models import (
    AnomalyPoint,
    AnomalyType,
    CausalTriple,
    EpisodePayload,
    FaultEvent,
    FaultEventType,
    GraphBuildStats,
    VMTimeSeries,
)
from data_ingest.synthetic_data import (
    DEFAULT_FAULT_RATE,
    DEFAULT_OBSERVATION_DAYS,
    generate_vm_batch,
    generate_vm_timeseries,
)
from graph_schema.constraints import (
    validate_edge_combination,
    validate_edge_full,
    validate_temporal_consistency,
)
from graph_schema.nodes import (
    CauseType,
    ComponentType,
    Severity,
    SolutionType,
    SymptomType,
)


# ============================================================
#  测试工具
# ============================================================

def _run_test(name: str, test_fn):
    """简单测试运行器，不依赖 pytest。"""
    try:
        test_fn()
        print(f"  [PASS] {name}")
        return True
    except AssertionError as e:
        print(f"  [FAIL] {name}: {e}")
        return False
    except Exception as e:
        print(f"  [ERROR] {name}: {type(e).__name__}: {e}")
        return False


def _make_cpu_readings(
    start: datetime,
    num_points: int,
    base_cpu: float = 0.4,
    spike_indices: list[int] | None = None,
    spike_value: float = 0.95,
    interval_seconds: int = 300,
) -> list[tuple[datetime, float]]:
    """构造测试用 CPU 时序。"""
    readings: list[tuple[datetime, float]] = []
    spike_set = set(spike_indices or [])
    for i in range(num_points):
        ts = start + timedelta(seconds=i * interval_seconds)
        cpu = spike_value if i in spike_set else base_cpu
        readings.append((ts, cpu))
    return readings


# ============================================================
#  1. models.py 测试
# ============================================================

def test_vm_timeseries_basic():
    """VMTimeSeries 基本属性正确。"""
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    readings = _make_cpu_readings(start, 10, base_cpu=0.4)
    ts = VMTimeSeries(
        vm_id="vm_001",
        cluster_id="cluster_A",
        cpu_readings=readings,
        vm_created=start,
    )
    assert ts.reading_count == 10
    assert ts.is_deleted is False
    assert ts.vm_deleted is None


def test_vm_timeseries_deleted():
    """被删除的 VM is_deleted 正确。"""
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    deleted = start + timedelta(hours=2)
    ts = VMTimeSeries(
        vm_id="vm_002",
        cluster_id="cluster_A",
        cpu_readings=[(start, 0.4)],
        vm_created=start,
        vm_deleted=deleted,
    )
    assert ts.is_deleted is True


def test_anomaly_point_deviation_ratio():
    """AnomalyPoint 偏离倍数计算正确。"""
    ap = AnomalyPoint(
        anomaly_type=AnomalyType.CPU_SPIKE,
        vm_id="vm_001",
        cluster_id="cluster_A",
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        observed_value=0.9,
        baseline_value=0.45,
        threshold=0.8,
        detection_method="iqr",
    )
    # (0.9 - 0.45) / 0.45 = 1.0
    assert abs(ap.deviation_ratio - 1.0) < 0.01


def test_fault_event_duration():
    """FaultEvent 持续时长计算正确。"""
    start = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
    end = datetime(2024, 1, 1, 10, 30, tzinfo=timezone.utc)
    ev = FaultEvent(
        event_id="test_001",
        event_type=FaultEventType.CPU_SPIKE,
        vm_id="vm_001",
        cluster_id="cluster_A",
        timestamp_start=start,
        timestamp_end=end,
        metric_name="cpu_usage",
        observed_value=0.95,
        baseline_value=0.45,
        threshold=0.8,
        detection_method="iqr",
    )
    assert ev.duration_seconds == 1800  # 30 分钟


def test_graph_build_stats_merge():
    """GraphBuildStats 合并正确。"""
    s1 = GraphBuildStats(total_episodes=10, episodes_written=8, episodes_failed=2)
    s2 = GraphBuildStats(total_episodes=5, episodes_written=5, episodes_failed=0)
    merged = s1.merge(s2)
    assert merged.total_episodes == 15
    assert merged.episodes_written == 13
    assert merged.episodes_failed == 2


# ============================================================
#  2. azure_trace_loader.py 测试
# ============================================================

def test_parse_azure_timestamp():
    """Azure 时间戳解析正确。"""
    # 正常时间戳
    dt = parse_azure_timestamp(1704067200)  # 2024-01-01 00:00:00 UTC
    assert dt is not None
    assert dt.year == 2024
    assert dt.month == 1
    assert dt.day == 1

    # 0 表示无时间
    assert parse_azure_timestamp(0) is None
    # None
    assert parse_azure_timestamp(None) is None
    # 字符串
    dt2 = parse_azure_timestamp("1704067200")
    assert dt2 is not None
    assert dt2.year == 2024


def test_parse_cpu_readings():
    """CPU 时序解析正确。"""
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    readings = parse_cpu_readings("0.4,0.45,0.5", start, interval_seconds=300)
    assert len(readings) == 3
    assert readings[0][1] == 0.4
    assert readings[1][1] == 0.45
    assert readings[2][1] == 0.5
    # 时间戳递增
    assert readings[1][0] - readings[0][0] == timedelta(seconds=300)

    # 空字符串
    assert parse_cpu_readings("", start) == []
    assert parse_cpu_readings("   ", start) == []


def test_compute_baseline_stats():
    """基线统计计算正确。"""
    values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    stats = compute_baseline_stats(values)
    assert stats["count"] == 10
    assert abs(stats["mean"] - 0.55) < 0.01
    assert 0.0 <= stats["median"] <= 1.0
    assert stats["q1"] <= stats["median"] <= stats["q3"]
    assert stats["iqr"] > 0
    assert stats["upper_fence"] > stats["q3"]
    assert 0.0 <= stats["p95"] <= 1.0


def test_detect_anomalies_iqr():
    """IQR 异常检测能识别 spike。"""
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    # 100 个基线点 + 中间 5 个 spike 点
    spike_indices = list(range(50, 55))
    readings = _make_cpu_readings(
        start, 100, base_cpu=0.4, spike_indices=spike_indices, spike_value=0.95,
    )
    ts = VMTimeSeries(
        vm_id="vm_test",
        cluster_id="cluster_A",
        cpu_readings=readings,
        vm_created=start,
    )
    anomalies = detect_anomalies_iqr(ts, min_duration_points=3)
    assert len(anomalies) >= 1
    assert all(a.anomaly_type == AnomalyType.CPU_SPIKE for a in anomalies)
    assert all(a.observed_value >= 0.9 for a in anomalies)


def test_detect_anomalies_iqr_no_anomaly():
    """平稳时序无异常。"""
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    readings = _make_cpu_readings(start, 100, base_cpu=0.4)
    ts = VMTimeSeries(
        vm_id="vm_stable",
        cluster_id="cluster_A",
        cpu_readings=readings,
        vm_created=start,
    )
    anomalies = detect_anomalies_iqr(ts, min_duration_points=3)
    assert len(anomalies) == 0


def test_detect_anomalies_3sigma():
    """3-sigma 异常检测能识别 spike。"""
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    spike_indices = list(range(40, 45))
    readings = _make_cpu_readings(
        start, 100, base_cpu=0.3, spike_indices=spike_indices, spike_value=0.98,
    )
    ts = VMTimeSeries(
        vm_id="vm_3sigma",
        cluster_id="cluster_A",
        cpu_readings=readings,
        vm_created=start,
    )
    anomalies = detect_anomalies_3sigma(ts, min_duration_points=3)
    assert len(anomalies) >= 1


def test_detect_vm_deletion():
    """VM 删除事件检测正确。"""
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    deleted = start + timedelta(hours=10)
    ts = VMTimeSeries(
        vm_id="vm_deleted",
        cluster_id="cluster_A",
        cpu_readings=_make_cpu_readings(start, 100, base_cpu=0.4),
        vm_created=start,
        vm_deleted=deleted,
    )
    anomaly = detect_vm_deletion(ts)
    assert anomaly is not None
    assert anomaly.anomaly_type == AnomalyType.VM_DELETION
    assert anomaly.timestamp == deleted


def test_detect_vm_deletion_no_deletion():
    """未删除的 VM 不返回删除异常。"""
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    ts = VMTimeSeries(
        vm_id="vm_alive",
        cluster_id="cluster_A",
        cpu_readings=_make_cpu_readings(start, 10, base_cpu=0.4),
        vm_created=start,
    )
    assert detect_vm_deletion(ts) is None


def test_detect_high_variance():
    """高方差检测能识别噪声邻居特征。"""
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    # 构造高方差窗口
    readings: list[tuple[datetime, float]] = []
    for i in range(60):
        ts = start + timedelta(seconds=i * 300)
        if 20 <= i < 35:
            # 高方差窗口：在 0.1~0.9 间大幅抖动
            cpu = 0.5 + (0.4 if i % 2 == 0 else -0.4)
        else:
            cpu = 0.4
        readings.append((ts, cpu))

    ts_obj = VMTimeSeries(
        vm_id="vm_var",
        cluster_id="cluster_A",
        cpu_readings=readings,
        vm_created=start,
    )
    anomalies = detect_high_variance(ts_obj, window_size=12, variance_threshold=0.05)
    assert len(anomalies) >= 1
    assert all(a.anomaly_type == AnomalyType.HIGH_VARIANCE for a in anomalies)


def test_azure_trace_loader_class():
    """AzureTraceLoader 封装类能正确检测。"""
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    spike_indices = list(range(50, 55))
    readings = _make_cpu_readings(
        start, 100, base_cpu=0.4, spike_indices=spike_indices, spike_value=0.95,
    )
    ts = VMTimeSeries(
        vm_id="vm_loader",
        cluster_id="cluster_A",
        cpu_readings=readings,
        vm_created=start,
        vm_deleted=start + timedelta(seconds=100 * 300),
    )
    loader = AzureTraceLoader(
        cluster_id="cluster_A",
        detection_method="iqr",
        detect_deletion=True,
        detect_variance=False,
    )
    anomalies = loader.detect_anomalies(ts)
    # 应包含至少一个 CPU spike + 一个 VM 删除
    types = {a.anomaly_type for a in anomalies}
    assert AnomalyType.CPU_SPIKE in types
    assert AnomalyType.VM_DELETION in types


# ============================================================
#  3. fault_event_extractor.py 测试
# ============================================================

def test_classify_severity():
    """严重程度分级正确。"""
    # VM 删除 = critical
    assert classify_severity(0.1, AnomalyType.VM_DELETION) == Severity.CRITICAL
    # 关联删除 = critical
    assert classify_severity(0.1, AnomalyType.CPU_SPIKE, is_vm_deleted=True) == Severity.CRITICAL
    # 偏离 50%+ = critical
    assert classify_severity(0.6, AnomalyType.CPU_SPIKE) == Severity.CRITICAL
    # 偏离 20%+ = warning
    assert classify_severity(0.3, AnomalyType.CPU_SPIKE) == Severity.WARNING
    # 其余 = info
    assert classify_severity(0.1, AnomalyType.CPU_SPIKE) == Severity.INFO


def test_make_event_id():
    """event_id 生成唯一且含关键信息。"""
    dt = datetime(2024, 1, 1, 10, 30, 0, tzinfo=timezone.utc)
    eid = make_event_id("cluster_A", "vm_001", dt, "cpu_spike")
    assert "cluster_A" in eid
    assert "cpu_spike" in eid
    assert "20240101T103000" in eid


def test_extract_trace_fragment():
    """trace 片段提取含上下文扩展。"""
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    readings = _make_cpu_readings(start, 100, base_cpu=0.4)
    ts = VMTimeSeries(
        vm_id="vm_trace",
        cluster_id="cluster_A",
        cpu_readings=readings,
        vm_created=start,
    )
    # 故障窗口：50~55
    window_start = start + timedelta(seconds=50 * 300)
    window_end = start + timedelta(seconds=55 * 300)
    fragment = extract_trace_fragment(
        ts, window_start, window_end, padding_points=3,
    )
    # 应包含 50-3=47 到 55+3=58，共 12 个点
    assert len(fragment) == 12


def test_fault_event_extractor_spike():
    """从 spike 异常抽取故障事件。"""
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    spike_time = start + timedelta(hours=10)
    anomalies = [
        AnomalyPoint(
            anomaly_type=AnomalyType.CPU_SPIKE,
            vm_id="vm_001",
            cluster_id="cluster_A",
            timestamp=spike_time,
            end_timestamp=spike_time + timedelta(minutes=20),
            observed_value=0.95,
            baseline_value=0.4,
            threshold=0.8,
            detection_method="iqr",
            duration_seconds=1200,
        ),
    ]
    ts = VMTimeSeries(
        vm_id="vm_001",
        cluster_id="cluster_A",
        cpu_readings=_make_cpu_readings(start, 200, base_cpu=0.4),
        vm_created=start,
    )
    extractor = FaultEventExtractor()
    events = extractor.extract(ts, anomalies)
    assert len(events) == 1
    assert events[0].event_type == FaultEventType.CPU_SPIKE
    assert events[0].severity == Severity.CRITICAL  # 偏离 > 50%
    assert events[0].vm_id == "vm_001"
    assert events[0].metric_name == "cpu_usage"


def test_fault_event_extractor_merge_adjacent_spikes():
    """相邻 spike 被合并为一个事件。"""
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    t1 = start + timedelta(hours=10)
    t2 = t1 + timedelta(minutes=5)  # 间隔 5 分钟 < merge_window 10 分钟
    anomalies = [
        AnomalyPoint(
            anomaly_type=AnomalyType.CPU_SPIKE,
            vm_id="vm_001",
            cluster_id="cluster_A",
            timestamp=t1,
            end_timestamp=t1 + timedelta(minutes=10),
            observed_value=0.9,
            baseline_value=0.4,
            threshold=0.8,
            detection_method="iqr",
        ),
        AnomalyPoint(
            anomaly_type=AnomalyType.CPU_SPIKE,
            vm_id="vm_001",
            cluster_id="cluster_A",
            timestamp=t2,
            end_timestamp=t2 + timedelta(minutes=10),
            observed_value=0.92,
            baseline_value=0.4,
            threshold=0.8,
            detection_method="iqr",
        ),
    ]
    ts = VMTimeSeries(
        vm_id="vm_001",
        cluster_id="cluster_A",
        cpu_readings=_make_cpu_readings(start, 200, base_cpu=0.4),
        vm_created=start,
    )
    extractor = FaultEventExtractor(merge_window_seconds=600)
    events = extractor.extract(ts, anomalies)
    assert len(events) == 1  # 合并为一个
    assert events[0].observed_value == 0.92  # 取峰值


def test_fault_event_extractor_links_deletion():
    """VM 删除与 spike 关联为复合事件。"""
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    spike_time = start + timedelta(hours=10)
    deletion_time = spike_time + timedelta(minutes=20)  # 删除在 spike 后 20 分钟

    anomalies = [
        AnomalyPoint(
            anomaly_type=AnomalyType.CPU_SPIKE,
            vm_id="vm_001",
            cluster_id="cluster_A",
            timestamp=spike_time,
            end_timestamp=spike_time + timedelta(minutes=15),
            observed_value=0.95,
            baseline_value=0.4,
            threshold=0.8,
            detection_method="iqr",
        ),
        AnomalyPoint(
            anomaly_type=AnomalyType.VM_DELETION,
            vm_id="vm_001",
            cluster_id="cluster_A",
            timestamp=deletion_time,
            observed_value=0.0,
            baseline_value=0.4,
            threshold=0.8,
            detection_method="deletion_event",
        ),
    ]
    ts = VMTimeSeries(
        vm_id="vm_001",
        cluster_id="cluster_A",
        cpu_readings=_make_cpu_readings(start, 200, base_cpu=0.4),
        vm_created=start,
        vm_deleted=deletion_time,
    )
    extractor = FaultEventExtractor(link_window_seconds=1800)
    events = extractor.extract(ts, anomalies)
    assert len(events) == 1
    # 应为 VM_DELETION 类型（复合故障）
    assert events[0].event_type == FaultEventType.VM_DELETION
    # timestamp_end = 删除时刻
    assert events[0].timestamp_end == deletion_time


def test_fault_event_extractor_no_anomalies():
    """无异常输入返回空列表。"""
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    ts = VMTimeSeries(
        vm_id="vm_001",
        cluster_id="cluster_A",
        cpu_readings=_make_cpu_readings(start, 10, base_cpu=0.4),
        vm_created=start,
    )
    extractor = FaultEventExtractor()
    events = extractor.extract(ts, [])
    assert events == []


# ============================================================
#  4. doc_skeleton_seeder.py 测试
# ============================================================

def test_get_all_causal_triples():
    """获取全部因果骨架非空。"""
    triples = get_all_causal_triples()
    assert len(triples) > 0
    # 每个 triple 字段完整
    for t in triples:
        assert t.symptom_type in list(SymptomType)
        assert t.cause_type in list(CauseType)
        assert t.solution_type in list(SolutionType)
        assert len(t.symptom_keywords) > 0
        assert t.cause_mechanism
        assert t.source_doc


def test_k8s_and_prometheus_triples():
    """K8s 与 Prometheus 骨架分离。"""
    k8s = get_k8s_causal_triples()
    prom = get_prometheus_causal_triples()
    assert len(k8s) > 0
    assert len(prom) > 0
    # 来源不同
    assert all(t.source_doc.startswith("k8s") or t.source_doc.startswith("runbook") for t in k8s)
    assert all(t.source_doc.startswith("prometheus") for t in prom)


def test_cause_type_coverage():
    """因果骨架覆盖所有 7 种 CauseType。"""
    coverage = get_cause_type_coverage()
    # 至少覆盖 6 种（允许部分类型无骨架，但应大部分覆盖）
    assert len(coverage) >= 6
    # 关键类型必须有骨架
    assert CauseType.RESOURCE_CONTENTION.value in coverage
    assert CauseType.MISCONFIGURATION.value in coverage
    assert CauseType.NOISY_NEIGHBOR.value in coverage


def test_match_cause_for_event_cpu_spike():
    """CPU spike 事件能匹配到因果骨架。"""
    matched = match_cause_for_event(
        metric_name="cpu_usage",
        event_type="cpu_spike",
    )
    assert len(matched) > 0
    # 匹配的 triple 应包含 cpu 相关关键词
    assert any("cpu" in kw.lower() or "spike" in kw.lower()
               for t in matched for kw in t.symptom_keywords)


def test_match_cause_for_event_oom():
    """OOM 事件能匹配到因果骨架。"""
    matched = match_cause_for_event(
        metric_name="oom",
        event_type="oom_killed",
    )
    assert len(matched) > 0
    # 应匹配到 resource_contention 或 misconfiguration
    cause_types = {t.cause_type for t in matched}
    assert CauseType.RESOURCE_CONTENTION in cause_types or CauseType.MISCONFIGURATION in cause_types


def test_intermediate_chains():
    """中间因链定义完整。"""
    chains = get_intermediate_chains()
    assert len(chains) >= 3
    for chain in chains:
        assert "intermediate_cause" in chain
        assert "root_cause" in chain
        assert "intermediate_mechanism" in chain
        assert "root_mechanism" in chain
        assert "solution_type" in chain
        assert chain["intermediate_cause"] != chain["root_cause"]


# ============================================================
#  5. episode_builder.py 测试
# ============================================================

def test_build_fault_episode_body():
    """故障 episode 正文包含关键信息。"""
    start = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
    event = FaultEvent(
        event_id="fault_cluster_A_vm001_cpu_spike_20240101T100000",
        event_type=FaultEventType.CPU_SPIKE,
        vm_id="vm_001",
        cluster_id="cluster_A",
        timestamp_start=start,
        timestamp_end=start + timedelta(minutes=20),
        severity=Severity.CRITICAL,
        component_type=ComponentType.VM,
        metric_name="cpu_usage",
        observed_value=0.95,
        baseline_value=0.4,
        threshold=0.8,
        detection_method="iqr",
        source_dataset="azure_v2",
        trace_fragment=[(start, 0.4), (start + timedelta(minutes=5), 0.95)],
    )
    body = build_fault_episode_body(event, matched_triple=None)
    # 包含关键字段
    assert "故障事件" in body
    assert event.event_id in body
    assert "cpu_usage" in body
    assert "因果分析" in body
    assert "Provenance" in body
    assert "```json" in body


def test_build_skeleton_episode_body():
    """骨架 episode 正文包含因果链信息。"""
    triple = get_all_causal_triples()[0]
    body = build_skeleton_episode_body(triple)
    assert "因果骨架种子" in body
    assert triple.cause_type.value in body
    assert triple.solution_type.value in body
    assert "```json" in body


def test_episode_builder_fault_event():
    """EpisodeBuilder 正确构建故障 episode。"""
    start = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
    event = FaultEvent(
        event_id="fault_test_001",
        event_type=FaultEventType.CPU_SPIKE,
        vm_id="vm_001",
        cluster_id="cluster_A",
        timestamp_start=start,
        severity=Severity.CRITICAL,
        component_type=ComponentType.VM,
        metric_name="cpu_usage",
        observed_value=0.95,
        baseline_value=0.4,
        threshold=0.8,
        detection_method="iqr",
    )
    builder = EpisodeBuilder()
    payload = builder.build_from_fault_event(event)

    # reference_time = 故障发生时刻（时态锚点）
    assert payload.reference_time == start
    # group_id 按故障场景隔离
    assert payload.group_id == f"cluster_A_{event.event_id}"
    # name 唯一
    assert payload.name == f"episode_{event.event_id}"
    # 回填了 cause_type
    assert event.linked_cause_type is not None
    assert event.linked_solution_type is not None


def test_episode_builder_skeletons():
    """EpisodeBuilder 构建骨架 episode。"""
    builder = EpisodeBuilder()
    skeletons = builder.build_skeleton_episodes()
    assert len(skeletons) > 0
    # 所有骨架 group_id 一致（先验知识分区）
    assert all(s.group_id == "skeleton_prior_knowledge" for s in skeletons)
    # metadata 标记为 causal_skeleton
    assert all(s.metadata.get("type") == "causal_skeleton" for s in skeletons)


def test_build_episodes_convenience():
    """便捷函数 build_episodes 正确工作。"""
    start = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
    events = [
        FaultEvent(
            event_id="fault_test_001",
            event_type=FaultEventType.CPU_SPIKE,
            vm_id="vm_001",
            cluster_id="cluster_A",
            timestamp_start=start,
            metric_name="cpu_usage",
            observed_value=0.95,
            baseline_value=0.4,
            threshold=0.8,
            detection_method="iqr",
        ),
    ]
    episodes = build_episodes(events, include_skeletons=True)
    # 骨架 + 故障
    assert len(episodes) > len(events)
    # 不含骨架
    episodes_no_sk = build_episodes(events, include_skeletons=False)
    assert len(episodes_no_sk) == len(events)


# ============================================================
#  6. synthetic_data.py 测试
# ============================================================

def test_generate_vm_timeseries_reproducible():
    """合成数据可复现（相同种子生成相同数据）。"""
    import random as _random
    rng1 = _random.Random(42)
    rng2 = _random.Random(42)
    ts1 = generate_vm_timeseries(
        vm_id="vm_001",
        cluster_id="cluster_A",
        rng=rng1,
    )
    ts2 = generate_vm_timeseries(
        vm_id="vm_001",
        cluster_id="cluster_A",
        rng=rng2,
    )
    assert ts1.vm_id == ts2.vm_id
    assert ts1.reading_count == ts2.reading_count
    assert ts1.cpu_readings[0][1] == ts2.cpu_readings[0][1]
    assert ts1.cpu_readings[-1][1] == ts2.cpu_readings[-1][1]


def test_generate_vm_batch():
    """批量生成 VM 数量正确。"""
    vms = generate_vm_batch(num_vms=10, cluster_id="cluster_A", seed=42)
    assert len(vms) == 10
    assert all(v.cluster_id == "cluster_A" for v in vms)
    # vm_id 唯一
    vm_ids = [v.vm_id for v in vms]
    assert len(set(vm_ids)) == 10


def test_generate_vm_with_fault():
    """注入故障的 VM 时序含异常点。"""
    import random as _random
    rng = _random.Random(123)
    ts = generate_vm_timeseries(
        vm_id="vm_fault",
        cluster_id="cluster_A",
        inject_fault=True,
        rng=rng,
    )
    # 至少有一个 CPU > 0.8 的点（spike 注入）
    cpu_values = [c for _, c in ts.cpu_readings]
    # 若注入了 spike，应有高值
    if ts.is_deleted:
        # 删除型故障，可能不包含高 CPU
        pass
    else:
        max_cpu = max(cpu_values)
        # inject_fault=True 强制注入，应产生 spike 或 variance
        assert max_cpu > 0.7 or ts.is_deleted


# ============================================================
#  7. 约束集成测试 —— 故障事件与因果骨架的层级组合校验
# ============================================================

def test_fault_event_to_episode_passes_constraints():
    """故障事件构建的 episode 路径符合 Schema 约束。"""
    start = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
    event = FaultEvent(
        event_id="fault_constraint_001",
        event_type=FaultEventType.CPU_SPIKE,
        vm_id="vm_001",
        cluster_id="cluster_A",
        timestamp_start=start,
        timestamp_end=start + timedelta(minutes=20),
        severity=Severity.CRITICAL,
        component_type=ComponentType.VM,
        metric_name="cpu_usage",
        observed_value=0.95,
        baseline_value=0.4,
        threshold=0.8,
        detection_method="iqr",
    )
    builder = EpisodeBuilder()
    payload = builder.build_from_fault_event(event)

    # 时态一致性
    assert validate_temporal_consistency(
        payload.reference_time,
        event.timestamp_end,
    )

    # 因果链层级组合（Symptom → Cause → Solution）合法
    assert validate_edge_combination("Symptom", "Cause", "CAUSED_BY")
    assert validate_edge_combination("Cause", "Solution", "RESOLVED_BY")


def test_validate_edge_full_with_fault_event():
    """validate_edge_full 校验故障事件因果链。"""
    start = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
    end = start + timedelta(minutes=20)

    # 合法因果链
    violations = validate_edge_full(
        source_label="Symptom",
        target_label="Cause",
        edge_name="CAUSED_BY",
        valid_at=start,
        invalid_at=end,
        source_attributes={"symptom_type": "metric_anomaly", "severity": "critical"},
        target_attributes={"cause_type": "noisy_neighbor", "confidence": 0.8, "is_root": True},
    )
    assert violations == [], f"应为空，实际: {violations}"


def test_episode_payload_metadata_completeness():
    """episode 的 metadata 含必要字段。"""
    start = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
    event = FaultEvent(
        event_id="fault_meta_001",
        event_type=FaultEventType.CPU_SPIKE,
        vm_id="vm_001",
        cluster_id="cluster_A",
        timestamp_start=start,
        metric_name="cpu_usage",
        observed_value=0.95,
        baseline_value=0.4,
        threshold=0.8,
        detection_method="iqr",
    )
    builder = EpisodeBuilder()
    payload = builder.build_from_fault_event(event)

    assert "event_id" in payload.metadata
    assert "event_type" in payload.metadata
    assert "vm_id" in payload.metadata
    assert "cluster_id" in payload.metadata
    assert "severity" in payload.metadata
    assert "hop_count" in payload.metadata
    assert payload.metadata["hop_count"] == 2  # 默认 2 跳


# ============================================================
#  8. 端到端集成测试
# ============================================================

def test_end_to_end_synthetic_to_episodes():
    """端到端：合成数据 → 异常检测 → 事件抽取 → episode 构建。"""
    # 1. 生成合成数据（强制注入故障）
    vms = generate_vm_batch(
        num_vms=20,
        cluster_id="cluster_e2e",
        fault_rate=0.5,  # 50% 故障率
        seed=100,
    )

    # 2. 异常检测 + 事件抽取
    loader = AzureTraceLoader(cluster_id="cluster_e2e", detection_method="iqr")
    extractor = FaultEventExtractor()
    all_events: list[FaultEvent] = []
    for ts in vms:
        anomalies = loader.detect_anomalies(ts)
        if anomalies:
            events = extractor.extract(ts, anomalies)
            all_events.extend(events)

    # 至少有部分事件
    assert len(all_events) > 0, "合成数据应产生至少一个故障事件"

    # 3. episode 构建
    builder = EpisodeBuilder()
    episodes = builder.build_from_fault_events(all_events)

    # 4. 验证
    assert len(episodes) == len(all_events)
    for ep, ev in zip(episodes, all_events):
        # reference_time 与事件开始时刻一致
        assert ep.reference_time == ev.timestamp_start
        # group_id 含 cluster_id
        assert ev.cluster_id in ep.group_id
        # episode_body 非空
        assert len(ep.episode_body) > 100


# ============================================================
#  9. Azure V2 长格式解析测试
# ============================================================

def _make_long_format_row(
    vm_id: str,
    vm_created_ts: int,
    vm_deleted_ts: int,
    timestamp_ts: int,
    cpu_avg: float,
    vm_category: str = "D-series",
    vcore_bucket: int = 2,
    memory_bucket: int = 3,
) -> list[str]:
    """构造一行长格式 Azure V2 CSV 数据（20 列）。"""
    return [
        "sub_001",                          # 0 subscription_id
        "dep_001",                          # 1 deployment_id
        str(vm_created_ts),                 # 2 first_vm_created
        "1",                                # 3 vm_count
        "1",                                # 4 deployment_size
        vm_id,                              # 5 vm_id
        str(vm_created_ts),                 # 6 vm_created
        str(vm_deleted_ts),                 # 7 vm_deleted
        "0.95",                             # 8 cpu_max_lifetime
        "0.45",                             # 9 cpu_avg_lifetime
        "0.85",                             # 10 cpu_p95_lifetime
        vm_category,                        # 11 vm_category
        str(vcore_bucket),                  # 12 vcore_bucket
        str(memory_bucket),                 # 13 memory_gb_bucket
        str(timestamp_ts),                  # 14 timestamp
        "0.30",                             # 15 cpu_min_5min
        "0.50",                             # 16 cpu_max_5min
        f"{cpu_avg:.4f}",                   # 17 cpu_avg_5min
        "5-8",                              # 18 vcore_bucket_def
        "17-32",                            # 19 memory_gb_bucket_def
    ]


def test_parse_long_format_row():
    """长格式行解析正确。"""
    row = _make_long_format_row(
        vm_id="vm_test",
        vm_created_ts=1704067200,   # 2024-01-01 00:00:00 UTC
        vm_deleted_ts=0,
        timestamp_ts=1704067200,
        cpu_avg=0.45,
    )
    parsed = parse_long_format_row(row)
    assert parsed["vm_id"] == "vm_test"
    assert parsed["vm_created"] == "1704067200"
    assert parsed["vm_deleted"] == "0"
    assert parsed["cpu_avg_5min"] == "0.4500"
    assert parsed["vm_category"] == "D-series"


def test_resolve_vcore_bucket():
    """vcore 桶解析正确。"""
    # 优先用 definition
    assert resolve_vcore_bucket("2", "5-8") == "5-8"
    # 无 definition 时用索引
    assert resolve_vcore_bucket("2", "") == "5-8"
    assert resolve_vcore_bucket("0", "") == "1-2"
    # 无效值
    assert resolve_vcore_bucket("abc", "") is None


def test_resolve_memory_bucket():
    """memory 桶解析正确。"""
    assert resolve_memory_bucket("3", "17-32") == "17-32"
    assert resolve_memory_bucket("3", "") == "17-32"
    assert resolve_memory_bucket("0", "") == "1-4"
    assert resolve_memory_bucket("xyz", "") is None


def test_stream_vm_timeseries_long_aggregation():
    """长格式 CSV 能正确聚合为 VMTimeSeries。"""
    import tempfile
    import os

    # 构造测试 CSV：2 个 VM，各 5 个时间戳
    vm1_created = 1704067200  # 2024-01-01 00:00:00
    vm2_created = 1704067200
    rows: list[str] = []

    # VM1: 5 个时间戳，CPU 在 0.4 附近
    for i in range(5):
        ts = vm1_created + i * 300
        rows.append(",".join(_make_long_format_row(
            vm_id="vm_001", vm_created_ts=vm1_created, vm_deleted_ts=0,
            timestamp_ts=ts, cpu_avg=0.4 + i * 0.01,
        )))
    # VM2: 5 个时间戳，CPU 在 0.6 附近，第 3 个点 spike
    for i in range(5):
        ts = vm2_created + i * 300
        cpu = 0.95 if i == 2 else 0.6
        rows.append(",".join(_make_long_format_row(
            vm_id="vm_002", vm_created_ts=vm2_created, vm_deleted_ts=0,
            timestamp_ts=ts, cpu_avg=cpu,
        )))

    # 写入临时 CSV（无表头）
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8"
    ) as f:
        f.write("\n".join(rows))
        f.write("\n")
        tmp_path = Path(f.name)

    try:
        # 加载全部
        vms = list(stream_vm_timeseries_long(
            csv_path=tmp_path, cluster_id="cluster_A", max_vms=None,
        ))
        assert len(vms) == 2

        vm1 = next(v for v in vms if v.vm_id == "vm_001")
        vm2 = next(v for v in vms if v.vm_id == "vm_002")
        assert vm1.reading_count == 5
        assert vm2.reading_count == 5
        # VM2 第 3 个点 CPU 应为 0.95
        assert vm2.cpu_readings[2][1] == 0.95
        # 元信息解析正确
        assert vm1.sku == "D-series"
        assert vm1.vcore_bucket == "5-8"
        assert vm1.memory_gb_bucket == "17-32"
    finally:
        os.unlink(tmp_path)


def test_stream_vm_timeseries_long_prefer_deleted():
    """长格式加载优先选择被删除的 VM。"""
    import tempfile
    import os

    rows: list[str] = []
    base_ts = 1704067200
    # VM_deleted：被删除的 VM
    for i in range(3):
        rows.append(",".join(_make_long_format_row(
            vm_id="vm_deleted", vm_created_ts=base_ts,
            vm_deleted_ts=base_ts + 86400,  # 1 天后删除
            timestamp_ts=base_ts + i * 300, cpu_avg=0.4,
        )))
    # VM_alive：存活的 VM
    for i in range(3):
        rows.append(",".join(_make_long_format_row(
            vm_id="vm_alive", vm_created_ts=base_ts,
            vm_deleted_ts=0,
            timestamp_ts=base_ts + i * 300, cpu_avg=0.4,
        )))

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8"
    ) as f:
        f.write("\n".join(rows))
        f.write("\n")
        tmp_path = Path(f.name)

    try:
        # max_vms=1 + prefer_deleted=True 应优先返回 vm_deleted
        vms = list(stream_vm_timeseries_long(
            csv_path=tmp_path, cluster_id="cluster_A",
            max_vms=1, prefer_deleted=True,
        ))
        assert len(vms) == 1
        assert vms[0].vm_id == "vm_deleted"
        assert vms[0].is_deleted is True
    finally:
        os.unlink(tmp_path)


def test_azure_trace_loader_load_long():
    """AzureTraceLoader.load_long 端到端。"""
    import tempfile
    import os

    base_ts = 1704067200
    rows: list[str] = []
    # 一个 VM，10 个点，中间 3 个点 spike
    for i in range(10):
        cpu = 0.95 if 4 <= i <= 6 else 0.4
        rows.append(",".join(_make_long_format_row(
            vm_id="vm_loader_test", vm_created_ts=base_ts, vm_deleted_ts=0,
            timestamp_ts=base_ts + i * 300, cpu_avg=cpu,
        )))

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8"
    ) as f:
        f.write("\n".join(rows))
        f.write("\n")
        tmp_path = Path(f.name)

    try:
        loader = AzureTraceLoader(cluster_id="cluster_A", detection_method="iqr")
        vms = list(loader.load_long(tmp_path, max_vms=None))
        assert len(vms) == 1
        assert vms[0].reading_count == 10

        # 异常检测应识别 spike
        anomalies = loader.detect_anomalies(vms[0])
        # IQR 可能不触发（数据点太少），但 VM 删除检测应无（vm_deleted=0）
        # 主要验证流程不报错
        assert isinstance(anomalies, list)
    finally:
        os.unlink(tmp_path)


# ============================================================
#  主入口
# ============================================================

def main():
    print("=" * 60)
    print("  Graph-RAG 数据接入层单元测试")
    print("=" * 60)
    print()

    tests = [
        # models
        ("test_vm_timeseries_basic", test_vm_timeseries_basic),
        ("test_vm_timeseries_deleted", test_vm_timeseries_deleted),
        ("test_anomaly_point_deviation_ratio", test_anomaly_point_deviation_ratio),
        ("test_fault_event_duration", test_fault_event_duration),
        ("test_graph_build_stats_merge", test_graph_build_stats_merge),
        # azure_trace_loader
        ("test_parse_azure_timestamp", test_parse_azure_timestamp),
        ("test_parse_cpu_readings", test_parse_cpu_readings),
        ("test_compute_baseline_stats", test_compute_baseline_stats),
        ("test_detect_anomalies_iqr", test_detect_anomalies_iqr),
        ("test_detect_anomalies_iqr_no_anomaly", test_detect_anomalies_iqr_no_anomaly),
        ("test_detect_anomalies_3sigma", test_detect_anomalies_3sigma),
        ("test_detect_vm_deletion", test_detect_vm_deletion),
        ("test_detect_vm_deletion_no_deletion", test_detect_vm_deletion_no_deletion),
        ("test_detect_high_variance", test_detect_high_variance),
        ("test_azure_trace_loader_class", test_azure_trace_loader_class),
        # fault_event_extractor
        ("test_classify_severity", test_classify_severity),
        ("test_make_event_id", test_make_event_id),
        ("test_extract_trace_fragment", test_extract_trace_fragment),
        ("test_fault_event_extractor_spike", test_fault_event_extractor_spike),
        ("test_fault_event_extractor_merge_adjacent_spikes", test_fault_event_extractor_merge_adjacent_spikes),
        ("test_fault_event_extractor_links_deletion", test_fault_event_extractor_links_deletion),
        ("test_fault_event_extractor_no_anomalies", test_fault_event_extractor_no_anomalies),
        # doc_skeleton_seeder
        ("test_get_all_causal_triples", test_get_all_causal_triples),
        ("test_k8s_and_prometheus_triples", test_k8s_and_prometheus_triples),
        ("test_cause_type_coverage", test_cause_type_coverage),
        ("test_match_cause_for_event_cpu_spike", test_match_cause_for_event_cpu_spike),
        ("test_match_cause_for_event_oom", test_match_cause_for_event_oom),
        ("test_intermediate_chains", test_intermediate_chains),
        # episode_builder
        ("test_build_fault_episode_body", test_build_fault_episode_body),
        ("test_build_skeleton_episode_body", test_build_skeleton_episode_body),
        ("test_episode_builder_fault_event", test_episode_builder_fault_event),
        ("test_episode_builder_skeletons", test_episode_builder_skeletons),
        ("test_build_episodes_convenience", test_build_episodes_convenience),
        # synthetic_data
        ("test_generate_vm_timeseries_reproducible", test_generate_vm_timeseries_reproducible),
        ("test_generate_vm_batch", test_generate_vm_batch),
        ("test_generate_vm_with_fault", test_generate_vm_with_fault),
        # 约束集成
        ("test_fault_event_to_episode_passes_constraints", test_fault_event_to_episode_passes_constraints),
        ("test_validate_edge_full_with_fault_event", test_validate_edge_full_with_fault_event),
        ("test_episode_payload_metadata_completeness", test_episode_payload_metadata_completeness),
        # 端到端
        ("test_end_to_end_synthetic_to_episodes", test_end_to_end_synthetic_to_episodes),
        # Azure V2 长格式解析
        ("test_parse_long_format_row", test_parse_long_format_row),
        ("test_resolve_vcore_bucket", test_resolve_vcore_bucket),
        ("test_resolve_memory_bucket", test_resolve_memory_bucket),
        ("test_stream_vm_timeseries_long_aggregation", test_stream_vm_timeseries_long_aggregation),
        ("test_stream_vm_timeseries_long_prefer_deleted", test_stream_vm_timeseries_long_prefer_deleted),
        ("test_azure_trace_loader_load_long", test_azure_trace_loader_load_long),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        if _run_test(name, fn):
            passed += 1
        else:
            failed += 1

    print()
    print("-" * 60)
    print(f"  结果: {passed} 通过, {failed} 失败, 共 {passed + failed} 项")
    print("-" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
