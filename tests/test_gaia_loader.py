"""GAIALoader 单元测试 —— 覆盖故障注入解析、metric/trace 加载、拓扑重建、事件转换。

运行方式（在项目根目录）：
    python -m pytest tests/test_gaia_loader.py -v
或直接运行：
    python tests/test_gaia_loader.py

测试不依赖真实 GAIA 数据集，用 tmp_path + csv 构造 mock 数据，
每个测试独立可运行，不依赖网络。
"""
from __future__ import annotations

import csv
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

# 确保能导入项目模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_ingest.gaia_loader import (
    GAIAFaultInjection,
    GAIALoader,
    GAIAMetric,
    GAIATraceSpan,
    GAIATraceTopology,
    download_gaia,
    parse_fault_injection_message,
    parse_metric_filename,
)
from data_ingest.models import FaultEventType
from graph_schema.nodes import Severity


# ============================================================
#  辅助函数 —— 构造 mock GAIA 数据
# ============================================================

def _write_metric_csv(
    path: Path, n_rows: int = 5, start_ts: int = 1625145600000
) -> None:
    """写 metric CSV：timestamp,value 二列，timestamp 为 13 位毫秒。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["timestamp", "value"])
        for i in range(n_rows):
            w.writerow([str(start_ts + i * 60000), f"{50.0 + i:.2f}"])


def _write_trace_csv(path: Path) -> None:
    """写 trace CSV：11 列，2 行 span（frontend→dbservice1 调用关系）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "timestamp", "host_ip", "service_name", "trace_id", "span_id",
        "parent_id", "start_time", "end_time", "url", "status_code", "message",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(headers)
        # 父 span：frontend 根调用
        w.writerow([
            "2021-07-01 10:54:23", "0.0.0.1", "frontend", "t1", "s1", "",
            "2021-07-01 10:54:23", "2021-07-01 10:54:24", "/api", "200", "ok",
        ])
        # 子 span：dbservice1，parent_id=s1
        w.writerow([
            "2021-07-01 10:54:23", "0.0.0.4", "dbservice1", "t1", "s2", "s1",
            "2021-07-01 10:54:23", "2021-07-01 10:54:24", "/db", "200", "ok",
        ])


def _write_run_csv(path: Path) -> None:
    """写 run CSV：2 行，1 行 memory_anomalies 注入 + 1 行普通日志。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    injection_msg = (
        "2021-07-01 22:33:05,033 | WARNING | 0.0.0.4 | 172.17.0.3 | dbservice1 | "
        "[memory_anomalies] trigger a high memory program, start at "
        "2021-07-01 22:23:04.230332 and lasts 600 seconds and use 1g memory"
    )
    normal_msg = "2021-07-01 22:34:05 | INFO | system running normally"
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["datetime", "service", "message"])
        w.writerow(["2021-07-01 22:33:05", "dbservice1", injection_msg])
        w.writerow(["2021-07-01 22:34:05", "dbservice1", normal_msg])


def _build_gaia_dataset(root: Path) -> Path:
    """在 root 下构造 GAIA mock 数据集目录结构，返回 root。

    结构：
        MicroSS/metric/dbservice1_0.0.0.4_cpu_usage_idle_20210701.csv
        MicroSS/trace/20210701_dbservice1.csv
        MicroSS/business/        (空)
        MicroSS/run/run_20210701.csv
        Companion_Data/          (空)
    """
    _write_metric_csv(
        root / "MicroSS" / "metric"
        / "dbservice1_0.0.0.4_cpu_usage_idle_20210701.csv"
    )
    _write_trace_csv(root / "MicroSS" / "trace" / "20210701_dbservice1.csv")
    _write_run_csv(root / "MicroSS" / "run" / "run_20210701.csv")
    # 创建空目录以满足目录结构约定
    (root / "MicroSS" / "business").mkdir(parents=True, exist_ok=True)
    (root / "Companion_Data").mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture
def gaia_data_dir(tmp_path) -> Path:
    """pytest fixture：在临时目录构造 GAIA mock 数据集，返回根目录。"""
    return _build_gaia_dataset(tmp_path)


# ============================================================
#  1. 工具函数测试
# ============================================================

def test_parse_fault_injection_message_memory():
    """完整 message 含 [memory_anomalies] + start at + lasts + use → 全字段解析正确。"""
    message = (
        "2021-07-01 22:33:05,033 | WARNING | 0.0.0.4 | 172.17.0.3 | dbservice1 | "
        "[memory_anomalies] trigger a high memory program, start at "
        "2021-07-01 22:23:04.230332 and lasts 600 seconds and use 1g memory"
    )
    parsed = parse_fault_injection_message(message)
    assert parsed["injection_type"] == "memory_anomalies"
    assert parsed["start_time"] == datetime(
        2021, 7, 1, 22, 23, 4, 230332, tzinfo=timezone.utc
    )
    assert parsed["duration_seconds"] == 600
    assert parsed["severity_hint"] == "1g memory"
    assert parsed["extra"]["log_level"] == "WARNING"


def test_parse_fault_injection_message_cpu():
    """[cpu_anomalies] + start at + lasts → injection_type=cpu_anomalies。"""
    message = (
        "2021-07-01 22:33:05,033 | ERROR | 0.0.0.4 | 172.17.0.3 | dbservice1 | "
        "[cpu_anomalies] trigger high cpu, start at "
        "2021-07-01 22:23:04.000000 and lasts 300 seconds and use 90% cpu"
    )
    parsed = parse_fault_injection_message(message)
    assert parsed["injection_type"] == "cpu_anomalies"
    assert parsed["duration_seconds"] == 300
    assert parsed["start_time"] is not None


def test_parse_fault_injection_message_no_match():
    """普通日志（无 [xxx_anomalies]）→ injection_type 为 None。"""
    message = "2021-07-01 22:34:05 | INFO | system running normally"
    parsed = parse_fault_injection_message(message)
    assert parsed["injection_type"] is None
    assert parsed["start_time"] is None
    assert parsed["duration_seconds"] is None


def test_parse_metric_filename_valid():
    """合法 metric 文件名解析出 node/ip/metric_name/time_period。"""
    parsed = parse_metric_filename(
        "dbservice1_0.0.0.4_cpu_usage_idle_20210701.csv"
    )
    assert parsed == {
        "node": "dbservice1",
        "ip": "0.0.0.4",
        "metric_name": "cpu_usage_idle",
        "time_period": "20210701",
    }


def test_parse_metric_filename_invalid():
    """非法文件名（段数不足）返回空 dict。"""
    assert parse_metric_filename("random.csv") == {}


def test_download_gaia_prints_hint(capsys, tmp_path):
    """download_gaia 打印 git clone 提示。"""
    download_gaia(str(tmp_path / "gaia"))
    captured = capsys.readouterr()
    assert "git clone" in captured.out
    assert "GAIA" in captured.out


# ============================================================
#  2. GAIALoader 基础测试
# ============================================================

def test_loader_init_missing_dir(tmp_path):
    """data_dir 不存在时仍能初始化（懒加载）。"""
    loader = GAIALoader(str(tmp_path / "missing"))
    assert loader.list_metric_files() == []


def test_loader_init_empty_dir(tmp_path):
    """空目录时 list_metric_files 返回空。"""
    loader = GAIALoader(str(tmp_path))
    assert loader.list_metric_files() == []


# ============================================================
#  3. metric 加载测试
# ============================================================

def test_list_metric_files(gaia_data_dir):
    """list_metric_files 返回 mock 数据集中的 1 个 metric 文件。"""
    loader = GAIALoader(str(gaia_data_dir))
    files = loader.list_metric_files()
    assert len(files) == 1
    assert files[0].endswith("dbservice1_0.0.0.4_cpu_usage_idle_20210701.csv")


def test_load_metric(gaia_data_dir):
    """解析 metric CSV → GAIAMetric，timestamps 长度 = 行数，元信息正确。"""
    loader = GAIALoader(str(gaia_data_dir))
    files = loader.list_metric_files()
    metric = loader.load_metric(files[0])
    assert isinstance(metric, GAIAMetric)
    assert metric.node == "dbservice1"
    assert metric.ip == "0.0.0.4"
    assert metric.metric_name == "cpu_usage_idle"
    assert len(metric.timestamps) == 5
    assert len(metric.values) == 5


def test_load_metrics_for_node(gaia_data_dir):
    """load_metrics_for_node 按 node 过滤返回 1 个 GAIAMetric。"""
    loader = GAIALoader(str(gaia_data_dir))
    metrics = loader.load_metrics_for_node("dbservice1")
    assert len(metrics) == 1
    assert metrics[0].node == "dbservice1"


# ============================================================
#  4. trace 加载与拓扑重建测试
# ============================================================

def test_list_trace_files(gaia_data_dir):
    """list_trace_files 返回 1 个 trace 文件。"""
    loader = GAIALoader(str(gaia_data_dir))
    files = loader.list_trace_files()
    assert len(files) == 1
    assert files[0].endswith("20210701_dbservice1.csv")


def test_load_trace(gaia_data_dir):
    """解析 trace CSV → list[GAIATraceSpan]，验证 status_code=200。"""
    loader = GAIALoader(str(gaia_data_dir))
    files = loader.list_trace_files()
    spans = loader.load_trace(files[0])
    assert len(spans) == 2
    assert all(isinstance(s, GAIATraceSpan) for s in spans)
    assert all(s.status_code == 200 for s in spans)


def test_build_topology(gaia_data_dir):
    """从 mock trace 重建拓扑，services 含 dbservice1，edges 非空。"""
    loader = GAIALoader(str(gaia_data_dir))
    topo = loader.build_topology()
    assert isinstance(topo, GAIATraceTopology)
    assert "dbservice1" in topo.services
    assert len(topo.edges) >= 1
    # parent span=frontend, child=dbservice1 → 调用边
    assert ("frontend", "dbservice1") in topo.edges


# ============================================================
#  5. 故障注入与事件转换测试
# ============================================================

def test_load_fault_injections(gaia_data_dir):
    """run 目录 2 行中只 1 行是 [memory_anomalies]，返回 1 个注入记录。"""
    loader = GAIALoader(str(gaia_data_dir))
    injections = loader.load_fault_injections()
    assert len(injections) == 1
    inj = injections[0]
    assert isinstance(inj, GAIAFaultInjection)
    assert inj.injection_type == "memory_anomalies"
    assert inj.service == "dbservice1"
    assert inj.duration_seconds == 600


def test_extract_fault_events(gaia_data_dir):
    """注入记录 → FaultEvent，event_type=OOM_KILLED，source_dataset=gaia。"""
    loader = GAIALoader(str(gaia_data_dir))
    events = loader.extract_fault_events()
    assert len(events) == 1
    ev = events[0]
    assert ev.event_type == FaultEventType.OOM_KILLED
    assert ev.source_dataset == "gaia"
    assert ev.vm_id == "dbservice1"


def test_extract_fault_events_empty(gaia_data_dir):
    """injections=[] → 返回空列表。"""
    loader = GAIALoader(str(gaia_data_dir))
    events = loader.extract_fault_events(injections=[])
    assert events == []


def test_extract_fault_events_timestamp_end(gaia_data_dir):
    """duration_seconds=600 → timestamp_end = start + 600s。"""
    loader = GAIALoader(str(gaia_data_dir))
    start = datetime(2021, 7, 1, 22, 23, 4, tzinfo=timezone.utc)
    inj = GAIAFaultInjection(
        timestamp=start,
        service="dbservice1",
        injection_type="memory_anomalies",
        description="",
        start_time=start,
        duration_seconds=600,
        severity_hint="1g memory",
    )
    events = loader.extract_fault_events(injections=[inj])
    assert len(events) == 1
    ev = events[0]
    assert ev.timestamp_start == start
    assert ev.timestamp_end is not None
    assert ev.timestamp_end - ev.timestamp_start == timedelta(seconds=600)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
