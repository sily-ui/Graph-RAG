"""GAIA 数据集加载器（CloudWise 开源 MicroSS 微服务系统）。

GAIA 数据集仓库：https://github.com/CloudWise-OpenSource/GAIA-DataSet

本加载器覆盖 MicroSS 子目录的四类数据：
    metric/    每文件 = 单节点单指标时序（13 位毫秒时间戳）
    trace/     调用链 span（含 parent_id，可重建服务调用拓扑）
    business/  业务日志
    run/       故障注入记录 + 系统日志（核心数据源）

Companion_Data/ 子目录（metric_detection / metric_forecast / log）含异常
标签数据，本加载器暂不实现其专用接口，仅在 docstring 中标注存在。

设计原则（与 azure_trace_loader 一致）：
- 用 csv 模块流式读取，避免一次性加载大文件
- 用 re.compile 预编译正则，解析故障注入 message
- 不依赖 graphiti_core，仅复用 data_ingest.models 的 FaultEvent 等
- 所有时间字段统一用带时区 datetime（GAIA 原始时间无时区，统一按 UTC 解释）

核心方法 load_fault_injections 解析 run 目录的故障注入记录，并通过
extract_fault_events 转换为项目统一的 FaultEvent，是补足「集群拓扑 +
根因标注」能力的关键入口。
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from data_ingest.models import FaultEvent, FaultEventType
from graph_schema.nodes import ComponentType, Severity


# ============================================================
#  正则预编译 —— 解析故障注入 message 字段
# ============================================================

# message 形如：
#   '2021-07-01 22:33:05,033 | WARNING | 0.0.0.4 | 172.17.0.3 | dbservice1 |
#    [memory_anomalies] trigger a high memory program, start at
#    2021-07-01 22:23:04.230332 and lasts 600 seconds and use 1g memory'

# 故障注入类型，如 [memory_anomalies] / [cpu_anomalies] / [network_anomalies] / [disk_anomalies]
_RE_INJECTION_TYPE = re.compile(r"\[([a-z_]+_anomalies)\]")

# 故障开始时刻：start at YYYY-MM-DD HH:MM:SS[.ffffff]
_RE_START_TIME = re.compile(
    r"start at (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?)"
)

# 故障持续时长：lasts N seconds
_RE_DURATION = re.compile(r"lasts (\d+) seconds")

# 严重程度提示：use 1g memory / use 90% cpu 等（非贪婪到行尾或下一个 and）
_RE_SEVERITY = re.compile(r"use (.+?)(?=\s+and\s+|$)")

# 日志级别：| WARNING | / | ERROR | 等
_RE_LOG_LEVEL = re.compile(r"\|\s*(INFO|WARNING|ERROR|CRITICAL)\s*\|")

# message 起头的精确时间戳：2021-07-01 22:33:05,033（逗号后为毫秒）
_RE_LEADING_TS = re.compile(
    r"^(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2})(?:,(\d+))?"
)

# IPv4 识别（metric 文件名中的 ip 段）
_RE_IPV4 = re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")


# 故障注入类型 → FaultEventType 映射
_INJECTION_TYPE_TO_FAULT_EVENT: dict[str, FaultEventType] = {
    "memory_anomalies": FaultEventType.OOM_KILLED,
    "cpu_anomalies": FaultEventType.CPU_SPIKE,
    "network_anomalies": FaultEventType.LATENCY_SURGE,
    "disk_anomalies": FaultEventType.LATENCY_SURGE,
}

# 故障注入类型 → 异常指标名映射
_INJECTION_TYPE_TO_METRIC_NAME: dict[str, str] = {
    "memory_anomalies": "memory_usage",
    "cpu_anomalies": "cpu_usage",
    "network_anomalies": "network_latency_ms",
    "disk_anomalies": "disk_io_latency_ms",
}

# 日志级别 → Severity 映射
_LOG_LEVEL_TO_SEVERITY: dict[str, Severity] = {
    "INFO": Severity.INFO,
    "WARNING": Severity.WARNING,
    "ERROR": Severity.CRITICAL,
    "CRITICAL": Severity.CRITICAL,
}

# GAIA 默认集群标识（MicroSS 系统）
_GAIA_CLUSTER_ID = "gaia_micross"


# ============================================================
#  数据模型（GAIA 专用，用 dataclass 保持轻量）
# ============================================================

@dataclass
class GAIAMetric:
    """GAIA 单指标时序。"""
    node: str                       # 节点名
    ip: str                         # 节点 IP
    metric_name: str                # 指标名
    timestamps: list[datetime]      # 时间戳列表（UTC）
    values: list[float]             # 指标值列表，与 timestamps 等长


@dataclass
class GAIATraceSpan:
    """GAIA 调用链 span。"""
    timestamp: datetime             # span 记录时刻（UTC）
    host_ip: str
    service_name: str
    trace_id: str
    span_id: str
    parent_id: str
    start_time: datetime            # span 开始时刻（UTC）
    end_time: datetime              # span 结束时刻（UTC）
    url: str
    status_code: int
    message: str


@dataclass
class GAIAFaultInjection:
    """GAIA 故障注入记录（核心数据）。"""
    timestamp: datetime                     # 记录时刻（UTC）
    service: str                            # 被注入的服务
    injection_type: str                     # memory/cpu/network/disk_anomalies
    description: str                        # 完整 message 描述
    start_time: datetime | None = None      # 解析出的故障开始时刻
    duration_seconds: int | None = None     # 解析出的故障持续时长（秒）
    severity_hint: str | None = None        # 解析出的严重程度提示（如 "1g memory"）


@dataclass
class GAIATraceTopology:
    """从 trace 重建的服务调用拓扑。"""
    services: set[str]                                   # 所有服务
    edges: set[tuple[str, str]]                          # 调用关系 (caller, callee)
    traces: dict[str, list[GAIATraceSpan]] = field(      # 按 trace_id 分组
        default_factory=dict
    )


# ============================================================
#  时间戳解析工具
# ============================================================

def _parse_ms_timestamp(ts: str | int | float) -> datetime | None:
    """解析 13 位毫秒时间戳 → UTC datetime。"""
    try:
        ts_ms = int(float(ts))
    except (ValueError, TypeError):
        return None
    if ts_ms <= 0:
        return None
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)


def _parse_trace_datetime(s: str) -> datetime:
    """解析 trace/run 字符串时间，统一附加 UTC 时区。

    支持两种格式：
    - '2021-07-01 10:54:23'（秒精度）
    - '2021-07-01 10:54:22.632751'（微秒精度）
    """
    s = s.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    # 兜底：仅日期
    try:
        return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError as e:
        raise ValueError(f"无法解析时间字符串: {s!r}") from e


def _ensure_utc(dt: datetime | None) -> datetime | None:
    """确保 datetime 带 UTC 时区（naive 视为 UTC）。"""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


# ============================================================
#  工具函数
# ============================================================

def parse_fault_injection_message(message: str) -> dict[str, Any]:
    """用正则解析故障注入 message 字段。

    返回 dict 含 injection_type / start_time / duration_seconds /
    severity_hint / extra（含 log_level / leading_timestamp 等附加信息）。

    示例 message：
        '2021-07-01 22:33:05,033 | WARNING | 0.0.0.4 | 172.17.0.3 | dbservice1 |
         [memory_anomalies] trigger a high memory program, start at
         2021-07-01 22:23:04.230332 and lasts 600 seconds and use 1g memory'

    应解析出：
        {
            'injection_type': 'memory_anomalies',
            'start_time': datetime(2021, 7, 1, 22, 23, 4, 230332, tzinfo=UTC),
            'duration_seconds': 600,
            'severity_hint': '1g memory',
            'extra': {'log_level': 'WARNING', 'leading_timestamp': ...},
        }
    """
    result: dict[str, Any] = {
        "injection_type": None,
        "start_time": None,
        "duration_seconds": None,
        "severity_hint": None,
        "extra": {},
    }
    if not message:
        return result

    # 注入类型
    m = _RE_INJECTION_TYPE.search(message)
    if m:
        result["injection_type"] = m.group(1)

    # 开始时刻
    m = _RE_START_TIME.search(message)
    if m:
        try:
            result["start_time"] = _parse_trace_datetime(m.group(1))
        except ValueError:
            result["start_time"] = None

    # 持续时长
    m = _RE_DURATION.search(message)
    if m:
        try:
            result["duration_seconds"] = int(m.group(1))
        except ValueError:
            result["duration_seconds"] = None

    # 严重程度提示
    m = _RE_SEVERITY.search(message)
    if m:
        result["severity_hint"] = m.group(1).strip()

    # 日志级别
    m = _RE_LOG_LEVEL.search(message)
    if m:
        result["extra"]["log_level"] = m.group(1)

    # 起头精确时间戳（YYYY-MM-DD HH:MM:SS[,mmm]）
    m = _RE_LEADING_TS.search(message)
    if m:
        date_part, time_part, ms_part = m.group(1), m.group(2), m.group(3)
        ts_str = f"{date_part} {time_part}"
        try:
            leading = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
            if ms_part:
                # ",033" 为毫秒，补齐到 6 位微秒
                leading = leading + timedelta(
                    microseconds=int((ms_part + "000000")[:6])
                )
            result["extra"]["leading_timestamp"] = leading.replace(tzinfo=timezone.utc)
        except ValueError:
            result["extra"]["leading_timestamp"] = None

    return result


def parse_metric_filename(filename: str) -> dict[str, str]:
    """从 metric CSV 文件名解析出 node / ip / metric_name / time_period。

    文件名格式：``<node>_<ip>_<metric_name>_<time_period>.csv``
    其中 metric_name 可能含下划线（如 cpu_usage_idle），ip 含点（如 0.0.0.4）。

    解析失败返回空 dict。
    """
    name = filename
    # 仅取文件名部分（容忍传入完整路径）
    name = name.replace("\\", "/").rsplit("/", 1)[-1]
    if name.lower().endswith(".csv"):
        name = name[:-4]

    parts = name.split("_")
    if len(parts) < 4:
        return {}

    node = parts[0]
    # 定位 IP 段（含点的合法 IPv4）
    ip_idx = None
    for i in range(1, len(parts) - 1):
        if _RE_IPV4.match(parts[i]):
            ip_idx = i
            break
    if ip_idx is None:
        return {}

    ip = parts[ip_idx]
    time_period = parts[-1]
    metric_name = "_".join(parts[ip_idx + 1:-1])
    if not metric_name:
        return {}

    return {
        "node": node,
        "ip": ip,
        "metric_name": metric_name,
        "time_period": time_period,
    }


def download_gaia(data_dir: str) -> None:
    """打印 GAIA 数据集下载提示（不实际下载）。

    Parameters
    ----------
    data_dir : str
        解压后应含 MicroSS/ 和 Companion_Data/ 子目录。
    """
    print("[GAIA] 请手动下载数据集：")
    print("  git clone https://github.com/CloudWise-OpenSource/GAIA-DataSet.git " + data_dir)
    print("  解压后确保目录结构为：<data_dir>/MicroSS/{metric,trace,business,run}")


# ============================================================
#  GAIALoader 主类
# ============================================================

class GAIALoader:
    """GAIA MicroSS 数据集加载器。

    Parameters
    ----------
    data_dir : str
        GAIA-DataSet 解压根目录（含 MicroSS/ 和 Companion_Data/ 子目录）。
    """

    def __init__(self, data_dir: str) -> None:
        self.data_dir: Path = Path(data_dir)
        self.micross_dir: Path = self.data_dir / "MicroSS"
        self.metric_dir: Path = self.micross_dir / "metric"
        self.trace_dir: Path = self.micross_dir / "trace"
        self.business_dir: Path = self.micross_dir / "business"
        self.run_dir: Path = self.micross_dir / "run"

    # ---------- metric ----------

    def list_metric_files(self, node: str | None = None) -> list[str]:
        """列出所有 metric CSV 文件路径。

        Parameters
        ----------
        node : str | None
            若提供，仅返回文件名以 ``<node>_`` 起头的文件。
        """
        if not self.metric_dir.is_dir():
            return []
        files = sorted(self.metric_dir.glob("*.csv"))
        if node is None:
            return [str(f) for f in files]
        prefix = f"{node}_"
        return [str(f) for f in files if f.name.startswith(prefix)]

    def load_metric(self, csv_path: str) -> GAIAMetric:
        """加载单个 metric CSV 文件。

        CSV 格式：``timestamp,value``，timestamp 为 13 位毫秒时间戳。
        文件名解析失败时 node/ip/metric_name 置空字符串。
        """
        path = Path(csv_path)
        meta = parse_metric_filename(path.name)
        node = meta.get("node", "")
        ip = meta.get("ip", "")
        metric_name = meta.get("metric_name", "")

        timestamps: list[datetime] = []
        values: list[float] = []
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                ts = _parse_ms_timestamp(row.get("timestamp", ""))
                if ts is None:
                    continue
                raw_val = row.get("value", "")
                try:
                    val = float(raw_val)
                except (ValueError, TypeError):
                    continue
                timestamps.append(ts)
                values.append(val)

        return GAIAMetric(
            node=node,
            ip=ip,
            metric_name=metric_name,
            timestamps=timestamps,
            values=values,
        )

    def load_metrics_for_node(
        self, node: str, max_files: int | None = None
    ) -> list[GAIAMetric]:
        """加载指定节点的所有 metric。"""
        files = self.list_metric_files(node=node)
        if max_files is not None:
            files = files[:max_files]
        return [self.load_metric(f) for f in files]

    # ---------- trace ----------

    def list_trace_files(self) -> list[str]:
        """列出所有 trace CSV 文件。"""
        if not self.trace_dir.is_dir():
            return []
        return sorted(str(f) for f in self.trace_dir.glob("*.csv"))

    def load_trace(
        self, csv_path: str, max_rows: int | None = None
    ) -> list[GAIATraceSpan]:
        """加载单个 trace CSV，返回 span 列表。

        CSV 表头：timestamp,host_ip,service_name,trace_id,span_id,parent_id,
                  start_time,end_time,url,status_code,message
        """
        spans: list[GAIATraceSpan] = []
        path = Path(csv_path)
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if max_rows is not None and len(spans) >= max_rows:
                    break
                try:
                    timestamp = _parse_trace_datetime(row["timestamp"])
                    start_time = _parse_trace_datetime(row["start_time"])
                    end_time = _parse_trace_datetime(row["end_time"])
                except (ValueError, KeyError):
                    continue
                try:
                    status_code = int(row.get("status_code") or 0)
                except ValueError:
                    status_code = 0
                spans.append(
                    GAIATraceSpan(
                        timestamp=timestamp,
                        host_ip=row.get("host_ip", "").strip(),
                        service_name=row.get("service_name", "").strip(),
                        trace_id=row.get("trace_id", "").strip(),
                        span_id=row.get("span_id", "").strip(),
                        parent_id=row.get("parent_id", "").strip(),
                        start_time=start_time,
                        end_time=end_time,
                        url=row.get("url", "").strip(),
                        status_code=status_code,
                        message=row.get("message", "").strip(),
                    )
                )
        return spans

    def build_topology(
        self,
        trace_files: list[str] | None = None,
        max_spans: int = 10000,
    ) -> GAIATraceTopology:
        """从 trace 数据重建服务调用拓扑。

        通过 parent_id → span_id 关系建立 caller→callee 边；
        忽略根 span（parent_id 为空 / 0 / None）与跨 trace 失配的 parent。
        """
        if trace_files is None:
            trace_files = self.list_trace_files()

        services: set[str] = set()
        edges: set[tuple[str, str]] = set()
        traces: dict[str, list[GAIATraceSpan]] = {}
        total = 0

        for f in trace_files:
            if total >= max_spans:
                break
            remaining = max_spans - total if max_spans > 0 else None
            spans = self.load_trace(f, max_rows=remaining)
            for span in spans:
                services.add(span.service_name)
                traces.setdefault(span.trace_id, []).append(span)
                total += 1
            if total >= max_spans:
                break

        # 基于 parent_id 重建调用边
        for span_list in traces.values():
            by_span_id: dict[str, GAIATraceSpan] = {
                s.span_id: s for s in span_list if s.span_id
            }
            for span in span_list:
                pid = span.parent_id
                if not pid or pid in ("0", "None", "null", "NULL"):
                    continue
                parent = by_span_id.get(pid)
                if parent is None:
                    continue
                if parent.service_name and parent.service_name != span.service_name:
                    edges.add((parent.service_name, span.service_name))

        return GAIATraceTopology(
            services=services, edges=edges, traces=traces
        )

    # ---------- run / fault injection ----------

    def load_fault_injections(self) -> list[GAIAFaultInjection]:
        """加载 run 目录所有故障注入记录（核心方法）。

        仅保留 message 含 ``[xxx_anomalies]`` 的行，过滤普通系统日志。
        解析 message 字段以提取 injection_type / start_time /
        duration_seconds / severity_hint。
        """
        injections: list[GAIAFaultInjection] = []
        if not self.run_dir.is_dir():
            return injections

        for path in sorted(self.run_dir.glob("*.csv")):
            with path.open("r", encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    message = row.get("message", "") or ""
                    if not _RE_INJECTION_TYPE.search(message):
                        # 非故障注入行（普通系统日志），跳过
                        continue
                    parsed = parse_fault_injection_message(message)
                    injection_type = parsed["injection_type"] or ""

                    # 记录时刻：优先用 message 起头精确时间戳，其次 datetime 字段
                    timestamp = parsed["extra"].get("leading_timestamp")
                    if timestamp is None:
                        dt_str = (row.get("datetime") or "").strip()
                        if dt_str:
                            try:
                                timestamp = _parse_trace_datetime(dt_str)
                            except ValueError:
                                timestamp = datetime.now(tz=timezone.utc)
                        else:
                            timestamp = datetime.now(tz=timezone.utc)

                    injections.append(
                        GAIAFaultInjection(
                            timestamp=timestamp,
                            service=(row.get("service") or "").strip(),
                            injection_type=injection_type,
                            description=message,
                            start_time=parsed["start_time"],
                            duration_seconds=parsed["duration_seconds"],
                            severity_hint=parsed["severity_hint"],
                        )
                    )
        return injections

    def extract_fault_events(
        self,
        injections: list[GAIAFaultInjection] | None = None,
    ) -> list[FaultEvent]:
        """把故障注入记录转换为项目统一的 FaultEvent。

        关键映射：
        - service → vm_id
        - injection_type → event_type（memory→OOM_KILLED, cpu→CPU_SPIKE,
          network/disk→LATENCY_SURGE）
        - start_time → timestamp_start，start_time + duration → timestamp_end
        - component_type = ComponentType.SERVICE
        - source_dataset = "gaia"
        - observed_value/baseline_value/threshold 用占位值（GAIA 注入记录无具体数值）
        """
        if injections is None:
            injections = self.load_fault_injections()

        events: list[FaultEvent] = []
        for idx, inj in enumerate(injections):
            event_type = _INJECTION_TYPE_TO_FAULT_EVENT.get(
                inj.injection_type, FaultEventType.LATENCY_SURGE
            )
            metric_name = _INJECTION_TYPE_TO_METRIC_NAME.get(
                inj.injection_type, "unknown"
            )

            # 开始/结束时刻
            ts_start = _ensure_utc(inj.start_time) or _ensure_utc(inj.timestamp)
            ts_end: datetime | None = None
            if inj.start_time is not None and inj.duration_seconds:
                ts_end = _ensure_utc(inj.start_time) + timedelta(
                    seconds=inj.duration_seconds
                )

            # 严重程度：优先用 message 中的日志级别，其次按注入类型兜底
            parsed = parse_fault_injection_message(inj.description)
            log_level = parsed["extra"].get("log_level")
            if log_level and log_level in _LOG_LEVEL_TO_SEVERITY:
                severity = _LOG_LEVEL_TO_SEVERITY[log_level]
            elif inj.injection_type == "memory_anomalies":
                severity = Severity.CRITICAL
            else:
                severity = Severity.WARNING

            vm_id = inj.service or "unknown"
            event_id = (
                f"fault_gaia_{vm_id}_{int(ts_start.timestamp())}_"
                f"{inj.injection_type or 'unknown'}_{idx}"
            )

            events.append(
                FaultEvent(
                    event_id=event_id,
                    event_type=event_type,
                    vm_id=vm_id,
                    cluster_id=_GAIA_CLUSTER_ID,
                    timestamp_start=ts_start,
                    timestamp_end=ts_end,
                    severity=severity,
                    component_type=ComponentType.SERVICE,
                    metric_name=metric_name,
                    # GAIA 注入记录无具体数值，统一占位
                    observed_value=0.0,
                    baseline_value=0.0,
                    threshold=0.0,
                    detection_method="gaia_fault_injection",
                    source_dataset="gaia",
                )
            )
        return events
