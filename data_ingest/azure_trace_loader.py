"""Azure Public Dataset V2 数据加载器 —— 流式读取 + 异常检测 + 事件抽取。

Azure V2 数据集格式（基于 https://github.com/Azure/AzurePublicDataset/blob/master/AzurePublicDatasetV2.md）：

数据集是**长格式** CSV：每行 = 一个 VM × 一个 5 分钟时间戳。
一个 VM 的完整 30 天时序由约 8640 行组成，分散在多个文件中。

真实 schema（20 列，无表头，列顺序固定）：
    1.  encrypted_subscription_id     订阅 ID
    2.  encrypted_deployment_id       部署 ID
    3.  first_vm_created_timestamp    部署首次创建 VM 的时间戳（秒，从 0 起）
    4.  vm_count                      部署内 VM 数
    5.  deployment_size               部署规模
    6.  encrypted_vm_id               VM ID
    7.  vm_created_timestamp          VM 创建时间戳（秒）
    8.  vm_deleted_timestamp          VM 删除时间戳（秒，0=未删除）
    9.  cpu_max                       全周期最大 CPU
    10. cpu_avg                       全周期平均 CPU
    11. cpu_p95                       全周期 P95 CPU
    12. vm_category                   VM 类别（如 D-series/E-series）
    13. vcore_bucket                  vCPU 桶（整数索引）
    14. memory_gb_bucket              内存桶（整数索引）
    15. timestamp                     5 分钟时间戳（秒）
    16. cpu_min                       该 5 分钟内最小 CPU
    17. cpu_max_5min                  该 5 分钟内最大 CPU
    18. cpu_avg_5min                  该 5 分钟内平均 CPU
    19. vcore_bucket_definition       vCPU 桶定义（字符串）
    20. memory_gb_bucket_definition   内存桶定义（字符串）

数据集规模：235GB / 198 文件 / 30 天 / 269 万 VM / 19 亿条 CPU 读数

本模块设计为流式处理，避免一次性加载 235GB：
1. 按文件流式读取长格式 CSV
2. 按 vm_id 聚合多行为单个 VMTimeSeries
3. 异常检测算法（IQR / 3-sigma）在单 VM 时序上运行
4. 输出 AnomalyPoint 列表，供 fault_event_extractor 聚合

抽样策略（避免全量加载）：
- 优先加载被删除的 VM（vm_deleted != 0，故障密度高）
- 限定文件数（1-3 个文件约 1.2GB/个）
- 通过 max_vms 参数控制

异常检测算法说明：
- IQR 法：Q1=25%分位, Q3=75%分位, IQR=Q3-Q1, 上界=Q3+1.5*IQR
  适合 CPU 时序（非高斯分布，有长尾），对极端值敏感
- 3-sigma 法：阈值=均值+3*标准差，适合近似高斯分布的指标
- spike 检测：超 p95 阈值且持续 ≥ min_duration 分钟，输出窗口型异常
"""
from __future__ import annotations

import csv
import statistics
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from data_ingest.models import (
    AnomalyPoint,
    AnomalyType,
    VMTimeSeries,
)


# ============================================================
#  Azure V2 真实 schema（20 列，无表头，按列序号映射）
# ============================================================

# 列索引（0-based）—— Azure V2 CSV 无表头，按列序号解析
AZURE_V2_COLUMN_INDEX = {
    "subscription_id": 0,
    "deployment_id": 1,
    "first_vm_created": 2,
    "vm_count": 3,
    "deployment_size": 4,
    "vm_id": 5,
    "vm_created": 6,
    "vm_deleted": 7,
    "cpu_max_lifetime": 8,
    "cpu_avg_lifetime": 9,
    "cpu_p95_lifetime": 10,
    "vm_category": 11,
    "vcore_bucket": 12,
    "memory_gb_bucket": 13,
    "timestamp": 14,
    "cpu_min_5min": 15,
    "cpu_max_5min": 16,
    "cpu_avg_5min": 17,
    "vcore_bucket_def": 18,
    "memory_gb_bucket_def": 19,
}

# 兼容旧代码的列名映射（已弃用，保留向后兼容）
AZURE_V2_COLUMNS = {
    "vmid": "vm_id",
    "vmcreated": "vm_created",
    "vmdeleted": "vm_deleted",
    "vmcategory": "vm_category",
}

# 5 分钟采样间隔（秒）
SAMPLE_INTERVAL_SECONDS = 300


# ============================================================
#  时间戳解析
# ============================================================

def parse_azure_timestamp(ts: Any) -> datetime | None:
    """解析 Azure V2 的 Unix epoch 时间戳。

    Azure V2 用 Unix epoch 秒表示 VM 创建/删除时刻。
    返回 UTC 时区的 datetime，None 表示 ts<=0（未删除）。

    Parameters
    ----------
    ts : int | str | float
        Unix epoch 时间戳（秒）。0 或负数表示无时间。
    """
    if ts is None:
        return None
    try:
        ts_int = int(float(ts))
    except (ValueError, TypeError):
        return None
    if ts_int <= 0:
        return None
    return datetime.fromtimestamp(ts_int, tz=timezone.utc)


# ============================================================
#  长格式 CSV 行解析
# ============================================================

def parse_long_format_row(row: list[str] | tuple[str, ...]) -> dict[str, str]:
    """解析 Azure V2 长格式 CSV 的一行（无表头，按列序号）。

    Parameters
    ----------
    row : list[str] | tuple[str, ...]
        CSV 行的字段列表（csv.reader 的输出）

    Returns
    -------
    dict[str, str]
        按 AZURE_V2_COLUMN_INDEX 的 key 命名的字段 dict
    """
    if len(row) < 18:
        return {}
    return {
        "vm_id": row[AZURE_V2_COLUMN_INDEX["vm_id"]].strip(),
        "vm_created": row[AZURE_V2_COLUMN_INDEX["vm_created"]].strip(),
        "vm_deleted": row[AZURE_V2_COLUMN_INDEX["vm_deleted"]].strip(),
        "vm_category": row[AZURE_V2_COLUMN_INDEX["vm_category"]].strip(),
        "vcore_bucket": row[AZURE_V2_COLUMN_INDEX["vcore_bucket"]].strip(),
        "memory_gb_bucket": row[AZURE_V2_COLUMN_INDEX["memory_gb_bucket"]].strip(),
        "timestamp": row[AZURE_V2_COLUMN_INDEX["timestamp"]].strip(),
        "cpu_min_5min": row[AZURE_V2_COLUMN_INDEX["cpu_min_5min"]].strip(),
        "cpu_max_5min": row[AZURE_V2_COLUMN_INDEX["cpu_max_5min"]].strip(),
        "cpu_avg_5min": row[AZURE_V2_COLUMN_INDEX["cpu_avg_5min"]].strip(),
        "cpu_p95_lifetime": row[AZURE_V2_COLUMN_INDEX["cpu_p95_lifetime"]].strip() if len(row) > 17 else "",
        "vcore_bucket_def": row[AZURE_V2_COLUMN_INDEX["vcore_bucket_def"]].strip() if len(row) > 18 else "",
        "memory_gb_bucket_def": row[AZURE_V2_COLUMN_INDEX["memory_gb_bucket_def"]].strip() if len(row) > 19 else "",
    }


# ============================================================
#  vcore / memory 桶定义（来自 Azure V2 文档）
# ============================================================

# Azure V2 的 vcore bucket 索引到描述的映射
VCORE_BUCKET_MAP: dict[int, str] = {
    0: "1-2",
    1: "3-4",
    2: "5-8",
    3: "9-16",
    4: "17-32",
    5: "33+",
}

# Azure V2 的 memory bucket 索引到描述的映射
MEMORY_BUCKET_MAP: dict[int, str] = {
    0: "1-4",
    1: "5-8",
    2: "9-16",
    3: "17-32",
    4: "33-64",
    5: "65-128",
    6: "129+",
}


def resolve_vcore_bucket(bucket_str: str, definition: str = "") -> str | None:
    """解析 vcore 桶字段，返回人类可读的桶描述。

    Azure V2 的 vcore_bucket 是整数索引，vcore_bucket_definition 是描述字符串。
    优先用 definition，其次用索引映射，最后返回 None。
    """
    if definition and definition != "0":
        return definition
    try:
        idx = int(bucket_str)
        return VCORE_BUCKET_MAP.get(idx)
    except (ValueError, TypeError):
        return None


def resolve_memory_bucket(bucket_str: str, definition: str = "") -> str | None:
    """解析 memory 桶字段。"""
    if definition and definition != "0":
        return definition
    try:
        idx = int(bucket_str)
        return MEMORY_BUCKET_MAP.get(idx)
    except (ValueError, TypeError):
        return None


# ============================================================
#  兼容旧接口：宽格式 CPU 读数解析（供合成数据/预处理数据用）
# ============================================================

def parse_cpu_readings(raw: str, vm_created: datetime, interval_seconds: int = SAMPLE_INTERVAL_SECONDS) -> list[tuple[datetime, float]]:
    """解析逗号分隔的 CPU 时序字段（宽格式，非 Azure V2 原生）。

    Azure V2 原生是长格式（每行一个时间戳），此函数用于：
    - 合成数据生成的逗号分隔字符串
    - 预处理后的宽格式 CSV
    - 向后兼容旧测试

    Parameters
    ----------
    raw : str
        逗号分隔的 CPU 读数字符串，如 "0.45,0.43,0.47,..."
    vm_created : datetime
        VM 创建时刻（时序起点）
    interval_seconds : int
        采样间隔（秒），默认 300（5 分钟）
    """
    if not raw or not raw.strip():
        return []
    readings: list[tuple[datetime, float]] = []
    parts = raw.split(",")
    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue
        try:
            cpu_val = float(part)
        except ValueError:
            continue
        ts = vm_created + timedelta(seconds=i * interval_seconds)
        readings.append((ts, cpu_val))
    return readings


# ============================================================
#  异常检测算法
# ============================================================

def compute_baseline_stats(cpu_values: list[float]) -> dict[str, float]:
    """计算 CPU 时序的基线统计量。

    用于异常检测的基线参考，包含：
    - mean, std: 3-sigma 法用
    - q1, median, q3, iqr, upper_fence: IQR 法用
    - p95, p99: spike 检测用

    Parameters
    ----------
    cpu_values : list[float]
        CPU 利用率列表（0~1）
    """
    if not cpu_values:
        return {
            "mean": 0.0, "std": 0.0,
            "q1": 0.0, "median": 0.0, "q3": 0.0,
            "iqr": 0.0, "upper_fence": 1.0,
            "p95": 1.0, "p99": 1.0, "count": 0,
        }

    sorted_vals = sorted(cpu_values)
    n = len(sorted_vals)

    def percentile(p: float) -> float:
        """线性插值法计算分位数。"""
        if n == 1:
            return sorted_vals[0]
        k = (n - 1) * p
        f = int(k)
        c = k - f
        if f + 1 < n:
            return sorted_vals[f] * (1 - c) + sorted_vals[f + 1] * c
        return sorted_vals[f]

    q1 = percentile(0.25)
    median = percentile(0.50)
    q3 = percentile(0.75)
    iqr = q3 - q1
    upper_fence = q3 + 1.5 * iqr

    mean = statistics.mean(cpu_values)
    std = statistics.pstdev(cpu_values) if n > 1 else 0.0

    return {
        "mean": mean,
        "std": std,
        "q1": q1,
        "median": median,
        "q3": q3,
        "iqr": iqr,
        "upper_fence": min(upper_fence, 1.0),  # CPU 上限 1.0
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "count": n,
    }


def detect_anomalies_iqr(
    ts: VMTimeSeries,
    min_duration_points: int = 3,
) -> list[AnomalyPoint]:
    """IQR 法异常检测。

    阈值 = Q3 + 1.5*IQR（上界）。超出阈值的连续点合并成一个窗口型异常。
    窗口长度 >= min_duration_points 才输出（过滤单点抖动）。

    Parameters
    ----------
    ts : VMTimeSeries
        单个 VM 的 CPU 时序
    min_duration_points : int
        最小持续点数（默认 3 = 15 分钟），过滤瞬时抖动
    """
    if ts.reading_count < 10:
        return []  # 数据太少不做检测

    cpu_values = [r[1] for r in ts.cpu_readings]
    stats = compute_baseline_stats(cpu_values)
    threshold = stats["upper_fence"]
    baseline = stats["median"]

    anomalies: list[AnomalyPoint] = []
    window_start_idx: int | None = None
    window_peak: float = 0.0

    for i, (dt, cpu) in enumerate(ts.cpu_readings):
        if cpu > threshold:
            if window_start_idx is None:
                window_start_idx = i
                window_peak = cpu
            else:
                window_peak = max(window_peak, cpu)
        else:
            if window_start_idx is not None:
                window_len = i - window_start_idx
                if window_len >= min_duration_points:
                    start_dt = ts.cpu_readings[window_start_idx][0]
                    end_dt = ts.cpu_readings[i - 1][0]
                    anomalies.append(AnomalyPoint(
                        anomaly_type=AnomalyType.CPU_SPIKE,
                        vm_id=ts.vm_id,
                        cluster_id=ts.cluster_id,
                        timestamp=start_dt,
                        end_timestamp=end_dt,
                        observed_value=window_peak,
                        baseline_value=baseline,
                        threshold=threshold,
                        detection_method="iqr",
                        duration_seconds=int((end_dt - start_dt).total_seconds()) + SAMPLE_INTERVAL_SECONDS,
                    ))
                window_start_idx = None
                window_peak = 0.0

    # 处理末尾未闭合的窗口
    if window_start_idx is not None:
        window_len = ts.reading_count - window_start_idx
        if window_len >= min_duration_points:
            start_dt = ts.cpu_readings[window_start_idx][0]
            end_dt = ts.cpu_readings[-1][0]
            anomalies.append(AnomalyPoint(
                anomaly_type=AnomalyType.CPU_SPIKE,
                vm_id=ts.vm_id,
                cluster_id=ts.cluster_id,
                timestamp=start_dt,
                end_timestamp=end_dt,
                observed_value=window_peak,
                baseline_value=baseline,
                threshold=threshold,
                detection_method="iqr",
                duration_seconds=int((end_dt - start_dt).total_seconds()) + SAMPLE_INTERVAL_SECONDS,
            ))

    return anomalies


def detect_anomalies_3sigma(
    ts: VMTimeSeries,
    sigma_multiplier: float = 3.0,
    min_duration_points: int = 3,
) -> list[AnomalyPoint]:
    """3-sigma 法异常检测。

    阈值 = mean + sigma_multiplier * std。适合近似高斯分布的指标。
    """
    if ts.reading_count < 10:
        return []

    cpu_values = [r[1] for r in ts.cpu_readings]
    stats = compute_baseline_stats(cpu_values)
    threshold = min(stats["mean"] + sigma_multiplier * stats["std"], 1.0)
    baseline = stats["mean"]

    anomalies: list[AnomalyPoint] = []
    window_start_idx: int | None = None
    window_peak: float = 0.0

    for i, (dt, cpu) in enumerate(ts.cpu_readings):
        if cpu > threshold:
            if window_start_idx is None:
                window_start_idx = i
                window_peak = cpu
            else:
                window_peak = max(window_peak, cpu)
        else:
            if window_start_idx is not None:
                window_len = i - window_start_idx
                if window_len >= min_duration_points:
                    start_dt = ts.cpu_readings[window_start_idx][0]
                    end_dt = ts.cpu_readings[i - 1][0]
                    anomalies.append(AnomalyPoint(
                        anomaly_type=AnomalyType.CPU_SPIKE,
                        vm_id=ts.vm_id,
                        cluster_id=ts.cluster_id,
                        timestamp=start_dt,
                        end_timestamp=end_dt,
                        observed_value=window_peak,
                        baseline_value=baseline,
                        threshold=threshold,
                        detection_method="3-sigma",
                        duration_seconds=int((end_dt - start_dt).total_seconds()) + SAMPLE_INTERVAL_SECONDS,
                    ))
                window_start_idx = None
                window_peak = 0.0

    if window_start_idx is not None:
        window_len = ts.reading_count - window_start_idx
        if window_len >= min_duration_points:
            start_dt = ts.cpu_readings[window_start_idx][0]
            end_dt = ts.cpu_readings[-1][0]
            anomalies.append(AnomalyPoint(
                anomaly_type=AnomalyType.CPU_SPIKE,
                vm_id=ts.vm_id,
                cluster_id=ts.cluster_id,
                timestamp=start_dt,
                end_timestamp=end_dt,
                observed_value=window_peak,
                baseline_value=baseline,
                threshold=threshold,
                detection_method="3-sigma",
                duration_seconds=int((end_dt - start_dt).total_seconds()) + SAMPLE_INTERVAL_SECONDS,
            ))

    return anomalies


def detect_vm_deletion(ts: VMTimeSeries) -> AnomalyPoint | None:
    """检测 VM 删除事件。

    VM 删除（vmdeleted != 0）在 Azure V2 中是故障/驱逐的强信号。
    返回一个 VM_DELETION 类型的 AnomalyPoint。
    """
    if not ts.is_deleted or ts.vm_deleted is None:
        return None

    # 找删除前的最后一个 CPU 读数作为观测值
    last_cpu = ts.cpu_readings[-1][1] if ts.cpu_readings else 0.0
    cpu_values = [r[1] for r in ts.cpu_readings] if ts.cpu_readings else [0.0]
    stats = compute_baseline_stats(cpu_values)

    return AnomalyPoint(
        anomaly_type=AnomalyType.VM_DELETION,
        vm_id=ts.vm_id,
        cluster_id=ts.cluster_id,
        timestamp=ts.vm_deleted,
        end_timestamp=None,
        observed_value=last_cpu,
        baseline_value=stats["median"],
        threshold=stats["p95"],
        detection_method="deletion_event",
        duration_seconds=0,
    )


def detect_high_variance(
    ts: VMTimeSeries,
    window_size: int = 12,        # 1 小时窗口（12 个 5 分钟点）
    variance_threshold: float = 0.05,
) -> list[AnomalyPoint]:
    """高方差检测 —— 噪声邻居特征。

    滑动窗口计算方差，方差持续偏高表明 CPU 抖动剧烈，
    是 noisy_neighbor 的典型特征。

    Parameters
    ----------
    window_size : int
        滑动窗口大小（点数），默认 12 = 1 小时
    variance_threshold : float
        方差阈值，超过此值视为高方差
    """
    if ts.reading_count < window_size:
        return []

    cpu_values = [r[1] for r in ts.cpu_readings]
    stats = compute_baseline_stats(cpu_values)
    baseline = stats["median"]

    anomalies: list[AnomalyPoint] = []
    i = 0
    while i + window_size <= ts.reading_count:
        window = cpu_values[i:i + window_size]
        var = statistics.pvariance(window)
        if var > variance_threshold:
            window_peak = max(window)
            start_dt = ts.cpu_readings[i][0]
            end_dt = ts.cpu_readings[i + window_size - 1][0]
            # 只在窗口起点报告一次（避免重叠告警）
            anomalies.append(AnomalyPoint(
                anomaly_type=AnomalyType.HIGH_VARIANCE,
                vm_id=ts.vm_id,
                cluster_id=ts.cluster_id,
                timestamp=start_dt,
                end_timestamp=end_dt,
                observed_value=window_peak,
                baseline_value=baseline,
                threshold=baseline + (var ** 0.5),
                detection_method="variance_window",
                duration_seconds=int((end_dt - start_dt).total_seconds()),
            ))
            i += window_size  # 跳过整个窗口
        else:
            i += 1

    return anomalies


# ============================================================
#  长格式 CSV 流式聚合器
# ============================================================

class _VMAggregator:
    """聚合长格式 CSV 行为单个 VMTimeSeries。

    Azure V2 长格式：每行 = 一个 VM × 一个 5 分钟时间戳。
    同一 vm_id 的多行需要聚合才能得到完整时序。

    流程：
    1. 第一遍扫描：统计每个 vm_id 的行数与是否被删除
    2. 按 max_vms / prefer_deleted 筛选目标 vm_id
    3. 第二遍扫描：只聚合目标 vm_id 的行
    """

    def __init__(self, cluster_id: str):
        self.cluster_id = cluster_id

    def aggregate_rows(self, rows: list[dict[str, str]]) -> VMTimeSeries | None:
        """把同一 vm_id 的多行聚合成 VMTimeSeries。"""
        if not rows:
            return None

        first = rows[0]
        vm_id = first["vm_id"]
        if not vm_id:
            return None

        vm_created = parse_azure_timestamp(first["vm_created"])
        if vm_created is None:
            return None

        vm_deleted = parse_azure_timestamp(first["vm_deleted"])

        # 解析 CPU 读数（按 timestamp 排序）
        readings: list[tuple[datetime, float]] = []
        for r in rows:
            ts = parse_azure_timestamp(r["timestamp"])
            if ts is None:
                continue
            try:
                # 用 5 分钟平均 CPU 作为该时刻的代表值
                cpu_val = float(r["cpu_avg_5min"])
            except (ValueError, KeyError):
                continue
            readings.append((ts, cpu_val))

        # 按时间排序
        readings.sort(key=lambda x: x[0])

        # 解析元信息
        sku = first.get("vm_category", "").strip() or None
        vcore_bucket = resolve_vcore_bucket(
            first.get("vcore_bucket", "0"),
            first.get("vcore_bucket_def", ""),
        )
        memory_gb_bucket = resolve_memory_bucket(
            first.get("memory_gb_bucket", "0"),
            first.get("memory_gb_bucket_def", ""),
        )

        return VMTimeSeries(
            vm_id=vm_id,
            cluster_id=self.cluster_id,
            cpu_readings=readings,
            vm_created=vm_created,
            vm_deleted=vm_deleted,
            sku=sku,
            vcore_bucket=vcore_bucket,
            memory_gb_bucket=memory_gb_bucket,
        )


def stream_vm_timeseries_long(
    csv_path: Path,
    cluster_id: str,
    max_vms: int | None = None,
    prefer_deleted: bool = True,
) -> Iterator[VMTimeSeries]:
    """流式读取 Azure V2 长格式 CSV，按 vm_id 聚合后产出 VMTimeSeries。

    Azure V2 是长格式：每行 = 一个 VM × 一个 5 分钟时间戳。
    本函数两遍扫描：
    1. 第一遍：统计每个 vm_id 的行数与删除状态，按优先级筛选
    2. 第二遍：只聚合目标 vm_id 的所有行

    Parameters
    ----------
    csv_path : Path
        Azure V2 CSV 文件路径（无表头，20 列）
    cluster_id : str
        集群标识（用于 group_id 隔离）
    max_vms : int | None
        最多加载多少 VM，None=全部
    prefer_deleted : bool
        是否优先加载被删除的 VM（故障密度高）
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Azure V2 CSV 文件不存在: {csv_path}")

    # ---- 第一遍：扫描 vm_id 元信息（vm_created/vm_deleted/category/桶） ----
    vm_meta: dict[str, dict[str, str]] = {}
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 18:
                continue
            parsed = parse_long_format_row(row)
            vm_id = parsed["vm_id"]
            if not vm_id:
                continue
            # 只保留首次出现的元信息（同一 VM 的元信息在每行重复）
            if vm_id not in vm_meta:
                vm_meta[vm_id] = parsed

    if not vm_meta:
        return

    # ---- 筛选目标 vm_id ----
    if max_vms is not None:
        if prefer_deleted:
            deleted_ids = [
                vid for vid, meta in vm_meta.items()
                if parse_azure_timestamp(meta["vm_deleted"]) is not None
            ]
            alive_ids = [
                vid for vid, meta in vm_meta.items()
                if parse_azure_timestamp(meta["vm_deleted"]) is None
            ]
            # 优先删除的 VM，不够再补存活的
            target_ids = set(deleted_ids[:max_vms])
            if len(target_ids) < max_vms:
                remaining = max_vms - len(target_ids)
                target_ids.update(alive_ids[:remaining])
        else:
            target_ids = set(list(vm_meta.keys())[:max_vms])
    else:
        target_ids = set(vm_meta.keys())

    # ---- 第二遍：聚合目标 vm_id 的所有行 ----
    rows_by_vm: dict[str, list[dict[str, str]]] = {vid: [] for vid in target_ids}
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 18:
                continue
            vm_id = row[AZURE_V2_COLUMN_INDEX["vm_id"]].strip()
            if vm_id not in target_ids:
                continue
            rows_by_vm[vm_id].append(parse_long_format_row(row))

    # ---- 聚合并 yield ----
    aggregator = _VMAggregator(cluster_id=cluster_id)
    for vm_id in target_ids:
        rows = rows_by_vm.get(vm_id, [])
        if not rows:
            continue
        ts = aggregator.aggregate_rows(rows)
        if ts is not None:
            yield ts


# ============================================================
#  兼容旧接口（已弃用，保留向后兼容）
# ============================================================

def parse_vm_row(row: dict[str, str], cluster_id: str) -> VMTimeSeries | None:
    """[已弃用] 解析宽格式 CSV 行。保留供旧测试兼容。

    新代码请用 stream_vm_timeseries_long 处理真实 Azure V2 长格式。
    """
    try:
        vm_id = row.get("vm_id", row.get("vmid", "")).strip()
        if not vm_id:
            return None
        vm_created = parse_azure_timestamp(row.get("vm_created", row.get("vmcreated", 0)))
        if vm_created is None:
            return None
        vm_deleted = parse_azure_timestamp(row.get("vm_deleted", row.get("vmdeleted", 0)))
        cpu_raw = row.get("cpu", row.get("cpu_readings", ""))
        cpu_readings = parse_cpu_readings(cpu_raw, vm_created)
        return VMTimeSeries(
            vm_id=vm_id,
            cluster_id=cluster_id,
            cpu_readings=cpu_readings,
            vm_created=vm_created,
            vm_deleted=vm_deleted,
            sku=row.get("vm_category", row.get("vmcategory", "")).strip() or None,
            vcore_bucket=row.get("vcore_bucket") or None,
            memory_gb_bucket=row.get("memory_gb_bucket") or None,
        )
    except Exception:
        return None


def stream_vm_timeseries(
    csv_path: Path,
    cluster_id: str,
    max_vms: int | None = None,
    prefer_deleted: bool = True,
) -> Iterator[VMTimeSeries]:
    """[已弃用] 旧版宽格式加载器。新代码请用 stream_vm_timeseries_long。"""
    if not csv_path.exists():
        raise FileNotFoundError(f"Azure V2 CSV 文件不存在: {csv_path}")

    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        yielded = 0
        for row in reader:
            ts = parse_vm_row(row, cluster_id)
            if ts is not None:
                yield ts
                yielded += 1
                if max_vms is not None and yielded >= max_vms:
                    return


# ============================================================
#  高级 API —— AzureTraceLoader
# ============================================================

class AzureTraceLoader:
    """Azure V2 数据加载器 —— 封装加载 + 异常检测的完整流程。

    支持两种数据源：
    1. 真实 Azure V2 长格式 CSV（无表头，20 列）—— 用 load_long()
    2. 合成数据 / 宽格式 CSV —— 用 load() 或直接传 VMTimeSeries 列表

    使用示例：
        loader = AzureTraceLoader(cluster_id="cluster_A")

        # 真实 Azure V2 长格式
        for ts in loader.load_long(csv_path, max_vms=100):
            anomalies = loader.detect_anomalies(ts)

        # 合成数据
        from data_ingest.synthetic_data import generate_vm_batch
        for ts in generate_vm_batch(num_vms=50):
            anomalies = loader.detect_anomalies(ts)

    抽样策略：
    - max_vms 限制加载 VM 数
    - prefer_deleted=True 优先加载被删除的 VM（故障密度高）
    - 可切换异常检测算法（IQR / 3-sigma）
    """

    def __init__(
        self,
        cluster_id: str,
        detection_method: str = "iqr",
        min_duration_points: int = 3,
        detect_deletion: bool = True,
        detect_variance: bool = False,
    ):
        self.cluster_id = cluster_id
        self.detection_method = detection_method
        self.min_duration_points = min_duration_points
        self.detect_deletion = detect_deletion
        self.detect_variance = detect_variance

    def load(
        self,
        csv_path: Path,
        max_vms: int | None = None,
        prefer_deleted: bool = True,
    ) -> Iterator[VMTimeSeries]:
        """[已弃用] 加载宽格式 CSV。新代码请用 load_long。"""
        yield from stream_vm_timeseries(
            csv_path=csv_path,
            cluster_id=self.cluster_id,
            max_vms=max_vms,
            prefer_deleted=prefer_deleted,
        )

    def load_long(
        self,
        csv_path: Path,
        max_vms: int | None = None,
        prefer_deleted: bool = True,
    ) -> Iterator[VMTimeSeries]:
        """加载真实 Azure V2 长格式 CSV（无表头，20 列）。

        两遍扫描聚合：
        1. 第一遍扫描所有行，收集 vm_id 元信息并筛选目标
        2. 第二遍聚合目标 vm_id 的所有行

        Parameters
        ----------
        csv_path : Path
            Azure V2 CSV 文件路径
        max_vms : int | None
            最多加载多少 VM，None=全部
        prefer_deleted : bool
            是否优先加载被删除的 VM（故障密度高）
        """
        yield from stream_vm_timeseries_long(
            csv_path=csv_path,
            cluster_id=self.cluster_id,
            max_vms=max_vms,
            prefer_deleted=prefer_deleted,
        )

    def detect_anomalies(self, ts: VMTimeSeries) -> list[AnomalyPoint]:
        """对单个 VM 时序运行异常检测，返回所有异常点。"""
        anomalies: list[AnomalyPoint] = []

        # CPU spike 检测
        if self.detection_method == "iqr":
            anomalies.extend(detect_anomalies_iqr(ts, self.min_duration_points))
        elif self.detection_method == "3-sigma":
            anomalies.extend(detect_anomalies_3sigma(ts, min_duration_points=self.min_duration_points))
        elif self.detection_method == "both":
            anomalies.extend(detect_anomalies_iqr(ts, self.min_duration_points))
            anomalies.extend(detect_anomalies_3sigma(ts, min_duration_points=self.min_duration_points))

        # VM 删除检测
        if self.detect_deletion:
            deletion = detect_vm_deletion(ts)
            if deletion is not None:
                anomalies.append(deletion)

        # 高方差检测（可选，噪声邻居特征）
        if self.detect_variance:
            anomalies.extend(detect_high_variance(ts))

        return anomalies

    def load_and_detect(
        self,
        csv_path: Path,
        max_vms: int | None = None,
        prefer_deleted: bool = True,
        long_format: bool = True,
    ) -> Iterator[tuple[VMTimeSeries, list[AnomalyPoint]]]:
        """流式加载并检测异常，产出 (VMTimeSeries, anomalies) 二元组。

        Parameters
        ----------
        long_format : bool
            True=真实 Azure V2 长格式，False=宽格式（兼容旧接口）
        """
        loader_fn = self.load_long if long_format else self.load
        for ts in loader_fn(csv_path, max_vms, prefer_deleted):
            anomalies = self.detect_anomalies(ts)
            yield ts, anomalies
