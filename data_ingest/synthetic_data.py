"""合成数据生成器 —— 无 235GB 真实 Azure V2 数据时用于开发与测试。

本模块生成符合 Azure V2 数据特征的合成 CPU 时序，注入可控故障，
保证数据接入管线在无真实数据时也能跑通端到端流程。

合成策略：
1. 基线 CPU：Beta 分布模拟真实 VM CPU 利用率（集中在 0.2~0.6）
2. 噪声：高斯噪声模拟 5 分钟采样抖动
3. 故障注入（按比例）：
   - CPU spike：在故障窗口内 CPU 飙升到 0.9+
   - VM 删除：在故障点后设置 vm_deleted
   - 高方差：噪声邻居特征，窗口内方差放大
4. 元信息：随机分配 sku / vcore / memory 桶，符合 Azure V2 分布

种子控制：
- 用固定 random.seed 保证可复现
- 不同 VM 用不同子种子，保证多样性
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import Iterator

from data_ingest.models import VMTimeSeries


# ============================================================
#  合成参数
# ============================================================

DEFAULT_SAMPLE_INTERVAL_SECONDS = 300  # 5 分钟
DEFAULT_OBSERVATION_DAYS = 7           # 默认观察期 7 天（Azure V2 是 30 天，这里缩减）
DEFAULT_FAULT_RATE = 0.15              # 15% VM 会发生故障

# Azure V2 VM 规格分布（简化版）
SKU_CHOICES = ["D-series", "E-series", "F-series", "A-series"]
VCORE_BUCKETS = ["1-2", "3-4", "5-8", "9-16", "17-32"]
VCORE_WEIGHTS = [0.1, 0.25, 0.35, 0.2, 0.1]
MEMORY_BUCKETS = ["1-4", "5-8", "9-16", "17-32", "33-64", "65-128"]
MEMORY_WEIGHTS = [0.1, 0.2, 0.3, 0.2, 0.15, 0.05]

# 基线 CPU 的 Beta 分布参数（集中在 0.2~0.6）
CPU_BETA_ALPHA = 5.0
CPU_BETA_BETA = 8.0
# 噪声标准差
CPU_NOISE_STD = 0.03


# ============================================================
#  合成 VM 时序
# ============================================================

def _generate_baseline_cpu(
    num_points: int,
    rng: random.Random,
) -> list[float]:
    """生成基线 CPU 时序（Beta 分布 + 小噪声）。"""
    baseline = []
    for _ in range(num_points):
        # Beta 分布近似 CPU 利用率
        base = rng.betavariate(CPU_BETA_ALPHA, CPU_BETA_BETA)
        # 加少量噪声
        noise = rng.gauss(0, CPU_NOISE_STD)
        cpu = max(0.01, min(0.99, base + noise))
        baseline.append(cpu)
    return baseline


def _inject_cpu_spike(
    cpu_values: list[float],
    start_idx: int,
    duration_points: int,
    spike_peak: float,
    rng: random.Random,
) -> list[float]:
    """在指定位置注入 CPU spike。"""
    end_idx = min(start_idx + duration_points, len(cpu_values))
    for i in range(start_idx, end_idx):
        # 渐升渐降的 spike 形状
        progress = (i - start_idx) / max(duration_points, 1)
        # 三角形脉冲：先升后降
        if progress < 0.3:
            factor = progress / 0.3
        else:
            factor = 1.0 - (progress - 0.3) / 0.7 * 0.3
        spike_val = spike_peak * factor + rng.gauss(0, 0.02)
        cpu_values[i] = max(cpu_values[i], min(0.999, spike_val))
    return cpu_values


def _inject_high_variance(
    cpu_values: list[float],
    start_idx: int,
    duration_points: int,
    rng: random.Random,
) -> list[float]:
    """在指定窗口注入高方差（噪声邻居特征）。"""
    end_idx = min(start_idx + duration_points, len(cpu_values))
    base = sum(cpu_values[start_idx:end_idx]) / max(end_idx - start_idx, 1)
    for i in range(start_idx, end_idx):
        # 大幅抖动
        cpu_values[i] = max(0.05, min(0.95, base + rng.gauss(0, 0.2)))
    return cpu_values


def generate_vm_timeseries(
    vm_id: str,
    cluster_id: str,
    vm_created: datetime | None = None,
    observation_days: int = DEFAULT_OBSERVATION_DAYS,
    inject_fault: bool | None = None,
    rng: random.Random | None = None,
) -> VMTimeSeries:
    """生成单个 VM 的合成时序数据。

    Parameters
    ----------
    vm_id : str
        VM 唯一标识
    cluster_id : str
        集群标识
    vm_created : datetime | None
        VM 创建时刻，None=用观察期起点
    observation_days : int
        观察期天数
    inject_fault : bool | None
        是否注入故障，None=按 DEFAULT_FAULT_RATE 随机决定
    rng : random.Random | None
        随机数生成器，None=用默认
    """
    if rng is None:
        rng = random.Random(hash(vm_id) & 0xFFFFFFFF)
    if vm_created is None:
        # 默认 2024-01-01 UTC 起点
        vm_created = datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(
            seconds=rng.randint(0, 86400)
        )

    # 决定是否注入故障
    if inject_fault is None:
        inject_fault = rng.random() < DEFAULT_FAULT_RATE

    # 计算采样点数
    observation_seconds = observation_days * 86400
    num_points = observation_seconds // DEFAULT_SAMPLE_INTERVAL_SECONDS

    # 生成基线 CPU
    cpu_values = _generate_baseline_cpu(num_points, rng)

    # VM 创建时刻对应的索引（相对于观察期起点）
    # 简化：vm_created 就是观察期起点，所有点从 vm_created 开始
    timestamps = [
        vm_created + timedelta(seconds=i * DEFAULT_SAMPLE_INTERVAL_SECONDS)
        for i in range(num_points)
    ]

    # 注入故障
    vm_deleted: datetime | None = None
    if inject_fault and num_points > 100:
        # 故障类型：spike / deletion / variance
        fault_type = rng.choices(
            ["spike", "deletion", "variance", "spike+deletion"],
            weights=[0.4, 0.2, 0.2, 0.2],
        )[0]

        # 故障发生位置：观察期中段
        fault_start = rng.randint(num_points // 3, num_points * 2 // 3)
        fault_duration = rng.randint(3, 12)  # 15~60 分钟

        if "spike" in fault_type:
            spike_peak = rng.uniform(0.85, 0.99)
            cpu_values = _inject_cpu_spike(
                cpu_values, fault_start, fault_duration, spike_peak, rng,
            )

        if "variance" in fault_type:
            cpu_values = _inject_high_variance(
                cpu_values, fault_start, fault_duration, rng,
            )

        if "deletion" in fault_type:
            # VM 在故障后某时刻被删除
            deletion_offset = rng.randint(fault_duration, fault_duration + 36)  # 0~3 小时后
            deletion_idx = min(fault_start + deletion_offset, num_points - 1)
            vm_deleted = timestamps[deletion_idx]
            # 删除后的 CPU 设为 0
            for i in range(deletion_idx, num_points):
                cpu_values[i] = 0.0

    # 元信息
    sku = rng.choice(SKU_CHOICES)
    vcore_bucket = rng.choices(VCORE_BUCKETS, weights=VCORE_WEIGHTS)[0]
    memory_gb_bucket = rng.choices(MEMORY_BUCKETS, weights=MEMORY_WEIGHTS)[0]

    # 组装时序
    cpu_readings = list(zip(timestamps, cpu_values))
    # 如果 VM 被删除，截断删除后的读数
    if vm_deleted is not None:
        cpu_readings = [(t, c) for t, c in cpu_readings if t <= vm_deleted]

    return VMTimeSeries(
        vm_id=vm_id,
        cluster_id=cluster_id,
        cpu_readings=cpu_readings,
        vm_created=vm_created,
        vm_deleted=vm_deleted,
        sku=sku,
        vcore_bucket=vcore_bucket,
        memory_gb_bucket=memory_gb_bucket,
    )


def generate_vm_batch(
    num_vms: int,
    cluster_id: str = "cluster_A",
    vm_id_prefix: str = "vm",
    observation_days: int = DEFAULT_OBSERVATION_DAYS,
    fault_rate: float = DEFAULT_FAULT_RATE,
    seed: int = 42,
) -> list[VMTimeSeries]:
    """批量生成 VM 时序数据。

    Parameters
    ----------
    num_vms : int
        生成 VM 数量
    cluster_id : str
        集群标识
    vm_id_prefix : str
        VM ID 前缀
    observation_days : int
        观察期天数
    fault_rate : float
        故障率（0~1）
    seed : int
        随机种子（保证可复现）
    """
    rng = random.Random(seed)
    vm_list: list[VMTimeSeries] = []
    for i in range(num_vms):
        vm_id = f"{vm_id_prefix}_{i:06d}"
        # 用子种子保证每个 VM 不同但可复现
        sub_seed = seed + i
        sub_rng = random.Random(sub_seed)
        # 按故障率决定是否注入故障
        inject = sub_rng.random() < fault_rate
        ts = generate_vm_timeseries(
            vm_id=vm_id,
            cluster_id=cluster_id,
            observation_days=observation_days,
            inject_fault=inject,
            rng=sub_rng,
        )
        vm_list.append(ts)
    return vm_list


def generate_vm_stream(
    num_vms: int,
    cluster_id: str = "cluster_A",
    vm_id_prefix: str = "vm",
    observation_days: int = DEFAULT_OBSERVATION_DAYS,
    fault_rate: float = DEFAULT_FAULT_RATE,
    seed: int = 42,
) -> Iterator[VMTimeSeries]:
    """流式生成 VM 时序（避免一次性占用内存）。"""
    rng = random.Random(seed)
    for i in range(num_vms):
        vm_id = f"{vm_id_prefix}_{i:06d}"
        sub_seed = seed + i
        sub_rng = random.Random(sub_seed)
        inject = sub_rng.random() < fault_rate
        yield generate_vm_timeseries(
            vm_id=vm_id,
            cluster_id=cluster_id,
            observation_days=observation_days,
            inject_fault=inject,
            rng=sub_rng,
        )
