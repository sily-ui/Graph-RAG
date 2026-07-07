"""SMD (Server Machine Dataset) 数据加载器 —— 多维时序 + 异常标签 + 故障事件抽取。

SMD 数据集来自 OmniAnomaly 论文
（https://github.com/NetManAIOps/OmniAnomaly/tree/master/ServerMachineDataset），
含 28 台服务器 × 38 维指标 × 5 周时序，带异常标签。

数据集结构（data_dir 解压根目录）：
    train/machine-{group}-{idx}.txt              训练集时序（无标签）
    test/machine-{group}-{idx}.txt               测试集时序
    test_label/machine-{group}-{idx}.txt         测试集异常标签（0/1，1 行/分钟）
    interpretation_label/machine-{group}-{idx}.txt  每个异常窗口的贡献维度

28 台机器编号：machine-1-1..8、machine-2-1..10、machine-3-1..10
文件格式：每行 38 个 tab 分隔数值，无表头
时序粒度：1 分钟一条
总长度：train 708405 行 + test 708420 行（约 5 周）
采集起点：2018-08-08 00:00:00 UTC（论文说明）

与 azure_trace_loader 的区别：
- SMD 是多维（38 维），Azure V2 是单维 CPU
- SMD 自带异常标签与解释标签，Azure V2 需算法检测
- SMD 的故障事件由 interpretation_label 直接给出贡献维度
- 本模块不依赖 graphiti_core，纯数据加载层，复用 data_ingest.models.FaultEvent
"""
from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from data_ingest.models import FaultEvent, FaultEventType
from graph_schema.nodes import ComponentType, Severity


# ============================================================
#  常量定义
# ============================================================

# SMD 采集起点（论文说明 2018/8/8 开始）
SMD_EPOCH: datetime = datetime(2018, 8, 8, 0, 0, 0, tzinfo=timezone.utc)

# 采样间隔（秒）—— 1 分钟一条
SMD_SAMPLE_INTERVAL_SECONDS: int = 60

# 指标维度数
SMD_NUM_DIMENSIONS: int = 38

# SMD 38 维指标标准名（OmniAnomaly 论文）
# 前 10 个有明确语义，其余用 dim_N 通用名
SMD_METRIC_NAMES: list[str] = [
    "cpu_user_rate",       # 0
    "cpu_system_rate",     # 1
    "cpu_idle_rate",       # 2
    "memory_used_rate",    # 3
    "memory_free_rate",    # 4
    "memory_cached",       # 5
    "net_in_bytes",        # 6
    "net_out_bytes",       # 7
    "net_in_packets",      # 8
    "net_out_packets",     # 9
    # 10-15 网络扩展指标
    "dim_10", "dim_11", "dim_12", "dim_13", "dim_14", "dim_15",
    # 16-25 磁盘 IO 指标
    "dim_16", "dim_17", "dim_18", "dim_19", "dim_20",
    "dim_21", "dim_22", "dim_23", "dim_24", "dim_25",
    # 26-37 系统指标（context_switch / load_avg_1m / fork_rate ...）
    "dim_26", "dim_27", "dim_28", "dim_29", "dim_30",
    "dim_31", "dim_32", "dim_33", "dim_34", "dim_35",
    "dim_36", "dim_37",
]

# 所有合法机器 ID（28 台）
SMD_MACHINE_IDS: list[str] = (
    [f"machine-1-{i}" for i in range(1, 9)]    # group 1: 8 台
    + [f"machine-2-{i}" for i in range(1, 11)]  # group 2: 10 台
    + [f"machine-3-{i}" for i in range(1, 11)]  # group 3: 10 台
)

# 文件名解析正则：machine-{group}-{idx}.txt
_SMD_FILENAME_RE = re.compile(r"^machine-(\d+)-(\d+)\.txt$")


# ============================================================
#  数据模型（本模块内定义，避免污染 models.py）
# ============================================================

@dataclass
class SMDEntity:
    """SMD 单台服务器多维时序。

    Attributes
    ----------
    machine_id : str
        机器标识，如 "machine-1-1"
    group_id : str
        机器组标识，如 "1"
    train_series : np.ndarray
        训练集时序，shape (T_train, 38)
    test_series : np.ndarray
        测试集时序，shape (T_test, 38)
    test_labels : np.ndarray
        测试集异常标签，shape (T_test,) 0/1
    anomaly_windows : list[tuple[int, int]]
        异常窗口起止索引（闭区间，相对 test_series）
    interpretation_labels : list[list[int]]
        每个异常窗口的贡献维度索引列表
    """
    machine_id: str
    group_id: str
    train_series: np.ndarray
    test_series: np.ndarray
    test_labels: np.ndarray
    anomaly_windows: list[tuple[int, int]]
    interpretation_labels: list[list[int]]


# ============================================================
#  工具函数
# ============================================================

def parse_smd_filename(filename: str) -> tuple[str, str] | None:
    """从 machine-1-1.txt 解析出 (machine_id, group_id)。

    Parameters
    ----------
    filename : str
        文件名或完整路径，取末尾 basename 匹配

    Returns
    -------
    tuple[str, str] | None
        (machine_id="machine-1-1", group_id="1")，不匹配返回 None
    """
    name = Path(filename).name
    m = _SMD_FILENAME_RE.match(name)
    if m is None:
        return None
    group_id, idx = m.group(1), m.group(2)
    machine_id = f"machine-{group_id}-{idx}"
    return machine_id, group_id


def download_smd(data_dir: str) -> None:
    """提示用户从 GitHub 获取 SMD 数据集。

    如果 data_dir 不存在或为空，打印 clone 指引；不实际执行下载。
    """
    data_path = Path(data_dir)
    if data_path.exists() and any(data_path.iterdir()):
        return
    print(
        f"[SMD] 数据目录 {data_dir} 不存在或为空。\n"
        "请从 GitHub 获取 SMD 数据集：\n"
        "  git clone https://github.com/NetManAIOps/OmniAnomaly.git tmp_omni\n"
        f"  mv tmp_omni/ServerMachineDataset {data_dir}\n"
        "  rm -rf tmp_omni"
    )


# ============================================================
#  SMDLoader —— 加载 + 窗口抽取 + 故障事件转换
# ============================================================

class SMDLoader:
    """SMD 数据集加载器 —— 封装加载、异常窗口抽取、故障事件转换。

    data_dir 是 SMD 解压根目录（含 train/test/test_label/interpretation_label
    四个子目录）。使用示例：

        loader = SMDLoader("data/ServerMachineDataset")
        for entity in loader.load_all(max_machines=3, max_rows=1000):
            events = loader.extract_fault_events(entity)
            for ev in events:
                print(ev.event_id, ev.event_type, ev.severity)

    时间戳约定（SMD 无显式时间戳，按论文说明）：
    - t=0 → 2018-08-08 00:00:00 UTC
    - 训练集 row i → SMD_EPOCH + i 分钟
    - 测试集 row j → SMD_EPOCH + T_train + j 分钟（紧接训练集）
    """

    def __init__(self, data_dir: str):
        """初始化加载器。

        Parameters
        ----------
        data_dir : str
            SMD 解压根目录（含 train/test/test_label/interpretation_label 子目录）
        """
        self.data_dir = Path(data_dir)
        self.train_dir = self.data_dir / "train"
        self.test_dir = self.data_dir / "test"
        self.test_label_dir = self.data_dir / "test_label"
        self.interp_label_dir = self.data_dir / "interpretation_label"

    # -------------------- 机器发现 --------------------

    def list_machines(self) -> list[str]:
        """返回所有可用机器 ID 列表（扫描 train 目录）。"""
        if not self.train_dir.exists():
            return []
        machines: list[str] = []
        seen: set[str] = set()
        for f in sorted(self.train_dir.glob("machine-*.txt")):
            parsed = parse_smd_filename(f.name)
            if parsed is None:
                continue
            mid, _ = parsed
            if mid not in seen:
                seen.add(mid)
                machines.append(mid)
        return machines

    # -------------------- 底层文件读取 --------------------

    def _load_matrix(self, path: Path, max_rows: int | None = None) -> np.ndarray:
        """加载 tab 分隔的 38 列数值矩阵。

        单行文件会被 np.loadtxt 降为 1D，调用方需 atleast_2d。
        """
        return np.loadtxt(path, delimiter="\t", max_rows=max_rows)

    def _load_labels(self, path: Path, max_rows: int | None = None) -> np.ndarray:
        """加载单列 0/1 标签为 int8 数组。"""
        arr = np.loadtxt(path, delimiter="\t", max_rows=max_rows)
        return np.atleast_1d(arr).astype(np.int8)

    def _load_interp_labels(self, path: Path) -> list[list[int]]:
        """加载解释标签：每行一个异常窗口的贡献维度索引。

        兼容多种分隔符（逗号/空格/tab/分号/括号），用正则提取所有数字序列。
        """
        if not path.exists():
            return []
        result: list[list[int]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                dims = [int(m) for m in re.findall(r"\d+", line)]
                result.append(dims)
        return result

    # -------------------- 单机加载 --------------------

    def load_machine(self, machine_id: str, max_rows: int | None = None) -> SMDEntity:
        """加载单台机器的全部数据。

        Parameters
        ----------
        machine_id : str
            机器标识，如 "machine-1-1"
        max_rows : int | None
            限制训练/测试集行数（调试用），None=全量
        """
        parsed = parse_smd_filename(f"{machine_id}.txt")
        if parsed is None:
            raise ValueError(f"非法 machine_id: {machine_id}")
        _, group_id = parsed

        train_path = self.train_dir / f"{machine_id}.txt"
        test_path = self.test_dir / f"{machine_id}.txt"
        label_path = self.test_label_dir / f"{machine_id}.txt"
        interp_path = self.interp_label_dir / f"{machine_id}.txt"

        if not train_path.exists():
            raise FileNotFoundError(f"训练文件不存在: {train_path}")

        # 训练集（必选）
        train_series = np.atleast_2d(
            self._load_matrix(train_path, max_rows=max_rows)
        )

        # 测试集（可选，缺失则空）
        if test_path.exists():
            test_series = np.atleast_2d(
                self._load_matrix(test_path, max_rows=max_rows)
            )
        else:
            test_series = np.empty((0, SMD_NUM_DIMENSIONS))

        # 异常标签（可选，缺失则全 0）
        if label_path.exists():
            test_labels = self._load_labels(label_path, max_rows=max_rows)
        else:
            test_labels = np.zeros(test_series.shape[0], dtype=np.int8)

        # 对齐标签与测试集长度
        if test_labels.shape[0] > test_series.shape[0]:
            test_labels = test_labels[: test_series.shape[0]]
        elif test_labels.shape[0] < test_series.shape[0]:
            # 标签不足则补 0（容错）
            padded = np.zeros(test_series.shape[0], dtype=np.int8)
            padded[: test_labels.shape[0]] = test_labels
            test_labels = padded

        anomaly_windows = self.find_anomaly_windows(test_labels)
        interpretation_labels = self._load_interp_labels(interp_path)

        return SMDEntity(
            machine_id=machine_id,
            group_id=group_id,
            train_series=train_series,
            test_series=test_series,
            test_labels=test_labels,
            anomaly_windows=anomaly_windows,
            interpretation_labels=interpretation_labels,
        )

    # -------------------- 流式加载 --------------------

    def load_all(
        self,
        max_machines: int | None = None,
        max_rows: int | None = None,
    ) -> Iterator[SMDEntity]:
        """流式加载所有机器。

        Parameters
        ----------
        max_machines : int | None
            最多加载多少台机器，None=全部
        max_rows : int | None
            每台机器限制行数（调试用），None=全量
        """
        machines = self.list_machines()
        if max_machines is not None:
            machines = machines[:max_machines]
        for mid in machines:
            yield self.load_machine(mid, max_rows=max_rows)

    # -------------------- 异常窗口抽取 --------------------

    def find_anomaly_windows(
        self,
        labels: np.ndarray,
        min_gap: int = 5,
    ) -> list[tuple[int, int]]:
        """从 0/1 标签序列抽取异常窗口（连续 1 段）。

        两个异常点索引差 ≤ min_gap 视为同一窗口（合并临近点），
        用 numpy 向量化实现。

        Parameters
        ----------
        labels : np.ndarray
            0/1 标签序列
        min_gap : int
            合并阈值：相邻两个 1 的索引差 ≤ min_gap 归入同一窗口

        Returns
        -------
        list[tuple[int, int]]
            异常窗口 (start, end) 闭区间列表
        """
        if labels.size == 0:
            return []
        ones = np.where(labels > 0)[0]
        if ones.size == 0:
            return []
        if ones.size == 1:
            return [(int(ones[0]), int(ones[0]))]

        # 索引差 > min_gap 的位置切分新段
        diffs = np.diff(ones)
        split_idx = np.where(diffs > min_gap)[0] + 1
        segments = np.split(ones, split_idx)

        return [(int(seg[0]), int(seg[-1])) for seg in segments if seg.size > 0]

    # -------------------- 故障事件抽取 --------------------

    def extract_fault_events(
        self,
        entity: SMDEntity,
        max_events: int | None = None,
    ) -> list[FaultEvent]:
        """从 SMD 异常窗口抽取故障事件，转换为项目统一的 FaultEvent。

        关键映射：
        - machine_id → vm_id（沿用现有字段名）
        - group_id → cluster_id
        - 异常窗口中位数索引 → timestamp_start
        - 异常窗口结束索引 → timestamp_end
        - 主要贡献维度（interpretation_labels）→ metric_name
        - 偏离倍数 = max(窗口内中位值) / 训练集同维度中位数
        - severity：偏离 ≥3 → CRITICAL，≥2 → CRITICAL（ERROR 不在枚举，提升），
          ≥1.5 → WARNING，否则 INFO
        - event_type：贡献维度落在 0-2 → CPU_SPIKE，3-5 → OOM_KILLED，
          6-9 → LATENCY_SURGE，默认 CPU_SPIKE
        - source_dataset = "smd"
        """
        events: list[FaultEvent] = []

        if entity.train_series.size == 0:
            return events

        # 训练集基线统计（每维）
        train_median = np.median(entity.train_series, axis=0)
        train_p95 = np.percentile(entity.train_series, 95, axis=0)

        # 测试集起始分钟 = 训练集行数（测试紧接训练）
        test_start_minute = int(entity.train_series.shape[0])

        windows = entity.anomaly_windows
        interp_labels = entity.interpretation_labels

        for i, (start, end) in enumerate(windows):
            # 贡献维度（对齐解释标签；缺失则用全部维度）
            if i < len(interp_labels) and interp_labels[i]:
                contrib_dims = interp_labels[i]
            else:
                contrib_dims = list(range(SMD_NUM_DIMENSIONS))
            contrib_dims = [d for d in contrib_dims if 0 <= d < SMD_NUM_DIMENSIONS]
            if not contrib_dims:
                contrib_dims = [0]

            # 异常窗口内的测试数据
            window_data = entity.test_series[start: end + 1]
            if window_data.size == 0:
                continue

            # 每个贡献维度的窗口中位数，取偏离最大者
            window_medians = np.median(window_data[:, contrib_dims], axis=0)
            best_dev = -float("inf")
            best_dim = contrib_dims[0]
            obs_value = 0.0
            base_value = 0.0
            for j, dim in enumerate(contrib_dims):
                base = float(train_median[dim])
                obs = float(window_medians[j])
                if base != 0.0:
                    dev = obs / base
                else:
                    dev = float("inf") if obs > 0 else 0.0
                if dev > best_dev:
                    best_dev = dev
                    best_dim = dim
                    obs_value = obs
                    base_value = base

            # severity（ERROR 不在 Severity 枚举，≥2 提升为 CRITICAL）
            if best_dev >= 3.0:
                severity = Severity.CRITICAL
            elif best_dev >= 2.0:
                severity = Severity.CRITICAL
            elif best_dev >= 1.5:
                severity = Severity.WARNING
            else:
                severity = Severity.INFO

            # event_type：按贡献维度范围推断
            if any(0 <= d <= 2 for d in contrib_dims):
                event_type = FaultEventType.CPU_SPIKE
            elif any(3 <= d <= 5 for d in contrib_dims):
                event_type = FaultEventType.OOM_KILLED
            elif any(6 <= d <= 9 for d in contrib_dims):
                event_type = FaultEventType.LATENCY_SURGE
            else:
                event_type = FaultEventType.CPU_SPIKE

            metric_name = SMD_METRIC_NAMES[best_dim]

            # 时间戳：中位数索引 → start，结束索引 → end
            median_idx = (start + end) // 2
            timestamp_start = SMD_EPOCH + timedelta(
                minutes=test_start_minute + median_idx
            )
            timestamp_end = SMD_EPOCH + timedelta(
                minutes=test_start_minute + end
            )

            # 阈值取训练集同维度 p95
            threshold = float(train_p95[best_dim])

            # trace_fragment：窗口内 best_dim 时序（采样上限 500 点）
            trace_fragment: list[tuple[datetime, float]] = [
                (
                    SMD_EPOCH + timedelta(minutes=test_start_minute + start + k),
                    float(v),
                )
                for k, v in enumerate(window_data[:, best_dim])
            ]
            if len(trace_fragment) > 500:
                step = max(1, len(trace_fragment) // 500)
                trace_fragment = trace_fragment[::step][:500]

            event_id = (
                f"fault_smd_{entity.group_id}_{entity.machine_id}_{start}_{end}"
            )

            events.append(FaultEvent(
                event_id=event_id,
                event_type=event_type,
                vm_id=entity.machine_id,
                cluster_id=entity.group_id,
                timestamp_start=timestamp_start,
                timestamp_end=timestamp_end,
                severity=severity,
                component_type=ComponentType.VM,
                metric_name=metric_name,
                observed_value=obs_value,
                baseline_value=base_value,
                threshold=threshold,
                trace_fragment=trace_fragment,
                detection_method="smd_label",
                source_dataset="smd",
            ))

            if max_events is not None and len(events) >= max_events:
                break

        return events

    # -------------------- 流式故障事件 --------------------

    def stream_machine_faults(
        self,
        max_machines: int | None = None,
        max_rows: int | None = None,
    ) -> Iterator[FaultEvent]:
        """流式产出所有机器的所有故障事件。

        Parameters
        ----------
        max_machines : int | None
            最多加载多少台机器，None=全部
        max_rows : int | None
            每台机器限制行数（调试用），None=全量
        """
        for entity in self.load_all(
            max_machines=max_machines, max_rows=max_rows
        ):
            for event in self.extract_fault_events(entity):
                yield event
