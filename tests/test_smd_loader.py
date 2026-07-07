"""SMDLoader 单元测试 —— 覆盖文件名解析、加载、异常窗口抽取、故障事件转换。

运行方式（在项目根目录）：
    python -m pytest tests/test_smd_loader.py -v
或直接运行：
    python tests/test_smd_loader.py

测试不依赖真实 SMD 数据集，用 tmp_path + numpy 构造 mock 数据，
每个测试独立可运行，不依赖网络。
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

# 确保能导入项目模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_ingest.models import FaultEvent, FaultEventType
from data_ingest.smd_loader import (
    SMDLoader,
    SMDEntity,
    SMD_EPOCH,
    SMD_METRIC_NAMES,
    SMD_NUM_DIMENSIONS,
    download_smd,
    parse_smd_filename,
)
from graph_schema.nodes import Severity


# ============================================================
#  辅助函数 —— 构造 mock SMD 数据
# ============================================================

def _write_matrix(path: Path, arr: np.ndarray, fmt: str = "%.6f") -> None:
    """写 tab 分隔的数值矩阵文件（无表头）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, arr, delimiter="\t", fmt=fmt)


def _write_labels(path: Path, labels: np.ndarray) -> None:
    """写单列 0/1 标签文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, labels, delimiter="\t", fmt="%d")


def _build_smd_dataset(
    root: Path,
    machine_id: str = "machine-1-1",
    n_rows: int = 100,
    anomaly_start: int = 50,
    anomaly_end: int = 60,
    seed: int = 42,
) -> Path:
    """在 root 下构造完整的 SMD mock 数据集目录结构，返回 root。

    结构：
        train/machine-1-1.txt              100 行 × 38 列 正态分布
        test/machine-1-1.txt               100 行 × 38 列
        test_label/machine-1-1.txt         100 行 0/1，第 50-60 行为 1
        interpretation_label/machine-1-1.txt  一行 "0,1" 表示贡献维度
    """
    rng = np.random.default_rng(seed)
    # loc=50 保证中位数为正，偏离比稳定（避免 0/0）
    train = rng.normal(loc=50.0, scale=5.0, size=(n_rows, SMD_NUM_DIMENSIONS))
    test = rng.normal(loc=50.0, scale=5.0, size=(n_rows, SMD_NUM_DIMENSIONS))
    labels = np.zeros(n_rows, dtype=np.int8)
    labels[anomaly_start: anomaly_end + 1] = 1

    _write_matrix(root / "train" / f"{machine_id}.txt", train)
    _write_matrix(root / "test" / f"{machine_id}.txt", test)
    _write_labels(root / "test_label" / f"{machine_id}.txt", labels)
    # 解释标签：一行 "0,1" 表示该异常窗口贡献维度为 0、1
    interp_path = root / "interpretation_label" / f"{machine_id}.txt"
    interp_path.parent.mkdir(parents=True, exist_ok=True)
    interp_path.write_text("0,1\n", encoding="utf-8")
    return root


@pytest.fixture
def smd_data_dir(tmp_path) -> Path:
    """pytest fixture：在临时目录构造 SMD mock 数据集，返回根目录。"""
    return _build_smd_dataset(tmp_path)


def _make_entity(
    train: np.ndarray,
    test: np.ndarray,
    labels: np.ndarray,
    interp: list[list[int]] | None = None,
    machine_id: str = "machine-1-1",
    group_id: str = "1",
    loader: SMDLoader | None = None,
) -> SMDEntity:
    """直接用数组构造 SMDEntity（绕过文件加载），用于受控测试。"""
    if loader is None:
        loader = SMDLoader(".")  # 不实际读文件
    windows = loader.find_anomaly_windows(labels)
    return SMDEntity(
        machine_id=machine_id,
        group_id=group_id,
        train_series=train,
        test_series=test,
        test_labels=labels,
        anomaly_windows=windows,
        interpretation_labels=interp or [],
    )


# ============================================================
#  1. 工具函数测试
# ============================================================

def test_parse_smd_filename_valid():
    """合法文件名 machine-1-1.txt 解析出 (machine_id, group_id)。"""
    parsed = parse_smd_filename("machine-1-1.txt")
    assert parsed == ("machine-1-1", "1")


def test_parse_smd_filename_with_path():
    """带路径的文件名只取 basename 匹配。"""
    parsed = parse_smd_filename("/path/to/machine-2-5.txt")
    assert parsed == ("machine-2-5", "2")


def test_parse_smd_filename_invalid():
    """非法文件名（无数字段）返回 None。"""
    assert parse_smd_filename("machine-abc.txt") is None


def test_download_smd_prints_hint(capsys, tmp_path):
    """download_smd 在目录不存在时打印 git clone 提示。"""
    target = tmp_path / "nonexistent_smd"
    download_smd(str(target))
    captured = capsys.readouterr()
    assert "git clone" in captured.out
    assert "OmniAnomaly" in captured.out


# ============================================================
#  2. SMDLoader 基础测试
# ============================================================

def test_loader_init_missing_dir(tmp_path):
    """data_dir 不存在时仍能初始化（懒加载），list_machines 返回空。"""
    loader = SMDLoader(str(tmp_path / "missing"))
    assert loader.list_machines() == []


def test_loader_init_empty_dir(tmp_path):
    """空目录时 list_machines 返回空。"""
    loader = SMDLoader(str(tmp_path))
    assert loader.list_machines() == []


# ============================================================
#  3. 加载与机器发现测试
# ============================================================

def test_list_machines(smd_data_dir):
    """list_machines 返回 mock 数据集中的机器列表。"""
    loader = SMDLoader(str(smd_data_dir))
    assert loader.list_machines() == ["machine-1-1"]


def test_load_machine(smd_data_dir):
    """load_machine 返回 SMDEntity，各数组 shape 正确。"""
    loader = SMDLoader(str(smd_data_dir))
    entity = loader.load_machine("machine-1-1")
    assert entity.machine_id == "machine-1-1"
    assert entity.group_id == "1"
    assert entity.train_series.shape == (100, SMD_NUM_DIMENSIONS)
    assert entity.test_series.shape == (100, SMD_NUM_DIMENSIONS)
    assert entity.test_labels.shape == (100,)


def test_load_machine_with_max_rows(smd_data_dir):
    """max_rows=50 限制加载行数。"""
    loader = SMDLoader(str(smd_data_dir))
    entity = loader.load_machine("machine-1-1", max_rows=50)
    assert entity.train_series.shape == (50, SMD_NUM_DIMENSIONS)
    assert entity.test_series.shape == (50, SMD_NUM_DIMENSIONS)


def test_load_machine_not_found(smd_data_dir):
    """加载不存在的机器抛 FileNotFoundError。"""
    loader = SMDLoader(str(smd_data_dir))
    with pytest.raises(FileNotFoundError):
        loader.load_machine("machine-9-9")


def test_load_all(smd_data_dir):
    """load_all 迭代器产出全部机器的 entity。"""
    loader = SMDLoader(str(smd_data_dir))
    entities = list(loader.load_all())
    assert len(entities) == 1
    assert entities[0].machine_id == "machine-1-1"


# ============================================================
#  4. 异常窗口抽取测试
# ============================================================

def test_find_anomaly_windows():
    """连续 1 段抽取为窗口，间隔大于 min_gap 切分。"""
    loader = SMDLoader(".")
    labels = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0], dtype=np.int8)
    # ones=[2,3,4,7,8]，索引 4→7 间隔 3，min_gap=2 时切分为两段
    windows = loader.find_anomaly_windows(labels, min_gap=2)
    assert windows == [(2, 4), (7, 8)]


def test_find_anomaly_windows_with_min_gap():
    """min_gap=3 时间隔小于等于 3 的 1 合并为同一窗口。"""
    loader = SMDLoader(".")
    labels = np.array([0, 0, 1, 1, 0, 1, 1, 0], dtype=np.int8)
    # ones=[2,3,5,6]，间隔 1,2,1 均 ≤3，合并为 (2,6)
    windows = loader.find_anomaly_windows(labels, min_gap=3)
    assert windows == [(2, 6)]


# ============================================================
#  5. 故障事件抽取测试
# ============================================================

def test_extract_fault_events(smd_data_dir):
    """从 mock entity 抽出 FaultEvent，验证关键字段映射。"""
    loader = SMDLoader(str(smd_data_dir))
    entity = loader.load_machine("machine-1-1")
    events = loader.extract_fault_events(entity)
    assert len(events) == 1
    ev = events[0]
    # 贡献维度 0,1 落在 0-2 → CPU_SPIKE
    assert ev.event_type == FaultEventType.CPU_SPIKE
    assert ev.vm_id == "machine-1-1"
    assert ev.metric_name in SMD_METRIC_NAMES
    assert ev.source_dataset == "smd"
    assert isinstance(ev.severity, Severity)


def test_extract_fault_events_max_events():
    """max_events 限制返回事件数。"""
    loader = SMDLoader(".")
    # 构造两个异常窗口
    train = np.full((20, SMD_NUM_DIMENSIONS), 50.0)
    test = np.full((20, SMD_NUM_DIMENSIONS), 50.0)
    labels = np.zeros(20, dtype=np.int8)
    labels[2:5] = 1
    labels[10:13] = 1
    entity = _make_entity(train, test, labels, interp=[[0], [0]], loader=loader)
    events = loader.extract_fault_events(entity, max_events=1)
    assert len(events) == 1


def test_stream_machine_faults(smd_data_dir):
    """stream_machine_faults 流式产出 FaultEvent。"""
    loader = SMDLoader(str(smd_data_dir))
    events = list(loader.stream_machine_faults())
    assert len(events) == 1
    assert isinstance(events[0], FaultEvent)
    assert events[0].source_dataset == "smd"


# ============================================================
#  6. 边界与容错测试
# ============================================================

def test_load_machine_single_row_file(tmp_path):
    """单行文件（np.loadtxt 会降维）能正确处理为 (1, 38)。"""
    root = tmp_path
    machine_id = "machine-1-1"
    single_row = np.array(
        [[float(c) for c in range(SMD_NUM_DIMENSIONS)]], dtype=np.float64
    )
    _write_matrix(root / "train" / f"{machine_id}.txt", single_row)
    _write_matrix(root / "test" / f"{machine_id}.txt", single_row)
    _write_labels(
        root / "test_label" / f"{machine_id}.txt", np.array([0], dtype=np.int8)
    )
    loader = SMDLoader(str(root))
    entity = loader.load_machine(machine_id)
    assert entity.train_series.shape == (1, SMD_NUM_DIMENSIONS)
    assert entity.test_series.shape == (1, SMD_NUM_DIMENSIONS)


def test_load_machine_label_length_mismatch(tmp_path):
    """标签行数与时序行数不一致时不崩溃（容错补 0）。"""
    root = tmp_path
    machine_id = "machine-1-1"
    n_rows = 100
    train = np.full((n_rows, SMD_NUM_DIMENSIONS), 50.0)
    test = np.full((n_rows, SMD_NUM_DIMENSIONS), 50.0)
    # 标签只有 80 行（少于时序）
    labels = np.zeros(80, dtype=np.int8)
    _write_matrix(root / "train" / f"{machine_id}.txt", train)
    _write_matrix(root / "test" / f"{machine_id}.txt", test)
    _write_labels(root / "test_label" / f"{machine_id}.txt", labels)
    loader = SMDLoader(str(root))
    entity = loader.load_machine(machine_id)
    # 标签被补齐到 100 行
    assert entity.test_labels.shape[0] == n_rows


# ============================================================
#  7. 时间戳与严重程度映射测试
# ============================================================

def test_timestamp_mapping():
    """SMD_EPOCH + i 分钟映射正确，FaultEvent.timestamp_start 在 2018-08-08 之后。"""
    loader = SMDLoader(".")
    # train 10 行，test 标签窗口 (3,4)
    train = np.full((10, SMD_NUM_DIMENSIONS), 50.0)
    test = np.full((10, SMD_NUM_DIMENSIONS), 50.0)
    labels = np.zeros(10, dtype=np.int8)
    labels[3:5] = 1
    entity = _make_entity(train, test, labels, interp=[[0]], loader=loader)
    events = loader.extract_fault_events(entity)
    assert len(events) == 1
    ev = events[0]
    # test_start_minute = 10，median_idx = (3+4)//2 = 3
    expected_start = SMD_EPOCH + timedelta(minutes=10 + 3)
    expected_end = SMD_EPOCH + timedelta(minutes=10 + 4)
    assert ev.timestamp_start == expected_start
    assert ev.timestamp_end == expected_end
    assert ev.timestamp_start > datetime(2018, 8, 8, tzinfo=timezone.utc)


def test_severity_mapping():
    """偏离倍数 ≥3 → CRITICAL，≥1.5 → WARNING。"""
    loader = SMDLoader(".")
    # 训练基线 dim 0 中位数 = 10
    train = np.full((10, SMD_NUM_DIMENSIONS), 10.0)
    labels = np.zeros(10, dtype=np.int8)
    labels[3:5] = 1

    # CRITICAL：窗口 dim 0 = 30 → dev = 3.0
    test_crit = np.full((10, SMD_NUM_DIMENSIONS), 10.0)
    test_crit[3:5, 0] = 30.0
    entity_crit = _make_entity(train, test_crit, labels, interp=[[0]], loader=loader)
    events_crit = loader.extract_fault_events(entity_crit)
    assert len(events_crit) == 1
    assert events_crit[0].severity == Severity.CRITICAL

    # WARNING：窗口 dim 0 = 15 → dev = 1.5
    test_warn = np.full((10, SMD_NUM_DIMENSIONS), 10.0)
    test_warn[3:5, 0] = 15.0
    entity_warn = _make_entity(train, test_warn, labels, interp=[[0]], loader=loader)
    events_warn = loader.extract_fault_events(entity_warn)
    assert len(events_warn) == 1
    assert events_warn[0].severity == Severity.WARNING


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
