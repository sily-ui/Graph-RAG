"""知识图谱边合法性约束 —— 保证图谱结构的语义正确性。

本模块定义并校验四层概念模型中边的合法性规则：

    ┌──────────┐  HAS_SYMPTOM   ┌──────────┐
    │Component │───────────────→│ Symptom  │
    └──────────┘                └────┬─────┘
                                     │ CAUSED_BY / TRIGGERED_BY
                                     ▼
                                ┌──────────┐
                        ┌──────→│  Cause   │←──────┐
                        │PROPAGATED_TO     │       │
                        │       └────┬─────┘       │
                        │            │ CAUSED_BY    │ PROPAGATED_TO
                        │            ▼              │
                        │       ┌──────────┐       │
                        │       │  Cause   │───────┘
                        │       │ (root)   │
                        │       └────┬─────┘
                        │            │
                                     ▼
                                ┌──────────┐
                                │ Solution │
                                └──────────┘
                          RESOLVED_BY / MITIGATED_BY / PREVENTED_BY

合法性规则：
1. 边的 source→target 层级组合必须在 EDGE_TYPE_MAP 中定义
2. 因果链必须终止于 is_root=true 的 Cause 节点
3. 时态一致性：valid_at 必须早于 invalid_at（若两者均存在）
4. 因果边的 lag_seconds 必须为非负整数

这些约束在两种场景下生效：
- 数据摄入时（add_episode 前）：拦截非法边，防止脏数据入库
- 查询评估时：校验预测路径的合法性
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from graph_schema.edges import (
    CAUSAL_EDGE_NAMES,
    SOLUTION_EDGE_NAMES,
    EdgeName,
    EDGE_TYPE_MAP,
)
from graph_schema.nodes import GraphLayer


# ============================================================
#  边合法性校验
# ============================================================

class ConstraintViolation(Exception):
    """边合法性校验失败时抛出。"""

    def __init__(self, source_label: str, target_label: str, edge_name: str, reason: str):
        self.source_label = source_label
        self.target_label = target_label
        self.edge_name = edge_name
        self.reason = reason
        super().__init__(
            f"非法边: ({source_label})-[{edge_name}]->({target_label}): {reason}"
        )


def validate_edge_combination(
    source_label: str,
    target_label: str,
    edge_name: str,
) -> bool:
    """校验边的层级组合是否合法。

    检查 (source_label, target_label) → edge_name 是否在 EDGE_TYPE_MAP 中定义。
    这是图谱语义正确性的核心约束——防止出现 Solution→Symptom 这类无意义边。

    Parameters
    ----------
    source_label : str
        源节点的层级 label（如 "Component"、"Symptom"）
    target_label : str
        目标节点的层级 label
    edge_name : str
        边名称（如 "CAUSED_BY"、"RESOLVED_BY"）

    Returns
    -------
    bool
        True = 合法，False = 非法

    Examples
    --------
    >>> validate_edge_combination("Component", "Symptom", "HAS_SYMPTOM")
    True
    >>> validate_edge_combination("Solution", "Symptom", "RESOLVED_BY")
    False  # 解法不能指向症状
    """
    allowed = EDGE_TYPE_MAP.get((source_label, target_label), [])
    return edge_name in allowed


def validate_temporal_consistency(
    valid_at: datetime | None,
    invalid_at: datetime | None,
) -> bool:
    """校验时态一致性：valid_at 必须早于 invalid_at。

    这是时态因果建模的基础约束。若 valid_at=症状开始时刻，
    invalid_at=症状结束时刻，则开始必须早于结束。

    Parameters
    ----------
    valid_at : datetime | None
        事实变为真的时刻（None=不约束）
    invalid_at : datetime | None
        事实失效的时刻（None=仍有效/不约束）

    Returns
    -------
    bool
        True = 一致，False = 不一致
    """
    if valid_at is not None and invalid_at is not None:
        return valid_at < invalid_at
    return True


def validate_causal_path_terminates_at_root(
    path_labels: list[str],
) -> bool:
    """校验因果路径是否终止于根因节点。

    因果链（由 CAUSAL_EDGE_NAMES 组成的路径）的最后一个 Cause 节点
    必须标记为 is_root=true。这是多跳推理的终止条件。

    此函数校验路径的层级结构是否合法（至少包含 Cause 层），
    is_root 标记的校验需要查询具体节点属性。

    Parameters
    ----------
    path_labels : list[str]
        路径中各节点的层级 label 序列，
        如 ["Symptom", "Cause", "Cause"] 表示 3 跳因果链

    Returns
    -------
    bool
        True = 路径包含 Cause 层且最后一个 Cause 之后是 Solution 或路径结束
    """
    if not path_labels:
        return False

    # 因果路径至少要经过一个 Cause 节点
    if "Cause" not in path_labels:
        return False

    # Cause 之后的节点只能是 Cause（传播）或 Solution（解法）
    cause_indices = [i for i, label in enumerate(path_labels) if label == "Cause"]
    if not cause_indices:
        return False

    last_cause_idx = cause_indices[-1]
    # 最后一个 Cause 之后的节点只允许 Solution
    for i in range(last_cause_idx + 1, len(path_labels)):
        if path_labels[i] not in ("Solution",):
            return False

    return True


def validate_entity_for_layer(
    label: str,
    attributes: dict[str, Any],
) -> list[str]:
    """校验节点的 attributes 是否包含其层级所需的必要字段。

    每层节点都有必填的自定义字段（存储在 attributes 中），
    缺少这些字段会导致查询时无法正确过滤或排序。

    Parameters
    ----------
    label : str
        节点层级 label（如 "Component"、"Symptom"）
    attributes : dict[str, Any]
        节点的 attributes dict

    Returns
    -------
    list[str]
        缺失字段列表，空列表表示校验通过

    Examples
    --------
    >>> validate_entity_for_layer("Component", {"component_type": "vm", "cluster_id": "c1"})
    []
    >>> validate_entity_for_layer("Cause", {"cause_type": "misconfiguration"})
    ['confidence']  # Cause 层缺 confidence 字段
    """
    required_fields: dict[str, list[str]] = {
        "Component": ["component_type", "cluster_id"],
        "Symptom": ["symptom_type", "severity"],
        "Cause": ["cause_type", "confidence", "is_root"],
        "Solution": ["solution_type"],
    }

    required = required_fields.get(label, [])
    missing = [f for f in required if f not in attributes]
    return missing


# ============================================================
#  便捷校验函数 —— 一次性校验所有约束
# ============================================================

def validate_edge_full(
    source_label: str,
    target_label: str,
    edge_name: str,
    valid_at: datetime | None = None,
    invalid_at: datetime | None = None,
    source_attributes: dict[str, Any] | None = None,
    target_attributes: dict[str, Any] | None = None,
) -> list[str]:
    """一次性校验边的所有约束，返回所有违规项。

    此函数组合了上述所有校验，便于在数据摄入前统一检查。
    不抛异常，而是返回违规列表，便于批量处理时汇总报错。

    Parameters
    ----------
    source_label, target_label, edge_name : str
        源/目标层级和边名称
    valid_at, invalid_at : datetime | None
        时态字段
    source_attributes, target_attributes : dict | None
        源/目标节点的属性

    Returns
    -------
    list[str]
        违规描述列表，空列表表示全部通过
    """
    violations: list[str] = []

    # 1. 层级组合合法性
    if not validate_edge_combination(source_label, target_label, edge_name):
        violations.append(
            f"非法层级组合: ({source_label})-[{edge_name}]->({target_label})"
        )

    # 2. 时态一致性
    if not validate_temporal_consistency(valid_at, invalid_at):
        violations.append(
            f"时态不一致: valid_at={valid_at} >= invalid_at={invalid_at}"
        )

    # 3. 源节点属性完整性
    if source_attributes is not None:
        missing = validate_entity_for_layer(source_label, source_attributes)
        if missing:
            violations.append(f"源节点缺字段: {missing}")

    # 4. 目标节点属性完整性
    if target_attributes is not None:
        missing = validate_entity_for_layer(target_label, target_attributes)
        if missing:
            violations.append(f"目标节点缺字段: {missing}")

    return violations


# ============================================================
#  多跳路径模板 —— 对照实验的 2/3/4 跳定义
# ============================================================

# 每个模板定义一条合法的因果路径的层级序列和允许的边名称
# 用于测试集构造和路径合法性校验
PATH_TEMPLATES: dict[int, dict] = {
    2: {
        "description": "2 跳: 症状 → 根因 → 解法",
        "labels": ["Symptom", "Cause", "Solution"],
        "edges": [["CAUSED_BY", "TRIGGERED_BY"], ["RESOLVED_BY", "MITIGATED_BY"]],
    },
    3: {
        "description": "3 跳: 症状 → 中间因 → 根因 → 解法",
        "labels": ["Symptom", "Cause", "Cause", "Solution"],
        "edges": [
            ["CAUSED_BY", "TRIGGERED_BY"],
            ["CAUSED_BY", "TRIGGERED_BY", "PROPAGATED_TO"],
            ["RESOLVED_BY", "MITIGATED_BY"],
        ],
    },
    4: {
        "description": "4 跳: 组件 → 症状 → 中间因 → 根因",
        "labels": ["Component", "Symptom", "Cause", "Cause"],
        "edges": [
            ["HAS_SYMPTOM"],
            ["CAUSED_BY", "TRIGGERED_BY"],
            ["CAUSED_BY", "TRIGGERED_BY", "PROPAGATED_TO"],
        ],
    },
}
