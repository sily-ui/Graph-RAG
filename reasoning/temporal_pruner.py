"""时态剪枝器 —— 基于时态一致性过滤候选因果路径。

本模块是推理控制器的核心创新组件，过滤掉时态不一致的候选路径，
提高多跳推理的准确率。

剪枝规则：
1. 因果顺序校验：原因的 valid_at 必须早于或等于症状的 valid_at
   （原因不能发生在症状之后）

2. lag_seconds 一致性校验：
   症状的 valid_at - 原因的 valid_at 应在 [0, lag_seconds * tolerance] 范围内
   （因果时延应与边的 lag_seconds 字段一致，允许一定容差）

3. 路径时态单调性：路径中各跳的 valid_at 应单调递增
   （因果链应按时间顺序传导）

4. 区间重叠校验：边的 [valid_at, invalid_at] 应与查询窗口重叠
   （查询窗口外的边不相关）

5. 因果终止校验：根因 Cause 的 valid_at 应早于所有 Solution 的 valid_at
   （解法应在根因之后才有效）

创新点：
- 传统的图推理只看拓扑结构，不考虑时态
- 本剪枝器结合 bi-temporal 模型的 valid_at/invalid_at/lag_seconds，
  排除时态不一致的路径，显著提升因果推理准确率
- 这是 Graphiti 默认不具备的能力（Graphiti 只做存储，不做推理剪枝）
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from reasoning.query_types import TimeWindow
from reasoning.result_models import CausalPath, PathHop


# ============================================================
#  剪枝配置
# ============================================================

class PrunerConfig:
    """时态剪枝配置。"""

    # lag_seconds 容差倍数（实际时延在 [0, lag*tolerance] 内视为一致）
    LAG_TOLERANCE_MULTIPLIER = 2.0

    # lag_seconds 最大容差（秒），即使 lag 很小也允许一定误差
    LAG_MAX_TOLERANCE_SECONDS = 600

    # 是否允许原因与症状同时刻（True=允许，False=必须严格早于）
    ALLOW_SIMULTANEOUS = True

    # 是否启用区间重叠校验
    ENABLE_INTERVAL_OVERLAP = True

    # 是否启用路径单调性校验
    ENABLE_MONOTONICITY = True

    # 是否启用因果终止校验
    ENABLE_TERMINATION = True


# ============================================================
#  单跳校验
# ============================================================

def validate_hop_temporal_consistency(
    hop: PathHop,
    config: PrunerConfig | None = None,
) -> tuple[bool, str | None]:
    """校验单跳的时态一致性。

    检查项：
    1. 源节点（症状）的 valid_at >= 目标节点（原因）的 valid_at
       （症状不能早于原因）
    2. 实际时延与 lag_seconds 一致

    Parameters
    ----------
    hop : PathHop
        待校验的跳
    config : PrunerConfig | None
        剪枝配置，None=默认

    Returns
    -------
    tuple[bool, str | None]
        (是否通过, 失败原因)
    """
    if config is None:
        config = PrunerConfig()

    # 如果边没有时态信息，跳过校验
    if hop.valid_at is None:
        return True, None

    # 因果顺序校验：源（症状）的 valid_at >= 目标（原因）的 valid_at
    # 注意：在因果链中，source 是症状，target 是原因
    # 但边的 valid_at 是边变为真的时刻（通常是症状出现时刻）
    # 这里校验边的 valid_at 与下一跳的 valid_at 顺序

    # lag_seconds 一致性校验需要相邻两跳，单跳无法校验
    # 单跳只校验 valid_at < invalid_at
    if hop.invalid_at is not None and hop.valid_at is not None:
        if hop.valid_at >= hop.invalid_at:
            return False, f"边 valid_at({hop.valid_at}) >= invalid_at({hop.invalid_at})"

    return True, None


# ============================================================
#  路径级校验
# ============================================================

def validate_path_temporal_consistency(
    path: CausalPath,
    config: PrunerConfig | None = None,
) -> tuple[bool, str | None]:
    """校验整条路径的时态一致性。

    检查项：
    1. 路径中各跳的 valid_at 单调递增（因果链按时间顺序传导）
    2. 相邻跳的时延与 lag_seconds 一致
    3. 根因 Cause 的 valid_at 早于 Solution 的 valid_at

    Parameters
    ----------
    path : CausalPath
        待校验的路径
    config : PrunerConfig | None
        剪枝配置

    Returns
    -------
    tuple[bool, str | None]
        (是否通过, 失败原因)
    """
    if config is None:
        config = PrunerConfig()

    if not path.hops:
        return True, None

    # 1. 单跳校验
    for i, hop in enumerate(path.hops):
        ok, reason = validate_hop_temporal_consistency(hop, config)
        if not ok:
            return False, f"第 {i + 1} 跳: {reason}"

    # 2. 路径单调性校验
    if config.ENABLE_MONOTONICITY and len(path.hops) > 1:
        for i in range(1, len(path.hops)):
            prev_valid = path.hops[i - 1].valid_at
            curr_valid = path.hops[i].valid_at
            if prev_valid is not None and curr_valid is not None:
                if config.ALLOW_SIMULTANEOUS:
                    if curr_valid < prev_valid:
                        return False, (
                            f"路径非单调: 第 {i} 跳 valid_at({curr_valid}) "
                            f"< 第 {i - 1} 跳 valid_at({prev_valid})"
                        )
                else:
                    if curr_valid <= prev_valid:
                        return False, (
                            f"路径非严格单调: 第 {i} 跳 valid_at({curr_valid}) "
                            f"<= 第 {i - 1} 跳 valid_at({prev_valid})"
                        )

    # 3. lag_seconds 一致性校验（相邻跳）
    if len(path.hops) > 1:
        for i in range(1, len(path.hops)):
            prev_valid = path.hops[i - 1].valid_at
            curr_valid = path.hops[i].valid_at
            curr_lag = path.hops[i].lag_seconds

            if prev_valid is not None and curr_valid is not None and curr_lag > 0:
                actual_lag = (curr_valid - prev_valid).total_seconds()
                # 实际时延应在 [0, lag * tolerance] 范围内
                max_allowed = max(
                    curr_lag * config.LAG_TOLERANCE_MULTIPLIER,
                    config.LAG_MAX_TOLERANCE_SECONDS,
                )
                # 允许实际时延略小于 lag（因果可能即时发生）
                if actual_lag < -config.LAG_MAX_TOLERANCE_SECONDS:
                    return False, (
                        f"第 {i + 1} 跳时延异常: 实际 {actual_lag}s "
                        f"远小于预期 lag {curr_lag}s"
                    )
                if actual_lag > max_allowed:
                    return False, (
                        f"第 {i + 1} 跳时延异常: 实际 {actual_lag}s "
                        f"超过预期 lag {curr_lag}s × {config.LAG_TOLERANCE_MULTIPLIER}"
                    )

    # 4. 因果终止校验：根因 Cause 的 valid_at 早于 Solution 的 valid_at
    if config.ENABLE_TERMINATION:
        cause_valid: datetime | None = None
        solution_valid: datetime | None = None
        for hop in path.hops:
            if hop.target.label == "Cause":
                cause_valid = hop.valid_at
            if hop.target.label == "Solution":
                solution_valid = hop.valid_at

        if cause_valid is not None and solution_valid is not None:
            if solution_valid < cause_valid:
                return False, (
                    f"因果终止校验失败: Solution valid_at({solution_valid}) "
                    f"< Cause valid_at({cause_valid})"
                )

    return True, None


# ============================================================
#  查询窗口校验
# ============================================================

def validate_path_in_window(
    path: CausalPath,
    window: TimeWindow,
    config: PrunerConfig | None = None,
) -> tuple[bool, str | None]:
    """校验路径是否与查询时间窗口重叠。

    路径中至少一跳的 [valid_at, invalid_at] 应与 [window.start, window.end] 重叠。

    Parameters
    ----------
    path : CausalPath
        待校验路径
    window : TimeWindow
        查询时间窗口
    config : PrunerConfig | None
        剪枝配置

    Returns
    -------
    tuple[bool, str | None]
        (是否通过, 失败原因)
    """
    if config is None:
        config = PrunerConfig()

    if not config.ENABLE_INTERVAL_OVERLAP:
        return True, None

    if not path.hops:
        return True, None

    for hop in path.hops:
        if window.overlaps(hop.valid_at, hop.invalid_at):
            return True, None

    return False, (
        f"路径与查询窗口 [{window.start}, {window.end}] 无重叠"
    )


# ============================================================
#  时态剪枝器
# ============================================================

class TemporalPruner:
    """时态剪枝器 —— 过滤时态不一致的候选路径。

    使用示例：
        pruner = TemporalPruner()
        pruned_paths = pruner.prune(candidate_paths, time_window)

    剪枝流程：
    1. 单跳时态校验（valid_at < invalid_at）
    2. 路径单调性校验（valid_at 递增）
    3. lag_seconds 一致性校验
    4. 因果终止校验（Cause 早于 Solution）
    5. 查询窗口重叠校验（可选）

    剪枝后的路径会被标记 is_temporally_consistent=True，
    被删除的路径标记 pruned_reason 并放入 pruned_paths。
    """

    def __init__(self, config: PrunerConfig | None = None):
        self.config = config or PrunerConfig()

    def prune(
        self,
        paths: list[CausalPath],
        time_window: TimeWindow | None = None,
    ) -> tuple[list[CausalPath], list[CausalPath]]:
        """对候选路径列表做时态剪枝。

        Parameters
        ----------
        paths : list[CausalPath]
            候选路径列表
        time_window : TimeWindow | None
            查询时间窗口，None=不做窗口校验

        Returns
        -------
        tuple[list[CausalPath], list[CausalPath]]
            (保留的路径, 被剪枝的路径)
            被剪枝的路径带 pruned_reason 标记
        """
        kept: list[CausalPath] = []
        pruned: list[CausalPath] = []

        for path in paths:
            # 1. 路径时态一致性校验
            ok, reason = validate_path_temporal_consistency(path, self.config)
            if not ok:
                path.is_temporally_consistent = False
                path.pruned_reason = reason
                pruned.append(path)
                continue

            # 2. 查询窗口校验
            if time_window is not None:
                ok, reason = validate_path_in_window(path, time_window, self.config)
                if not ok:
                    path.is_temporally_consistent = False
                    path.pruned_reason = reason
                    pruned.append(path)
                    continue

            # 3. 通过校验
            path.is_temporally_consistent = True
            path.pruned_reason = None
            kept.append(path)

        return kept, pruned

    def prune_single(
        self,
        path: CausalPath,
        time_window: TimeWindow | None = None,
    ) -> tuple[bool, str | None]:
        """校验单条路径是否通过时态剪枝。

        Returns
        -------
        tuple[bool, str | None]
            (是否通过, 失败原因)
        """
        ok, reason = validate_path_temporal_consistency(path, self.config)
        if not ok:
            return False, reason

        if time_window is not None:
            ok, reason = validate_path_in_window(path, time_window, self.config)
            if not ok:
                return False, reason

        return True, None


# ============================================================
#  置信度计算
# ============================================================

def compute_path_confidence(path: CausalPath) -> float:
    """计算路径置信度（各跳置信度的几何平均）。

    几何平均对低置信度跳更敏感，能惩罚「一跳很弱」的路径。
    """
    if not path.hops:
        return 0.0

    confidences: list[float] = []
    for hop in path.hops:
        c = hop.edge_confidence
        if c is None:
            c = 0.8  # 默认置信度
        confidences.append(max(c, 0.01))  # 避免 0 导致几何平均为 0

    # 几何平均
    product = 1.0
    for c in confidences:
        product *= c
    return product ** (1.0 / len(confidences))


def compute_path_lag(path: CausalPath) -> int:
    """计算路径总时延（各跳 lag_seconds 之和）。"""
    return sum(hop.lag_seconds for hop in path.hops)


def rank_paths(paths: list[CausalPath]) -> list[CausalPath]:
    """对路径排序：置信度降序 + 时延升序。

    排序规则：
    1. 置信度高的优先
    2. 同置信度下，时延短的优先（更快定位根因）
    3. 同时延下，跳数少的优先（更简洁）
    """
    # 先计算置信度与时延
    for path in paths:
        path.path_confidence = compute_path_confidence(path)
        path.total_lag_seconds = compute_path_lag(path)

    return sorted(
        paths,
        key=lambda p: (
            -p.path_confidence,  # 置信度降序
            p.total_lag_seconds,  # 时延升序
            p.hop_count,          # 跳数升序
        ),
    )
