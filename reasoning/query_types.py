"""查询类型定义 —— 推理控制器的输入数据模型。

本模块定义查询意图枚举与结构化查询模型，衔接 LLM 解析与 Cypher 生成。

查询意图分类（覆盖服务器故障排查的典型问题）：
1. SINGLE_ENTITY     单实体查询：查某个 VM/组件的故障
2. CAUSAL_CHAIN      因果链查询：某症状的根因是什么
3. TIME_RANGE        时态范围查询：某时间段内的故障
4. MULTI_HOP_PATH    多跳路径查询：完整因果链（2/3/4 跳）
5. SOLUTION_LOOKUP   解法查询：某故障的解法
6. COMPARISON        对比查询：对比多个 VM 的故障模式
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ============================================================
#  查询意图枚举
# ============================================================

class QueryType(str, Enum):
    """查询类型枚举 —— 决定 Cypher 生成器的查询模板。

    每种类型对应一种 Cypher 模板与时态剪枝策略：
    - SINGLE_ENTITY:    MATCH 单节点 + HAS_SYMPTOM
    - CAUSAL_CHAIN:     MATCH Symptom-[CAUSED_BY]->Cause（1-2 跳）
    - TIME_RANGE:       MATCH + WHERE valid_at 过滤
    - MULTI_HOP_PATH:   MATCH 路径模式（2/3/4 跳）
    - SOLUTION_LOOKUP:  MATCH Cause-[RESOLVED_BY]->Solution
    - COMPARISON:       UNION 多个 SINGLE_ENTITY
    """
    SINGLE_ENTITY = "single_entity"
    CAUSAL_CHAIN = "causal_chain"
    TIME_RANGE = "time_range"
    MULTI_HOP_PATH = "multi_hop_path"
    SOLUTION_LOOKUP = "solution_lookup"
    COMPARISON = "comparison"


class QueryIntent(BaseModel):
    """查询意图 —— LLM 从自然语言解析出的结构化意图。

    LLMInterpreter.parse_query() 把自然语言查询转成 QueryIntent，
    CypherGenerator 根据 QueryIntent 生成对应的 Cypher。
    """
    query_type: QueryType
    target_entity: str | None = Field(
        default=None,
        description="目标实体标识，如 vm_id='vm_001' 或 cluster_id='cluster_A'"
    )
    target_entity_type: str | None = Field(
        default=None,
        description="目标实体类型，如 'vm'/'cluster'/'pod'"
    )
    symptom_keywords: list[str] = Field(
        default_factory=list,
        description="症状关键词，如 ['cpu_usage','spike']"
    )
    cause_keywords: list[str] = Field(
        default_factory=list,
        description="根因关键词，如 ['noisy_neighbor','resource_contention']"
    )
    hop_count: int | None = Field(
        default=None,
        ge=2,
        le=4,
        description="多跳路径的跳数（2/3/4），仅 MULTI_HOP_PATH 用"
    )
    time_window: TimeWindow | None = Field(
        default=None,
        description="时态查询窗口，仅 TIME_RANGE 用"
    )
    severity_filter: str | None = Field(
        default=None,
        description="严重程度过滤，如 'critical'/'warning'"
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="返回结果数上限"
    )


class TimeWindow(BaseModel):
    """时态查询窗口 —— 时态范围查询的过滤条件。"""
    start: datetime = Field(description="窗口起始时刻")
    end: datetime = Field(description="窗口结束时刻")

    def contains(self, dt: datetime) -> bool:
        """检查时刻是否在窗口内。"""
        return self.start <= dt <= self.end

    def overlaps(self, other_start: datetime | None, other_end: datetime | None) -> bool:
        """检查 [other_start, other_end] 与本窗口是否重叠。

        用于时态剪枝：边的 valid_at/invalid_at 区间与查询窗口重叠才保留。
        None 表示无限区间。
        """
        if other_start is None and other_end is None:
            return True  # 无限区间与任何窗口重叠
        if other_start is None:
            return other_end >= self.start  # 区间 (-∞, other_end]
        if other_end is None:
            return other_start <= self.end  # 区间 [other_start, +∞)
        return other_start <= self.end and other_end >= self.start


class StructuredQuery(BaseModel):
    """结构化查询 —— 推理控制器的内部表示。

    由 LLMInterpreter 从自然语言生成，传给 CypherGenerator 生成 Cypher。
    保留原始自然语言用于溯源。
    """
    natural_language: str = Field(description="原始自然语言查询")
    intent: QueryIntent
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="元信息，如解析置信度、LLM 模型名等"
    )
