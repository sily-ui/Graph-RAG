"""推理结果数据模型 —— 推理控制器的输出数据结构。

本模块定义多跳路径、推理结果等数据结构，衔接图谱查询与 LLM 解释。

数据流：
    Neo4j 查询结果（dict）
         │
         ▼
    NodeInfo（节点信息）+ PathHop（边信息）
         │
         ▼
    CausalPath（完整因果路径，含时态信息）
         │
         ▼
    TemporalPruner.prune()（时态剪枝）
         │
         ▼
    ReasoningResult（最终结果：路径 + 自然语言答案 + 置信度）
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ============================================================
#  节点信息
# ============================================================

class NodeInfo(BaseModel):
    """图谱节点信息 —— 从 Neo4j 查询结果提取的节点快照。

    Graphiti 的节点存为 EntityNode，自定义字段在 attributes dict 中。
    本模型把常用字段拍平，便于上层使用。
    """
    uuid: str = Field(description="节点 UUID（Graphiti 自动生成）")
    name: str = Field(description="节点名称（唯一标识）")
    label: str = Field(description="层级 label，如 'Component'/'Symptom'/'Cause'/'Solution'")
    summary: str = Field(default="", description="节点摘要（LLM 生成）")
    group_id: str = Field(default="", description="分组 ID")
    attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="自定义属性，如 cause_type/confidence/is_root/solution_type"
    )

    # 便捷属性访问
    @property
    def cause_type(self) -> str | None:
        return self.attributes.get("cause_type")

    @property
    def is_root(self) -> bool:
        return bool(self.attributes.get("is_root", False))

    @property
    def confidence(self) -> float | None:
        val = self.attributes.get("confidence")
        return float(val) if val is not None else None

    @property
    def symptom_type(self) -> str | None:
        return self.attributes.get("symptom_type")

    @property
    def severity(self) -> str | None:
        return self.attributes.get("severity")

    @property
    def solution_type(self) -> str | None:
        return self.attributes.get("solution_type")

    @property
    def component_type(self) -> str | None:
        return self.attributes.get("component_type")

    @property
    def cluster_id(self) -> str | None:
        return self.attributes.get("cluster_id")


# ============================================================
#  边信息（路径中的一跳）
# ============================================================

class PathHop(BaseModel):
    """路径中的一跳 —— 描述两个节点之间的边。

    一条 CausalPath 由多个 PathHop 组成，每个 PathHop 是路径中的一条边。
    """
    edge_name: str = Field(description="边名称，如 'CAUSED_BY'/'RESOLVED_BY'")
    source: NodeInfo = Field(description="源节点")
    target: NodeInfo = Field(description="目标节点")
    valid_at: datetime | None = Field(
        default=None,
        description="边变为真的时刻（业务时间）"
    )
    invalid_at: datetime | None = Field(
        default=None,
        description="边失效的时刻（None=仍有效）"
    )
    attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="边的自定义属性，如 lag_seconds/mechanism/confidence/effectiveness"
    )

    @property
    def lag_seconds(self) -> int:
        return int(self.attributes.get("lag_seconds", 0))

    @property
    def mechanism(self) -> str | None:
        return self.attributes.get("mechanism")

    @property
    def edge_confidence(self) -> float | None:
        val = self.attributes.get("confidence")
        return float(val) if val is not None else None

    @property
    def effectiveness(self) -> float | None:
        val = self.attributes.get("effectiveness")
        return float(val) if val is not None else None


# ============================================================
#  完整因果路径
# ============================================================

class CausalPath(BaseModel):
    """完整因果路径 —— 多跳推理的核心结果。

    一条 CausalPath 描述从起点到终点的完整因果链：
    - 2 跳：Symptom → Cause → Solution
    - 3 跳：Symptom → Cause → Cause → Solution
    - 4 跳：Component → Symptom → Cause → Cause

    路径置信度 = 各跳置信度的几何平均（惩罚长路径中的低置信度跳）
    路径时延 = 各跳 lag_seconds 之和
    """
    hops: list[PathHop] = Field(description="路径的各跳，按因果顺序排列")
    path_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="路径整体置信度（各跳几何平均）"
    )
    total_lag_seconds: int = Field(
        default=0,
        ge=0,
        description="路径总时延（各跳 lag_seconds 之和）"
    )
    is_temporally_consistent: bool = Field(
        default=True,
        description="时态一致性（valid_at 单调递增）"
    )
    pruned_reason: str | None = Field(
        default=None,
        description="如被时态剪枝删除，记录原因（None=未被剪枝）"
    )

    @property
    def hop_count(self) -> int:
        """路径跳数。"""
        return len(self.hops)

    @property
    def start_node(self) -> NodeInfo | None:
        """路径起点。"""
        return self.hops[0].source if self.hops else None

    @property
    def end_node(self) -> NodeInfo | None:
        """路径终点。"""
        return self.hops[-1].target if self.hops else None

    @property
    def labels(self) -> list[str]:
        """路径的层级 label 序列，如 ['Symptom','Cause','Solution']。"""
        if not self.hops:
            return []
        labels = [self.hops[0].source.label]
        for hop in self.hops:
            labels.append(hop.target.label)
        return labels

    @property
    def root_cause(self) -> NodeInfo | None:
        """路径的根因节点（最后一个 Cause 节点）。"""
        for hop in reversed(self.hops):
            if hop.target.label == "Cause" and hop.target.is_root:
                return hop.target
            if hop.target.label == "Cause":
                return hop.target
        # 没找到 Cause，返回起点
        return self.start_node

    def to_natural_language(self) -> str:
        """把路径转成自然语言描述（供 LLM 解释器参考）。"""
        if not self.hops:
            return "(空路径)"
        parts = [f"{self.hops[0].source.name}({self.hops[0].source.label})"]
        for hop in self.hops:
            parts.append(
                f" --[{hop.edge_name} lag={hop.lag_seconds}s]--> "
                f"{hop.target.name}({hop.target.label})"
            )
        result = "".join(parts)
        result += f"\n  路径置信度: {self.path_confidence:.3f}"
        result += f"\n  总时延: {self.total_lag_seconds}s"
        result += f"\n  时态一致: {'是' if self.is_temporally_consistent else '否'}"
        return result


# ============================================================
#  推理结果
# ============================================================

class ReasoningResult(BaseModel):
    """推理结果 —— 推理控制器的最终输出。

    包含：
    - 自然语言答案（LLM 生成）
    - 候选路径列表（时态剪枝后保留的路径）
    - 推理过程元信息（耗时、Cypher、模型等）
    - 置信度（基于路径置信度与 LLM 置信度综合）
    """
    query: str = Field(description="原始自然语言查询")
    answer: str = Field(description="LLM 生成的自然语言答案")
    paths: list[CausalPath] = Field(
        default_factory=list,
        description="候选因果路径（已时态剪枝）"
    )
    pruned_paths: list[CausalPath] = Field(
        default_factory=list,
        description="被剪枝删除的路径（用于评估）"
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="综合置信度"
    )
    cypher_used: str = Field(default="", description="执行的 Cypher 查询")
    elapsed_seconds: float = Field(default=0.0, description="推理总耗时（秒）")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="元信息，如 LLM 模型、token 数等"
    )

    @property
    def best_path(self) -> CausalPath | None:
        """置信度最高的路径。"""
        if not self.paths:
            return None
        return max(self.paths, key=lambda p: p.path_confidence)

    @property
    def path_count(self) -> int:
        return len(self.paths)

    @property
    def pruned_count(self) -> int:
        return len(self.pruned_paths)

    def summary(self) -> str:
        """结果摘要。"""
        lines = [
            f"查询: {self.query}",
            f"答案: {self.answer}",
            f"候选路径: {self.path_count} 条 (剪枝 {self.pruned_count} 条)",
            f"综合置信度: {self.confidence:.3f}",
            f"耗时: {self.elapsed_seconds:.2f}s",
        ]
        if self.best_path:
            lines.append(f"最佳路径: {self.best_path.labels}")
        return "\n".join(lines)
