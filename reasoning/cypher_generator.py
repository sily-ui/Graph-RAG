"""Cypher 查询生成器 —— 根据 StructuredQuery 生成 Neo4j Cypher。

本模块是推理控制器的查询生成组件，针对 6 种查询类型生成对应 Cypher。

关键设计（基于 graph_schema/edges.py 的 Graphiti 存储约束）：
1. 所有关系类型在 Neo4j 中统一为 RELATES_TO，必须用 r.name 过滤
   MATCH (s)-[r:RELATES_TO]->(c) WHERE r.name='CAUSED_BY'
   而非 MATCH (s)-[:CAUSED_BY]->(c)

2. 时态过滤用边的 valid_at / invalid_at：
   WHERE r.valid_at <= $window_end
     AND (r.invalid_at IS NULL OR r.invalid_at >= $window_start)

3. 节点层级用 label 过滤：
   WHERE ('Symptom' IN labels(s))
   Graphiti 的节点可能有多个 label，用 IN labels() 检查

4. 自定义属性在 attributes dict 中：
   WHERE s.attributes.cause_type = 'noisy_neighbor'
   或用 apoc.cypher.runFirstColumn 动态访问

5. 多跳路径用变长路径：
   MATCH path = (s)-[:RELATES_TO*2..4]->(t)
   然后在 WHERE 中过滤 r.name 与节点 label
"""
from __future__ import annotations

from typing import Any

from reasoning.query_types import QueryIntent, QueryType, StructuredQuery, TimeWindow


# ============================================================
#  Cypher 模板常量
# ============================================================

# 因果链边名称（用于多跳路径过滤）
CAUSAL_EDGE_NAMES = ["CAUSED_BY", "TRIGGERED_BY", "PROPAGATED_TO"]
SOLUTION_EDGE_NAMES = ["RESOLVED_BY", "MITIGATED_BY", "PREVENTED_BY"]
ALL_EDGE_NAMES = CAUSAL_EDGE_NAMES + SOLUTION_EDGE_NAMES + ["HAS_SYMPTOM"]


# ============================================================
#  参数转义
# ============================================================

def _escape_string(s: str) -> str:
    """转义 Cypher 字符串参数（防注入）。"""
    if s is None:
        return "''"
    return "'" + s.replace("\\", "\\\\").replace("'", "\\'") + "'"


def _format_datetime(dt: Any) -> str:
    """格式化 datetime 为 Cypher datetime() 字面量。"""
    if dt is None:
        return "null"
    if hasattr(dt, "isoformat"):
        return f"datetime('{dt.isoformat()}')"
    return f"datetime('{dt}')"


def _format_list(items: list[str]) -> str:
    """格式化字符串列表为 Cypher list 字面量。"""
    if not items:
        return "[]"
    return "[" + ", ".join(_escape_string(i) for i in items) + "]"


# ============================================================
#  WHERE 子句构建
# ============================================================

def _build_entity_filter(
    intent: QueryIntent,
    node_var: str = "n",
) -> str:
    """构建目标实体过滤 WHERE 子句。"""
    conditions: list[str] = []
    if intent.target_entity and intent.target_entity_type:
        # 根据 entity_type 构建过滤
        if intent.target_entity_type.lower() in ("vm", "component"):
            conditions.append(f"{node_var}.name = {_escape_string(intent.target_entity)}")
        elif intent.target_entity_type.lower() == "cluster":
            conditions.append(f"{node_var}.attributes.cluster_id = {_escape_string(intent.target_entity)}")

    if intent.severity_filter:
        conditions.append(f"{node_var}.attributes.severity = {_escape_string(intent.severity_filter)}")

    return " AND ".join(conditions) if conditions else ""


def _build_temporal_filter(
    time_window: TimeWindow | None,
    edge_var: str = "r",
) -> str:
    """构建时态过滤 WHERE 子句。"""
    if time_window is None:
        return ""
    conditions = [
        f"{edge_var}.valid_at <= {_format_datetime(time_window.end)}",
        f"({edge_var}.invalid_at IS NULL OR {edge_var}.invalid_at >= {_format_datetime(time_window.start)})",
    ]
    return " AND ".join(conditions)


# ============================================================
#  Cypher 生成器
# ============================================================

class CypherGenerator:
    """Cypher 查询生成器 —— 根据 StructuredQuery 生成 Cypher。

    使用示例：
        gen = CypherGenerator()
        cypher, params = gen.generate(structured_query)
        # 用 neo4j driver 执行
        with driver.session() as session:
            result = session.run(cypher, **params)

    生成策略：
    - 每种 QueryType 对应一个 _generate_xxx 方法
    - 返回 (cypher_str, params_dict) 二元组
    - 参数化查询避免注入，复杂字面量用 $param 占位
    """

    def generate(self, query: StructuredQuery) -> tuple[str, dict[str, Any]]:
        """根据结构化查询生成 Cypher。

        Returns
        -------
        tuple[str, dict]
            (cypher_str, params_dict)
        """
        intent = query.intent
        qt = intent.query_type

        if qt == QueryType.SINGLE_ENTITY:
            return self._generate_single_entity(intent)
        elif qt == QueryType.CAUSAL_CHAIN:
            return self._generate_causal_chain(intent)
        elif qt == QueryType.TIME_RANGE:
            return self._generate_time_range(intent)
        elif qt == QueryType.MULTI_HOP_PATH:
            return self._generate_multi_hop_path(intent)
        elif qt == QueryType.SOLUTION_LOOKUP:
            return self._generate_solution_lookup(intent)
        elif qt == QueryType.COMPARISON:
            return self._generate_comparison(intent)
        else:
            raise ValueError(f"未知查询类型: {qt}")

    # ------------------------------------------------------------
    #  1. 单实体查询：查某个 VM/组件的故障
    # ------------------------------------------------------------

    def _generate_single_entity(self, intent: QueryIntent) -> tuple[str, dict[str, Any]]:
        """生成单实体查询 Cypher。

        查询模式：
            MATCH (c:Component)-[r1:RELATES_TO]->(s:Symptom)
            WHERE c.name = $vm_id AND r1.name = 'HAS_SYMPTOM'
            RETURN c, r1, s
        """
        if not intent.target_entity:
            return "", {}

        params: dict[str, Any] = {
            "vm_id": intent.target_entity,
            "limit": intent.limit,
        }

        where_clauses = [
            "c.name = $vm_id",
            "r1.name = 'HAS_SYMPTOM'",
        ]
        if intent.severity_filter:
            where_clauses.append("s.attributes.severity = $severity")
            params["severity"] = intent.severity_filter

        cypher = f"""
MATCH (c:Component)-[r1:RELATES_TO]->(s)
WHERE {' AND '.join(where_clauses)}
  AND 'Symptom' IN labels(s)
RETURN c, r1, s
ORDER BY r1.valid_at DESC
LIMIT $limit
""".strip()
        return cypher, params

    # ------------------------------------------------------------
    #  2. 因果链查询：某症状的根因
    # ------------------------------------------------------------

    def _generate_causal_chain(self, intent: QueryIntent) -> tuple[str, dict[str, Any]]:
        """生成因果链查询 Cypher。

        查询模式（1-2 跳）：
            MATCH (s:Symptom)-[r1:RELATES_TO]->(c:Cause)
            WHERE r1.name IN ['CAUSED_BY','TRIGGERED_BY']
              AND s.name CONTAINS $keyword
            OPTIONAL MATCH (c)-[r2:RELATES_TO]->(sol:Solution)
            WHERE r2.name IN ['RESOLVED_BY','MITIGATED_BY']
            RETURN s, r1, c, r2, sol
        """
        params: dict[str, Any] = {
            "limit": intent.limit,
        }

        where_clauses = [
            "r1.name IN ['CAUSED_BY', 'TRIGGERED_BY']",
            "'Cause' IN labels(c)",
        ]

        # 症状关键词过滤
        if intent.symptom_keywords:
            params["symptom_kw"] = intent.symptom_keywords[0]
            where_clauses.append(
                f"(s.name CONTAINS $symptom_kw OR s.summary CONTAINS $symptom_kw)"
            )

        # 目标实体过滤
        if intent.target_entity:
            params["entity_name"] = intent.target_entity
            where_clauses.append("s.name = $entity_name")

        # 根因关键词过滤
        if intent.cause_keywords:
            params["cause_kw"] = intent.cause_keywords[0]
            where_clauses.append(
                f"(c.name CONTAINS $cause_kw OR c.attributes.cause_type = $cause_kw)"
            )

        # 时态过滤
        if intent.time_window:
            params["window_start"] = intent.time_window.start
            params["window_end"] = intent.time_window.end
            where_clauses.append("r1.valid_at <= $window_end")
            where_clauses.append("(r1.invalid_at IS NULL OR r1.invalid_at >= $window_start)")

        cypher = f"""
MATCH (s)-[r1:RELATES_TO]->(c)
WHERE 'Symptom' IN labels(s)
  AND {' AND '.join(where_clauses)}
OPTIONAL MATCH (c)-[r2:RELATES_TO]->(sol)
WHERE r2.name IN ['RESOLVED_BY', 'MITIGATED_BY']
  AND 'Solution' IN labels(sol)
RETURN s, r1, c, r2, sol
ORDER BY r1.attributes.confidence DESC
LIMIT $limit
""".strip()
        return cypher, params

    # ------------------------------------------------------------
    #  3. 时态范围查询
    # ------------------------------------------------------------

    def _generate_time_range(self, intent: QueryIntent) -> tuple[str, dict[str, Any]]:
        """生成时态范围查询 Cypher。

        查询模式：
            MATCH (n)-[r:RELATES_TO]->(m)
            WHERE r.valid_at >= $start AND r.valid_at <= $end
            RETURN n, r, m
        """
        if not intent.time_window:
            return "", {}

        params: dict[str, Any] = {
            "start": intent.time_window.start,
            "end": intent.time_window.end,
            "limit": intent.limit,
        }

        where_clauses = [
            "r.valid_at >= $start",
            "r.valid_at <= $end",
        ]

        if intent.target_entity:
            params["entity_name"] = intent.target_entity
            where_clauses.append("(n.name = $entity_name OR m.name = $entity_name)")

        if intent.severity_filter:
            params["severity"] = intent.severity_filter
            where_clauses.append("(n.attributes.severity = $severity OR m.attributes.severity = $severity)")

        cypher = f"""
MATCH (n)-[r:RELATES_TO]->(m)
WHERE {' AND '.join(where_clauses)}
RETURN n, r, m
ORDER BY r.valid_at ASC
LIMIT $limit
""".strip()
        return cypher, params

    # ------------------------------------------------------------
    #  4. 多跳路径查询（核心：2/3/4 跳）
    # ------------------------------------------------------------

    def _generate_multi_hop_path(self, intent: QueryIntent) -> tuple[str, dict[str, Any]]:
        """生成多跳路径查询 Cypher。

        基于 PATH_TEMPLATES 的 2/3/4 跳路径模板。
        用变长路径 + 节点 label 过滤。

        2 跳：Symptom → Cause → Solution
        3 跳：Symptom → Cause → Cause → Solution
        4 跳：Component → Symptom → Cause → Cause

        Cypher 模式（以 3 跳为例）：
            MATCH path = (s)-[r1:RELATES_TO]->(c1)-[r2:RELATES_TO]->(c2)-[r3:RELATES_TO]->(sol)
            WHERE 'Symptom' IN labels(s) AND 'Cause' IN labels(c1)
              AND 'Cause' IN labels(c2) AND 'Solution' IN labels(sol)
              AND r1.name IN ['CAUSED_BY','TRIGGERED_BY']
              AND r2.name IN ['CAUSED_BY','TRIGGERED_BY','PROPAGATED_TO']
              AND r3.name IN ['RESOLVED_BY','MITIGATED_BY']
            RETURN path
        """
        hop_count = intent.hop_count or 2
        if hop_count not in (2, 3, 4):
            hop_count = 2

        params: dict[str, Any] = {
            "limit": intent.limit,
        }

        # 根据 hop_count 构建 path 模式
        # 2 跳: (s)-[r1]->(c)-[r2]->(sol)
        # 3 跳: (s)-[r1]->(c1)-[r2]->(c2)-[r3]->(sol)
        # 4 跳: (comp)-[r1]->(s)-[r2]->(c1)-[r3]->(c2)
        path_pattern, label_filter, edge_filter = self._build_path_pattern(hop_count)

        where_clauses = [label_filter, edge_filter]

        # 目标实体过滤
        if intent.target_entity:
            params["entity_name"] = intent.target_entity
            if hop_count == 4:
                # 4 跳起点是 Component
                where_clauses.append("comp.name = $entity_name")
            else:
                where_clauses.append("s.name = $entity_name")

        # 症状关键词过滤
        if intent.symptom_keywords:
            params["symptom_kw"] = intent.symptom_keywords[0]
            where_clauses.append("(s.name CONTAINS $symptom_kw OR s.summary CONTAINS $symptom_kw)")

        # 时态过滤（多跳路径用首跳的 valid_at）
        if intent.time_window:
            params["window_start"] = intent.time_window.start
            params["window_end"] = intent.time_window.end
            where_clauses.append("r1.valid_at <= $window_end")
            where_clauses.append("(r1.invalid_at IS NULL OR r1.invalid_at >= $window_start)")

        cypher = f"""
MATCH {path_pattern}
WHERE {' AND '.join(where_clauses)}
RETURN path
ORDER BY length(path), r1.attributes.confidence DESC
LIMIT $limit
""".strip()
        return cypher, params

    def _build_path_pattern(self, hop_count: int) -> tuple[str, str, str]:
        """构建多跳路径的 MATCH 模式与 WHERE 过滤。

        Returns
        -------
        tuple[str, str, str]
            (path_pattern, label_filter, edge_filter)
        """
        if hop_count == 2:
            # Symptom → Cause → Solution
            path_pattern = (
                "path = (s)-[r1:RELATES_TO]->(c)-[r2:RELATES_TO]->(sol)"
            )
            label_filter = (
                "'Symptom' IN labels(s) AND 'Cause' IN labels(c) "
                "AND 'Solution' IN labels(sol)"
            )
            edge_filter = (
                "r1.name IN ['CAUSED_BY', 'TRIGGERED_BY'] "
                "AND r2.name IN ['RESOLVED_BY', 'MITIGATED_BY']"
            )
        elif hop_count == 3:
            # Symptom → Cause → Cause → Solution
            path_pattern = (
                "path = (s)-[r1:RELATES_TO]->(c1)-[r2:RELATES_TO]->(c2)-[r3:RELATES_TO]->(sol)"
            )
            label_filter = (
                "'Symptom' IN labels(s) AND 'Cause' IN labels(c1) "
                "AND 'Cause' IN labels(c2) AND 'Solution' IN labels(sol)"
            )
            edge_filter = (
                "r1.name IN ['CAUSED_BY', 'TRIGGERED_BY'] "
                "AND r2.name IN ['CAUSED_BY', 'TRIGGERED_BY', 'PROPAGATED_TO'] "
                "AND r3.name IN ['RESOLVED_BY', 'MITIGATED_BY']"
            )
        elif hop_count == 4:
            # Component → Symptom → Cause → Cause
            path_pattern = (
                "path = (comp)-[r1:RELATES_TO]->(s)-[r2:RELATES_TO]->(c1)-[r3:RELATES_TO]->(c2)"
            )
            label_filter = (
                "'Component' IN labels(comp) AND 'Symptom' IN labels(s) "
                "AND 'Cause' IN labels(c1) AND 'Cause' IN labels(c2)"
            )
            edge_filter = (
                "r1.name = 'HAS_SYMPTOM' "
                "AND r2.name IN ['CAUSED_BY', 'TRIGGERED_BY'] "
                "AND r3.name IN ['CAUSED_BY', 'TRIGGERED_BY', 'PROPAGATED_TO']"
            )
        else:
            raise ValueError(f"不支持的跳数: {hop_count}")

        return path_pattern, label_filter, edge_filter

    # ------------------------------------------------------------
    #  5. 解法查询
    # ------------------------------------------------------------

    def _generate_solution_lookup(self, intent: QueryIntent) -> tuple[str, dict[str, Any]]:
        """生成解法查询 Cypher。

        查询模式：
            MATCH (c:Cause)-[r:RELATES_TO]->(sol:Solution)
            WHERE r.name IN ['RESOLVED_BY','MITIGATED_BY']
              AND c.attributes.cause_type = $cause_type
            RETURN c, r, sol
        """
        params: dict[str, Any] = {
            "limit": intent.limit,
        }

        where_clauses = [
            "r.name IN ['RESOLVED_BY', 'MITIGATED_BY', 'PREVENTED_BY']",
            "'Solution' IN labels(sol)",
        ]

        if intent.cause_keywords:
            params["cause_type"] = intent.cause_keywords[0]
            where_clauses.append("c.attributes.cause_type = $cause_type")

        if intent.target_entity:
            params["entity_name"] = intent.target_entity
            where_clauses.append("c.name = $entity_name")

        cypher = f"""
MATCH (c)-[r:RELATES_TO]->(sol)
WHERE 'Cause' IN labels(c)
  AND {' AND '.join(where_clauses)}
RETURN c, r, sol
ORDER BY r.attributes.effectiveness DESC
LIMIT $limit
""".strip()
        return cypher, params

    # ------------------------------------------------------------
    #  6. 对比查询
    # ------------------------------------------------------------

    def _generate_comparison(self, intent: QueryIntent) -> tuple[str, dict[str, Any]]:
        """生成对比查询 Cypher（UNION 多个单实体查询）。

        查询模式：
            MATCH (c:Component)-[r1:RELATES_TO]->(s:Symptom)
            WHERE c.name = $vm1 AND r1.name = 'HAS_SYMPTOM'
            RETURN c.name AS vm, s, r1.valid_at AS ts
            UNION
            MATCH (c:Component)-[r1:RELATES_TO]->(s:Symptom)
            WHERE c.name = $vm2 AND r1.name = 'HAS_SYMPTOM'
            RETURN c.name AS vm, s, r1.valid_at AS ts
        """
        if not intent.target_entity:
            return "", {}

        # 假设 target_entity 是逗号分隔的多个 VM ID
        vm_ids = [v.strip() for v in intent.target_entity.split(",") if v.strip()]
        if not vm_ids:
            return "", {}

        params: dict[str, Any] = {"limit": intent.limit}
        # 动态参数名 vm1, vm2, ...
        for i, vm_id in enumerate(vm_ids, 1):
            params[f"vm{i}"] = vm_id

        # 构建 UNION 查询
        parts = []
        for i in range(1, len(vm_ids) + 1):
            parts.append(f"""
MATCH (c:Component)-[r1:RELATES_TO]->(s)
WHERE c.name = $vm{i} AND r1.name = 'HAS_SYMPTOM'
  AND 'Symptom' IN labels(s)
RETURN c.name AS vm, s.name AS symptom, s.attributes.severity AS severity,
       r1.valid_at AS valid_at, r1.invalid_at AS invalid_at
""".strip())

        cypher = "\nUNION\n".join(parts) + f"\nLIMIT $limit"
        return cypher, params


# ============================================================
#  便捷函数
# ============================================================

def generate_cypher(query: StructuredQuery) -> tuple[str, dict[str, Any]]:
    """便捷函数：生成 Cypher。"""
    gen = CypherGenerator()
    return gen.generate(query)
