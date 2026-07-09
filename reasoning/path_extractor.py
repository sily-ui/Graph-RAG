"""多跳路径抽取器 —— 执行 Cypher 并把结果转成 CausalPath。

本模块是推理控制器的查询执行组件，把 Neo4j 返回的原始记录
转换成结构化的 CausalPath 列表，供时态剪枝器处理。

数据流：
    CypherGenerator 生成 Cypher
         │
         ▼
    PathExtractor.execute(cypher, params)
         │  ↓ neo4j driver 执行
         ▼
    原始查询结果（list of Record）
         │
         ▼
    _parse_record_to_path(record)
         │  ↓ 节点/边解析
         ▼
    CausalPath 列表
         │
         ▼
    TemporalPruner.prune()（时态剪枝）

关键设计：
1. 节点解析：从 Neo4j Node 提取 uuid/name/labels/attributes
2. 边解析：从 Neo4j Relationship 提取 name/valid_at/invalid_at/attributes
3. 路径构建：从 path 对象或节点/边列表构建 CausalPath
4. 置信度计算：调用 temporal_pruner.compute_path_confidence
"""
from __future__ import annotations

import logging
from typing import Any

from reasoning.result_models import CausalPath, NodeInfo, PathHop
from reasoning.temporal_pruner import compute_path_confidence, compute_path_lag

logger = logging.getLogger(__name__)


# ============================================================
#  Neo4j 节点/边解析
# ============================================================

def parse_neo4j_node(node: Any) -> NodeInfo:
    """从 Neo4j Node 对象解析出 NodeInfo。

    Graphiti 的节点存为 EntityNode，字段映射：
    - node.id / node.element_id → uuid
    - node['name'] → name
    - labels(node) → label（取第一个层级 label）
    - node['summary'] → summary
    - node['group_id'] → group_id
    - 其余自定义字段在 node['attributes'] 或 node._properties 中

    Parameters
    ----------
    node : neo4j.Node | dict
        Neo4j 节点对象或 dict（测试用）
    """
    # 支持 dict（测试用）和 neo4j.Node（生产用）
    if isinstance(node, dict):
        properties = dict(node)
        labels = properties.pop("_labels", [])
        if not labels:
            # 推断 label
            for layer in ["Component", "Symptom", "Cause", "Solution"]:
                if properties.get("label") == layer or layer.lower() in str(properties).lower():
                    labels = [layer]
                    break
    else:
        # neo4j.Node 对象
        properties = dict(node)
        try:
            labels = list(node.labels)
        except AttributeError:
            labels = []

    # 提取字段
    uuid = properties.get("uuid", properties.get("id", ""))
    name = properties.get("name", "")
    summary = properties.get("summary", "")
    group_id = properties.get("group_id", "")

    # label 优先取层级 label
    label = ""
    for layer in ["Component", "Symptom", "Cause", "Solution"]:
        if layer in labels:
            label = layer
            break
    if not label and labels:
        label = labels[0]

    # attributes：Graphiti 把自定义字段放在 attributes dict
    attributes = properties.get("attributes", {})
    if not isinstance(attributes, dict):
        attributes = {}
    # 也把顶层的一些自定义字段合并进 attributes（兼容非 Graphiti 数据）
    for key in ["cause_type", "is_root", "confidence", "symptom_type", "severity",
                "solution_type", "component_type", "cluster_id", "vm_id",
                "mechanism", "effectiveness", "lag_seconds"]:
        if key in properties and key not in attributes:
            attributes[key] = properties[key]

    return NodeInfo(
        uuid=str(uuid),
        name=str(name),
        label=label,
        summary=str(summary),
        group_id=str(group_id),
        attributes=attributes,
    )


def _to_python_datetime(value: Any) -> Any:
    """把 neo4j.time.DateTime / Date / Duration 转成 Python datetime。

    Neo4j Python driver 返回的是 neo4j.time.DateTime，pydantic 校验 datetime 类型不接受。
    """
    if value is None:
        return None
    # neo4j.time.DateTime / Date 有 to_native() / iso_format() 接口
    if hasattr(value, "to_native"):
        try:
            return value.to_native()
        except Exception:
            pass
    # fallback: 用 iso_format 解析
    if hasattr(value, "iso_format"):
        from datetime import datetime
        try:
            return datetime.fromisoformat(value.iso_format())
        except Exception:
            return value
    return value


def parse_neo4j_relationship(rel: Any) -> dict[str, Any]:
    """从 Neo4j Relationship 对象解析出边属性 dict。

    Graphiti 的边存为 RELATES_TO 关系，字段映射：
    - rel.type → 关系类型（统一为 RELATES_TO）
    - rel['name'] → 边名称（如 'CAUSED_BY'）
    - rel['valid_at'] → valid_at
    - rel['invalid_at'] → invalid_at
    - rel['attributes'] → 自定义属性（lag_seconds 等）

    Returns
    -------
    dict[str, Any]
        包含 name/valid_at/invalid_at/attributes 的 dict
    """
    if isinstance(rel, dict):
        properties = dict(rel)
    else:
        properties = dict(rel)

    # 提取时态字段（转 Python datetime 以满足 pydantic）
    valid_at = _to_python_datetime(properties.get("valid_at"))
    invalid_at = _to_python_datetime(properties.get("invalid_at"))

    # attributes
    attributes = properties.get("attributes", {})
    if not isinstance(attributes, dict):
        attributes = {}
    # 合并顶层自定义字段
    for key in ["lag_seconds", "mechanism", "confidence", "effectiveness",
                "is_immediate", "detection_method"]:
        if key in properties and key not in attributes:
            attributes[key] = properties[key]

    return {
        "name": properties.get("name", ""),
        "valid_at": valid_at,
        "invalid_at": invalid_at,
        "attributes": attributes,
    }


# ============================================================
#  路径构建
# ============================================================

def build_path_from_nodes_and_edges(
    nodes: list[Any],
    edges: list[Any],
) -> CausalPath:
    """从节点列表与边列表构建 CausalPath。

    nodes 与 edges 交替：nodes[i] --edges[i]--> nodes[i+1]

    Parameters
    ----------
    nodes : list
        节点列表（neo4j.Node 或 dict）
    edges : list
        边列表（neo4j.Relationship 或 dict）
    """
    if len(nodes) != len(edges) + 1:
        raise ValueError(
            f"节点数({len(nodes)})与边数({len(edges)})不匹配，应为 节点数=边数+1"
        )

    hops: list[PathHop] = []
    for i, edge in enumerate(edges):
        source = parse_neo4j_node(nodes[i])
        target = parse_neo4j_node(nodes[i + 1])
        edge_props = parse_neo4j_relationship(edge)

        hop = PathHop(
            edge_name=edge_props["name"],
            source=source,
            target=target,
            valid_at=edge_props["valid_at"],
            invalid_at=edge_props["invalid_at"],
            attributes=edge_props["attributes"],
        )
        hops.append(hop)

    path = CausalPath(hops=hops)
    path.path_confidence = compute_path_confidence(path)
    path.total_lag_seconds = compute_path_lag(path)
    return path


def build_path_from_neo4j_path(neo4j_path: Any) -> CausalPath:
    """从 Neo4j 的 path 对象构建 CausalPath。

    Neo4j 返回的 path 对象包含 nodes() 与 relationships() 方法。

    Parameters
    ----------
    neo4j_path : neo4j.Path | dict
        Neo4j 路径对象或 dict（测试用，含 nodes/relationships 键）
    """
    if isinstance(neo4j_path, dict):
        nodes = neo4j_path.get("nodes", [])
        relationships = neo4j_path.get("relationships", [])
    else:
        nodes = list(neo4j_path.nodes)
        relationships = list(neo4j_path.relationships)

    return build_path_from_nodes_and_edges(nodes, relationships)


def build_path_from_record(
    record: dict[str, Any] | Any,
) -> CausalPath | None:
    """从单条查询记录构建 CausalPath。

    支持两种返回格式：
    1. 含 'path' 字段：直接用 build_path_from_neo4j_path
    2. 含节点/边字段（如 s, r1, c, r2, sol）：按字段名顺序组装

    Parameters
    ----------
    record : dict | neo4j.Record
        查询记录
    """
    # 转 dict
    if isinstance(record, dict):
        rec_dict = record
    else:
        try:
            rec_dict = dict(record)
        except (TypeError, ValueError):
            return None

    # 情况 1：含 path 字段
    if "path" in rec_dict:
        return build_path_from_neo4j_path(rec_dict["path"])

    # 情况 2：含节点/边字段
    # 常见模式：s, r1, c, r2, sol（2 跳）
    #          s, r1, c1, r2, c2, r3, sol（3 跳）
    #          comp, r1, s, r2, c1, r3, c2（4 跳）
    # 按 key 中的数字顺序组装
    node_keys = sorted(
        [k for k in rec_dict if k.startswith("c") or k in ("s", "sol", "comp")],
        key=lambda x: (0 if x in ("s", "comp") else 1 if x == "c" else
                       2 if x.startswith("c1") else 3 if x.startswith("c2") else 4)
    )
    edge_keys = sorted(
        [k for k in rec_dict if k.startswith("r") and k[1:].isdigit()],
        key=lambda x: int(x[1:])
    )

    if not node_keys or not edge_keys:
        return None

    nodes = [rec_dict[k] for k in node_keys if rec_dict[k] is not None]
    edges = [rec_dict[k] for k in edge_keys if rec_dict[k] is not None]

    if len(nodes) != len(edges) + 1:
        # 可能有些字段是 None（OPTIONAL MATCH），过滤后重建
        return None

    return build_path_from_nodes_and_edges(nodes, edges)


# ============================================================
#  多跳路径抽取器
# ============================================================

class PathExtractor:
    """多跳路径抽取器 —— 执行 Cypher 并把结果转成 CausalPath 列表。

    使用示例：
        extractor = PathExtractor(driver)
        paths = extractor.execute(cypher, params)
        for p in paths:
            print(p.labels, p.path_confidence)

    依赖 neo4j Python driver，但本类的方法也支持测试模式
    （直接传 records 列表，不连真实数据库）。
    """

    def __init__(self, driver: Any | None = None):
        """
        Parameters
        ----------
        driver : neo4j.Driver | None
            Neo4j 驱动，None=测试模式（不连数据库）
        """
        self.driver = driver

    def execute(
        self,
        cypher: str,
        params: dict[str, Any] | None = None,
    ) -> list[CausalPath]:
        """执行 Cypher 查询并返回 CausalPath 列表。

        Parameters
        ----------
        cypher : str
            Cypher 查询语句
        params : dict | None
            查询参数
        """
        if self.driver is None:
            logger.warning("PathExtractor.driver 为 None，返回空结果（测试模式）")
            return []

        params = params or {}
        paths: list[CausalPath] = []

        with self.driver.session() as session:
            result = session.run(cypher, **params)
            for record in result:
                path = build_path_from_record(record)
                if path is not None:
                    paths.append(path)

        logger.info(f"Cypher 执行完成，抽取 {len(paths)} 条路径")
        return paths

    def extract_from_records(
        self,
        records: list[dict[str, Any]],
    ) -> list[CausalPath]:
        """从查询记录列表抽取路径（测试用）。

        不连数据库，直接传入 records 列表。
        """
        paths: list[CausalPath] = []
        for record in records:
            path = build_path_from_record(record)
            if path is not None:
                paths.append(path)
        return paths


# ============================================================
#  便捷函数
# ============================================================

def extract_paths_from_query_result(
    records: list[dict[str, Any]],
) -> list[CausalPath]:
    """便捷函数：从查询结果记录列表抽取路径。"""
    extractor = PathExtractor(driver=None)
    return extractor.extract_from_records(records)
