"""图构建质量指标 —— GraphRAG-Bench §3.2 借鉴。

References
----------
- arXiv:2506.02404 (GraphRAG-Bench, Xiao et al. 2025) §3.2 "Corpus collection
  and processing" 提出了图构建质量评估的必要性，本模块借鉴定量化为：
  - Entity Recall    = |exp.nodes ∩ pred.nodes| / |exp.nodes|
  - Relation Recall  = |exp.triples ∩ pred.triples| / |exp.triples|
  - Pipeline F1      = 2 * ER * EP / (ER + EP)

三个指标把"图构建 + 检索"从最终答案中隔离出来，回答"是图没建对，
还是图对了但 LLM 没生成对"。

与 [eval/metrics.py](file:///root/Graph-RAG/eval/metrics.py) 中的路径级
Recall/Precision 不同——本模块以**节点 / 三元组集合**为粒度，不考虑顺序。
"""
from __future__ import annotations

import logging

from reasoning.result_models import PathHop

logger = logging.getLogger(__name__)


def _hop_name_eq(a: str, b: str) -> bool:
    """节点名等价：与 metrics.py:_hop_name_eq 同实现。"""
    a, b = a.lower().strip(), b.lower().strip()
    return a == b or a in b or b in a


def _extract_expected_nodes(expected_path) -> set[str]:
    """从 ExpectedHop 序列里取出所有节点 name（小写去重）。"""
    nodes: set[str] = set()
    if not expected_path:
        return nodes
    for h in expected_path:
        nodes.add(h.source_name.lower().strip())
        nodes.add(h.target_name.lower().strip())
    return nodes


def _extract_predicted_nodes(predicted_hops: list[PathHop]) -> set[str]:
    """从 PathHop 序列里取出所有节点 name（小写去重）。"""
    nodes: set[str] = set()
    for h in predicted_hops or []:
        if h.source and h.source.name:
            nodes.add(h.source.name.lower().strip())
        if h.target and h.target.name:
            nodes.add(h.target.name.lower().strip())
    return nodes


def _extract_expected_triples(expected_path) -> set[tuple[str, str, str]]:
    """从 ExpectedHop 序列里取出 (source, edge, target) 三元组。"""
    triples: set[tuple[str, str, str]] = set()
    if not expected_path:
        return triples
    for h in expected_path:
        triples.add((
            h.source_name.lower().strip(),
            h.edge_name.lower().strip(),
            h.target_name.lower().strip(),
        ))
    return triples


def _extract_predicted_triples(predicted_hops: list[PathHop]) -> set[tuple[str, str, str]]:
    """从 PathHop 序列里取出三元组；边名做大小写无关匹配。"""
    triples: set[tuple[str, str, str]] = set()
    for h in predicted_hops or []:
        if not h.source or not h.target:
            continue
        triples.add((
            h.source.name.lower().strip(),
            (h.edge_name or "").lower().strip(),
            h.target.name.lower().strip(),
        ))
    return triples


def compute_entity_recall(expected_path, predicted_hops: list[PathHop]) -> float:
    """Entity Recall：|pred.nodes ∩ exp.nodes| / |exp.nodes|.

    Returns 0.0 if expected is empty (no positive class).
    """
    exp_nodes = _extract_expected_nodes(expected_path)
    pred_nodes = _extract_predicted_nodes(predicted_hops)
    if not exp_nodes:
        return 0.0
    inter = {n for n in pred_nodes if any(_hop_name_eq(n, e) for e in exp_nodes)}
    return len(inter) / len(exp_nodes)


def compute_entity_precision(expected_path, predicted_hops: list[PathHop]) -> float:
    """Entity Precision：|pred.nodes ∩ exp.nodes| / |pred.nodes|.

    Returns 1.0 if both empty (vacuously true); 0.0 if pred is empty but exp is not.
    """
    exp_nodes = _extract_expected_nodes(expected_path)
    pred_nodes = _extract_predicted_nodes(predicted_hops)
    if not pred_nodes:
        return 1.0 if not exp_nodes else 0.0
    inter = {n for n in pred_nodes if any(_hop_name_eq(n, e) for e in exp_nodes)}
    return len(inter) / len(pred_nodes)


def compute_relation_recall(expected_path, predicted_hops: list[PathHop]) -> float:
    """Relation Recall：|pred.triples ∩ exp.triples| / |exp.triples|."""
    exp_t = _extract_expected_triples(expected_path)
    pred_t = _extract_predicted_triples(predicted_hops)
    if not exp_t:
        return 0.0
    return len(exp_t & pred_t) / len(exp_t)


def compute_pipeline_f1(recall: float, precision: float) -> float:
    """Pipeline F1 = 2 * R * P / (R + P). 当 R+P=0 时返回 0.0。"""
    if recall + precision <= 0:
        return 0.0
    return 2.0 * recall * precision / (recall + precision)


def compute_graph_construction_metrics(
    expected_path,
    predicted_hops: list[PathHop],
) -> dict[str, float]:
    """一次性返回三项图构建指标的 dict。

    Returns
    -------
    {"entity_recall", "entity_precision", "relation_recall", "pipeline_f1"}
    """
    er = compute_entity_recall(expected_path, predicted_hops)
    ep = compute_entity_precision(expected_path, predicted_hops)
    rr = compute_relation_recall(expected_path, predicted_hops)
    return {
        "entity_recall": er,
        "entity_precision": ep,
        "relation_recall": rr,
        "pipeline_f1": compute_pipeline_f1(er, ep),
    }
