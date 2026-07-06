"""模块 3（自研推理控制器）单元测试。

覆盖：
1. 查询类型与意图模型
2. Cypher 生成器（6 种查询类型）
3. 时态剪枝器（单跳/路径/窗口校验）
4. 路径抽取器（Neo4j 记录解析）
5. LLM 解释器（规则兜底解析）
6. 推理控制器（端到端流程）
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from reasoning.controller import ReasoningController
from reasoning.cypher_generator import CypherGenerator
from reasoning.llm_interpreter import LLMInterpreter
from reasoning.path_extractor import (
    PathExtractor,
    build_path_from_nodes_and_edges,
    parse_neo4j_node,
    parse_neo4j_relationship,
)
from reasoning.query_types import (
    QueryIntent,
    QueryType,
    StructuredQuery,
    TimeWindow,
)
from reasoning.result_models import CausalPath, NodeInfo, PathHop, ReasoningResult
from reasoning.temporal_pruner import (
    PrunerConfig,
    TemporalPruner,
    compute_path_confidence,
    compute_path_lag,
    rank_paths,
    validate_hop_temporal_consistency,
    validate_path_in_window,
    validate_path_temporal_consistency,
)


# ============================================================
#  测试辅助：构造测试数据
# ============================================================

def _make_node(
    name: str,
    label: str,
    attributes: dict | None = None,
    uuid: str | None = None,
) -> NodeInfo:
    """构造测试节点。"""
    return NodeInfo(
        uuid=uuid or f"uuid_{name}",
        name=name,
        label=label,
        summary=f"{label} node {name}",
        group_id="test_group",
        attributes=attributes or {},
    )


def _make_hop(
    source: NodeInfo,
    target: NodeInfo,
    edge_name: str,
    valid_at: datetime | None = None,
    invalid_at: datetime | None = None,
    attributes: dict | None = None,
) -> PathHop:
    """构造测试跳。"""
    return PathHop(
        edge_name=edge_name,
        source=source,
        target=target,
        valid_at=valid_at,
        invalid_at=invalid_at,
        attributes=attributes or {},
    )


def _make_symptom_cause_solution_path(
    vm_name: str = "vm_001",
    cause_type: str = "noisy_neighbor",
    confidence: float = 0.85,
    lag_seconds: int = 30,
    valid_at_base: datetime | None = None,
) -> CausalPath:
    """构造 2 跳路径：Symptom → Cause → Solution。"""
    base = valid_at_base or datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    symptom = _make_node(
        f"cpu_spike_{vm_name}",
        "Symptom",
        {"symptom_type": "cpu_usage", "severity": "critical"},
    )
    cause = _make_node(
        f"cause_{cause_type}",
        "Cause",
        {"cause_type": cause_type, "confidence": 0.9, "is_root": True},
    )
    solution = _make_node(
        f"sol_throttle_{cause_type}",
        "Solution",
        {"solution_type": "throttle"},
    )

    hop1 = _make_hop(
        symptom, cause, "CAUSED_BY",
        valid_at=base,
        invalid_at=base + timedelta(hours=1),
        attributes={"mechanism": "邻居抢占 CPU", "lag_seconds": lag_seconds, "confidence": confidence},
    )
    hop2 = _make_hop(
        cause, solution, "RESOLVED_BY",
        valid_at=base + timedelta(seconds=lag_seconds),
        invalid_at=None,
        attributes={"effectiveness": 0.92, "is_immediate": True},
    )

    path = CausalPath(hops=[hop1, hop2])
    path.path_confidence = compute_path_confidence(path)
    path.total_lag_seconds = compute_path_lag(path)
    return path


# ============================================================
#  1. 查询类型与意图模型
# ============================================================

def test_query_type_enum():
    """查询类型枚举完整。"""
    assert QueryType.SINGLE_ENTITY.value == "single_entity"
    assert QueryType.CAUSAL_CHAIN.value == "causal_chain"
    assert QueryType.TIME_RANGE.value == "time_range"
    assert QueryType.MULTI_HOP_PATH.value == "multi_hop_path"
    assert QueryType.SOLUTION_LOOKUP.value == "solution_lookup"
    assert QueryType.COMPARISON.value == "comparison"


def test_time_window_contains():
    """TimeWindow.contains 正确。"""
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 2, tzinfo=timezone.utc)
    window = TimeWindow(start=start, end=end)

    assert window.contains(datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc))
    assert window.contains(start)
    assert window.contains(end)
    assert not window.contains(datetime(2023, 12, 31, tzinfo=timezone.utc))


def test_time_window_overlaps():
    """TimeWindow.overlaps 区间重叠校验正确。"""
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 2, tzinfo=timezone.utc)
    window = TimeWindow(start=start, end=end)

    # 完全包含
    assert window.overlaps(start, end)
    # 部分重叠
    assert window.overlaps(
        datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        datetime(2024, 1, 3, tzinfo=timezone.utc),
    )
    # 不重叠
    assert not window.overlaps(
        datetime(2024, 1, 3, tzinfo=timezone.utc),
        datetime(2024, 1, 4, tzinfo=timezone.utc),
    )
    # 无限区间
    assert window.overlaps(None, None)
    assert window.overlaps(None, datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc))


def test_structured_query_model():
    """StructuredQuery 模型构建正确。"""
    intent = QueryIntent(
        query_type=QueryType.CAUSAL_CHAIN,
        target_entity="vm_001",
        target_entity_type="vm",
    )
    sq = StructuredQuery(
        natural_language="vm_001 为什么 CPU 飙高",
        intent=intent,
    )
    assert sq.natural_language == "vm_001 为什么 CPU 飙高"
    assert sq.intent.query_type == QueryType.CAUSAL_CHAIN
    assert sq.intent.target_entity == "vm_001"


# ============================================================
#  2. Cypher 生成器
# ============================================================

def test_cypher_single_entity():
    """单实体查询 Cypher 生成正确。"""
    gen = CypherGenerator()
    intent = QueryIntent(
        query_type=QueryType.SINGLE_ENTITY,
        target_entity="vm_001",
        target_entity_type="vm",
    )
    sq = StructuredQuery(natural_language="vm_001 的故障", intent=intent)
    cypher, params = gen.generate(sq)

    assert "MATCH" in cypher
    assert "$vm_id" in cypher
    assert params["vm_id"] == "vm_001"
    assert "HAS_SYMPTOM" in cypher
    assert "Component" in cypher


def test_cypher_causal_chain():
    """因果链查询 Cypher 生成正确。"""
    gen = CypherGenerator()
    intent = QueryIntent(
        query_type=QueryType.CAUSAL_CHAIN,
        target_entity="vm_001",
        symptom_keywords=["cpu_spike"],
    )
    sq = StructuredQuery(natural_language="vm_001 的根因", intent=intent)
    cypher, params = gen.generate(sq)

    assert "CAUSED_BY" in cypher
    assert "TRIGGERED_BY" in cypher
    assert "Cause" in cypher
    assert params.get("entity_name") == "vm_001"


def test_cypher_time_range():
    """时态范围查询 Cypher 生成正确。"""
    gen = CypherGenerator()
    intent = QueryIntent(
        query_type=QueryType.TIME_RANGE,
        time_window=TimeWindow(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 2, tzinfo=timezone.utc),
        ),
    )
    sq = StructuredQuery(natural_language="2024-01-01 的故障", intent=intent)
    cypher, params = gen.generate(sq)

    assert "$start" in cypher
    assert "$end" in cypher
    assert "valid_at" in cypher
    assert "start" in params and "end" in params


def test_cypher_multi_hop_2():
    """2 跳多跳路径 Cypher 生成正确。"""
    gen = CypherGenerator()
    intent = QueryIntent(
        query_type=QueryType.MULTI_HOP_PATH,
        hop_count=2,
        target_entity="vm_001",
    )
    sq = StructuredQuery(natural_language="vm_001 的完整因果链", intent=intent)
    cypher, params = gen.generate(sq)

    assert "path" in cypher
    assert "Symptom" in cypher
    assert "Cause" in cypher
    assert "Solution" in cypher
    assert "CAUSED_BY" in cypher
    assert "RESOLVED_BY" in cypher


def test_cypher_multi_hop_3():
    """3 跳多跳路径 Cypher 生成正确。"""
    gen = CypherGenerator()
    intent = QueryIntent(
        query_type=QueryType.MULTI_HOP_PATH,
        hop_count=3,
    )
    sq = StructuredQuery(natural_language="完整因果链", intent=intent)
    cypher, _ = gen.generate(sq)

    assert "c1" in cypher and "c2" in cypher  # 两个 Cause 节点
    assert "PROPAGATED_TO" in cypher


def test_cypher_multi_hop_4():
    """4 跳多跳路径 Cypher 生成正确。"""
    gen = CypherGenerator()
    intent = QueryIntent(
        query_type=QueryType.MULTI_HOP_PATH,
        hop_count=4,
    )
    sq = StructuredQuery(natural_language="从组件到根因的路径", intent=intent)
    cypher, _ = gen.generate(sq)

    assert "Component" in cypher
    assert "HAS_SYMPTOM" in cypher


def test_cypher_solution_lookup():
    """解法查询 Cypher 生成正确。"""
    gen = CypherGenerator()
    intent = QueryIntent(
        query_type=QueryType.SOLUTION_LOOKUP,
        cause_keywords=["noisy_neighbor"],
    )
    sq = StructuredQuery(natural_language="noisy_neighbor 的解法", intent=intent)
    cypher, params = gen.generate(sq)

    assert "RESOLVED_BY" in cypher
    assert "MITIGATED_BY" in cypher
    assert "Solution" in cypher
    assert params.get("cause_type") == "noisy_neighbor"


def test_cypher_comparison():
    """对比查询 Cypher 生成正确（UNION）。"""
    gen = CypherGenerator()
    intent = QueryIntent(
        query_type=QueryType.COMPARISON,
        target_entity="vm_001,vm_002",
    )
    sq = StructuredQuery(natural_language="vm_001 和 vm_002 对比", intent=intent)
    cypher, params = gen.generate(sq)

    assert "UNION" in cypher
    assert "vm1" in params and "vm2" in params
    assert params["vm1"] == "vm_001"
    assert params["vm2"] == "vm_002"


# ============================================================
#  3. 时态剪枝器
# ============================================================

def test_validate_hop_valid():
    """合法单跳通过校验。"""
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    hop = _make_hop(
        _make_node("s", "Symptom"),
        _make_node("c", "Cause"),
        "CAUSED_BY",
        valid_at=base,
        invalid_at=base + timedelta(hours=1),
    )
    ok, reason = validate_hop_temporal_consistency(hop)
    assert ok, f"应通过校验，但失败: {reason}"


def test_validate_hop_invalid_interval():
    """valid_at >= invalid_at 的跳校验失败。"""
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    hop = _make_hop(
        _make_node("s", "Symptom"),
        _make_node("c", "Cause"),
        "CAUSED_BY",
        valid_at=base + timedelta(hours=1),
        invalid_at=base,
    )
    ok, reason = validate_hop_temporal_consistency(hop)
    assert not ok
    assert "valid_at" in reason


def test_validate_path_monotonicity():
    """路径时态单调性校验。"""
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    symptom = _make_node("s", "Symptom")
    cause = _make_node("c", "Cause", {"is_root": True})
    sol = _make_node("sol", "Solution")

    # 合法：valid_at 递增
    hop1 = _make_hop(symptom, cause, "CAUSED_BY", valid_at=base)
    hop2 = _make_hop(cause, sol, "RESOLVED_BY", valid_at=base + timedelta(seconds=30))
    path = CausalPath(hops=[hop1, hop2])
    ok, _ = validate_path_temporal_consistency(path)
    assert ok

    # 非法：valid_at 递减
    hop1_bad = _make_hop(symptom, cause, "CAUSED_BY", valid_at=base + timedelta(seconds=30))
    hop2_bad = _make_hop(cause, sol, "RESOLVED_BY", valid_at=base)
    path_bad = CausalPath(hops=[hop1_bad, hop2_bad])
    ok, reason = validate_path_temporal_consistency(path_bad)
    assert not ok
    assert "单调" in reason or "非单调" in reason


def test_validate_path_lag_consistency():
    """lag_seconds 一致性校验。"""
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    symptom = _make_node("s", "Symptom")
    cause = _make_node("c", "Cause", {"is_root": True})
    sol = _make_node("sol", "Solution")

    # 合法：实际时延与 lag_seconds 一致
    hop1 = _make_hop(symptom, cause, "CAUSED_BY", valid_at=base,
                     attributes={"lag_seconds": 30})
    hop2 = _make_hop(cause, sol, "RESOLVED_BY", valid_at=base + timedelta(seconds=30),
                     attributes={"lag_seconds": 0})
    path = CausalPath(hops=[hop1, hop2])
    ok, _ = validate_path_temporal_consistency(path)
    assert ok

    # 非法：实际时延远超 lag_seconds
    hop2_bad = _make_hop(cause, sol, "RESOLVED_BY",
                         valid_at=base + timedelta(hours=24),
                         attributes={"lag_seconds": 5})
    path_bad = CausalPath(hops=[hop1, hop2_bad])
    ok, reason = validate_path_temporal_consistency(path_bad)
    assert not ok
    assert "时延" in reason


def test_validate_path_in_window():
    """路径与查询窗口重叠校验。"""
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    symptom = _make_node("s", "Symptom")
    cause = _make_node("c", "Cause")
    hop = _make_hop(
        symptom, cause, "CAUSED_BY",
        valid_at=base,
        invalid_at=base + timedelta(hours=1),
    )
    path = CausalPath(hops=[hop])

    # 重叠窗口
    window_overlap = TimeWindow(
        start=base + timedelta(minutes=30),
        end=base + timedelta(minutes=45),
    )
    ok, _ = validate_path_in_window(path, window_overlap)
    assert ok

    # 不重叠窗口
    window_no_overlap = TimeWindow(
        start=base + timedelta(days=2),
        end=base + timedelta(days=3),
    )
    ok, reason = validate_path_in_window(path, window_no_overlap)
    assert not ok
    assert "无重叠" in reason


def test_temporal_pruner_prune():
    """TemporalPruner.prune 剪枝流程正确。"""
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    symptom = _make_node("s", "Symptom")
    cause = _make_node("c", "Cause", {"is_root": True})
    sol = _make_node("sol", "Solution")

    # 合法路径
    hop1 = _make_hop(symptom, cause, "CAUSED_BY", valid_at=base,
                     attributes={"lag_seconds": 30, "confidence": 0.9})
    hop2 = _make_hop(cause, sol, "RESOLVED_BY",
                     valid_at=base + timedelta(seconds=30),
                     attributes={"lag_seconds": 0, "confidence": 0.85})
    good_path = CausalPath(hops=[hop1, hop2])

    # 非法路径（时态倒序）
    hop1_bad = _make_hop(symptom, cause, "CAUSED_BY",
                         valid_at=base + timedelta(seconds=30),
                         attributes={"lag_seconds": 30})
    hop2_bad = _make_hop(cause, sol, "RESOLVED_BY", valid_at=base,
                         attributes={"lag_seconds": 0})
    bad_path = CausalPath(hops=[hop1_bad, hop2_bad])

    pruner = TemporalPruner()
    kept, pruned = pruner.prune([good_path, bad_path])

    assert len(kept) == 1
    assert len(pruned) == 1
    assert kept[0] is good_path
    assert pruned[0].pruned_reason is not None


def test_compute_path_confidence():
    """路径置信度计算正确（几何平均）。"""
    symptom = _make_node("s", "Symptom")
    cause = _make_node("c", "Cause")
    hop1 = _make_hop(symptom, cause, "CAUSED_BY",
                     attributes={"confidence": 0.9})
    hop2 = _make_hop(cause, _make_node("sol", "Solution"), "RESOLVED_BY",
                     attributes={"confidence": 0.4})
    path = CausalPath(hops=[hop1, hop2])

    conf = compute_path_confidence(path)
    # 几何平均: sqrt(0.9 * 0.4) ≈ 0.6
    assert 0.59 < conf < 0.61


def test_rank_paths():
    """路径排序：置信度降序 + 时延升序。"""
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    symptom = _make_node("s", "Symptom")
    cause = _make_node("c", "Cause")
    sol = _make_node("sol", "Solution")

    # 高置信度长时延
    hop1a = _make_hop(symptom, cause, "CAUSED_BY", valid_at=base,
                      attributes={"confidence": 0.95, "lag_seconds": 100})
    hop2a = _make_hop(cause, sol, "RESOLVED_BY", valid_at=base + timedelta(seconds=100),
                      attributes={"confidence": 0.95, "lag_seconds": 0})
    path_a = CausalPath(hops=[hop1a, hop2a])

    # 低置信度短时延
    hop1b = _make_hop(symptom, cause, "CAUSED_BY", valid_at=base,
                      attributes={"confidence": 0.5, "lag_seconds": 10})
    hop2b = _make_hop(cause, sol, "RESOLVED_BY", valid_at=base + timedelta(seconds=10),
                      attributes={"confidence": 0.5, "lag_seconds": 0})
    path_b = CausalPath(hops=[hop1b, hop2b])

    ranked = rank_paths([path_b, path_a])
    assert ranked[0] is path_a  # 高置信度优先


# ============================================================
#  4. 路径抽取器
# ============================================================

def test_parse_neo4j_node_dict():
    """从 dict 解析节点。"""
    node_dict = {
        "uuid": "uuid_001",
        "name": "cpu_spike_vm_001",
        "summary": "CPU spike on vm_001",
        "group_id": "cluster_A",
        "attributes": {"cause_type": "noisy_neighbor", "is_root": True},
        "_labels": ["Cause", "Entity"],
    }
    node = parse_neo4j_node(node_dict)
    assert node.uuid == "uuid_001"
    assert node.name == "cpu_spike_vm_001"
    assert node.label == "Cause"
    assert node.cause_type == "noisy_neighbor"
    assert node.is_root is True


def test_parse_neo4j_relationship_dict():
    """从 dict 解析边。"""
    rel_dict = {
        "name": "CAUSED_BY",
        "valid_at": datetime(2024, 1, 1, tzinfo=timezone.utc),
        "invalid_at": None,
        "attributes": {"lag_seconds": 30, "confidence": 0.85},
    }
    rel = parse_neo4j_relationship(rel_dict)
    assert rel["name"] == "CAUSED_BY"
    assert rel["attributes"]["lag_seconds"] == 30
    assert rel["valid_at"] == datetime(2024, 1, 1, tzinfo=timezone.utc)


def test_build_path_from_nodes_and_edges():
    """从节点/边列表构建路径。"""
    nodes = [
        {"name": "s", "label": "Symptom", "_labels": ["Symptom"]},
        {"name": "c", "label": "Cause", "_labels": ["Cause"],
         "attributes": {"is_root": True}},
        {"name": "sol", "label": "Solution", "_labels": ["Solution"]},
    ]
    edges = [
        {"name": "CAUSED_BY", "attributes": {"lag_seconds": 30, "confidence": 0.9}},
        {"name": "RESOLVED_BY", "attributes": {"effectiveness": 0.9}},
    ]
    path = build_path_from_nodes_and_edges(nodes, edges)
    assert path.hop_count == 2
    assert path.start_node.name == "s"
    assert path.end_node.name == "sol"
    assert path.labels == ["Symptom", "Cause", "Solution"]
    assert path.hops[0].lag_seconds == 30


def test_path_extractor_extract_from_records():
    """从查询记录列表抽取路径。"""
    records = [
        {
            "path": {
                "nodes": [
                    {"name": "s", "_labels": ["Symptom"]},
                    {"name": "c", "_labels": ["Cause"],
                     "attributes": {"is_root": True}},
                    {"name": "sol", "_labels": ["Solution"]},
                ],
                "relationships": [
                    {"name": "CAUSED_BY", "attributes": {"lag_seconds": 30}},
                    {"name": "RESOLVED_BY", "attributes": {}},
                ],
            }
        }
    ]
    extractor = PathExtractor(driver=None)
    paths = extractor.extract_from_records(records)
    assert len(paths) == 1
    assert paths[0].hop_count == 2
    assert paths[0].labels == ["Symptom", "Cause", "Solution"]


def test_path_root_cause():
    """路径根因节点提取。"""
    path = _make_symptom_cause_solution_path()
    root = path.root_cause
    assert root is not None
    assert root.label == "Cause"
    assert root.is_root is True
    assert root.cause_type == "noisy_neighbor"


# ============================================================
#  5. LLM 解释器（规则兜底）
# ============================================================

def test_llm_interpreter_rule_based_parse():
    """规则兜底解析查询意图。"""
    interpreter = LLMInterpreter(client=None)

    # 因果链查询
    sq = interpreter.parse_query("vm_001 为什么 CPU 飙高")
    assert sq.intent.query_type == QueryType.CAUSAL_CHAIN
    assert sq.intent.target_entity == "vm_001"

    # 解法查询
    sq = interpreter.parse_query("noisy_neighbor 的解法是什么")
    assert sq.intent.query_type == QueryType.SOLUTION_LOOKUP

    # 多跳路径
    sq = interpreter.parse_query("vm_001 的完整因果路径")
    assert sq.intent.query_type == QueryType.MULTI_HOP_PATH
    assert sq.intent.hop_count == 2

    # 时态范围
    sq = interpreter.parse_query("2024-01-01 期间的故障")
    assert sq.intent.query_type == QueryType.TIME_RANGE


def test_llm_interpreter_rule_based_explain():
    """规则兜底解释结果。"""
    interpreter = LLMInterpreter(client=None)
    path = _make_symptom_cause_solution_path()

    sq = StructuredQuery(
        natural_language="vm_001 为什么 CPU 飙高",
        intent=QueryIntent(query_type=QueryType.CAUSAL_CHAIN, target_entity="vm_001"),
    )

    answer = interpreter.explain(sq, [path])
    assert "根因" in answer
    assert "noisy_neighbor" in answer
    assert "置信度" in answer


def test_llm_interpreter_explain_no_result():
    """无结果时的解释。"""
    interpreter = LLMInterpreter(client=None)
    sq = StructuredQuery(
        natural_language="不存在的查询",
        intent=QueryIntent(query_type=QueryType.CAUSAL_CHAIN),
    )
    answer = interpreter.explain(sq, [])
    assert "未找到" in answer


def test_llm_interpreter_estimate_confidence():
    """综合置信度估算。"""
    interpreter = LLMInterpreter(client=None)
    path = _make_symptom_cause_solution_path(confidence=0.9)
    # path_confidence 已计算

    # 全部保留
    conf = interpreter.estimate_confidence([path], [])
    assert 0 < conf <= 1.0

    # 一半剪枝
    pruned_path = _make_symptom_cause_solution_path(confidence=0.3)
    conf2 = interpreter.estimate_confidence([path], [pruned_path])
    assert conf2 < conf  # 通过率降低


# ============================================================
#  6. 推理控制器（端到端）
# ============================================================

def test_controller_ask_rule_based():
    """推理控制器端到端（规则兜底模式）。"""
    controller = ReasoningController()  # 无 LLM，无 driver

    result = controller.ask("vm_001 为什么 CPU 飙高")

    assert isinstance(result, ReasoningResult)
    assert result.query == "vm_001 为什么 CPU 飙高"
    assert result.cypher_used  # 有生成 Cypher
    assert result.elapsed_seconds > 0
    # 无 driver，路径为空
    assert result.path_count == 0
    assert "未找到" in result.answer


def test_controller_query_structured():
    """结构化查询接口。"""
    controller = ReasoningController()
    intent = QueryIntent(
        query_type=QueryType.MULTI_HOP_PATH,
        hop_count=2,
        target_entity="vm_001",
    )
    sq = StructuredQuery(natural_language="测试查询", intent=intent)
    result = controller.query(sq)

    assert isinstance(result, ReasoningResult)
    assert "path" in result.cypher_used
    assert result.metadata["query_type"] == "multi_hop_path"


def test_controller_explain_query():
    """诊断接口返回中间结果。"""
    controller = ReasoningController()
    diag = controller.explain_query("vm_001 的根因")

    assert "structured_query" in diag
    assert "cypher" in diag
    assert "candidate_count" in diag
    assert "kept_count" in diag
    assert "pruned_count" in diag
    assert diag["candidate_count"] == 0  # 无 driver


def test_controller_end_to_end_with_paths():
    """端到端：注入路径测试完整流程。

    用 mock path_extractor 注入路径，验证剪枝+解释流程。
    """
    controller = ReasoningController()

    # Mock path_extractor：返回预构造路径
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    good_path = _make_symptom_cause_solution_path(valid_at_base=base)

    # 构造一条坏路径（时态倒序）
    symptom = _make_node("s_bad", "Symptom")
    cause = _make_node("c_bad", "Cause", {"is_root": True})
    sol = _make_node("sol_bad", "Solution")
    bad_path = CausalPath(hops=[
        _make_hop(symptom, cause, "CAUSED_BY",
                  valid_at=base + timedelta(seconds=30),
                  attributes={"lag_seconds": 30, "confidence": 0.8}),
        _make_hop(cause, sol, "RESOLVED_BY", valid_at=base,
                  attributes={"lag_seconds": 0, "confidence": 0.8}),
    ])

    # 替换 path_extractor.execute
    def mock_execute(cypher, params):
        return [good_path, bad_path]
    controller.path_extractor.execute = mock_execute

    result = controller.ask("vm_001 为什么 CPU 飙高")

    assert result.path_count == 1  # 坏路径被剪枝
    assert result.pruned_count == 1
    assert result.confidence > 0
    assert "根因" in result.answer or "未找到" in result.answer


# ============================================================
#  7. 结果模型
# ============================================================

def test_causal_path_labels():
    """CausalPath.labels 正确。"""
    path = _make_symptom_cause_solution_path()
    assert path.labels == ["Symptom", "Cause", "Solution"]


def test_causal_path_to_natural_language():
    """路径转自然语言。"""
    path = _make_symptom_cause_solution_path()
    nl = path.to_natural_language()
    assert "CAUSED_BY" in nl
    assert "RESOLVED_BY" in nl
    assert "置信度" in nl


def test_reasoning_result_summary():
    """ReasoningResult.summary 正确。"""
    result = ReasoningResult(
        query="测试",
        answer="答案",
        paths=[_make_symptom_cause_solution_path()],
        confidence=0.85,
        elapsed_seconds=0.5,
    )
    summary = result.summary()
    assert "测试" in summary
    assert "答案" in summary
    assert "0.850" in summary


def test_reasoning_result_best_path():
    """ReasoningResult.best_path 返回置信度最高的路径。"""
    path_high = _make_symptom_cause_solution_path(confidence=0.95)
    path_low = _make_symptom_cause_solution_path(confidence=0.3)
    path_high.path_confidence = 0.95
    path_low.path_confidence = 0.3

    result = ReasoningResult(
        query="测试",
        answer="",
        paths=[path_low, path_high],
    )
    assert result.best_path is path_high


# ============================================================
#  主入口
# ============================================================

def main():
    """运行所有测试。"""
    tests = [
        # 1. 查询类型与意图模型
        ("test_query_type_enum", test_query_type_enum),
        ("test_time_window_contains", test_time_window_contains),
        ("test_time_window_overlaps", test_time_window_overlaps),
        ("test_structured_query_model", test_structured_query_model),
        # 2. Cypher 生成器
        ("test_cypher_single_entity", test_cypher_single_entity),
        ("test_cypher_causal_chain", test_cypher_causal_chain),
        ("test_cypher_time_range", test_cypher_time_range),
        ("test_cypher_multi_hop_2", test_cypher_multi_hop_2),
        ("test_cypher_multi_hop_3", test_cypher_multi_hop_3),
        ("test_cypher_multi_hop_4", test_cypher_multi_hop_4),
        ("test_cypher_solution_lookup", test_cypher_solution_lookup),
        ("test_cypher_comparison", test_cypher_comparison),
        # 3. 时态剪枝器
        ("test_validate_hop_valid", test_validate_hop_valid),
        ("test_validate_hop_invalid_interval", test_validate_hop_invalid_interval),
        ("test_validate_path_monotonicity", test_validate_path_monotonicity),
        ("test_validate_path_lag_consistency", test_validate_path_lag_consistency),
        ("test_validate_path_in_window", test_validate_path_in_window),
        ("test_temporal_pruner_prune", test_temporal_pruner_prune),
        ("test_compute_path_confidence", test_compute_path_confidence),
        ("test_rank_paths", test_rank_paths),
        # 4. 路径抽取器
        ("test_parse_neo4j_node_dict", test_parse_neo4j_node_dict),
        ("test_parse_neo4j_relationship_dict", test_parse_neo4j_relationship_dict),
        ("test_build_path_from_nodes_and_edges", test_build_path_from_nodes_and_edges),
        ("test_path_extractor_extract_from_records", test_path_extractor_extract_from_records),
        ("test_path_root_cause", test_path_root_cause),
        # 5. LLM 解释器
        ("test_llm_interpreter_rule_based_parse", test_llm_interpreter_rule_based_parse),
        ("test_llm_interpreter_rule_based_explain", test_llm_interpreter_rule_based_explain),
        ("test_llm_interpreter_explain_no_result", test_llm_interpreter_explain_no_result),
        ("test_llm_interpreter_estimate_confidence", test_llm_interpreter_estimate_confidence),
        # 6. 推理控制器
        ("test_controller_ask_rule_based", test_controller_ask_rule_based),
        ("test_controller_query_structured", test_controller_query_structured),
        ("test_controller_explain_query", test_controller_explain_query),
        ("test_controller_end_to_end_with_paths", test_controller_end_to_end_with_paths),
        # 7. 结果模型
        ("test_causal_path_labels", test_causal_path_labels),
        ("test_causal_path_to_natural_language", test_causal_path_to_natural_language),
        ("test_reasoning_result_summary", test_reasoning_result_summary),
        ("test_reasoning_result_best_path", test_reasoning_result_best_path),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            test_fn()
            print(f"  [PASS] {name}")
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"  [ERROR] {name}: {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"模块 3 测试结果: {passed} 通过 / {failed} 失败 / {len(tests)} 总计")
    print(f"{'='*60}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
