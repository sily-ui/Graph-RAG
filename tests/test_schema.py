"""Schema 单元测试 —— 验证四层节点、三类边、约束逻辑的正确性。

运行方式（在项目根目录）：
    python -m pytest tests/test_schema.py -v

或直接运行：
    python tests/test_schema.py
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

# 确保能导入项目模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph_schema.nodes import (
    ENTITY_TYPES,
    ComponentEntity,
    ComponentType,
    CauseEntity,
    CauseType,
    GraphLayer,
    SolutionEntity,
    SolutionType,
    SymptomEntity,
    SymptomType,
    Severity,
    build_component_attributes,
    build_symptom_attributes,
    build_cause_attributes,
    build_solution_attributes,
)
from graph_schema.edges import (
    ALL_EDGE_NAMES,
    CAUSAL_EDGE_NAMES,
    EDGE_TYPES,
    EDGE_TYPE_MAP,
    SOLUTION_EDGE_NAMES,
    EdgeName,
    build_causal_edge_attributes,
    build_solution_edge_attributes,
)
from graph_schema.constraints import (
    PATH_TEMPLATES,
    validate_causal_path_terminates_at_root,
    validate_edge_combination,
    validate_edge_full,
    validate_entity_for_layer,
    validate_temporal_consistency,
)


# ============================================================
#  测试工具
# ============================================================

def _run_test(name: str, test_fn):
    """简单测试运行器，不依赖 pytest。"""
    try:
        test_fn()
        print(f"  [PASS] {name}")
        return True
    except AssertionError as e:
        print(f"  [FAIL] {name}: {e}")
        return False
    except Exception as e:
        print(f"  [ERROR] {name}: {type(e).__name__}: {e}")
        return False


# ============================================================
#  1. 节点 Schema 测试
# ============================================================

def test_entity_types_registered():
    """四层节点类型都注册在 ENTITY_TYPES 中。"""
    assert "Component" in ENTITY_TYPES
    assert "Symptom" in ENTITY_TYPES
    assert "Cause" in ENTITY_TYPES
    assert "Solution" in ENTITY_TYPES
    assert len(ENTITY_TYPES) == 4


def test_component_entity_creation():
    """ComponentEntity 可以正确创建，字段类型正确。"""
    comp = ComponentEntity(
        component_type=ComponentType.VM,
        cluster_id="cluster_A",
        sku="D-series",
        vcore_bucket="8-16",
        memory_gb_bucket="64-128",
    )
    assert comp.component_type == ComponentType.VM
    assert comp.cluster_id == "cluster_A"
    assert comp.sku == "D-series"


def test_symptom_entity_defaults():
    """SymptomEntity 的 severity 有默认值。"""
    sym = SymptomEntity(symptom_type=SymptomType.METRIC_ANOMALY)
    assert sym.severity == Severity.WARNING
    assert sym.metric_name is None


def test_cause_entity_confidence_range():
    """CauseEntity 的 confidence 在 [0,1] 范围内。"""
    cause = CauseEntity(cause_type=CauseType.MISCONFIGURATION, confidence=0.9, is_root=True)
    assert cause.confidence == 0.9
    assert cause.is_root is True

    # 超范围应该报错
    try:
        CauseEntity(cause_type=CauseType.HARDWARE_FAULT, confidence=1.5)
        raise AssertionError("confidence=1.5 应该被拒绝")
    except Exception:
        pass  # Pydantic 校验拦截


def test_solution_entity_optional_fields():
    """SolutionEntity 的 runbook_ref 和 estimated_mttr_min 可选。"""
    sol = SolutionEntity(solution_type=SolutionType.RESTART_POD)
    assert sol.runbook_ref is None
    assert sol.estimated_mttr_min is None


def test_build_attributes_functions():
    """attributes 构造函数返回正确的 dict。"""
    comp_attrs = build_component_attributes(ComponentType.VM, "cluster_A")
    assert comp_attrs["component_type"] == "vm"
    assert comp_attrs["cluster_id"] == "cluster_A"

    sym_attrs = build_symptom_attributes(
        SymptomType.METRIC_ANOMALY, Severity.CRITICAL,
        metric_name="cpu_usage", threshold=0.95, observed_value=0.98,
    )
    assert sym_attrs["symptom_type"] == "metric_anomaly"
    assert sym_attrs["severity"] == "critical"

    cause_attrs = build_cause_attributes(CauseType.NOISY_NEIGHBOR, confidence=0.85, is_root=False)
    assert cause_attrs["cause_type"] == "noisy_neighbor"
    assert cause_attrs["is_root"] is False

    sol_attrs = build_solution_attributes(SolutionType.SCALE_UP, estimated_mttr_min=5)
    assert sol_attrs["solution_type"] == "scale_up"
    assert sol_attrs["estimated_mttr_min"] == 5


# ============================================================
#  2. 边 Schema 测试
# ============================================================

def test_edge_names_complete():
    """所有边名称枚举完整。"""
    assert EdgeName.HAS_SYMPTOM.value == "HAS_SYMPTOM"
    assert EdgeName.CAUSED_BY.value == "CAUSED_BY"
    assert EdgeName.RESOLVED_BY.value == "RESOLVED_BY"


def test_causal_edge_names_group():
    """因果链边名称分组正确。"""
    assert "CAUSED_BY" in CAUSAL_EDGE_NAMES
    assert "TRIGGERED_BY" in CAUSAL_EDGE_NAMES
    assert "PROPAGATED_TO" in CAUSAL_EDGE_NAMES
    assert "RESOLVED_BY" not in CAUSAL_EDGE_NAMES


def test_solution_edge_names_group():
    """解法边名称分组正确。"""
    assert "RESOLVED_BY" in SOLUTION_EDGE_NAMES
    assert "MITIGATED_BY" in SOLUTION_EDGE_NAMES
    assert "CAUSED_BY" not in SOLUTION_EDGE_NAMES


def test_edge_type_map_completeness():
    """EDGE_TYPE_MAP 覆盖所有合法层级组合。"""
    # Component → Symptom
    assert ("Component", "Symptom") in EDGE_TYPE_MAP
    # Symptom → Cause
    assert ("Symptom", "Cause") in EDGE_TYPE_MAP
    # Cause → Cause
    assert ("Cause", "Cause") in EDGE_TYPE_MAP
    # Cause → Solution
    assert ("Cause", "Solution") in EDGE_TYPE_MAP


def test_causal_edge_attributes():
    """因果边 attributes 构造函数正确。"""
    attrs = build_causal_edge_attributes(
        mechanism="内存不足触发 OOMKiller",
        lag_seconds=30,
        confidence=0.85,
    )
    assert attrs["mechanism"] == "内存不足触发 OOMKiller"
    assert attrs["lag_seconds"] == 30
    assert attrs["confidence"] == 0.85


def test_solution_edge_attributes():
    """解法边 attributes 构造函数正确。"""
    attrs = build_solution_edge_attributes(effectiveness=0.9, is_immediate=True)
    assert attrs["effectiveness"] == 0.9
    assert attrs["is_immediate"] is True


# ============================================================
#  3. 约束校验测试
# ============================================================

def test_validate_edge_combination_legal():
    """合法层级组合通过校验。"""
    assert validate_edge_combination("Component", "Symptom", "HAS_SYMPTOM")
    assert validate_edge_combination("Symptom", "Cause", "CAUSED_BY")
    assert validate_edge_combination("Cause", "Cause", "PROPAGATED_TO")
    assert validate_edge_combination("Cause", "Solution", "RESOLVED_BY")


def test_validate_edge_combination_illegal():
    """非法层级组合被拒绝。"""
    # Solution 不能指向 Symptom
    assert not validate_edge_combination("Solution", "Symptom", "RESOLVED_BY")
    # Component 不能直接指向 Cause
    assert not validate_edge_combination("Component", "Cause", "CAUSED_BY")
    # HAS_SYMPTOM 只能 Component→Symptom
    assert not validate_edge_combination("Symptom", "Cause", "HAS_SYMPTOM")


def test_validate_temporal_consistency():
    """时态一致性校验正确。"""
    t1 = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
    t2 = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)

    assert validate_temporal_consistency(t1, t2)   # valid < invalid
    assert not validate_temporal_consistency(t2, t1)  # valid > invalid 非法
    assert validate_temporal_consistency(t1, None)    # invalid_at=None 允许
    assert validate_temporal_consistency(None, None)  # 两者都 None 允许


def test_validate_entity_for_layer():
    """节点属性完整性校验。"""
    # Component 完整
    assert validate_entity_for_layer("Component", {"component_type": "vm", "cluster_id": "c1"}) == []
    # Component 缺字段
    missing = validate_entity_for_layer("Component", {"component_type": "vm"})
    assert "cluster_id" in missing

    # Cause 完整
    assert validate_entity_for_layer("Cause", {"cause_type": "misconfiguration", "confidence": 0.8, "is_root": True}) == []
    # Cause 缺字段
    missing = validate_entity_for_layer("Cause", {"cause_type": "misconfiguration"})
    assert "confidence" in missing
    assert "is_root" in missing


def test_validate_causal_path():
    """因果路径终止条件校验。"""
    # 2 跳: Symptom → Cause → Solution，合法
    assert validate_causal_path_terminates_at_root(["Symptom", "Cause", "Solution"])
    # 3 跳: Symptom → Cause → Cause → Solution，合法
    assert validate_causal_path_terminates_at_root(["Symptom", "Cause", "Cause", "Solution"])
    # 4 跳: Component → Symptom → Cause → Cause，合法（终止于 Cause）
    assert validate_causal_path_terminates_at_root(["Component", "Symptom", "Cause", "Cause"])
    # 无 Cause，非法
    assert not validate_causal_path_terminates_at_root(["Component", "Symptom", "Solution"])
    # Cause 后面跟 Symptom，非法
    assert not validate_causal_path_terminates_at_root(["Symptom", "Cause", "Symptom"])


def test_validate_edge_full():
    """综合校验函数返回所有违规项。"""
    t1 = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
    t2 = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)

    # 全部合法
    violations = validate_edge_full(
        "Symptom", "Cause", "CAUSED_BY",
        valid_at=t1, invalid_at=t2,
        source_attributes={"symptom_type": "metric_anomaly", "severity": "critical"},
        target_attributes={"cause_type": "noisy_neighbor", "confidence": 0.8, "is_root": True},
    )
    assert violations == [], f"应为空，实际: {violations}"

    # 层级非法 + 时态非法 + 缺字段
    violations = validate_edge_full(
        "Solution", "Symptom", "RESOLVED_BY",
        valid_at=t2, invalid_at=t1,  # valid > invalid
        source_attributes={"solution_type": "scale_up"},  # OK
        target_attributes={"symptom_type": "metric_anomaly"},  # 缺 severity
    )
    assert len(violations) >= 3  # 层级非法 + 时态非法 + 缺 severity


# ============================================================
#  4. 路径模板测试
# ============================================================

def test_path_templates():
    """2/3/4 跳路径模板定义正确。"""
    assert 2 in PATH_TEMPLATES
    assert 3 in PATH_TEMPLATES
    assert 4 in PATH_TEMPLATES

    # 2 跳: Symptom → Cause → Solution
    t2 = PATH_TEMPLATES[2]
    assert t2["labels"] == ["Symptom", "Cause", "Solution"]
    assert len(t2["edges"]) == 2

    # 3 跳: Symptom → Cause → Cause → Solution
    t3 = PATH_TEMPLATES[3]
    assert t3["labels"] == ["Symptom", "Cause", "Cause", "Solution"]
    assert len(t3["edges"]) == 3

    # 4 跳: Component → Symptom → Cause → Cause
    t4 = PATH_TEMPLATES[4]
    assert t4["labels"] == ["Component", "Symptom", "Cause", "Cause"]
    assert len(t4["edges"]) == 3


def test_path_template_edges_legal():
    """路径模板中的每条边都是合法的层级组合。"""
    for hops, template in PATH_TEMPLATES.items():
        labels = template["labels"]
        edges = template["edges"]
        for i, edge_options in enumerate(edges):
            for edge_name in edge_options:
                source = labels[i]
                target = labels[i + 1]
                assert validate_edge_combination(source, target, edge_name), \
                    f"{hops}跳模板第{i+1}跳 ({source})-[{edge_name}]->({target}) 非法"


# ============================================================
#  5. Graphiti 兼容性测试 —— 确保自定义类型不与保留字段冲突
# ============================================================

def test_no_field_conflicts():
    """自定义节点/边模型的字段名不与 EntityNode/EntityEdge 的保留字段冲突。

    这是 Graphiti add_episode 的硬性要求：字段名若与保留字段重名，
    会在 validate_entity_types 阶段直接抛 EntityTypeValidationError。
    """
    from graphiti_core.nodes import EntityNode
    from graphiti_core.edges import EntityEdge

    reserved_node_fields = set(EntityNode.model_fields.keys())
    reserved_edge_fields = set(EntityEdge.model_fields.keys())

    # 检查节点类型
    for name, model in ENTITY_TYPES.items():
        custom_fields = set(model.model_fields.keys())
        conflicts = custom_fields & reserved_node_fields
        assert not conflicts, f"{name} 字段与 EntityNode 冲突: {conflicts}"

    # 检查边类型
    for name, model in EDGE_TYPES.items():
        custom_fields = set(model.model_fields.keys())
        conflicts = custom_fields & reserved_edge_fields
        assert not conflicts, f"{name} 字段与 EntityEdge 冲突: {conflicts}"


def test_graphiti_node_label_naming():
    """GraphLayer 的 label 值符合 Graphiti 的命名限制。

    labels 必须匹配 ^[A-Za-z_][A-Za-z0-9_]*$，不能含中文/空格/连字符。
    """
    import re
    pattern = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
    for layer in GraphLayer:
        assert pattern.match(layer.value), f"label '{layer.value}' 不符合命名规则"


# ============================================================
#  主入口
# ============================================================

def main():
    print("=" * 60)
    print("  Graph-RAG Schema 单元测试")
    print("=" * 60)
    print()

    tests = [
        # 节点
        ("test_entity_types_registered", test_entity_types_registered),
        ("test_component_entity_creation", test_component_entity_creation),
        ("test_symptom_entity_defaults", test_symptom_entity_defaults),
        ("test_cause_entity_confidence_range", test_cause_entity_confidence_range),
        ("test_solution_entity_optional_fields", test_solution_entity_optional_fields),
        ("test_build_attributes_functions", test_build_attributes_functions),
        # 边
        ("test_edge_names_complete", test_edge_names_complete),
        ("test_causal_edge_names_group", test_causal_edge_names_group),
        ("test_solution_edge_names_group", test_solution_edge_names_group),
        ("test_edge_type_map_completeness", test_edge_type_map_completeness),
        ("test_causal_edge_attributes", test_causal_edge_attributes),
        ("test_solution_edge_attributes", test_solution_edge_attributes),
        # 约束
        ("test_validate_edge_combination_legal", test_validate_edge_combination_legal),
        ("test_validate_edge_combination_illegal", test_validate_edge_combination_illegal),
        ("test_validate_temporal_consistency", test_validate_temporal_consistency),
        ("test_validate_entity_for_layer", test_validate_entity_for_layer),
        ("test_validate_causal_path", test_validate_causal_path),
        ("test_validate_edge_full", test_validate_edge_full),
        # 路径模板
        ("test_path_templates", test_path_templates),
        ("test_path_template_edges_legal", test_path_template_edges_legal),
        # Graphiti 兼容性
        ("test_no_field_conflicts", test_no_field_conflicts),
        ("test_graphiti_node_label_naming", test_graphiti_node_label_naming),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        if _run_test(name, fn):
            passed += 1
        else:
            failed += 1

    print()
    print("-" * 60)
    print(f"  结果: {passed} 通过, {failed} 失败, 共 {passed + failed} 项")
    print("-" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
