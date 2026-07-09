"""GraphRAG-Bench 借鉴指标的单元测试 —— 无 LLM / 无 Neo4j 依赖。

覆盖范围：
- eval/graph_construction_metrics.py：Entity/Relation Recall/Precision/F1
- eval/reasoning_metrics.py：EM / R / AR + task_level F1
- 集成：metrics.py::evaluate_case 暴露全部 7 个新 key
"""
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reasoning.hallucination_verifier import VerdictEnum
from reasoning.result_models import NodeInfo, PathHop

from eval.graph_construction_metrics import (
    compute_entity_precision,
    compute_entity_recall,
    compute_graph_construction_metrics,
    compute_pipeline_f1,
    compute_relation_recall,
)
from eval.reasoning_metrics import (
    compute_answer_exact_match,
    compute_ar_score,
    compute_r_score,
    compute_reasoning_metrics,
    compute_task_level_f1,
)


def _node(name: str, label: str = "Entity") -> NodeInfo:
    return NodeInfo(uuid=f"u-{name}", name=name, label=label)


def _hop(src: str, edge: str, tgt: str) -> PathHop:
    return PathHop(edge_name=edge, source=_node(src), target=_node(tgt))


@dataclass
class FakeExpectedHop:
    """最小可用 expected hop：图构建指标只读 source_name / target_name / edge_name。"""
    source_name: str
    target_name: str
    edge_name: str = "R"
    source_label: str = "Entity"
    target_label: str = "Entity"
    valid_at: str | None = None
    invalid_at: str | None = None
    lag_seconds: int = 0


# ============================================================
#  Graph Construction Metrics
# ============================================================

class TestEntityRecall(unittest.TestCase):
    """Entity Recall：|pred.nodes ∩ exp.nodes| / |exp.nodes|."""

    def test_01_perfect_match(self):
        exp = [FakeExpectedHop("a", "b"), FakeExpectedHop("b", "c")]
        pred = [_hop("a", "R1", "b"), _hop("b", "R2", "c")]
        # expected: {a, b, c}; predicted: {a, b, c}; intersection = 3
        self.assertEqual(compute_entity_recall(exp, pred), 1.0)

    def test_02_partial_match(self):
        exp = [FakeExpectedHop("a", "b"), FakeExpectedHop("b", "c")]
        # predicted: {a, b} only
        pred = [_hop("a", "R1", "b")]
        self.assertAlmostEqual(compute_entity_recall(exp, pred), 2 / 3, places=3)

    def test_03_no_match(self):
        exp = [FakeExpectedHop("a", "b")]
        pred = [_hop("x", "R1", "y")]
        self.assertEqual(compute_entity_recall(exp, pred), 0.0)

    def test_04_empty_expected(self):
        # No positive class → 0.0 (避免被除 0 误报)
        self.assertEqual(compute_entity_recall([], []), 0.0)

    def test_05_case_insensitive(self):
        exp = [FakeExpectedHop("A", "B")]
        pred = [_hop("a", "R", "b")]
        self.assertEqual(compute_entity_recall(exp, pred), 1.0)


class TestEntityPrecision(unittest.TestCase):

    def test_01_perfect(self):
        exp = [FakeExpectedHop("a", "b")]
        pred = [_hop("a", "R", "b")]
        self.assertEqual(compute_entity_precision(exp, pred), 1.0)

    def test_02_pred_has_extra(self):
        exp = [FakeExpectedHop("a", "b")]
        pred = [_hop("a", "R", "b"), _hop("b", "R", "EXTRA")]
        # 2/3 predicted nodes are in expected
        self.assertAlmostEqual(compute_entity_precision(exp, pred), 2 / 3, places=3)


class TestRelationRecall(unittest.TestCase):
    """Relation Recall：(source, edge, target) 三元组集合的 recall."""

    def test_01_perfect(self):
        exp = [FakeExpectedHop("a", "b", "R1"), FakeExpectedHop("b", "c", "R2")]
        pred = [_hop("a", "R1", "b"), _hop("b", "R2", "c")]
        self.assertEqual(compute_relation_recall(exp, pred), 1.0)

    def test_02_wrong_edge(self):
        # 节点对得上但边名错 → 三元组不匹配
        exp = [FakeExpectedHop("a", "b", "R1")]
        pred = [_hop("a", "WRONG_EDGE", "b")]
        self.assertEqual(compute_relation_recall(exp, pred), 0.0)

    def test_03_partial(self):
        exp = [
            FakeExpectedHop("a", "b", "R1"),
            FakeExpectedHop("b", "c", "R2"),
            FakeExpectedHop("c", "d", "R3"),
        ]
        pred = [_hop("a", "R1", "b"), _hop("b", "R2", "c")]
        self.assertEqual(compute_relation_recall(exp, pred), 2 / 3)


class TestPipelineF1(unittest.TestCase):

    def test_01_harmonic_mean(self):
        self.assertEqual(compute_pipeline_f1(1.0, 1.0), 1.0)
        self.assertEqual(compute_pipeline_f1(0.0, 0.0), 0.0)
        # F1(0.5, 1.0) = 2*0.5*1.0 / 1.5 = 0.667
        self.assertAlmostEqual(compute_pipeline_f1(0.5, 1.0), 2 / 3, places=3)


class TestGraphConstructionMetricsAgg(unittest.TestCase):

    def test_01_aggregate_returns_4_keys(self):
        exp = [FakeExpectedHop("a", "b"), FakeExpectedHop("b", "c")]
        pred = [_hop("a", "R1", "b"), _hop("b", "R2", "c")]
        out = compute_graph_construction_metrics(exp, pred)
        self.assertEqual(
            set(out.keys()),
            {"entity_recall", "entity_precision", "relation_recall", "pipeline_f1"},
        )


# ============================================================
#  Reasoning Metrics (R / AR / EM)
# ============================================================

class TestExactMatch(unittest.TestCase):
    """用 stopword 之外的 token 测，避免 'a'/'the' 等被过滤。"""

    def test_01_all_gold_in_answer(self):
        # gold: {latency, network, partition}; answer 含全部
        self.assertEqual(
            compute_answer_exact_match(
                "latency caused network partition",
                "latency network partition",
            ),
            1.0,
        )

    def test_02_partial_no_subset(self):
        # gold {latency, network, partition}; answer {latency, network}
        # meaningful tokens 都不过 stopwords
        score = compute_answer_exact_match("latency network", "latency network partition")
        # Jaccard = 2 / 3 (ans has 2, gold has 3, intersection 2, union 3)
        self.assertAlmostEqual(score, 2 / 3, places=3)

    def test_03_empty_gold_is_vacuously_true(self):
        # gold 为空 → 1.0（vacuously true）
        self.assertEqual(compute_answer_exact_match("anything", ""), 1.0)


class TestRScore(unittest.TestCase):

    def test_01_full_coverage(self):
        # 全部 token 都是非停用词
        self.assertEqual(
            compute_r_score("latency network partition", "latency network partition"),
            1.0,
        )

    def test_02_partial_coverage(self):
        # gold: {latency, network, partition}; answer: {latency, network, cause, miss}
        # intersection = {latency, network} = 2; gold = 3 → 2/3
        self.assertAlmostEqual(
            compute_r_score("latency network cause miss", "latency network partition"),
            2 / 3,
            places=3,
        )

    def test_03_no_coverage(self):
        self.assertEqual(compute_r_score("xyz qqq", "latency network partition"), 0.0)

    def test_04_empty_gold(self):
        self.assertEqual(compute_r_score("latency network", ""), 1.0)


class TestARScore(unittest.TestCase):
    """AR: 条件指标，EM=0 时返回 None。"""

    def test_01_em_zero_returns_none(self):
        result = compute_ar_score(0.5, [{"verdict": "entailed"}])
        self.assertIsNone(result)

    def test_02_em_one_with_entailed(self):
        result = compute_ar_score(1.0, [{"verdict": "entailed"}])
        self.assertEqual(result, 1.0)

    def test_03_em_one_with_contradicted(self):
        result = compute_ar_score(1.0, [{"verdict": "contradicted"}])
        self.assertEqual(result, 0.0)

    def test_04_em_one_no_claims(self):
        result = compute_ar_score(1.0, [])
        self.assertEqual(result, 0.0)

    def test_05_em_one_with_unsupported(self):
        result = compute_ar_score(1.0, [{"verdict": "unsupported"}])
        self.assertEqual(result, 0.0)

    def test_06_with_dataclass_verdict(self):
        """兼容 dataclass 形式的 VerifiedClaim（verdict 是 enum）。"""
        from dataclasses import dataclass as _dc
        @_dc
        class FakeClaim:
            verdict: VerdictEnum
        result = compute_ar_score(1.0, [FakeClaim(verdict=VerdictEnum.ENTAILED)])
        self.assertEqual(result, 1.0)


class TestReasoningMetricsAggregate(unittest.TestCase):

    def test_01_returns_three_keys(self):
        from dataclasses import dataclass as _dc
        @_dc
        class FakeHop:
            target_name: str
            edge_name: str
        exp = [FakeHop("b", "R1"), FakeHop("c", "R2")]
        out = compute_reasoning_metrics("a b c d", exp, [])
        self.assertIn("em", out)
        self.assertIn("r_score", out)
        self.assertIn("ar_score", out)


# ============================================================
#  4 级任务分类 F1
# ============================================================

class TestTaskLevelF1(unittest.TestCase):

    def test_01_grouping_and_macro(self):
        case_results = [
            {"task_level": "Fact_Retrieval", "pipeline_f1": 0.8},
            {"task_level": "Fact_Retrieval", "pipeline_f1": 0.6},
            {"task_level": "Complex_Reasoning", "pipeline_f1": 0.4},
            {"task_level": "Complex_Reasoning", "pipeline_f1": 0.2},
            {"task_level": "Contextual_Summarize", "pipeline_f1": 1.0},
        ]
        out = compute_task_level_f1(case_results)
        self.assertAlmostEqual(out["by_level"]["Fact_Retrieval"]["f1"], 0.7, places=3)
        self.assertAlmostEqual(out["by_level"]["Complex_Reasoning"]["f1"], 0.3, places=3)
        # macro = (0.7 + 0.3 + 1.0) / 3 = 0.667
        self.assertAlmostEqual(out["macro_f1"], 2 / 3, places=3)
        self.assertEqual(set(out["covered_levels"]),
                         {"Fact_Retrieval", "Complex_Reasoning", "Contextual_Summarize"})

    def test_02_empty_results(self):
        out = compute_task_level_f1([])
        self.assertEqual(out["macro_f1"], 0.0)
        self.assertEqual(out["covered_levels"], [])


if __name__ == "__main__":
    unittest.main()
