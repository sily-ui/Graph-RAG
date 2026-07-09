"""端到端集成测试 —— GraphRAG-Bench 借鉴指标的 metrics.py 集成。

验证：
- evaluate_case() 输出 dict 包含 7 个新 key
- aggregate_metrics() 把新指标汇总到 PerHopStats
- report_to_markdown() 把新指标渲染到表格
"""
import sys
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reasoning.result_models import NodeInfo, PathHop

from eval.metrics import (
    aggregate_metrics,
    evaluate_case,
    report_to_markdown,
)


@dataclass
class FakeCase:
    case_id: str
    hop_count: int
    domain: str
    expected_path: list
    supporting_facts: list
    query_time: str
    ground_truth_free_text: str
    metadata: dict = field(default_factory=dict)
    question_type: str = "OE"
    task_level: str = "Complex_Reasoning"
    answer: str = ""


def _node(name: str) -> NodeInfo:
    return NodeInfo(uuid=f"u-{name}", name=name, label="Entity")


def _hop(src: str, edge: str, tgt: str) -> PathHop:
    return PathHop(edge_name=edge, source=_node(src), target=_node(tgt))


class TestEvaluateCaseIntegratesNewMetrics(unittest.TestCase):
    """evaluate_case 应返回 7 个新 key."""

    NEW_KEYS = [
        "entity_recall", "entity_precision", "relation_recall", "pipeline_f1",
        "r_score", "ar_score", "em",
    ]

    def setUp(self):
        # 构造最小可用 expected_path：2 跳 (a, b) + (b, c)
        # 直接用 type() 避免 import 循环
        self.case = FakeCase(
            case_id="test_001",
            hop_count=2,
            domain="graph",
            expected_path=[
                type("H", (), {
                    "source_name": "a", "target_name": "b",
                    "edge_name": "R1", "source_label": "Entity",
                    "target_label": "Entity", "valid_at": None,
                    "invalid_at": None, "lag_seconds": 0,
                })(),
                type("H", (), {
                    "source_name": "b", "target_name": "c",
                    "edge_name": "R2", "source_label": "Entity",
                    "target_label": "Entity", "valid_at": None,
                    "invalid_at": None, "lag_seconds": 0,
                })(),
            ],
            supporting_facts=[],
            query_time="2026-01-01T00:00:00",
            ground_truth_free_text="a b c",
            answer="a caused b which led to c",
        )
        self.predicted = [
            _hop("a", "R1", "b"),
            _hop("b", "R2", "c"),
        ]

    def test_01_all_new_keys_present(self):
        out = evaluate_case(
            case=self.case,
            predicted_hops=self.predicted,
            verified_claims=[],
        )
        for k in self.NEW_KEYS:
            self.assertIn(k, out, f"key {k!r} 应在 evaluate_case 输出中")
            self.assertIsInstance(out[k], float, f"{k} 应是 float")

    def test_02_perfect_match_yields_high_pipeline_f1(self):
        out = evaluate_case(
            case=self.case,
            predicted_hops=self.predicted,
            verified_claims=[],
        )
        # 完全匹配 → Pipeline F1 应为 1.0
        self.assertAlmostEqual(out["pipeline_f1"], 1.0, places=3)
        self.assertAlmostEqual(out["entity_recall"], 1.0, places=3)
        self.assertAlmostEqual(out["relation_recall"], 1.0, places=3)

    def test_03_r_score_with_gold_tokens(self):
        out = evaluate_case(
            case=self.case,
            predicted_hops=self.predicted,
            verified_claims=[],
        )
        # answer 包含 b/c，但 gold 含 b/c → R 应是 1.0
        self.assertGreaterEqual(out["r_score"], 0.5)

    def test_04_ar_with_no_claims_is_zero(self):
        out = evaluate_case(
            case=self.case,
            predicted_hops=self.predicted,
            verified_claims=[],
        )
        # 无核验 claim → AR = 0.0 (不返回 None 时填 0)
        self.assertEqual(out["ar_score"], 0.0)


class TestAggregateMetricsIncludesNewFields(unittest.TestCase):

    def setUp(self):
        self.case_results = [
            {
                "case_id": f"c{i:03d}",
                "hop_count": 2,
                "domain": "graph",
                "path_error_rate": 0.5,
                "hallucination_rate_overall": 0.0,
                "hallucination_rate_per_hop": 0.0,
                "recall": 0.5,
                "precision": 0.5,
                "temporal_accuracy": 1.0,
                "provenance_completeness": 1.0,
                "entity_recall": 0.8,
                "entity_precision": 0.6,
                "relation_recall": 0.7,
                "pipeline_f1": 0.7,
                "r_score": 0.6,
                "ar_score": 0.5,
                "em": 0.5,
            }
            for i in range(5)
        ]

    def test_01_overall_has_new_fields(self):
        report = aggregate_metrics("B4_Test", self.case_results)
        # 验证 7 个新字段都存在
        for f in ["entity_recall", "entity_precision", "relation_recall",
                  "pipeline_f1", "r_score", "ar_score", "em"]:
            self.assertTrue(
                hasattr(report.overall, f),
                f"PerHopStats 应有字段 {f!r}",
            )

    def test_02_per_hop_has_new_fields(self):
        report = aggregate_metrics("B4_Test", self.case_results)
        for h, stats in report.per_hop.items():
            for f in ["entity_recall", "entity_precision", "relation_recall",
                      "pipeline_f1", "r_score", "ar_score", "em"]:
                self.assertTrue(hasattr(stats, f), f"hop {h} 缺字段 {f}")


class TestReportToMarkdownIncludesNewColumns(unittest.TestCase):

    def test_01_markdown_has_new_columns(self):
        from eval.metrics import MetricsReport, PerHopStats
        report = MetricsReport(baseline_name="B4_Test")
        s = PerHopStats(
            hop_count=2,
            case_count=10,
            path_error_rate=0.5,
            hallucination_rate_overall=0.0,
            hallucination_rate_per_hop=0.0,
            recall=0.5,
            precision=0.5,
            temporal_accuracy=1.0,
            provenance_completeness=1.0,
            entity_recall=0.8,
            entity_precision=0.6,
            relation_recall=0.7,
            pipeline_f1=0.7,
            r_score=0.6,
            ar_score=0.5,
            em=0.5,
        )
        report.per_hop[2] = s
        report.overall = s
        md = report_to_markdown(report)
        # 验证 markdown 里出现 4 个图构建列名 + 3 个推理列名
        for col in ["EntityR", "EntityP", "RelR", "PipeF1", "R↑", "AR↑", "EM↑"]:
            self.assertIn(col, md, f"markdown 缺列 {col!r}")


if __name__ == "__main__":
    unittest.main()
