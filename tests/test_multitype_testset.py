"""多题型测试集兼容性测试。

验证：
- 老 JSONL（无 question_type / task_level 字段）能正常 load
- 新增 build_mc_variants / build_tf_variants 至少产生 1 条 MC / 1 条 TF
- question_type_scorer 对每个题型正确打分
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.question_type_scorer import (
    build_mc_variants,
    build_tf_variants,
    score_fb,
    score_mc,
    score_tf,
)
from eval.testset_builder import (
    ExpectedHop,
    SupportingFact,
    TestCase,
    _classify_task_level,
    load_testset,
)


def _make_testcase(
    case_id: str = "x_001",
    question_type: str = "OE",
    task_level: str = "Complex_Reasoning",
    hop_count: int = 3,
    expected_target_names: list[str] | None = None,
) -> TestCase:
    """构造一个最小可用的 TestCase."""
    targets = expected_target_names or ["b", "c", "d"]
    srcs = ["a"] + targets[:-1]
    exp = [
        ExpectedHop(
            edge_name="R1",
            source_name=s,
            source_label="Entity",
            target_name=t,
            target_label="Entity",
        )
        for s, t in zip(srcs, targets)
    ]
    return TestCase(
        case_id=case_id,
        domain="graph",
        hop_count=hop_count,
        query="a 怎么了",
        expected_path=exp,
        supporting_facts=[],
        query_time="2026-01-01T00:00:00",
        ground_truth_free_text="a -> b -> c",
        question_type=question_type,
        task_level=task_level,
    )


class TestBackwardsCompat(unittest.TestCase):
    """老 JSONL（无 question_type 字段）应能正常 load."""

    def test_01_load_existing_testset(self):
        cases = load_testset("eval/testset.jsonl")
        self.assertEqual(len(cases), 150)
        # 第一条不应有 question_type 字段（老格式）
        first = cases[0]
        self.assertIn("case_id", first)
        # 加载后转为 TestCase 时，缺失字段会用默认值
        tc = TestCase(
            case_id=first["case_id"],
            domain=first["domain"],
            hop_count=first["hop_count"],
            query=first["query"],
            expected_path=[ExpectedHop(**h) for h in first["expected_path"]],
            supporting_facts=[SupportingFact(**f) for f in first["supporting_facts"]],
            query_time=first["query_time"],
            ground_truth_free_text=first["ground_truth_free_text"],
            metadata=first.get("metadata", {}),
        )
        # 默认值应自动填上
        self.assertEqual(tc.question_type, "OE")
        self.assertEqual(tc.task_level, "Complex_Reasoning")

    def test_02_classify_task_level(self):
        self.assertEqual(_classify_task_level(2, {"nodes": [1, 2, 3]}), "Fact_Retrieval")
        self.assertEqual(_classify_task_level(3, {"nodes": [1, 2, 3, 4]}), "Complex_Reasoning")
        # 4 跳 5 节点 → Contextual_Summarize
        self.assertEqual(
            _classify_task_level(4, {"nodes": [1, 2, 3, 4, 5]}),
            "Contextual_Summarize",
        )


class TestMCVariantGeneration(unittest.TestCase):

    def test_01_build_mc_variants_returns_testcase(self):
        tc = _make_testcase(hop_count=3)
        mcs = build_mc_variants(tc)
        self.assertEqual(len(mcs), 1)
        mc = mcs[0]
        self.assertEqual(mc.question_type, "MC")
        self.assertIn("correct_option", mc.metadata)
        self.assertIn(mc.metadata["correct_option"], ["A", "B", "C", "D"])
        # 题面应含 4 个选项
        self.assertIn("A.", mc.query)
        self.assertIn("B.", mc.query)
        self.assertIn("C.", mc.query)
        self.assertIn("D.", mc.query)

    def test_02_mc_correct_option_is_in_options(self):
        tc = _make_testcase(hop_count=3)
        mcs = build_mc_variants(tc)
        options = mcs[0].metadata["options"]
        correct = mcs[0].metadata["correct_option"]
        correct_idx = "ABCD".index(correct)
        # 正确选项对应的实体名 = expected_path[0].target_name
        self.assertEqual(options[correct_idx], tc.expected_path[0].target_name)


class TestTFVariantGeneration(unittest.TestCase):

    def test_01_build_tf_variants_returns_testcase(self):
        tc = _make_testcase(hop_count=2)
        tfs = build_tf_variants(tc)
        self.assertEqual(len(tfs), 1)
        tf = tfs[0]
        self.assertEqual(tf.question_type, "TF")
        self.assertIn("expected_true", tf.metadata)
        # 题面应含"判断"或"正确"
        self.assertIn("判断", tf.query)


class TestScorers(unittest.TestCase):
    """3 个题型打分器."""

    def test_01_score_mc(self):
        self.assertEqual(score_mc("答案是 A", "A"), 1.0)
        self.assertEqual(score_mc("A", "A"), 1.0)
        self.assertEqual(score_mc("选 B", "A"), 0.0)
        self.assertEqual(score_mc("", "A"), 0.0)

    def test_02_score_tf_true(self):
        self.assertEqual(score_tf("这是正确的陈述", True), 1.0)
        self.assertEqual(score_tf("这是错误的陈述", True), 0.0)
        self.assertEqual(score_tf("", True), 0.0)

    def test_03_score_tf_false(self):
        self.assertEqual(score_tf("这是错误的陈述", False), 1.0)
        self.assertEqual(score_tf("这是正确的陈述", False), 0.0)

    def test_04_score_fb(self):
        self.assertEqual(score_fb("答案是 network_partition", "network_partition"), 1.0)
        self.assertEqual(score_fb("没有 token", "network_partition"), 0.0)


class TestNewMultitypeFile(unittest.TestCase):
    """验证 eval/testset_multitype.jsonl 文件结构正确."""

    def test_01_file_exists_and_has_30_cases(self):
        path = Path("eval/testset_multitype.jsonl")
        self.assertTrue(path.exists(), "testset_multitype.jsonl 应存在")
        cases = load_testset(str(path))
        self.assertEqual(len(cases), 30)
        # 统计 MC 和 TF
        mc_count = sum(1 for c in cases if c.get("question_type") == "MC")
        tf_count = sum(1 for c in cases if c.get("question_type") == "TF")
        self.assertEqual(mc_count, 15)
        self.assertEqual(tf_count, 15)

    def test_02_mc_cases_have_correct_option(self):
        cases = load_testset("eval/testset_multitype.jsonl")
        for c in cases:
            if c.get("question_type") == "MC":
                meta = c.get("metadata", {})
                self.assertIn("correct_option", meta)
                self.assertIn(meta["correct_option"], ["A", "B", "C", "D"])
                self.assertIn("options", meta)
                self.assertEqual(len(meta["options"]), 4)
            elif c.get("question_type") == "TF":
                meta = c.get("metadata", {})
                self.assertIn("expected_true", meta)
                self.assertIsInstance(meta["expected_true"], bool)


if __name__ == "__main__":
    unittest.main()
