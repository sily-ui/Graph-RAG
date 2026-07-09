"""claim_decomposer 的 markdown JSON 解析单元测试。

覆盖 LLM 输出可能出现的所有格式：
1. 裸 JSON
2. ```json ... ``` 包裹
3. ``` ... ``` 包裹（无语言标识）
4. 前后混入解释文字
5. BOM 字符
6. 嵌套大括号内的真实 JSON
7. 单条 claim / 多条 claim
8. 异常格式 → 抛 ValueError
"""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reasoning.claim_decomposer import _parse_llm_json_response


class TestParseLLMJsonResponse(unittest.TestCase):
    """测试 _parse_llm_json_response 在各种 LLM 输出格式下的鲁棒性。"""

    def test_01_plain_json(self):
        """裸 JSON。"""
        text = '{"claims": [{"claim_text": "cpu_spike 引发 resource_contention", "hop_index": 0}]}'
        result = _parse_llm_json_response(text)
        self.assertIn("claims", result)
        self.assertEqual(len(result["claims"]), 1)
        self.assertEqual(result["claims"][0]["hop_index"], 0)

    def test_02_markdown_json_block(self):
        """```json\n{...}\n``` 包裹。"""
        text = '```json\n{"claims": [{"claim_text": "test", "hop_index": 1}]}\n```'
        result = _parse_llm_json_response(text)
        self.assertEqual(len(result["claims"]), 1)
        self.assertEqual(result["claims"][0]["hop_index"], 1)

    def test_03_markdown_plain_block(self):
        """无语言标识的代码块。"""
        text = '```\n{"claims": [{"claim_text": "memory_leak", "hop_index": -1}]}\n```'
        result = _parse_llm_json_response(text)
        self.assertEqual(result["claims"][0]["hop_index"], -1)

    def test_04_text_before_and_after(self):
        """前后混入解释文字。"""
        text = '''好的，这是拆解结果：
        {"claims": [{"claim_text": "x", "hop_index": 0, "source_nodes": ["n1"]}]}
        希望对您有帮助！'''
        result = _parse_llm_json_response(text)
        self.assertEqual(result["claims"][0]["source_nodes"], ["n1"])

    def test_05_bom_character(self):
        """带 BOM 字符。"""
        text = '\ufeff{"claims": [{"claim_text": "bom-test", "hop_index": 0}]}'
        result = _parse_llm_json_response(text)
        self.assertEqual(result["claims"][0]["claim_text"], "bom-test")

    def test_06_nested_braces_in_strings(self):
        """字符串内含大括号。"""
        text = '{"claims": [{"claim_text": "string has { braces } inside", "hop_index": 2}]}'
        result = _parse_llm_json_response(text)
        self.assertEqual(result["claims"][0]["claim_text"], "string has { braces } inside")

    def test_07_multiple_claims(self):
        """多条 claim。"""
        text = json.dumps({
            "claims": [
                {"claim_text": "claim1", "hop_index": 0, "confidence": 0.9},
                {"claim_text": "claim2", "hop_index": 1, "confidence": 0.8},
                {"claim_text": "claim3", "hop_index": -1, "confidence": 0.7},
            ]
        })
        result = _parse_llm_json_response(text)
        self.assertEqual(len(result["claims"]), 3)
        self.assertEqual([c["confidence"] for c in result["claims"]], [0.9, 0.8, 0.7])

    def test_08_markdown_with_extra_text(self):
        """```json 块前后都有解释文字。"""
        text = '''我帮你拆解一下。

```json
{"claims": [
  {"claim_text": "machine-1-1 出现 cpu_spike", "hop_index": 0, "source_nodes": ["machine-1-1", "cpu_spike"]}
]}
```

以上是拆解结果。'''
        result = _parse_llm_json_response(text)
        self.assertEqual(len(result["claims"]), 1)
        self.assertEqual(result["claims"][0]["source_nodes"], ["machine-1-1", "cpu_spike"])

    def test_09_escaped_quotes_in_strings(self):
        """字符串内含转义双引号。"""
        text = r'{"claims": [{"claim_text": "he said \"hello\" to me", "hop_index": 0}]}'
        result = _parse_llm_json_response(text)
        self.assertEqual(result["claims"][0]["claim_text"], 'he said "hello" to me')

    def test_10_empty_response(self):
        """空响应。"""
        with self.assertRaises(ValueError) as cm:
            _parse_llm_json_response("")
        self.assertIn("空", str(cm.exception))

    def test_11_whitespace_only(self):
        """只有空白。"""
        with self.assertRaises(ValueError):
            _parse_llm_json_response("   \n  \t  ")

    def test_12_invalid_json_raises(self):
        """完全无法解析的垃圾。"""
        with self.assertRaises(ValueError) as cm:
            _parse_llm_json_response("This is not JSON at all, sorry.")
        self.assertIn("无法从 LLM 响应中解析 JSON", str(cm.exception))

    def test_13_unclosed_brace(self):
        """未闭合的 JSON。"""
        with self.assertRaises(ValueError):
            _parse_llm_json_response('{"claims": [{"claim_text": "unclosed')

    def test_14_stepfun_real_format(self):
        """模拟 StepFun 真实输出（带 markdown + 中文 + 嵌套数组）。"""
        text = '''```json
{
  "claims": [
    {
      "claim_text": "machine-1-1 通过 HAS_SYMPTOM 表现出 cpu_spike 症状",
      "hop_index": 0,
      "source_nodes": ["machine-1-1", "cpu_spike"],
      "source_edges": ["HAS_SYMPTOM"],
      "confidence": 0.95
    },
    {
      "claim_text": "cpu_spike 由 resource_contention 引发",
      "hop_index": 1,
      "source_nodes": ["cpu_spike", "resource_contention"],
      "source_edges": ["CAUSED_BY"],
      "confidence": 0.92
    }
  ]
}
```'''
        result = _parse_llm_json_response(text)
        self.assertEqual(len(result["claims"]), 2)
        self.assertEqual(result["claims"][0]["source_edges"], ["HAS_SYMPTOM"])
        self.assertEqual(result["claims"][1]["source_edges"], ["CAUSED_BY"])

    def test_15_deepseek_real_format(self):
        """模拟 DeepSeek 真实输出（裸 JSON 但前后有 'Here is the result:'）。"""
        text = 'Here is the result:\n{"claims": [{"claim_text": "x", "hop_index": -1, "source_nodes": ["n"], "source_edges": ["E"]}]}\nDone.'
        result = _parse_llm_json_response(text)
        self.assertEqual(result["claims"][0]["source_edges"], ["E"])


class TestClaimDecomposer(unittest.TestCase):
    """端到端测试 ClaimDecomposer。"""

    def test_rule_based_fallback(self):
        """无 LLM 客户端时走规则模式。"""
        from reasoning.claim_decomposer import ClaimDecomposer
        from reasoning.result_models import PathHop, CausalPath, NodeInfo

        # 构造假路径
        node_a = NodeInfo(uuid="u1", name="machine-1-1", label="Component")
        node_b = NodeInfo(uuid="u2", name="cpu_spike", label="Symptom")
        node_c = NodeInfo(uuid="u3", name="resource_contention", label="Cause")
        hop1 = PathHop(
            edge_name="HAS_SYMPTOM",
            source=node_a, target=node_b,
            valid_at="2021-01-01T00:00:00+00:00",
            invalid_at=None, lag_seconds=0,
        )
        path = CausalPath(
            path_id="p1",
            start_node=node_a,
            end_node=node_c,
            hops=[hop1],
            path_confidence=0.9,
        )

        decomposer = ClaimDecomposer(client=None)
        answer = "machine-1-1 出现 cpu_spike 症状。根因是 resource_contention。"
        decomp = decomposer.decompose(answer, [path])

        self.assertEqual(decomp.parser, "rule")
        self.assertGreater(len(decomp.claims), 0)
        # 检查 source_nodes 至少有一个被识别
        all_source_nodes = {n for c in decomp.claims for n in c.source_nodes}
        self.assertIn("machine-1-1", all_source_nodes)
        self.assertIn("cpu_spike", all_source_nodes)


if __name__ == "__main__":
    unittest.main(verbosity=2)
