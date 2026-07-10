"""_llm_explain 的 prompt 内容断言测试。

验证改 prompt 后：
1. system prompt 含"原样使用节点名/边名"指令
2. user content 含"必须原样出现在回答中的节点名/边名"清单
3. 节点名/边名清单里确实列出了 paths 里的节点和边

不实际调用 LLM，全部 mock。
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from reasoning.llm_interpreter import LLMInterpreter  # noqa: E402
from reasoning.query_types import QueryIntent, QueryType, StructuredQuery  # noqa: E402
from reasoning.result_models import CausalPath, NodeInfo, PathHop  # noqa: E402


def _make_node(name: str, label: str) -> NodeInfo:
    return NodeInfo(uuid=f"u-{name}", name=name, label=label)


def _make_hop(src: str, edge: str, tgt: str, src_label="Symptom", tgt_label="Cause") -> PathHop:
    return PathHop(
        edge_name=edge,
        source=_make_node(src, src_label),
        target=_make_node(tgt, tgt_label),
    )


def _make_client_capturing_messages():
    """构造 LLMInterpreter，其 client.chat 记录传入的 messages 并返回固定字符串。"""
    interpreter = LLMInterpreter.__new__(LLMInterpreter)
    captured: dict = {}

    def fake_chat(messages, temperature=0.3, max_tokens=800):
        captured["messages"] = messages
        captured["temperature"] = temperature
        captured["max_tokens"] = max_tokens
        return "因果链：cpu_spike -[CAUSED_BY]-> resource_contention -[RESOLVED_BY]-> scale_up"

    client = MagicMock()
    client.chat = fake_chat
    client.model = "test-model"
    interpreter.client = client
    return interpreter, captured


def _make_query() -> StructuredQuery:
    return StructuredQuery(
        natural_language="cpu_spike 异常的根因链路",
        intent=QueryIntent(query_type=QueryType.CAUSAL_CHAIN, target_entity="cpu_spike"),
    )


def _make_paths() -> list[CausalPath]:
    """2 跳路径：cpu_spike -[CAUSED_BY]-> resource_contention -[RESOLVED_BY]-> scale_up"""
    path = CausalPath(
        hops=[
            _make_hop("cpu_spike", "CAUSED_BY", "resource_contention",
                      src_label="Symptom", tgt_label="Cause"),
            _make_hop("resource_contention", "RESOLVED_BY", "scale_up",
                      src_label="Cause", tgt_label="Solution"),
        ],
        path_confidence=0.85,
    )
    return [path]


# ============================================================
#  1. system prompt 含"原样使用"指令
# ============================================================
def test_system_prompt_requires_verbatim_node_names():
    interpreter, captured = _make_client_capturing_messages()
    interpreter.explain(_make_query(), _make_paths(), [])

    system_msg = captured["messages"][0]["content"]
    assert "原样使用" in system_msg, "system prompt 应含'原样使用'指令"
    assert "不得意译" in system_msg or "不得改写" in system_msg


# ============================================================
#  2. system prompt 要求"因果链："行
# ============================================================
def test_system_prompt_requires_chain_line():
    interpreter, captured = _make_client_capturing_messages()
    interpreter.explain(_make_query(), _make_paths(), [])

    system_msg = captured["messages"][0]["content"]
    assert "因果链" in system_msg, "system prompt 应要求输出'因果链：'行"


# ============================================================
#  3. user content 含节点名/边名清单
# ============================================================
def test_user_content_lists_must_include_tokens():
    interpreter, captured = _make_client_capturing_messages()
    interpreter.explain(_make_query(), _make_paths(), [])

    user_msg = captured["messages"][1]["content"]
    # 节点名清单里应出现路径上所有节点
    for node_name in ["cpu_spike", "resource_contention", "scale_up"]:
        assert node_name in user_msg, f"user content 应列出节点名 {node_name}"
    # 边名清单里应出现所有边
    for edge_name in ["CAUSED_BY", "RESOLVED_BY"]:
        assert edge_name in user_msg, f"user content 应列出边名 {edge_name}"
    # 应有"必须原样出现在回答中"的清单标签
    assert "必须原样出现在回答中的节点名" in user_msg
    assert "必须原样出现在回答中的边名" in user_msg


# ============================================================
#  4. 调用参数：temperature=0.3, max_tokens=800
# ============================================================
def test_chat_call_params_preserved():
    interpreter, captured = _make_client_capturing_messages()
    interpreter.explain(_make_query(), _make_paths(), [])

    assert captured["temperature"] == 0.3
    assert captured["max_tokens"] == 800


# ============================================================
#  5. 规则兜底 _rule_based_explain 也输出"因果链："行
# ============================================================
def test_rule_based_explain_includes_chain_line():
    interpreter = LLMInterpreter(client=None)  # 降级模式
    answer = interpreter.explain(_make_query(), _make_paths(), [])

    assert "因果链：" in answer, "规则兜底答案应含'因果链：'行"
    # 因果链行应包含所有节点名和边名
    for token in ["cpu_spike", "CAUSED_BY", "resource_contention",
                  "RESOLVED_BY", "scale_up"]:
        assert token in answer, f"规则兜底答案应含 token {token}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
