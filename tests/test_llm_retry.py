"""LLMClient.chat 的失败恢复机制单元测试。

测试矩阵：
1. 正常返回 → 一次成功
2. finish_reason='length' + 重试 → max_tokens 翻倍后成功
3. finish_reason='length' + 重试用尽 → 抛 RuntimeError
4. 网络超时（APITimeoutError）→ 退避重试
5. 非网络错误（如 BadRequestError）→ 不重试，直接抛
6. 空响应 + finish_reason='stop' → 抛 RuntimeError
7. 重试时 max_tokens 翻倍上限 8000

所有测试都不实际调用 LLM，全部 mock。
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from reasoning.llm_interpreter import LLMClient  # noqa: E402


def _make_client():
    """构造 LLMClient 实例，绕过 _validate_config 直接赋值属性。"""
    c = LLMClient.__new__(LLMClient)
    c.api_key = "test-key-123"
    c.base_url = "https://api.test.com"
    c.model = "test-model"
    c.timeout = 60
    c._client = MagicMock()
    return c


def _make_response(content: str = "", finish_reason: str = "stop"):
    """构造 OpenAI 风格响应。"""
    r = MagicMock()
    r.choices = [MagicMock()]
    r.choices[0].message.content = content
    r.choices[0].finish_reason = finish_reason
    return r


# ============================================================
#  1. 正常返回
# ============================================================
def test_normal_response_returns_immediately():
    c = _make_client()
    c._client.chat.completions.create.return_value = _make_response(
        "正常结果", finish_reason="stop"
    )

    result = c.chat([{"role": "user", "content": "hi"}], max_tokens=500)

    assert result == "正常结果"
    # 验证 max_tokens=500 被传递
    call_kwargs = c._client.chat.completions.create.call_args.kwargs
    assert call_kwargs["max_tokens"] == 500
    # 验证只调用 1 次（不重试）
    assert c._client.chat.completions.create.call_count == 1


# ============================================================
#  2. length 截断 + 重试后成功
# ============================================================
def test_length_truncation_retries_with_doubled_max_tokens():
    c = _make_client()
    # 第一次：length 截断空内容；第二次：成功
    c._client.chat.completions.create.side_effect = [
        _make_response("", finish_reason="length"),
        _make_response("完整内容", finish_reason="stop"),
    ]

    result = c.chat(
        [{"role": "user", "content": "hi"}],
        max_tokens=1000,
        max_retries=2,
    )

    assert result == "完整内容"
    assert c._client.chat.completions.create.call_count == 2

    # 第二次调用 max_tokens 应该是 2000（翻倍）
    second_call_kwargs = c._client.chat.completions.create.call_args_list[1].kwargs
    assert second_call_kwargs["max_tokens"] == 2000


# ============================================================
#  3. length 截断 + 重试用尽
# ============================================================
def test_length_truncation_exhausts_retries_then_raises():
    c = _make_client()
    # 连续 3 次（首次 + 2 次重试）都是 length 空响应
    c._client.chat.completions.create.side_effect = [
        _make_response("", finish_reason="length"),
        _make_response("", finish_reason="length"),
        _make_response("", finish_reason="length"),
    ]

    with pytest.raises(RuntimeError, match="LLM 返回空内容"):
        c.chat(
            [{"role": "user", "content": "hi"}],
            max_tokens=1000,
            max_retries=2,
        )

    assert c._client.chat.completions.create.call_count == 3


# ============================================================
#  4. 网络超时：APITimeoutError → 退避重试
# ============================================================
def test_network_timeout_retries_with_backoff():
    c = _make_client()
    # 模拟 openai.APITimeoutError
    from openai import APITimeoutError
    c._client.chat.completions.create.side_effect = [
        APITimeoutError("连接超时"),
        _make_response("成功", finish_reason="stop"),
    ]

    result = c.chat(
        [{"role": "user", "content": "hi"}],
        max_tokens=1000,
        max_retries=2,
    )

    assert result == "成功"
    assert c._client.chat.completions.create.call_count == 2


# ============================================================
#  5. 非网络错误（如 BadRequestError）→ 不重试
# ============================================================
def test_non_network_error_does_not_retry():
    c = _make_client()
    from openai import BadRequestError
    c._client.chat.completions.create.side_effect = BadRequestError(
        "invalid api key", response=MagicMock(), body=None
    )

    with pytest.raises(BadRequestError):
        c.chat(
            [{"role": "user", "content": "hi"}],
            max_tokens=1000,
            max_retries=2,
        )

    # 验证只调了 1 次（不重试）
    assert c._client.chat.completions.create.call_count == 1


# ============================================================
#  6. 空响应 + finish_reason='stop' → 抛 RuntimeError（不重试）
# ============================================================
def test_empty_content_with_stop_reason_raises():
    c = _make_client()
    c._client.chat.completions.create.return_value = _make_response(
        "", finish_reason="stop"
    )

    with pytest.raises(RuntimeError, match="LLM 返回空内容"):
        c.chat(
            [{"role": "user", "content": "hi"}],
            max_tokens=1000,
            max_retries=2,
        )

    # 'stop' 不是 length，不重试
    assert c._client.chat.completions.create.call_count == 1


# ============================================================
#  7. max_tokens 翻倍上限 8000
# ============================================================
def test_max_tokens_doubling_capped_at_8000():
    c = _make_client()
    # max_retries=2 → 共 3 次调用机会（attempt=0,1,2）
    # 全部返回 length 空响应
    c._client.chat.completions.create.side_effect = [
        _make_response("", finish_reason="length"),  # attempt=0: 1000 → 2000
        _make_response("", finish_reason="length"),  # attempt=1: 2000 → 4000
        _make_response("", finish_reason="length"),  # attempt=2: 4000 → raise
    ]

    with pytest.raises(RuntimeError, match="LLM 返回空内容"):
        c.chat(
            [{"role": "user", "content": "hi"}],
            max_tokens=1000,
            max_retries=2,
        )

    # 验证 3 次调用的 max_tokens 序列：1000, 2000, 4000
    max_tokens_seq = [
        call.kwargs["max_tokens"]
        for call in c._client.chat.completions.create.call_args_list
    ]
    assert max_tokens_seq == [1000, 2000, 4000]


# ============================================================
#  8. 有内容但 finish_reason='length' → 检查是否完整结尾
# ============================================================
def test_truncated_content_with_complete_ending_accepted():
    c = _make_client()
    # 拿到内容但 length 截断，且以句号结尾 → 接受
    c._client.chat.completions.create.return_value = _make_response(
        "这是完整的一段解释。", finish_reason="length"
    )

    result = c.chat(
        [{"role": "user", "content": "hi"}],
        max_tokens=1000,
        max_retries=2,
    )

    # 内容以句号结尾 → 接受，不重试
    assert result == "这是完整的一段解释。"
    assert c._client.chat.completions.create.call_count == 1


def test_truncated_content_with_incomplete_ending_retries():
    c = _make_client()
    # length 截断，内容看起来不完整
    c._client.chat.completions.create.side_effect = [
        _make_response("不完整的内容", finish_reason="length"),
        _make_response("完整内容。", finish_reason="stop"),
    ]

    result = c.chat(
        [{"role": "user", "content": "hi"}],
        max_tokens=1000,
        max_retries=2,
    )

    assert result == "完整内容。"
    assert c._client.chat.completions.create.call_count == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
