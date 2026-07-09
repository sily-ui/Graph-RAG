"""多题型打分器 —— GraphRAG-Bench §3.1 借鉴。

References
----------
- arXiv:2506.02404 (GraphRAG-Bench, Xiao et al. 2025) §3.1 提出 5 种题型
  (MC / MS / TF / FB / OE)，每种题型有独立的"判分逻辑"。

适配说明
--------
- MC（多选一）：LLM 输出包含"X 是"或"答案是 X"或"X"（独立字符）→ 1.0
- TF（判断）：LLM 输出包含"正确/错误"或"true/false" → 1.0
- FB（填空）：与 OE 相同的 token F1
- OE（开放）：沿用 GraphRAG-Bench §3.3 的 R/AR/EM 计算
- MS（多选多）：未实现（标注成本高，留作未来工作）

输入：TestCase + BaselineResult（包含 answer 字段）
输出：dict[str, float] 题型相关分数 + 与 OE 一致的 R/AR/EM
"""
from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


# MC 选项字母 → 提取顺序（A/B/C/D）
_MC_OPTION_PATTERN = re.compile(r"\b([A-D])\b")
# 故意不放 "是"（"是"在中文里出现太频繁，是/否/对的/错的 都会有 "是"）
_TF_TRUE_PATTERNS = ["正确", "对的", "true", "yes", "✓"]
_TF_FALSE_PATTERNS = ["错误", "错的", "false", "no", "✗", "否", "不是", "不正确"]


def score_mc(answer: str, correct_option: str) -> float:
    """MC 判分：答案里出现 `correct_option` 字母（A/B/C/D）→ 1.0。

    兼容格式：
    - "答案是 A"
    - "A"
    - "选 B"
    - "→ C"
    """
    if not answer or not correct_option:
        return 0.0
    matches = _MC_OPTION_PATTERN.findall(answer)
    return 1.0 if correct_option.upper() in matches else 0.0


def score_tf(answer: str, expected_true: bool) -> float:
    """TF 判分：答案里出现"正确/错误"中的一种 → 1.0。

    expected_true=True → 答"正确"才得分（且不能含"错误"）
    expected_true=False → 答"错误"才得分（且不能含"正确"）
    """
    if not answer:
        return 0.0
    a = answer.lower()
    has_true = any(p.lower() in a for p in _TF_TRUE_PATTERNS)
    has_false = any(p.lower() in a for p in _TF_FALSE_PATTERNS)
    if expected_true:
        # 期望"正确"：出现 false 关键词 → 错（无论是否同时含 true 词）
        if has_false:
            return 0.0
        return 1.0 if has_true else 0.0
    else:
        # 期望"错误"：出现 true 关键词 → 错
        if has_true:
            return 0.0
        return 1.0 if has_false else 0.0


def score_fb(answer: str, gold_token: str) -> float:
    """FB 判分：答案里包含 gold_token → 1.0。"""
    if not answer or not gold_token:
        return 0.0
    return 1.0 if gold_token.lower() in answer.lower() else 0.0


def score_case(case: Any, baseline_result: Any) -> dict[str, float]:
    """根据 case.question_type 分发打分。

    Parameters
    ----------
    case : TestCase
        必含 question_type 和 metadata（如 correct_option/expected_true/gold_token）
    baseline_result : BaselineResult | dict
        含 answer 字段

    Returns
    -------
    dict[str, float]
        至少包含 question_type_score；OE 还会带 r_score/ar_score/em（由 metrics.py 算）
    """
    qtype = getattr(case, "question_type", "OE")
    answer = ""
    if hasattr(baseline_result, "answer"):
        answer = baseline_result.answer or ""
    elif isinstance(baseline_result, dict):
        answer = baseline_result.get("answer", "")
    answer = answer or ""

    meta = getattr(case, "metadata", {}) or {}
    if isinstance(case, dict):
        meta = case.get("metadata", {}) or {}

    if qtype == "MC":
        correct = meta.get("correct_option", "A")
        return {"question_type_score": score_mc(answer, correct)}
    if qtype == "TF":
        expected_true = bool(meta.get("expected_true", True))
        return {"question_type_score": score_tf(answer, expected_true)}
    if qtype == "FB":
        gold = meta.get("gold_token", "")
        return {"question_type_score": score_fb(answer, gold)}
    # OE: 不在这里算（由 metrics.py 的 R/AR/EM 接管）
    return {"question_type_score": 0.0}


# ============================================================
#  MC / TF 变体生成器（从已有 TestCase 派生新 case）
# ============================================================

def build_mc_variants(case: Any, rng: Any = None) -> list[Any]:
    """从一条 OE 路径派生 1 个 MC 变体。

    MC 题面："{symptom} 的根因是？
    A. {correct_cause}
    B. {distractor1}
    C. {distractor2}
    D. {distractor3}"

    distractor 池：图库中其他 Cause/Solution 节点名。
    """
    import random as _random
    rng = rng or _random.Random(42)

    if not case.expected_path:
        return []
    # 取 expected_path[0].target_name 作为"correct"（第一个 cause）
    first_hop = case.expected_path[0]
    correct = first_hop.target_name

    # 收集候选干扰项：所有 expected_path 里出现过的非正确答案
    pool: set[str] = set()
    for h in case.expected_path:
        if h.target_name and h.target_name != correct:
            pool.add(h.target_name)
        if h.source_name and h.source_name != correct:
            pool.add(h.source_name)

    if len(pool) < 3:
        # 干扰项不足 → 用通用 placeholder
        while len(pool) < 3:
            pool.add(f"other_cause_{len(pool)}")
    distractors = rng.sample(sorted(pool), 3)
    options = [correct] + distractors
    rng.shuffle(options)
    correct_idx = options.index(correct)
    option_letter = "ABCD"[correct_idx]

    symptom = ""
    if case.expected_path and case.expected_path[0].source_name:
        symptom = case.expected_path[0].source_name

    question = (
        f"{symptom} 的根因是？\n"
        f"A. {options[0]}\n"
        f"B. {options[1]}\n"
        f"C. {options[2]}\n"
        f"D. {options[3]}"
    )

    from copy import deepcopy
    new_case = deepcopy(case)
    new_case.query = question
    new_case.question_type = "MC"
    new_case.metadata = dict(new_case.metadata or {})
    new_case.metadata["correct_option"] = option_letter
    new_case.metadata["options"] = options
    new_case.metadata["original_question_type"] = "OE"
    return [new_case]


def build_tf_variants(case: Any, rng: Any = None) -> list[Any]:
    """从一条 OE 路径派生 1 个 TF 变体（判断"陈述是否正确"）。

    命题策略：以 expected_path[0] 边名为陈述（"X 由 Y 引起"），correct=True。
    """
    import random as _random
    rng = rng or _random.Random(43)

    if not case.expected_path:
        return []
    h0 = case.expected_path[0]
    statement = f"'{h0.source_name}' 由 '{h0.target_name}' 引起（关系：{h0.edge_name}）"

    from copy import deepcopy
    new_case = deepcopy(case)
    new_case.query = f"判断以下陈述是否正确：{statement}"
    new_case.question_type = "TF"
    new_case.metadata = dict(new_case.metadata or {})
    new_case.metadata["expected_true"] = True
    new_case.metadata["statement"] = statement
    new_case.metadata["original_question_type"] = "OE"
    return [new_case]
