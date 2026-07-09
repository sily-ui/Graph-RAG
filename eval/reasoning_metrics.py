"""推理质量指标 —— GraphRAG-Bench §3.3 借鉴 (R/AR) + When-to-Use-Graphs 4 级任务分类。

References
----------
- arXiv:2506.02404 (GraphRAG-Bench, Xiao et al. 2025) §3.3 "Holistic evaluation
  framework" 提出 **Reasoning score (R)** 和 **Accurate Reasoning score (AR)**：
  - R：预测答案的"推理过程"与 gold rationale 的语义一致性
  - AR：答案正确时，推理过程是否也正确（条件指标）

- arXiv:2506.05690 (When to Use Graphs in RAG, Xiang et al. 2025) Table 1
  提出 4 级任务分类（Fact Retrieval / Complex Reasoning / Contextual Summarize
  / Creative Generation），本模块在 compute_task_level_f1() 里按 task_level
  分组计算 macro F1。

适配说明
--------
- GraphRAG-Bench 用 ROUGE-L 算 R，本项目用**轻量 token F1**（无外部依赖，
  可在 150×4 case 评估里 1s 内算完，避免拖慢评测）。
- EM 用"gold tokens 全部出现在 answer"判定，宽松匹配（Jaccard 兜底）。
- AR 仅在 EM=1 时计算；EM=0 时返回 None，不计入宏平均。
"""
from __future__ import annotations

import logging
import re
from typing import Any, Iterable

logger = logging.getLogger(__name__)


# ============================================================
#  工具：文本 → token 集合
# ============================================================

# 简化版中文停用词（避免与"是/了/的"等通用词误判）
_STOPWORDS_ZH = {
    "是", "的", "了", "和", "与", "或", "在", "是", "有", "无",
    "为", "与", "及", "等", "中", "上", "下", "不", "也", "都",
    "我", "你", "他", "她", "它", "我们", "你们", "他们",
    "这", "那", "这个", "那个", "这些", "那些",
    "会", "可", "要", "能", "请", "问", "答", "回",
}
_STOPWORDS_EN = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "to", "of", "in", "on", "at", "for", "and", "or", "but",
    "this", "that", "these", "those", "it", "its",
}


def _tokenize(text: str) -> list[str]:
    """简单分词：中文按字符 + 英文按单词。

    GraphRAG-Bench 用 ROUGE-L 需要 NLTK 等外部依赖；本实现只用 Python 内置，
    在 150 case 评估上速度可接受。
    """
    if not text:
        return []
    # 英文 / 数字 / 下划线 → 单词
    en_tokens = re.findall(r"[A-Za-z0-9_\-]+", text)
    # 中文字符 → 单字
    zh_tokens = re.findall(r"[\u4e00-\u9fff]", text)
    return en_tokens + zh_tokens


def _meaningful_tokens(text: str) -> set[str]:
    """过滤停用词后的 token 集合（小写）。"""
    tokens = _tokenize(text)
    out: set[str] = set()
    for t in tokens:
        low = t.lower()
        if not low:
            continue
        if low in _STOPWORDS_EN or low in _STOPWORDS_ZH:
            continue
        out.add(low)
    return out


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ============================================================
#  R / AR / EM 三个核心指标
# ============================================================

def compute_answer_exact_match(answer: str, gold_text: str) -> float:
    """EM: gold tokens 全部出现在 answer → 1.0；否则 Jaccard。

    GraphRAG-Bench §3.3 形式：
        EM = 1 if all gold tokens in answer else 0
    本项目放宽为 Jaccard 兜底（避免单 token 缺失导致 EM=0 噪声大）。
    """
    ans_tokens = _meaningful_tokens(answer)
    gold_tokens = _meaningful_tokens(gold_text)
    if not gold_tokens:
        return 1.0 if not ans_tokens else 0.0
    if gold_tokens.issubset(ans_tokens):
        return 1.0
    return _jaccard(ans_tokens, gold_tokens)


def compute_r_score(answer: str, gold_rationale: str) -> float:
    """R: |answer_tokens ∩ gold_rationale_tokens| / |gold_rationale_tokens|.

    衡量"预测答案覆盖了多少 gold rationale 的关键词"。
    当 gold_rationale 为空时返回 1.0（vacuously true）。
    """
    ans_tokens = _meaningful_tokens(answer)
    gold_tokens = _meaningful_tokens(gold_rationale)
    if not gold_tokens:
        return 1.0
    return len(ans_tokens & gold_tokens) / len(gold_tokens)


def compute_ar_score(em: float, verified_claims: Iterable) -> float | None:
    """AR: 当 EM=1 时，核验结果是否含 ≥1 ENTAILED 且 0 CONTRADICTED。

    Returns
    -------
    - 1.0: 答对且推理也对
    - 0.0: 答对但推理错
    - None: EM=0 时未定义（不计入宏平均）

    注意：verified_claims 可为 [VerifiedClaim] 或 [dict]，
    兼容两种 schema。
    """
    if em < 1.0:
        return None

    has_entailed = False
    has_contradicted = False
    for c in verified_claims or []:
        # 兼容 dataclass 和 dict
        if isinstance(c, dict):
            verdict = c.get("verdict")
        else:
            verdict = getattr(c, "verdict", None)
        v = getattr(verdict, "value", verdict)
        if v == "entailed":
            has_entailed = True
        elif v == "contradicted":
            has_contradicted = True
    if has_contradicted:
        return 0.0
    return 1.0 if has_entailed else 0.0


# ============================================================
#  4 级任务分类 F1 (arXiv:2506.05690 Table 1)
# ============================================================

def compute_task_level_f1(
    case_results: list[dict],
    f1_key: str = "pipeline_f1",
) -> dict[str, Any]:
    """按 task_level 分组计算 macro F1。

    Parameters
    ----------
    case_results : list[dict]
        每条 case 的评估结果（来自 evaluate_case）
    f1_key : str
        用于分组的 F1 字段名（默认 "pipeline_f1"）

    Returns
    -------
    dict
        {
            "by_level": {level: {"count": N, "f1": float}, ...},
            "macro_f1": float,
            "covered_levels": [str, ...],
        }
    """
    by_level: dict[str, list[float]] = {}
    for r in case_results:
        # task_level 可能在 case.metadata 里，也可能在 case 本身
        level = r.get("task_level") or "Complex_Reasoning"
        score = float(r.get(f1_key, 0.0))
        by_level.setdefault(level, []).append(score)

    by_level_stats: dict[str, dict[str, float]] = {}
    for level, scores in sorted(by_level.items()):
        by_level_stats[level] = {
            "count": len(scores),
            "f1": sum(scores) / len(scores) if scores else 0.0,
        }
    levels = list(by_level_stats.keys())
    macro_f1 = (
        sum(s["f1"] for s in by_level_stats.values()) / len(by_level_stats)
        if by_level_stats else 0.0
    )
    return {
        "by_level": by_level_stats,
        "macro_f1": macro_f1,
        "covered_levels": levels,
    }


# ============================================================
#  聚合入口
# ============================================================

def compute_reasoning_metrics(
    answer: str,
    expected_path: list,
    verified_claims: Iterable,
) -> dict[str, float | None]:
    """一次性返回 R / EM / AR 三项推理质量指标。

    Gold rationale 由 expected_path 拼成：
        " ".join(h.target_name for h in expected_path) + " " +
        " ".join(h.edge_name for h in expected_path)

    Returns
    -------
    {"em", "r_score", "ar_score"}  其中 ar_score 可能为 None
    """
    gold_rationale = " ".join(
        [h.target_name for h in (expected_path or [])] +
        [h.edge_name for h in (expected_path or [])]
    )
    em = compute_answer_exact_match(answer or "", gold_rationale)
    r = compute_r_score(answer or "", gold_rationale)
    ar = compute_ar_score(em, verified_claims)
    return {"em": em, "r_score": r, "ar_score": ar}
