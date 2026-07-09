"""六项评估指标 —— Graph-RAG 实验评估核心。

六项指标（按方案定义）：
1. PathErrorRate: 逐跳对齐，缺失/多余均计错误
2. HallucinationRate: 整体 + 逐跳，原子声明是否被路径蕴含
3. Recall: |predicted ∩ ground_truth| / |ground_truth|
4. Precision: |predicted ∩ ground_truth| / |predicted|
5. TemporalAccuracy: 预测路径所有边 valid_at ≤ query_time ≤ invalid_at 的比例
6. ProvenanceCompleteness: 有 episode 的预测边占比

所有指标都支持按 2/3/4 跳分别统计（PerHop + Overall）。

输入约定：
- expected_path: list[ExpectedHop]（来自 testset_builder）
- predicted_path: list[PathHop]（来自 reasoning.path_extractor）
- predicted_claims: list[VerifiedClaim]（来自 hallucination_verifier）
- query_time: datetime

新增（GraphRAG-Bench 借鉴，arXiv:2506.02404 §3.2）：
7. EntityRecall / EntityPrecision：节点集合匹配度
8. RelationRecall：(source, edge, target) 三元组匹配度
9. PipelineF1：Entity 的 R/P 调和均值
10. R / AR / EM：推理质量（见 eval/reasoning_metrics.py，arXiv:2506.02404 §3.3）
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from reasoning.result_models import CausalPath, PathHop
from reasoning.hallucination_verifier import VerifiedClaim, VerdictEnum
from eval.graph_construction_metrics import compute_graph_construction_metrics

logger = logging.getLogger(__name__)


# ============================================================
#  指标结果数据类
# ============================================================

@dataclass
class PerHopStats:
    """单个跳数下的指标统计。"""
    hop_count: int
    case_count: int = 0
    path_error_rate: float = 0.0
    hallucination_rate_overall: float = 0.0
    hallucination_rate_per_hop: float = 0.0
    recall: float = 0.0
    precision: float = 0.0
    temporal_accuracy: float = 0.0
    provenance_completeness: float = 0.0
    # GraphRAG-Bench 借鉴 (arXiv:2506.02404 §3.2)
    entity_recall: float = 0.0
    entity_precision: float = 0.0
    relation_recall: float = 0.0
    pipeline_f1: float = 0.0


@dataclass
class MetricsReport:
    """整体评估报告。"""
    baseline_name: str
    per_hop: dict[int, PerHopStats] = field(default_factory=dict)
    overall: PerHopStats = field(default_factory=lambda: PerHopStats(hop_count=0))
    case_results: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "baseline_name": self.baseline_name,
            "per_hop": {h: vars(s) for h, s in self.per_hop.items()},
            "overall": vars(self.overall),
            "case_results": self.case_results,
        }


# ============================================================
#  单条 case 的指标计算
# ============================================================

def _hop_name_eq(expected_name: str, predicted_node_name: str) -> bool:
    """期望实体名 vs 预测实体名是否等价（宽松匹配：包含/反向包含）。"""
    a, b = expected_name.lower().strip(), predicted_node_name.lower().strip()
    return a == b or a in b or b in a


def compute_path_error_rate(
    expected_path: list,
    predicted_hops: list[PathHop],
) -> tuple[float, list[dict]]:
    """路径错误率：逐跳对齐，缺失/多余均计错误。

    Returns
    -------
    (error_rate, per_hop_details)
        error_rate ∈ [0, 1]
        per_hop_details: list of {hop_idx, expected, predicted, correct}
    """
    if not expected_path and not predicted_hops:
        return 0.0, []

    details = []
    correct_count = 0
    max_len = max(len(expected_path), len(predicted_hops))
    if max_len == 0:
        return 0.0, []
    pred_len = len(predicted_hops)

    for i in range(max_len):
        if i < len(expected_path):
            exp = expected_path[i]
            exp_target = exp.target_name
            exp_source = exp.source_name
        else:
            # 预测多余的跳（expected 已穷尽）
            exp_target = None
            exp_source = None

        if i < pred_len:
            ph = predicted_hops[i]
            pred_target = ph.target.name
            pred_source = ph.source.name
            pred_edge = ph.edge_name
        else:
            # expected 跳缺失（预测太短）
            pred_target = pred_source = pred_edge = None

        # 缺失 / 多余都算错误：只有 exp 和 pred 都非空且匹配才 correct
        if pred_target is None or pred_source is None:
            is_correct = False
        elif exp_target is None or exp_source is None:
            is_correct = False  # 预测多余
        else:
            target_ok = _hop_name_eq(exp_target, pred_target)
            source_ok = _hop_name_eq(exp_source, pred_source)
            is_correct = bool(target_ok and source_ok)
        if is_correct:
            correct_count += 1

        details.append({
            "hop_idx": i,
            "expected": {"source": exp_source, "target": exp_target},
            "predicted": {"source": pred_source, "target": pred_target, "edge": pred_edge},
            "correct": is_correct,
        })

    error_rate = 1.0 - (correct_count / max_len)
    return error_rate, details


def compute_recall_precision(
    expected_path: list,
    predicted_hops: list[PathHop],
) -> tuple[float, float]:
    """Recall + Precision：按 (source_name, target_name) 元组集合。

    集合元素统一小写，去重。
    """
    def to_set(path_or_hops) -> set[tuple[str, str]]:
        s = set()
        if not path_or_hops:
            return s
        # ExpectedPath
        if hasattr(path_or_hops[0], "target_name"):
            for h in path_or_hops:
                s.add((h.source_name.lower(), h.target_name.lower()))
        else:
            for h in path_or_hops:
                s.add((h.source.name.lower(), h.target.name.lower()))
        return s

    exp_set = to_set(expected_path)
    pred_set = to_set(predicted_hops)

    if not exp_set:
        recall = 0.0
    else:
        recall = len(exp_set & pred_set) / len(exp_set)
    if not pred_set:
        precision = 0.0
    else:
        precision = len(exp_set & pred_set) / len(pred_set)
    return recall, precision


def compute_temporal_accuracy(
    predicted_hops: list[PathHop],
    query_time: datetime,
) -> float:
    """时态准确率：所有边 valid_at ≤ query_time ≤ invalid_at 的比例。

    若边 invalid_at 为 None（仍有效），视作通过。
    若边 valid_at 为 None（无时态锚点），视作通过（无证据不算错）。
    若边 valid_at > query_time（边在查询时刻之后才生效），视作失败。
    若边 invalid_at < query_time（边在查询时刻已失效），视作失败。
    若无任何预测路径，返回 0.0（系统未给出任何可校验的时态信息）。
    """
    if not predicted_hops:
        return 0.0
    correct = 0
    for h in predicted_hops:
        if h.valid_at is None:
            # 无时态锚点，视作通过（与 v1 行为一致，便于向后兼容）
            correct += 1
            continue
        if h.valid_at > query_time:
            # 边在查询时刻之后才生效 → 时态不一致
            continue
        if h.invalid_at is not None and h.invalid_at < query_time:
            # 边在查询时刻已失效 → 时态不一致
            continue
        correct += 1
    return correct / len(predicted_hops)


def compute_provenance_completeness(
    predicted_hops: list[PathHop],
) -> float:
    """Provenance 完备率：有 attributes（episode / source / reference）的边占比。"""
    if not predicted_hops:
        return 1.0
    has_prov = 0
    for h in predicted_hops:
        if h.attributes:
            has_prov += 1
    return has_prov / len(predicted_hops)


def compute_hallucination_rate(
    verified_claims: list[VerifiedClaim],
) -> tuple[float, float, dict[int, dict]]:
    """幻觉率（整体 + 逐跳）。

    Returns
    -------
    (overall_rate, per_hop_avg_rate, per_hop_stats)
        per_hop_stats: {hop_index: {total, entailed, contradicted, unsupported, rate}}
    """
    if not verified_claims:
        return 0.0, 0.0, {}

    # 整体：CONTRADICTED 视为幻觉
    contradicted = sum(1 for c in verified_claims if c.verdict == VerdictEnum.CONTRADICTED)
    overall_rate = contradicted / len(verified_claims)

    # 逐跳
    per_hop: dict[int, list[VerifiedClaim]] = {}
    for c in verified_claims:
        per_hop.setdefault(c.hop_index, []).append(c)

    per_hop_stats: dict[int, dict] = {}
    rates: list[float] = []
    for hop, claims in per_hop.items():
        cnt_contra = sum(1 for c in claims if c.verdict == VerdictEnum.CONTRADICTED)
        cnt_ent = sum(1 for c in claims if c.verdict == VerdictEnum.ENTAILED)
        cnt_uns = sum(1 for c in claims if c.verdict == VerdictEnum.UNSUPPORTED)
        rate = cnt_contra / len(claims) if claims else 0.0
        per_hop_stats[hop] = {
            "total": len(claims),
            "entailed": cnt_ent,
            "contradicted": cnt_contra,
            "unsupported": cnt_uns,
            "rate": rate,
        }
        rates.append(rate)

    per_hop_avg = sum(rates) / len(rates) if rates else 0.0
    return overall_rate, per_hop_avg, per_hop_stats


# ============================================================
#  单条 case 的指标汇总
# ============================================================

def evaluate_case(
    case: Any,
    predicted_hops: list[PathHop],
    verified_claims: list[VerifiedClaim] | None = None,
) -> dict:
    """评估单条 case，返回所有指标。"""
    query_time = datetime.fromisoformat(case.query_time)
    exp_path = case.expected_path

    per, per_hop_details = compute_path_error_rate(exp_path, predicted_hops)
    recall, precision = compute_recall_precision(exp_path, predicted_hops)
    temporal = compute_temporal_accuracy(predicted_hops, query_time)
    provenance = compute_provenance_completeness(predicted_hops)
    hallu_overall, hallu_per_hop, hallu_stats = compute_hallucination_rate(
        verified_claims or []
    )
    # GraphRAG-Bench §3.2 借鉴：图构建质量
    gc_metrics = compute_graph_construction_metrics(exp_path, predicted_hops)

    return {
        "case_id": case.case_id,
        "hop_count": case.hop_count,
        "domain": case.domain,
        "path_error_rate": per,
        "hallucination_rate_overall": hallu_overall,
        "hallucination_rate_per_hop": hallu_per_hop,
        "recall": recall,
        "precision": precision,
        "temporal_accuracy": temporal,
        "provenance_completeness": provenance,
        "predicted_hop_count": len(predicted_hops),
        "expected_hop_count": len(exp_path),
        "claim_count": len(verified_claims) if verified_claims else 0,
        "per_hop_details": per_hop_details,
        "hallucination_stats": hallu_stats,
        # GraphRAG-Bench §3.2 借鉴
        "entity_recall": gc_metrics["entity_recall"],
        "entity_precision": gc_metrics["entity_precision"],
        "relation_recall": gc_metrics["relation_recall"],
        "pipeline_f1": gc_metrics["pipeline_f1"],
    }


# ============================================================
#  整体报告
# ============================================================

def aggregate_metrics(
    baseline_name: str,
    case_results: list[dict],
) -> MetricsReport:
    """聚合所有 case 的指标，按 2/3/4 跳 + Overall 报告。"""
    report = MetricsReport(baseline_name=baseline_name)
    report.case_results = case_results

    by_hop: dict[int, list[dict]] = {}
    for r in case_results:
        by_hop.setdefault(r["hop_count"], []).append(r)

    def _avg(rs: list[dict], key: str) -> float:
        if not rs:
            return 0.0
        return sum(r.get(key, 0.0) for r in rs) / len(rs)

    # 逐跳
    for h, rs in sorted(by_hop.items()):
        stats = PerHopStats(
            hop_count=h,
            case_count=len(rs),
            path_error_rate=_avg(rs, "path_error_rate"),
            hallucination_rate_overall=_avg(rs, "hallucination_rate_overall"),
            hallucination_rate_per_hop=_avg(rs, "hallucination_rate_per_hop"),
            recall=_avg(rs, "recall"),
            precision=_avg(rs, "precision"),
            temporal_accuracy=_avg(rs, "temporal_accuracy"),
            provenance_completeness=_avg(rs, "provenance_completeness"),
            entity_recall=_avg(rs, "entity_recall"),
            entity_precision=_avg(rs, "entity_precision"),
            relation_recall=_avg(rs, "relation_recall"),
            pipeline_f1=_avg(rs, "pipeline_f1"),
        )
        report.per_hop[h] = stats

    # Overall
    all_rs = case_results
    if all_rs:
        report.overall = PerHopStats(
            hop_count=0,
            case_count=len(all_rs),
            path_error_rate=_avg(all_rs, "path_error_rate"),
            hallucination_rate_overall=_avg(all_rs, "hallucination_rate_overall"),
            hallucination_rate_per_hop=_avg(all_rs, "hallucination_rate_per_hop"),
            recall=_avg(all_rs, "recall"),
            precision=_avg(all_rs, "precision"),
            temporal_accuracy=_avg(all_rs, "temporal_accuracy"),
            provenance_completeness=_avg(all_rs, "provenance_completeness"),
            entity_recall=_avg(all_rs, "entity_recall"),
            entity_precision=_avg(all_rs, "entity_precision"),
            relation_recall=_avg(all_rs, "relation_recall"),
            pipeline_f1=_avg(all_rs, "pipeline_f1"),
        )
    return report


def report_to_markdown(report: MetricsReport) -> str:
    """把报告渲染成 Markdown 表格（方便论文插图）。"""
    lines = [f"## 评估报告 — {report.baseline_name}", ""]
    # GraphRAG-Bench §3.2 借鉴：4 个图构建质量列
    lines.append("| 跳数 | 样本数 | PathError↓ | Hallu(整体)↓ | Hallu(逐跳)↓ | Recall↑ | Precision↑ | TemporalAcc↑ | Provenance↑ | EntityR↑ | EntityP↑ | RelationR↑ | PipeF1↑ |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for h in sorted(report.per_hop.keys()):
        s = report.per_hop[h]
        lines.append(
            f"| {h} | {s.case_count} | {s.path_error_rate:.3f} | "
            f"{s.hallucination_rate_overall:.3f} | {s.hallucination_rate_per_hop:.3f} | "
            f"{s.recall:.3f} | {s.precision:.3f} | {s.temporal_accuracy:.3f} | "
            f"{s.provenance_completeness:.3f} | {s.entity_recall:.3f} | "
            f"{s.entity_precision:.3f} | {s.relation_recall:.3f} | {s.pipeline_f1:.3f} |"
        )
    o = report.overall
    lines.append(
        f"| **Overall** | {o.case_count} | **{o.path_error_rate:.3f}** | "
        f"**{o.hallucination_rate_overall:.3f}** | **{o.hallucination_rate_per_hop:.3f}** | "
        f"**{o.recall:.3f}** | **{o.precision:.3f}** | **{o.temporal_accuracy:.3f}** | "
        f"**{o.provenance_completeness:.3f}** | **{o.entity_recall:.3f}** | "
        f"**{o.entity_precision:.3f}** | **{o.relation_recall:.3f}** | **{o.pipeline_f1:.3f}** |"
    )
    return "\n".join(lines)
