"""B4: 完整 Graph-RAG baseline（自研 Cypher + BFS + 时态剪枝 + provenance 核验）。

直接调用 ReasoningController.ask() + ClaimDecomposer + HallucinationVerifier。
"""
from __future__ import annotations

import logging
import time

from eval.baselines.common import BaselineResult, BaselineRunner
from reasoning.claim_decomposer import ClaimDecomposer
from reasoning.controller import ReasoningController
from reasoning.hallucination_verifier import HallucinationVerifier
from datetime import datetime, timedelta

from reasoning.query_types import TimeWindow
from reasoning.result_models import PathHop


def _build_time_window(query_time_iso: str) -> TimeWindow | None:
    """把测试集的 query_time（ISO 字符串）转成 TimeWindow。

    用 [query_time - 1s, query_time + 1s] 作为窗口，让 pruner 的
    overlaps 校验能正确判断边的 valid_at 是否在查询时刻有效。
    """
    if not query_time_iso:
        return None
    try:
        qt = datetime.fromisoformat(query_time_iso.replace("Z", "+00:00"))
        return TimeWindow(start=qt - timedelta(seconds=1), end=qt + timedelta(seconds=1))
    except (ValueError, TypeError):
        return None

logger = logging.getLogger(__name__)


class B4FullGraphRAG(BaselineRunner):
    """B4 — 完整 Graph-RAG（自研全链路）。

    隔离价值：作为完整方案，证明所有设计决策合并后的总收益。
    """

    name = "B4_Full_GraphRAG"

    def __init__(
        self,
        controller: ReasoningController,
        decomposer: ClaimDecomposer,
        verifier: HallucinationVerifier,
    ):
        self.controller = controller
        self.decomposer = decomposer
        self.verifier = verifier

    def predict(self, case) -> BaselineResult:
        t0 = time.time()
        try:
            # 1. 跑自研推理（把测试集的 query_time 作为时态窗口传入，让 TemporalPruner
            #    做区间重叠校验——这是 B4 区别于 B3 的核心：B3 用 PassThroughPruner
            #    完全跳过剪枝，B4 用真实 query_time 过滤掉 valid_at > query_time 的边）
            time_window = _build_time_window(case.query_time)
            reasoning = self.controller.ask(case.query, time_window=time_window)
            predicted_hops: list[PathHop] = []
            for path in reasoning.paths:
                predicted_hops.extend(path.hops)

            # 2. 拆解 + 核验
            decomposition = self.decomposer.decompose(reasoning.answer, reasoning.paths)
            report = self.verifier.verify(decomposition, reasoning.paths)

            return BaselineResult(
                case_id=case.case_id,
                baseline_name=self.name,
                answer=reasoning.answer,
                predicted_hops=predicted_hops,
                verified_claims=report.verified,
                elapsed_seconds=time.time() - t0,
                metadata={
                    "path_count": reasoning.path_count,
                    "pruned_count": reasoning.pruned_count,
                    "confidence": reasoning.confidence,
                    "decomposition_parser": decomposition.parser,
                    "verifier_parser": report.parser,
                },
            )
        except Exception as e:
            logger.exception(f"B4 推理失败: {case.case_id}")
            return BaselineResult(
                case_id=case.case_id,
                baseline_name=self.name,
                error=str(e),
                elapsed_seconds=time.time() - t0,
            )
