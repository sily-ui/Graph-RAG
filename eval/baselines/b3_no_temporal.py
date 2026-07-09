"""B3: 自研控制器 - 时态剪枝消融。

复用 B4 链路，但 TemporalPruner.prune() 永远保留所有路径（关闭时态校验）。
"""
from __future__ import annotations

import logging
import time

from eval.baselines.common import BaselineResult, BaselineRunner
from reasoning.claim_decomposer import ClaimDecomposer
from reasoning.controller import ReasoningController
from reasoning.hallucination_verifier import HallucinationVerifier
from reasoning.result_models import PathHop
from reasoning.temporal_pruner import (
    PrunerConfig,
    TemporalPruner,
    validate_path_temporal_consistency,
    validate_path_in_window,
)

logger = logging.getLogger(__name__)


class B3NoTemporal(BaselineRunner):
    """B3 — 关闭时态剪枝的自研控制器。

    隔离价值：证明「时态建模」对路径质量的边际贡献。
    B4 - B3 = 时态剪枝的净收益。
    """

    name = "B3_NoTemporal"

    def __init__(
        self,
        controller: ReasoningController,
        decomposer: ClaimDecomposer,
        verifier: HallucinationVerifier,
    ):
        self.controller = controller
        self.decomposer = decomposer
        self.verifier = verifier

        # 替换 pruner：构造一个"永远保留"的版本
        self.controller.pruner = _PassThroughPruner()

    def predict(self, case) -> BaselineResult:
        t0 = time.time()
        try:
            reasoning = self.controller.ask(case.query)
            predicted_hops: list[PathHop] = []
            for path in reasoning.paths:
                predicted_hops.extend(path.hops)

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
                    "pruned_count": 0,
                    "decomposition_parser": decomposition.parser,
                    "verifier_parser": report.parser,
                },
            )
        except Exception as e:
            logger.exception(f"B3 推理失败: {case.case_id}")
            return BaselineResult(
                case_id=case.case_id,
                baseline_name=self.name,
                error=str(e),
                elapsed_seconds=time.time() - t0,
            )


class _PassThroughPruner(TemporalPruner):
    """时态剪枝的"穿透"版本：所有路径都保留，不做任何校验。"""

    def __init__(self):
        # 用一个所有 flag 都关掉的 config
        super().__init__(config=PrunerConfig())
        # 重写 config：关闭所有校验
        self.config.ENABLE_MONOTONICITY = False
        self.config.ENABLE_TERMINATION = False
        self.config.ENABLE_INTERVAL_OVERLAP = False

    def prune(self, paths, time_window=None):
        """永远保留所有路径。"""
        kept = []
        pruned = []
        for p in paths:
            p.is_temporally_consistent = True
            p.pruned_reason = None
            kept.append(p)
        return kept, pruned
