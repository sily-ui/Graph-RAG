"""Baseline 通用接口与数据类。

四个 baseline 都实现 BaselineRunner.predict(case) → BaselineResult。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from reasoning.hallucination_verifier import VerifiedClaim
from reasoning.result_models import PathHop

logger = logging.getLogger(__name__)


@dataclass
class BaselineResult:
    """单个 baseline 对单条 case 的输出。"""
    case_id: str
    baseline_name: str
    answer: str = ""
    predicted_hops: list[PathHop] = field(default_factory=list)
    verified_claims: list[VerifiedClaim] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "baseline_name": self.baseline_name,
            "answer": self.answer,
            "predicted_hop_count": len(self.predicted_hops),
            "verified_claim_count": len(self.verified_claims),
            "elapsed_seconds": self.elapsed_seconds,
            "error": self.error,
            "metadata": self.metadata,
        }


class BaselineRunner:
    """Baseline 抽象基类。"""

    name: str = "abstract"

    def predict(self, case: Any) -> BaselineResult:
        """对单条 case 推理，返回结构化结果。

        子类必须实现。

        Parameters
        ----------
        case : TestCase
            测试用例（来自 testset_builder）
        """
        raise NotImplementedError(f"{self.name} 未实现 predict()")

    def cleanup(self) -> None:
        """清理资源（关闭 driver 等）。"""
        pass
