"""B1: 朴素 RAG baseline —— 纯 LLM 直答，不查图谱。

隔离「图谱结构」对答案质量的边际价值。
B4 - B1 ≈ 图谱检索带来的净增益。
"""
from __future__ import annotations

import logging
import time

from eval.baselines.common import BaselineResult, BaselineRunner
from reasoning.hallucination_verifier import HallucinationVerifier
from reasoning.llm_interpreter import LLMClient

logger = logging.getLogger(__name__)


# 朴素 RAG prompt：让 LLM 用自己的"先验知识"答故障根因
NAIVE_RAG_PROMPT = """你是一个云原生运维专家。请仅凭你的先验知识（不要使用任何外部工具），
回答以下关于服务器集群故障的根因问题。

要求：
1. 回答必须简洁（3-5 句话）
2. 列出可能的故障根因和解法
3. 回答中的每条事实声明必须用「[1]」「[2]」「[3]」编号，方便后续核验
4. 不要编造具体的服务名/机器名/时间戳，只描述通用模式

问题：{query}

回答："""


class B1NaiveRAG(BaselineRunner):
    """B1 — 纯 LLM 直答，不查图谱。

    用空路径 + LLM 回答，模拟"无图谱"baseline。
    幻觉率用 HallucinationVerifier 在空路径上核验 → 全部 UNSUPPORTED/无来源 → 高幻觉率。
    """

    name = "B1_NaiveRAG"

    def __init__(self, llm_client: LLMClient | None = None):
        self.llm_client = llm_client
        # 仍用 verifier，只是核验的"路径"是空的
        self.verifier = HallucinationVerifier(client=llm_client)

    def predict(self, case) -> BaselineResult:
        t0 = time.time()
        try:
            if self.llm_client is None:
                return BaselineResult(
                    case_id=case.case_id,
                    baseline_name=self.name,
                    answer="(no LLM client configured)",
                    predicted_hops=[],
                    verified_claims=[],
                    elapsed_seconds=time.time() - t0,
                )

            # 纯 LLM 回答
            answer = self.llm_client.chat(
                messages=[
                    {"role": "system", "content": "你是云原生运维专家。"},
                    {"role": "user", "content": NAIVE_RAG_PROMPT.format(query=case.query)},
                ],
                temperature=0.1,
            )

            # 拆解（规则模式：按 [1][2][3] 拆）
            from reasoning.claim_decomposer import ClaimDecomposer
            decomposer = ClaimDecomposer(client=None)
            decomp = decomposer.decompose(answer, [])

            # 核验（空路径 → 大部分应 UNSUPPORTED）
            report = self.verifier.verify(decomp, [])

            return BaselineResult(
                case_id=case.case_id,
                baseline_name=self.name,
                answer=answer,
                predicted_hops=[],
                verified_claims=report.verified,
                elapsed_seconds=time.time() - t0,
                metadata={"decomposition_parser": decomp.parser},
            )
        except Exception as e:
            logger.exception(f"B1 推理失败: {case.case_id}")
            return BaselineResult(
                case_id=case.case_id,
                baseline_name=self.name,
                error=str(e),
                elapsed_seconds=time.time() - t0,
            )
