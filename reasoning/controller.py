"""推理控制器 —— 模块 3 的顶层编排组件。

本模块是 Graph-RAG 自研推理控制器的入口，协调 LLM/Cypher/剪枝/解释四大组件。

完整推理流程：
    1. LLMInterpreter.parse_query()      自然语言 → StructuredQuery
    2. CypherGenerator.generate()        StructuredQuery → Cypher
    3. PathExtractor.execute()           Cypher → CausalPath 列表
    4. TemporalPruner.prune()            CausalPath 列表 → 剪枝后路径
    5. TemporalPruner.rank_paths()       按置信度+时延排序
    6. LLMInterpreter.explain()          路径 + 查询 → 自然语言答案
    7. LLMInterpreter.estimate_confidence() 综合置信度

创新点（对照实验的评估维度）：
1. 时态因果剪枝：排除时态不一致的路径，提升准确率
2. 多跳路径模板：基于 Schema 约束的 2/3/4 跳路径
3. LLM + 图谱协同：避免纯 LLM 幻觉，也避免纯图谱的语义缺失
4. 置信度排序：路径置信度（几何平均）+ 时延 + 跳数综合排序
"""
from __future__ import annotations

import logging
import time
from typing import Any

from reasoning.cypher_generator import CypherGenerator
from reasoning.llm_interpreter import LLMInterpreter
from reasoning.path_extractor import PathExtractor
from reasoning.query_types import StructuredQuery
from reasoning.result_models import ReasoningResult
from reasoning.temporal_pruner import TemporalPruner, rank_paths

logger = logging.getLogger(__name__)


# ============================================================
#  推理控制器
# ============================================================

class ReasoningController:
    """推理控制器 —— 编排 LLM + Cypher + 时态剪枝 + 解释。

    使用示例：
        controller = ReasoningController.from_config(config, driver)
        result = controller.ask("vm_001 为什么 CPU 飙高")
        print(result.answer)
        print(f"置信度: {result.confidence:.3f}")
        print(f"候选路径: {result.path_count} 条")

    降级模式（无 LLM/无 Neo4j）：
        controller = ReasoningController()  # 无 LLM，无 driver
        result = controller.ask("...")
        # 用规则兜底解析查询，返回空路径
    """

    def __init__(
        self,
        interpreter: LLMInterpreter | None = None,
        cypher_generator: CypherGenerator | None = None,
        path_extractor: PathExtractor | None = None,
        pruner: TemporalPruner | None = None,
    ):
        """
        Parameters
        ----------
        interpreter : LLMInterpreter | None
            LLM 解释器，None=降级模式（规则兜底）
        cypher_generator : CypherGenerator | None
            Cypher 生成器，None=默认实例
        path_extractor : PathExtractor | None
            路径抽取器，None=测试模式（不连数据库）
        pruner : TemporalPruner | None
            时态剪枝器，None=默认实例
        """
        self.interpreter = interpreter or LLMInterpreter(client=None)
        self.cypher_generator = cypher_generator or CypherGenerator()
        self.path_extractor = path_extractor or PathExtractor(driver=None)
        self.pruner = pruner or TemporalPruner()

    @classmethod
    def from_config(
        cls,
        neo4j_driver: Any = None,
        llm_api_key: str = "",
        llm_base_url: str = "",
        llm_model: str = "",
        llm_timeout: int = 60,
    ) -> ReasoningController:
        """从配置创建推理控制器。

        Parameters
        ----------
        neo4j_driver : neo4j.Driver | None
            Neo4j 驱动，None=测试模式
        llm_api_key : str
            LLM API 密钥，空=降级模式
        llm_base_url : str
            LLM API 基址
        llm_model : str
            LLM 模型名
        llm_timeout : int
            LLM 请求超时（秒）
        """
        interpreter: LLMInterpreter
        if llm_api_key:
            interpreter = LLMInterpreter.from_config(
                api_key=llm_api_key,
                base_url=llm_base_url,
                model=llm_model,
                timeout=llm_timeout,
            )
        else:
            interpreter = LLMInterpreter(client=None)

        path_extractor = PathExtractor(driver=neo4j_driver)

        return cls(
            interpreter=interpreter,
            path_extractor=path_extractor,
        )

    # ------------------------------------------------------------
    #  主入口：自然语言问答
    # ------------------------------------------------------------

    def ask(
        self,
        natural_language: str,
        time_window: Any | None = None,
    ) -> ReasoningResult:
        """自然语言问答主入口。

        Parameters
        ----------
        natural_language : str
            自然语言查询
        time_window : TimeWindow | None
            查询时间窗口（可选，覆盖 LLM 解析的窗口）

        Returns
        -------
        ReasoningResult
            推理结果（含答案、路径、置信度）
        """
        start_time = time.time()
        logger.info(f"开始推理: {natural_language[:100]}")

        # 1. LLM 解析查询意图
        structured_query = self.interpreter.parse_query(natural_language)
        if time_window is not None:
            structured_query.intent.time_window = time_window
        logger.info(
            f"查询意图: type={structured_query.intent.query_type.value}, "
            f"entity={structured_query.intent.target_entity}"
        )

        # 2. 生成 Cypher
        cypher, params = self.cypher_generator.generate(structured_query)
        logger.info(f"生成 Cypher: {cypher[:200]}...")

        # 3. 执行 Cypher，抽取路径
        candidate_paths = self.path_extractor.execute(cypher, params)
        logger.info(f"抽取候选路径: {len(candidate_paths)} 条")

        # 4. 时态剪枝
        kept_paths, pruned_paths = self.pruner.prune(
            candidate_paths,
            time_window=structured_query.intent.time_window,
        )
        logger.info(
            f"时态剪枝: 保留 {len(kept_paths)} 条, 剪枝 {len(pruned_paths)} 条"
        )

        # 5. 排序
        kept_paths = rank_paths(kept_paths)

        # 6. LLM 解释
        answer = self.interpreter.explain(structured_query, kept_paths, pruned_paths)

        # 7. 综合置信度
        confidence = self.interpreter.estimate_confidence(kept_paths, pruned_paths)

        elapsed = time.time() - start_time
        logger.info(f"推理完成: 置信度={confidence:.3f}, 耗时={elapsed:.2f}s")

        return ReasoningResult(
            query=natural_language,
            answer=answer,
            paths=kept_paths,
            pruned_paths=pruned_paths,
            confidence=confidence,
            cypher_used=cypher,
            elapsed_seconds=elapsed,
            metadata={
                "query_type": structured_query.intent.query_type.value,
                "parser": structured_query.metadata.get("parser", "unknown"),
                "llm_model": self.interpreter.client.model if self.interpreter.client else "",
            },
        )

    # ------------------------------------------------------------
    #  结构化查询接口（跳过 LLM 解析）
    # ------------------------------------------------------------

    def query(self, structured_query: StructuredQuery) -> ReasoningResult:
        """结构化查询接口 —— 跳过 LLM 解析，直接用 StructuredQuery。

        用于：
        - 测试场景（不需要 LLM）
        - 已知查询意图的场景（程序化调用）
        - 评估场景（固定查询集）

        Parameters
        ----------
        structured_query : StructuredQuery
            结构化查询
        """
        start_time = time.time()

        # 1. 生成 Cypher
        cypher, params = self.cypher_generator.generate(structured_query)

        # 2. 执行 Cypher
        candidate_paths = self.path_extractor.execute(cypher, params)

        # 3. 时态剪枝
        kept_paths, pruned_paths = self.pruner.prune(
            candidate_paths,
            time_window=structured_query.intent.time_window,
        )

        # 4. 排序
        kept_paths = rank_paths(kept_paths)

        # 5. 解释
        answer = self.interpreter.explain(structured_query, kept_paths, pruned_paths)

        # 6. 置信度
        confidence = self.interpreter.estimate_confidence(kept_paths, pruned_paths)

        elapsed = time.time() - start_time

        return ReasoningResult(
            query=structured_query.natural_language,
            answer=answer,
            paths=kept_paths,
            pruned_paths=pruned_paths,
            confidence=confidence,
            cypher_used=cypher,
            elapsed_seconds=elapsed,
            metadata={
                "query_type": structured_query.intent.query_type.value,
                "parser": "structured",
            },
        )

    # ------------------------------------------------------------
    #  批量查询接口
    # ------------------------------------------------------------

    def batch_ask(
        self,
        queries: list[str],
    ) -> list[ReasoningResult]:
        """批量问答。

        Parameters
        ----------
        queries : list[str]
            自然语言查询列表
        """
        results: list[ReasoningResult] = []
        for i, q in enumerate(queries, 1):
            logger.info(f"批量查询 {i}/{len(queries)}: {q[:80]}")
            result = self.ask(q)
            results.append(result)
        return results

    # ------------------------------------------------------------
    #  诊断接口
    # ------------------------------------------------------------

    def explain_query(
        self,
        natural_language: str,
        time_window: Any | None = None,
    ) -> dict[str, Any]:
        """诊断接口 —— 返回推理过程的中间结果。

        用于调试与评估，展示每一步的中间状态。
        """
        start_time = time.time()

        # 1. 解析
        structured_query = self.interpreter.parse_query(natural_language)
        if time_window is not None:
            structured_query.intent.time_window = time_window

        # 2. Cypher
        cypher, params = self.cypher_generator.generate(structured_query)

        # 3. 抽取
        candidate_paths = self.path_extractor.execute(cypher, params)

        # 4. 剪枝
        kept_paths, pruned_paths = self.pruner.prune(
            candidate_paths,
            time_window=structured_query.intent.time_window,
        )

        # 5. 排序
        kept_paths = rank_paths(kept_paths)

        elapsed = time.time() - start_time

        return {
            "natural_language": natural_language,
            "structured_query": structured_query.model_dump(),
            "cypher": cypher,
            "params": {k: str(v) for k, v in params.items()},  # 序列化
            "candidate_count": len(candidate_paths),
            "kept_count": len(kept_paths),
            "pruned_count": len(pruned_paths),
            "kept_paths": [p.model_dump() for p in kept_paths[:5]],
            "pruned_paths": [
                {
                    "labels": p.labels,
                    "confidence": p.path_confidence,
                    "reason": p.pruned_reason,
                }
                for p in pruned_paths[:5]
            ],
            "elapsed_seconds": elapsed,
        }
