"""Graph-RAG 自研推理控制器 —— 模块 3。

本包实现基于时态因果知识图谱的多跳推理控制器，是 Graph-RAG 方案的核心研究贡献。

架构：
    自然语言查询
         │
         ▼
    LLMInterpreter.parse_query()  ← LLM 解析查询意图
         │
         ▼
    StructuredQuery（结构化查询）
         │
         ▼
    CypherGenerator.generate()    ← 生成 Cypher 查询
         │
         ▼
    PathExtractor.extract()       ← Neo4j 执行 + 多跳路径抽取
         │
         ▼
    TemporalPruner.prune()        ← 时态剪枝（valid_at/lag_seconds）
         │
         ▼
    LLMInterpreter.explain()      ← LLM 解释结果
         │
         ▼
    ReasoningResult（最终答案 + 路径 + 置信度）

核心创新点：
1. 时态因果剪枝：基于 valid_at/invalid_at/lag_seconds 过滤候选路径，
   排除时态不一致的因果链（如症状早于原因的路径）
2. 多跳路径模板：基于 graph_schema.PATH_TEMPLATES 的 2/3/4 跳路径，
   保证抽取的因果链符合 Schema 约束
3. LLM + 图谱协同：LLM 负责意图解析与结果解释，图谱负责精确检索，
   避免纯 LLM 的幻觉问题
"""
from reasoning.claim_decomposer import (
    AtomicClaim,
    ClaimDecomposition,
    ClaimDecomposer,
)
from reasoning.controller import ReasoningController
from reasoning.cypher_generator import CypherGenerator
from reasoning.hallucination_verifier import (
    HallucinationReport,
    HallucinationVerifier,
    VerifiedClaim,
    VerdictEnum,
)
from reasoning.llm_interpreter import LLMInterpreter
from reasoning.path_extractor import PathExtractor
from reasoning.query_types import (
    QueryIntent,
    QueryType,
    StructuredQuery,
    TimeWindow,
)
from reasoning.result_models import (
    CausalPath,
    NodeInfo,
    PathHop,
    ReasoningResult,
)
from reasoning.temporal_pruner import TemporalPruner

__all__ = [
    "ReasoningController",
    "CypherGenerator",
    "LLMInterpreter",
    "PathExtractor",
    "TemporalPruner",
    "QueryIntent",
    "QueryType",
    "StructuredQuery",
    "TimeWindow",
    "CausalPath",
    "NodeInfo",
    "PathHop",
    "ReasoningResult",
    "ClaimDecomposer",
    "AtomicClaim",
    "ClaimDecomposition",
    "HallucinationVerifier",
    "HallucinationReport",
    "VerifiedClaim",
    "VerdictEnum",
]
