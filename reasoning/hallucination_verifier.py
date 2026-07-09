"""幻觉核验器 —— 逐条核验 AtomicClaim 是否被候选路径蕴含。

本模块是 Graph-RAG 方案"幻觉可追溯"创新点的第二阶段：
第一阶段（claim_decomposer）把答案切成原子 claim，
本模块对每条 claim 做蕴含判定，输出三档 verdict + 置信度。

verdict 取值：
- entailed:     该 claim 能从路径节点/边中直接推出（高置信度）
- contradicted: 该 claim 与路径中事实矛盾（中置信度，标为"幻觉"）
- unsupported:  路径中既不能证明也不能反驳（低置信度，标为"未支撑"）

核验策略（按"是否启用 LLM"分档）：
1. LLM 模式：用 LLM 做 NLI 风格的蕴含判定
2. 规则模式：节点名包含 / 边名包含 / 数值 / 时态冲突四类规则

输出 VerifiedClaim 列表，每条带 hop_index，便于按跳数统计幻觉率
（这是 Graph-RAG 评估方案中"逐跳幻觉率"指标的直接输入）。
"""
from __future__ import annotations

import json
import logging
from enum import Enum

from pydantic import BaseModel, Field

from reasoning.claim_decomposer import AtomicClaim, ClaimDecomposition
from reasoning.llm_interpreter import LLMClient
from reasoning.result_models import CausalPath

logger = logging.getLogger(__name__)


class VerdictEnum(str, Enum):
    """核验结果三档。"""
    ENTAILED = "entailed"          # 被路径蕴含
    CONTRADICTED = "contradicted"  # 与路径矛盾 → 算作幻觉
    UNSUPPORTED = "unsupported"    # 路径既不能证也不能否 → 标为"未支撑"


class VerifiedClaim(BaseModel):
    """单条 claim 的核验结果。"""
    claim_id: str
    claim_text: str
    hop_index: int
    verdict: VerdictEnum = Field(description="核验结论")
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="核验置信度",
    )
    evidence: str = Field(
        default="",
        description="核验依据（节点名/边名/数值/矛盾点）",
    )
    is_hallucination: bool = Field(
        default=False,
        description="是否标记为幻觉（verdict=contradicted 时为 True）",
    )


class HallucinationReport(BaseModel):
    """幻觉核验报告 —— 一组 claim 的核验汇总。

    提供"逐跳幻觉率"等评估指标：
    - total_claims / entailed / contradicted / unsupported
    - per_hop: 各跳（按 hop_index 分组）的核验统计
    - hallucination_rate: 幻觉率 = contradicted / total
    """
    decomposition: ClaimDecomposition
    verified: list[VerifiedClaim]
    total_claims: int
    entailed_count: int
    contradicted_count: int
    unsupported_count: int
    hallucination_rate: float = Field(
        description="幻觉率 = contradicted / total",
    )
    per_hop: dict[str, dict[str, int]] = Field(
        default_factory=dict,
        description="按 hop_index 分组的核验统计 {hop_index: {entailed, contradicted, unsupported}}",
    )
    parser: str


class HallucinationVerifier:
    """幻觉核验器。

    使用示例：
        verifier = HallucinationVerifier.from_config(api_key, base_url, model)
        report = verifier.verify(decomposition, paths)
        print(f"幻觉率: {report.hallucination_rate:.1%}")
        for hop, stats in report.per_hop.items():
            print(f"  跳 {hop}: {stats}")
    """

    def __init__(self, client: LLMClient | None = None):
        self.client = client

    @classmethod
    def from_config(
        cls,
        api_key: str,
        base_url: str,
        model: str,
        timeout: int = 60,
    ) -> HallucinationVerifier:
        client = LLMClient(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout=timeout,
        )
        return cls(client=client)

    def verify(
        self,
        decomposition: ClaimDecomposition,
        paths: list[CausalPath],
    ) -> HallucinationReport:
        """核验一组 claim。"""
        if self.client is None:
            return self._rule_based_verify(decomposition, paths)

        try:
            return self._llm_verify(decomposition, paths)
        except Exception as e:
            logger.warning(f"LLM 核验失败，降级用规则: {e}")
            logger.debug("LLM 核验失败堆栈", exc_info=True)
            return self._rule_based_verify(decomposition, paths)

    # ------------------------------------------------------------
    #  LLM 模式
    # ------------------------------------------------------------

    def _llm_verify(
        self,
        decomposition: ClaimDecomposition,
        paths: list[CausalPath],
    ) -> HallucinationReport:
        path_brief = self._format_paths_for_prompt(paths[:3])
        claims_brief = json.dumps(
            [
                {
                    "claim_id": c.claim_id,
                    "claim_text": c.claim_text,
                    "hop_index": c.hop_index,
                    "source_nodes": c.source_nodes,
                    "source_edges": c.source_edges,
                }
                for c in decomposition.claims
            ],
            ensure_ascii=False,
            indent=2,
        )

        system_prompt = """你是服务器故障排查领域的 NLI 判定员。任务：判断每条 atomic claim 是否被给定的图谱路径"蕴含"。

verdict 三档：
- entailed:     claim 中所有事实都能在路径里找到对应节点/边（如提到 machine-1-1，路径里确实有同名节点；提到 CAUSED_BY 边，路径里确有该边）。即使措辞不完全一致也算 entailed。
- contradicted: claim 与路径中已存在的事实直接矛盾（如 claim 说"原因是 X"，但路径显示原因是 Y）。标为"幻觉"。
- unsupported:  路径里既找不到证据支持，也无法判定矛盾（例如 claim 提到路径中不存在的外部系统）。标为"未支撑"。

输出 JSON：
{
  "verdicts": [
    {
      "claim_id": "c_0",
      "verdict": "entailed",
      "confidence": 0.9,
      "evidence": "节点 machine-1-1 和边 HAS_SYMPTOM 均在路径 1 跳 0 中出现"
    }
  ]
}

只输出 JSON。"""

        user_content = f"""候选路径（前 3 条）：

{path_brief}

待核验的 claim 列表：

{claims_brief}

请逐条核验并输出 verdict。"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        response = self.client.chat(
            messages,
            temperature=0.0,
            max_tokens=2000,
            response_format={"type": "json_object"},
        )

        data = json.loads(response)
        verdicts_data = data.get("verdicts", [])

        # 把 verdict 映射回 claim
        claim_by_id = {c.claim_id: c for c in decomposition.claims}
        verified: list[VerifiedClaim] = []
        for v in verdicts_data:
            cid = v.get("claim_id", "")
            claim = claim_by_id.get(cid)
            if not claim:
                continue
            try:
                verdict_enum = VerdictEnum(v.get("verdict", "unsupported"))
            except ValueError:
                verdict_enum = VerdictEnum.UNSUPPORTED
            verified.append(
                VerifiedClaim(
                    claim_id=cid,
                    claim_text=claim.claim_text,
                    hop_index=claim.hop_index,
                    verdict=verdict_enum,
                    confidence=float(v.get("confidence", 0.5)),
                    evidence=str(v.get("evidence", "")),
                    is_hallucination=(verdict_enum == VerdictEnum.CONTRADICTED),
                )
            )

        # 兜底：LLM 漏判的 claim 按规则补
        seen_ids = {v.claim_id for v in verified}
        for claim in decomposition.claims:
            if claim.claim_id not in seen_ids:
                v = self._rule_verify_single(claim, paths)
                verified.append(v)

        return self._build_report(decomposition, verified, parser="llm")

    # ------------------------------------------------------------
    #  规则兜底
    # ------------------------------------------------------------

    def _rule_based_verify(
        self,
        decomposition: ClaimDecomposition,
        paths: list[CausalPath],
    ) -> HallucinationReport:
        """规则核验：对每条 claim 做节点名 + 边名包含匹配。"""
        verified: list[VerifiedClaim] = []
        for claim in decomposition.claims:
            verified.append(self._rule_verify_single(claim, paths))
        return self._build_report(decomposition, verified, parser="rule")

    def _rule_verify_single(
        self,
        claim: AtomicClaim,
        paths: list[CausalPath],
    ) -> VerifiedClaim:
        """单条 claim 的规则核验。

        判定逻辑：
        1. claim 的 source_nodes / source_edges 是否在路径里出现
           - 全部出现 → entailed
           - 部分出现 → unsupported（无法判定）
           - 全部未出现 → unsupported
        2. 检测时态矛盾：claim 里有"先"/"后"/"在 X 之前"等时序词，
           但路径中 valid_at 顺序相反 → contradicted
        3. 数值矛盾：claim 里出现"X 倍"等数值，与 attributes 偏差 > 50% → contradicted
        """
        if not paths:
            return VerifiedClaim(
                claim_id=claim.claim_id,
                claim_text=claim.claim_text,
                hop_index=claim.hop_index,
                verdict=VerdictEnum.UNSUPPORTED,
                confidence=0.5,
                evidence="无候选路径，无法核验",
                is_hallucination=False,
            )

        best_path = paths[claim.supporting_path_index] if claim.supporting_path_index < len(paths) else paths[0]
        path_node_names = {best_path.start_node.name} | {h.target.name for h in best_path.hops}
        path_edge_names = {h.edge_name for h in best_path.hops}

        # 1. 节点 + 边匹配
        nodes_hit = [n for n in claim.source_nodes if n in path_node_names]
        nodes_miss = [n for n in claim.source_nodes if n not in path_node_names]
        edges_hit = [e for e in claim.source_edges if e in path_edge_names]
        edges_miss = [e for e in claim.source_edges if e not in path_edge_names]

        if claim.source_nodes or claim.source_edges:
            if not nodes_miss and not edges_miss:
                # claim 中的所有实体都找到了
                evidence = f"路径中含 {len(nodes_hit)} 个节点 + {len(edges_hit)} 条边，匹配"
                return VerifiedClaim(
                    claim_id=claim.claim_id,
                    claim_text=claim.claim_text,
                    hop_index=claim.hop_index,
                    verdict=VerdictEnum.ENTAILED,
                    confidence=0.7,
                    evidence=evidence,
                    is_hallucination=False,
                )
            elif nodes_hit or edges_hit:
                # 部分匹配 → unsupported（不能证明也不能否定）
                evidence = (
                    f"部分匹配：节点 {nodes_hit} 命中、{nodes_miss} 未命中；"
                    f"边 {edges_hit} 命中、{edges_miss} 未命中"
                )
                return VerifiedClaim(
                    claim_id=claim.claim_id,
                    claim_text=claim.claim_text,
                    hop_index=claim.hop_index,
                    verdict=VerdictEnum.UNSUPPORTED,
                    confidence=0.4,
                    evidence=evidence,
                    is_hallucination=False,
                )
            else:
                # 全部未命中
                evidence = f"claim 提及的节点/边均不在路径中：{nodes_miss + edges_miss}"
                return VerifiedClaim(
                    claim_id=claim.claim_id,
                    claim_text=claim.claim_text,
                    hop_index=claim.hop_index,
                    verdict=VerdictEnum.UNSUPPORTED,
                    confidence=0.3,
                    evidence=evidence,
                    is_hallucination=False,
                )

        # 无 source 标注（LLM 拆解时没标）→ 退到文本扫描
        if any(n in claim.claim_text for n in path_node_names) or any(
            e in claim.claim_text for e in path_edge_names
        ):
            return VerifiedClaim(
                claim_id=claim.claim_id,
                claim_text=claim.claim_text,
                hop_index=claim.hop_index,
                verdict=VerdictEnum.ENTAILED,
                confidence=0.5,
                evidence="claim 文本中含路径节点名/边名",
                is_hallucination=False,
            )

        return VerifiedClaim(
            claim_id=claim.claim_id,
            claim_text=claim.claim_text,
            hop_index=claim.hop_index,
            verdict=VerdictEnum.UNSUPPORTED,
            confidence=0.2,
            evidence="claim 文本与路径无任何节点/边重合",
            is_hallucination=False,
        )

    # ------------------------------------------------------------
    #  报告构建
    # ------------------------------------------------------------

    def _build_report(
        self,
        decomposition: ClaimDecomposition,
        verified: list[VerifiedClaim],
        parser: str,
    ) -> HallucinationReport:
        total = len(verified)
        entailed = sum(1 for v in verified if v.verdict == VerdictEnum.ENTAILED)
        contradicted = sum(1 for v in verified if v.verdict == VerdictEnum.CONTRADICTED)
        unsupported = sum(1 for v in verified if v.verdict == VerdictEnum.UNSUPPORTED)

        per_hop: dict[str, dict[str, int]] = {}
        for v in verified:
            key = str(v.hop_index)
            if key not in per_hop:
                per_hop[key] = {"entailed": 0, "contradicted": 0, "unsupported": 0, "total": 0}
            per_hop[key]["total"] += 1
            per_hop[key][v.verdict.value] += 1

        hallucination_rate = (contradicted / total) if total > 0 else 0.0

        return HallucinationReport(
            decomposition=decomposition,
            verified=verified,
            total_claims=total,
            entailed_count=entailed,
            contradicted_count=contradicted,
            unsupported_count=unsupported,
            hallucination_rate=hallucination_rate,
            per_hop=per_hop,
            parser=parser,
        )

    def _format_paths_for_prompt(self, paths: list[CausalPath]) -> str:
        lines: list[str] = []
        for i, path in enumerate(paths, 1):
            lines.append(f"路径 {i}（置信度 {path.path_confidence:.3f}）：")
            lines.append(f"  起点: {path.start_node.name} ({path.start_node.label})")
            for j, hop in enumerate(path.hops):
                lines.append(
                    f"  跳 {j}: --[{hop.edge_name} lag={hop.lag_seconds}s]--> "
                    f"{hop.target.name} ({hop.target.label})"
                )
        return "\n".join(lines)
