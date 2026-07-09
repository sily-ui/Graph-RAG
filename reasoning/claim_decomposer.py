"""原子声明拆解器 —— 把 LLM 答案拆成可核验的原子陈述。

本模块是 Graph-RAG 方案的"幻觉可追溯"创新点之一：
把自然语言答案切成 AtomicClaim 列表，每条 claim 显式标注：
- hop_index: 对应路径中的哪一跳（-1 = 跨跳的总结性陈述）
- supporting_path_index: 来自哪条候选路径
- source_nodes / source_edges: 涉及的节点名 / 边名（用于幻觉核验）
- confidence: 该 claim 从答案中拆出来的置信度

设计动机：
LLM 生成的整段答案粒度太粗，无法精确核验"哪一句是幻觉"。
拆成原子 claim 后，hallucination_verifier 可以逐条对照图谱节点/边做蕴含判定，
最终得到"逐跳幻觉率"指标（评估方案的核心量化维度）。

降级策略：
- 无 LLM 时按"句号+节点名提及"做启发式切分
- LLM 失败时记录原始段落为单条 claim，verifier 退到节点名包含匹配
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from pydantic import BaseModel, Field

from reasoning.llm_interpreter import LLMClient
from reasoning.result_models import CausalPath

logger = logging.getLogger(__name__)


class AtomicClaim(BaseModel):
    """单条原子声明 —— 可被图谱独立核验的最小事实单元。

    每条 claim 必须满足：
    - claim_text: 单一事实陈述（不含"并且"/"或者"等连接词）
    - hop_index: 该 claim 描述哪一跳（-1 = 跨跳/总结性）
    - source_nodes: 涉及的节点 name 列表
    - source_edges: 涉及的边名列表
    - confidence: 拆解置信度 [0,1]
    """
    claim_id: str = Field(description="claim 唯一 ID，格式 'c_{i}'")
    claim_text: str = Field(description="原子事实陈述")
    hop_index: int = Field(
        default=-1,
        ge=-1,
        description="对应路径跳索引（-1=总结性陈述，不对应单跳）",
    )
    supporting_path_index: int = Field(
        default=0,
        description="支撑该 claim 的候选路径索引（0-based）",
    )
    source_nodes: list[str] = Field(
        default_factory=list,
        description="涉及的节点 name（来自 path.hops[*].source/target.name）",
    )
    source_edges: list[str] = Field(
        default_factory=list,
        description="涉及的边 name（如 CAUSED_BY/RESOLVED_BY）",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="该 claim 从答案中拆出的置信度",
    )


class ClaimDecomposition(BaseModel):
    """声明拆解结果 —— 一条答案 → 多条 atomic claim。"""
    original_answer: str = Field(description="原始 LLM 答案")
    claims: list[AtomicClaim] = Field(description="拆解出的原子 claim 列表")
    parser: str = Field(description="拆解方式：'llm' | 'rule'")
    metadata: dict[str, Any] = Field(default_factory=dict)


class ClaimDecomposer:
    """声明拆解器 —— 拆 LLM 答案为原子 claim。

    使用示例：
        decomposer = ClaimDecomposer.from_config(api_key, base_url, model)
        decomp = decomposer.decompose(reasoning_result)
        for claim in decomp.claims:
            print(f"[{claim.hop_index}] {claim.claim_text}")
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
    ) -> ClaimDecomposer:
        client = LLMClient(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout=timeout,
        )
        return cls(client=client)

    def decompose(
        self,
        answer: str,
        paths: list[CausalPath],
    ) -> ClaimDecomposition:
        """把答案拆成原子 claim 列表。

        Parameters
        ----------
        answer : str
            LLM 生成的完整答案
        paths : list[CausalPath]
            推理控制器给出的候选路径（用于给 claim 标注 hop_index/source_nodes）
        """
        if self.client is None:
            return self._rule_based_decompose(answer, paths)

        try:
            return self._llm_decompose(answer, paths)
        except Exception as e:
            logger.warning(f"LLM 拆解失败，降级用规则: {e}")
            logger.debug("LLM 拆解失败堆栈", exc_info=True)
            return self._rule_based_decompose(answer, paths)

    def _llm_decompose(
        self,
        answer: str,
        paths: list[CausalPath],
    ) -> ClaimDecomposition:
        """用 LLM 把答案拆成原子 claim。"""
        # 构造 prompt：把答案和路径描述一起给 LLM，让它把答案切到具体节点/边
        path_brief = self._format_paths_for_prompt(paths[:3])

        system_prompt = """你是服务器故障排查领域的声明拆解助手。任务是把一段答案切成"原子事实陈述"。

每条 atomic claim 必须满足：
1. 单一事实：只表达一个事实，不含"并且"/"或者"/"此外"等连接多个事实的连接词
2. 可独立核验：每条 claim 必须显式提及至少一个图谱节点 name 或边名（CAUSED_BY / RESOLVED_BY / HAS_SYMPTOM 等）
3. 标注 hop_index：该 claim 描述路径中的哪一跳（0-based），跨跳总结性陈述用 -1

输出 JSON 格式：
{
  "claims": [
    {
      "claim_text": "machine-1-1 通过 HAS_SYMPTOM 表现出 cpu_spike 症状",
      "hop_index": 0,
      "source_nodes": ["machine-1-1", "cpu_spike"],
      "source_edges": ["HAS_SYMPTOM"],
      "confidence": 0.95
    }
  ]
}

只输出 JSON，不要其他文字。"""

        user_content = f"""图谱候选路径（按置信度排序，前 3 条）：

{path_brief}

LLM 生成的答案：
{answer}

请把上述答案拆成 atomic claim 列表。"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        response = self.client.chat(
            messages,
            temperature=0.0,
            max_tokens=1500,
            response_format={"type": "json_object"},
        )

        data = _parse_llm_json_response(response)
        claims_data = data.get("claims", [])
        if not isinstance(claims_data, list):
            raise ValueError("LLM 输出中 claims 不是 list")

        claims: list[AtomicClaim] = []
        for i, c in enumerate(claims_data):
            claims.append(
                AtomicClaim(
                    claim_id=f"c_{i}",
                    claim_text=str(c.get("claim_text", "")).strip(),
                    hop_index=int(c.get("hop_index", -1)),
                    supporting_path_index=0,
                    source_nodes=list(c.get("source_nodes", []) or []),
                    source_edges=list(c.get("source_edges", []) or []),
                    confidence=float(c.get("confidence", 0.8)),
                )
            )

        return ClaimDecomposition(
            original_answer=answer,
            claims=claims,
            parser="llm",
            metadata={"model": self.client.model if self.client else ""},
        )

    def _rule_based_decompose(
        self,
        answer: str,
        paths: list[CausalPath],
    ) -> ClaimDecomposition:
        """规则兜底拆解：按句号切句，每句从路径里匹配涉及的节点/边。"""
        # 按中英文句号切
        sentences = re.split(r"(?<=[。!?\.!?])\s*", answer)
        sentences = [s.strip() for s in sentences if s.strip()]

        # 收集路径中所有节点名 + 边名
        node_names: set[str] = set()
        edge_names: set[str] = set()
        hop_node_map: list[tuple[int, str, str]] = []  # (hop_index, node_name, edge_name)
        if paths:
            for i, path in enumerate(paths[:1]):  # 只对最佳路径做匹配
                for hop_idx, hop in enumerate(path.hops):
                    node_names.add(hop.source.name)
                    node_names.add(hop.target.name)
                    edge_names.add(hop.edge_name)
                    hop_node_map.append((hop_idx, hop.source.name, hop.edge_name))
                    hop_node_map.append((hop_idx, hop.target.name, hop.edge_name))

        claims: list[AtomicClaim] = []
        for i, sent in enumerate(sentences):
            # 找出该句中出现的节点/边
            sent_nodes = [n for n in node_names if n and n in sent]
            sent_edges = [e for e in edge_names if e in sent]

            # 推断 hop_index：取首次出现的节点对应的 hop
            hop_index = -1
            for hi, hn, _ in hop_node_map:
                if hn in sent:
                    hop_index = hi
                    break

            claims.append(
                AtomicClaim(
                    claim_id=f"c_{i}",
                    claim_text=sent,
                    hop_index=hop_index,
                    supporting_path_index=0,
                    source_nodes=sent_nodes,
                    source_edges=sent_edges,
                    confidence=0.6 if (sent_nodes or sent_edges) else 0.3,
                )
            )

        return ClaimDecomposition(
            original_answer=answer,
            claims=claims,
            parser="rule",
        )

    def _format_paths_for_prompt(self, paths: list[CausalPath]) -> str:
        """格式化路径供 LLM 阅读。"""
        lines: list[str] = []
        for i, path in enumerate(paths, 1):
            lines.append(f"路径 {i}（置信度 {path.path_confidence:.3f}，{path.hop_count} 跳）：")
            lines.append(f"  起点: {path.start_node.name} ({path.start_node.label})")
            for j, hop in enumerate(path.hops):
                lines.append(
                    f"  跳 {j}: --[{hop.edge_name} lag={hop.lag_seconds}s]--> "
                    f"{hop.target.name} ({hop.target.label})"
                )
            if path.end_node and path.end_node.label == "Solution":
                lines.append(f"  终点: {path.end_node.name} ({path.end_node.label})")
        return "\n".join(lines)


# ============================================================
#  LLM JSON 响应解析（处理 markdown 包裹、BOM、混合内容）
# ============================================================

import json as _json  # noqa: E402 — 工具函数需要
import re as _re  # noqa: E402


def _parse_llm_json_response(response: str) -> dict:
    """从 LLM 响应中提取 JSON dict。

    LLM 经常在 JSON 外包一层 markdown：
        ```json
        {"claims": [...]}
        ```

    或者前/后混入解释文字、代码块标记、BOM 字符等。这里用多重策略依次尝试：
    1. 直接 json.loads（理想情况）
    2. 去 BOM、strip 前后空白
    3. 抽 ```json ... ```  代码块
    4. 抽 ``` ... ```  代码块
    5. 用 brace matching 找第一个完整 {...}
    """
    if not response or not response.strip():
        raise ValueError("LLM 响应为空")

    text = response

    # 1. 去除 BOM
    if text.startswith("\ufeff"):
        text = text[1:]

    # 2. 去除前后空白
    text = text.strip()

    # 3. 直接 json.loads
    try:
        return _json.loads(text)
    except _json.JSONDecodeError:
        pass

    # 4. 抽 ```json ... ``` 代码块
    m = _re.search(r"```(?:json)?\s*\n(.*?)\n```", text, _re.DOTALL)
    if m:
        try:
            return _json.loads(m.group(1).strip())
        except _json.JSONDecodeError:
            pass

    # 5. 抽 ``` ... ``` （无语言标识）
    m = _re.search(r"```\s*\n?(.*?)\n?```", text, _re.DOTALL)
    if m:
        try:
            return _json.loads(m.group(1).strip())
        except _json.JSONDecodeError:
            pass

    # 6. brace matching：找第一个 { 到对应 }
    start = text.find("{")
    if start >= 0:
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            c = text[i]
            if escape:
                escape = False
                continue
            if c == "\\":
                escape = True
                continue
            if c == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        return _json.loads(candidate)
                    except _json.JSONDecodeError:
                        break

    # 7. 全失败：抛清晰错误
    preview = text[:200].replace("\n", "\\n")
    raise ValueError(
        f"无法从 LLM 响应中解析 JSON（响应前 200 字符: {preview!r}）"
    )
