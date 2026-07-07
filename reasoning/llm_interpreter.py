"""LLM 解释器 —— 查询意图解析 + 结果自然语言解释。

本模块是推理控制器的 LLM 协同组件，负责：
1. parse_query：把自然语言查询解析成 StructuredQuery
2. explain：把图谱查询结果（CausalPath 列表）转成自然语言答案

设计原则：
- LLM 只做「理解」与「解释」，不做「检索」（检索交给 Cypher）
- LLM 输出必须是结构化 JSON（parse_query）或简洁文本（explain）
- 提供降级方案：LLM 失败时用规则兜底，保证可用性

LLM 客户端用 OpenAI SDK 兼容接口（DeepSeek/OpenAI/通义千问等均支持）。
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from reasoning.query_types import QueryIntent, QueryType, StructuredQuery, TimeWindow
from reasoning.result_models import CausalPath, ReasoningResult

logger = logging.getLogger(__name__)


# ============================================================
#  LLM 客户端封装
# ============================================================

class LLMClient:
    """LLM 客户端封装（OpenAI 兼容接口）。

    支持 DeepSeek/OpenAI/通义千问等任何兼容 OpenAI Chat API 的服务。
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout: int = 60,
    ):
        """
        Parameters
        ----------
        api_key : str
            API 密钥
        base_url : str
            API 基址，如 https://api.deepseek.com
        model : str
            模型名，如 deepseek-v4-flash
        timeout : int
            请求超时（秒）
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self._client: Any = None

    def _get_client(self):
        """懒加载 OpenAI 客户端。"""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as e:
                raise ImportError(
                    "需要 openai 库：pip install openai"
                ) from e
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )
        return self._client

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 2000,
        response_format: dict | None = None,
    ) -> str:
        """调用 LLM Chat API。

        Parameters
        ----------
        messages : list[dict]
            OpenAI 格式的消息列表
        temperature : float
            采样温度（低=确定，高=多样）
        max_tokens : int
            最大输出 token 数
        response_format : dict | None
            响应格式约束，如 {"type": "json_object"}
        """
        client = self._get_client()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format

        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""


# ============================================================
#  LLM 解释器
# ============================================================

class LLMInterpreter:
    """LLM 解释器 —— 查询意图解析 + 结果解释。

    使用示例：
        interpreter = LLMInterpreter.from_config(config.gen_llm)
        query = interpreter.parse_query("vm_001 为什么 CPU 飙高")
        # query.intent.query_type → CAUSAL_CHAIN
        # query.intent.target_entity → "vm_001"

        result = ReasoningResult(...)
        result.answer = interpreter.explain(query, paths)
    """

    def __init__(self, client: LLMClient | None = None):
        """
        Parameters
        ----------
        client : LLMClient | None
            LLM 客户端，None=降级模式（用规则兜底，不调 LLM）
        """
        self.client = client

    @classmethod
    def from_config(
        cls,
        api_key: str,
        base_url: str,
        model: str,
        timeout: int = 60,
    ) -> LLMInterpreter:
        """从配置创建 LLM 解释器。"""
        client = LLMClient(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout=timeout,
        )
        return cls(client=client)

    # ------------------------------------------------------------
    #  1. 查询意图解析
    # ------------------------------------------------------------

    def parse_query(self, natural_language: str) -> StructuredQuery:
        """把自然语言查询解析成 StructuredQuery。

        LLM 任务：识别查询类型、目标实体、关键词、时态窗口等。
        降级方案：LLM 失败时用关键词规则兜底。

        Parameters
        ----------
        natural_language : str
            自然语言查询，如 "vm_001 在 2024-01-01 期间发生了什么故障"

        Returns
        -------
        StructuredQuery
            结构化查询
        """
        if self.client is None:
            return self._rule_based_parse(natural_language)

        try:
            return self._llm_parse(natural_language)
        except Exception as e:
            logger.warning(f"LLM 解析查询失败，降级用规则: {e}")
            return self._rule_based_parse(natural_language)

    def _llm_parse(self, natural_language: str) -> StructuredQuery:
        """用 LLM 解析查询意图。"""
        system_prompt = """你是服务器故障排查助手。把用户的自然语言查询解析成结构化 JSON。

查询类型（query_type）：
- single_entity: 查某个 VM/组件的故障
- causal_chain: 查某症状的根因
- time_range: 查某时间段内的故障
- multi_hop_path: 查完整因果链（2/3/4 跳）
- solution_lookup: 查某故障的解法
- comparison: 对比多个 VM 的故障

输出 JSON 格式：
{
  "query_type": "causal_chain",
  "target_entity": "vm_001",
  "target_entity_type": "vm",
  "symptom_keywords": ["cpu_usage", "spike"],
  "cause_keywords": [],
  "hop_count": null,
  "time_window": null,
  "severity_filter": null,
  "limit": 10
}

注意：
- target_entity 是实体 ID（如 vm_001），不是描述
- time_window 格式：{"start": "2024-01-01T00:00:00", "end": "2024-01-02T00:00:00"}
- hop_count 只在 multi_hop_path 时设置（2/3/4），其他为 null
- 只输出 JSON，不要其他文字"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": natural_language},
        ]

        response = self.client.chat(
            messages,
            temperature=0.0,
            max_tokens=500,
            response_format={"type": "json_object"},
        )

        # 解析 LLM 输出
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            logger.warning(f"LLM 输出非合法 JSON: {response[:200]}")
            return self._rule_based_parse(natural_language)

        # 构建 QueryIntent
        intent = self._build_intent_from_dict(data)

        return StructuredQuery(
            natural_language=natural_language,
            intent=intent,
            metadata={"parser": "llm", "model": self.client.model if self.client else ""},
        )

    def _rule_based_parse(self, natural_language: str) -> StructuredQuery:
        """规则兜底的查询解析。"""
        text = natural_language.lower()

        # 识别查询类型
        if any(kw in text for kw in ["对比", "比较", "compare", "vs", "和"]):
            query_type = QueryType.COMPARISON
        elif any(kw in text for kw in ["时间段", "期间", "between", "range", "2024", "2023"]):
            query_type = QueryType.TIME_RANGE
        elif any(kw in text for kw in ["解法", "解决", "修复", "solution", "fix", "resolve"]):
            query_type = QueryType.SOLUTION_LOOKUP
        elif any(kw in text for kw in ["根因", "原因", "为什么", "为何", "为啥", "why", "cause", "root"]):
            query_type = QueryType.CAUSAL_CHAIN
        elif any(kw in text for kw in ["路径", "链路", "完整", "path", "chain"]):
            query_type = QueryType.MULTI_HOP_PATH
        elif any(kw in text for kw in ["vm", "component", "组件", "节点"]):
            query_type = QueryType.SINGLE_ENTITY
        else:
            query_type = QueryType.CAUSAL_CHAIN  # 默认查根因

        # 提取 VM ID（简单正则）
        import re
        vm_match = re.search(r"vm[_\w]*\d+", text)
        target_entity = vm_match.group(0) if vm_match else None
        target_entity_type = "vm" if target_entity else None

        # 跳数
        hop_count = None
        if query_type == QueryType.MULTI_HOP_PATH:
            if "3跳" in text or "3 跳" in text:
                hop_count = 3
            elif "4跳" in text or "4 跳" in text:
                hop_count = 4
            else:
                hop_count = 2

        # 严重程度
        severity = None
        if "critical" in text or "严重" in text:
            severity = "critical"
        elif "warning" in text or "警告" in text:
            severity = "warning"

        intent = QueryIntent(
            query_type=query_type,
            target_entity=target_entity,
            target_entity_type=target_entity_type,
            hop_count=hop_count,
            severity_filter=severity,
        )

        return StructuredQuery(
            natural_language=natural_language,
            intent=intent,
            metadata={"parser": "rule"},
        )

    def _build_intent_from_dict(self, data: dict[str, Any]) -> QueryIntent:
        """从 LLM 输出的 dict 构建 QueryIntent。"""
        # query_type
        qt_str = data.get("query_type", "causal_chain")
        try:
            query_type = QueryType(qt_str)
        except ValueError:
            query_type = QueryType.CAUSAL_CHAIN

        # time_window
        time_window = None
        tw_data = data.get("time_window")
        if tw_data and isinstance(tw_data, dict):
            try:
                start = datetime.fromisoformat(tw_data["start"])
                end = datetime.fromisoformat(tw_data["end"])
                time_window = TimeWindow(start=start, end=end)
            except (KeyError, ValueError, TypeError):
                pass

        return QueryIntent(
            query_type=query_type,
            target_entity=data.get("target_entity"),
            target_entity_type=data.get("target_entity_type"),
            symptom_keywords=data.get("symptom_keywords", []) or [],
            cause_keywords=data.get("cause_keywords", []) or [],
            hop_count=data.get("hop_count"),
            time_window=time_window,
            severity_filter=data.get("severity_filter"),
            limit=data.get("limit", 10),
        )

    # ------------------------------------------------------------
    #  2. 结果解释
    # ------------------------------------------------------------

    def explain(
        self,
        query: StructuredQuery,
        paths: list[CausalPath],
        pruned_paths: list[CausalPath] | None = None,
    ) -> str:
        """把图谱查询结果转成自然语言答案。

        Parameters
        ----------
        query : StructuredQuery
            原始查询
        paths : list[CausalPath]
            候选路径（已时态剪枝）
        pruned_paths : list[CausalPath] | None
            被剪枝的路径（用于解释为什么排除）

        Returns
        -------
        str
            自然语言答案
        """
        if not paths:
            return self._explain_no_result(query, pruned_paths or [])

        if self.client is None:
            return self._rule_based_explain(query, paths, pruned_paths or [])

        try:
            return self._llm_explain(query, paths, pruned_paths or [])
        except Exception as e:
            logger.warning(f"LLM 解释失败，降级用规则: {e}")
            return self._rule_based_explain(query, paths, pruned_paths or [])

    def _llm_explain(
        self,
        query: StructuredQuery,
        paths: list[CausalPath],
        pruned_paths: list[CausalPath],
    ) -> str:
        """用 LLM 生成自然语言解释。"""
        # 构建路径描述
        path_descriptions = []
        for i, path in enumerate(paths[:5], 1):  # 最多 5 条
            path_descriptions.append(f"路径 {i}（置信度 {path.path_confidence:.3f}）：\n{path.to_natural_language()}")

        pruned_info = ""
        if pruned_paths:
            pruned_info = f"\n\n被时态剪枝排除的路径：{len(pruned_paths)} 条"

        system_prompt = """你是服务器故障排查助手。根据图谱查询结果回答用户问题。

要求：
1. 优先解释置信度最高的路径
2. 说明根因、传导链路、解法
3. 简洁明了，不超过 300 字
4. 如果路径时态不一致，说明原因"""

        user_content = f"""用户问题：{query.natural_language}

图谱查询到的因果路径（按置信度排序）：

{chr(10).join(path_descriptions)}
{pruned_info}

请根据上述路径回答用户问题。"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        return self.client.chat(messages, temperature=0.3, max_tokens=800)

    def _rule_based_explain(
        self,
        query: StructuredQuery,
        paths: list[CausalPath],
        pruned_paths: list[CausalPath],
    ) -> str:
        """规则兜底的结果解释。"""
        if not paths:
            return "未找到匹配的因果路径。"

        best = paths[0]
        lines = [f"查询：{query.natural_language}", ""]

        # 根因
        root = best.root_cause
        if root:
            cause_type = root.cause_type or "未知"
            lines.append(f"根因：{root.name}（类型：{cause_type}）")

        # 传导链路
        if best.hop_count >= 2:
            lines.append(f"\n传导链路（{best.hop_count} 跳）：")
            lines.append(f"  {best.start_node.name}({best.start_node.label})")
            for hop in best.hops:
                lines.append(
                    f"  --[{hop.edge_name} 时延{hop.lag_seconds}s]--> "
                    f"{hop.target.name}({hop.target.label})"
                )

        # 解法
        for hop in best.hops:
            if hop.target.label == "Solution":
                sol_type = hop.target.solution_type or "未知"
                eff = hop.effectiveness
                eff_str = f"（有效率 {eff:.0%}）" if eff is not None else ""
                lines.append(f"\n解法：{hop.target.name}（类型：{sol_type}）{eff_str}")

        # 置信度
        lines.append(f"\n路径置信度：{best.path_confidence:.3f}")
        lines.append(f"总时延：{best.total_lag_seconds}s")

        if pruned_paths:
            lines.append(f"\n（另有 {len(pruned_paths)} 条路径因时态不一致被排除）")

        return "\n".join(lines)

    def _explain_no_result(
        self,
        query: StructuredQuery,
        pruned_paths: list[CausalPath],
    ) -> str:
        """无结果时的解释。"""
        lines = [f"查询：{query.natural_language}", "", "未找到匹配的因果路径。"]
        if pruned_paths:
            lines.append(f"\n有 {len(pruned_paths)} 条候选路径因时态不一致被排除：")
            for i, path in enumerate(pruned_paths[:3], 1):
                reason = path.pruned_reason or "未知原因"
                lines.append(f"  {i}. {reason}")
        return "\n".join(lines)

    # ------------------------------------------------------------
    #  3. 综合置信度评估
    # ------------------------------------------------------------

    def estimate_confidence(
        self,
        paths: list[CausalPath],
        pruned_paths: list[CausalPath],
    ) -> float:
        """估算综合置信度。

        综合置信度 = 路径置信度 × 时态剪枝通过率
        - 路径置信度：最佳路径的置信度
        - 通过率：保留路径数 / 总候选路径数

        Returns
        -------
        float
            [0, 1] 的置信度
        """
        total = len(paths) + len(pruned_paths)
        if total == 0:
            return 0.0

        pass_rate = len(paths) / total
        best_confidence = max((p.path_confidence for p in paths), default=0.0)

        # 综合置信度：路径置信度占 70%，通过率占 30%
        return best_confidence * 0.7 + pass_rate * 0.3
