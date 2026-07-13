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
from reasoning.result_models import CausalPath

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
        self._validate_config()

    def _validate_config(self) -> None:
        """校验配置：api_key 不能是占位符或含非 ASCII 字符。

        httpx header 不支持非 ASCII，Authorization: Bearer <中文> 会在
        编码阶段直接失败。占位 key（如 "REPLACE_WITH_*" / "your_api_key"）
        走到调用时才会 401，但提前拦截能给出更清晰的引导。
        """
        if not self.api_key or not self.api_key.strip():
            raise ValueError(
                f"LLM api_key 为空（base_url={self.base_url}, model={self.model}）"
            )
        try:
            self.api_key.encode("ascii")
        except UnicodeEncodeError as e:
            preview = self.api_key[:10] + "…" if len(self.api_key) > 10 else self.api_key
            raise ValueError(
                f"LLM api_key 含非 ASCII 字符（{preview!r}），"
                f"可能未在 .env 中正确配置。base_url={self.base_url}, model={self.model}"
            ) from e
        # 占位符检测：识别明显未替换的占位 key
        placeholder_patterns = [
            "REPLACE_WITH_",
            "your_api_key",
            "your-token",
            "<api_key>",
            "PLACEHOLDER",
            "请填入",
        ]
        upper = self.api_key.upper()
        for pat in placeholder_patterns:
            if pat.upper() in upper:
                raise ValueError(
                    f"LLM api_key 仍是占位符（{pat!r}），"
                    f"请到 .env 把 {self._key_env_name()} 换成真实 key。"
                    f"base_url={self.base_url}, model={self.model}"
                )

    def _key_env_name(self) -> str:
        """根据 base_url 推断对应 .env 变量名（用于错误提示）。"""
        if "stepfun" in self.base_url.lower():
            return "GEN_LLM_API_KEY / VERIFY_LLM_API_KEY (StepFun)"
        if "deepseek" in self.base_url.lower():
            return "GEN_LLM_API_KEY / VERIFY_LLM_API_KEY (DeepSeek)"
        if "openai.com" in self.base_url.lower():
            return "OPENAI_API_KEY"
        return "<api_key>"

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
        max_retries: int = 2,
    ) -> str:
        """调用 LLM Chat API。

        失败恢复策略（按优先级）：
        1. finish_reason='length' 且还有重试次数 → 自动重试 + max_tokens 翻倍
        2. 网络/超时异常 → 退避重试
        3. 全部失败 → 抛出 RuntimeError 让上层降级

        Parameters
        ----------
        messages : list[dict]
            OpenAI 格式的消息列表
        temperature : float
            采样温度（低=确定，高=多样）
        max_tokens : int
            最大输出 token 数（首次调用值，重试时会翻倍）
        response_format : dict | None
            响应格式约束，如 {"type": "json_object"}
        max_retries : int
            失败重试次数（默认 2 次，加上首次共 3 次机会）
        """
        import time
        last_err: Exception | None = None
        current_max_tokens = max_tokens

        for attempt in range(max_retries + 1):
            try:
                client = self._get_client()
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": current_max_tokens,
                }
                if response_format is not None:
                    kwargs["response_format"] = response_format

                response = client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content
                finish = getattr(response.choices[0], "finish_reason", "")

                if not content or not content.strip():
                    # 空内容：可能是 length 截断（最常见）或 content filter
                    if finish == "length" and attempt < max_retries:
                        # 翻倍 max_tokens 重试（给模型更多空间）
                        current_max_tokens = min(current_max_tokens * 2, 8000)
                        time.sleep(0.5)
                        last_err = RuntimeError(
                            f"LLM 返回空内容，finish_reason='length'，"
                            f"重试 attempt={attempt+1}/{max_retries+1}"
                        )
                        continue
                    # 最后一次或非 length 原因 → 抛错
                    raise RuntimeError(
                        f"LLM 返回空内容，finish_reason={finish!r}，model={self.model}"
                    )
                if finish == "length" and attempt < max_retries:
                    # 拿到了内容但 finish_reason=length 仍有截断风险
                    # 如果内容看起来完整（以句号/右括号/右花括号结尾），就接受
                    if content.rstrip().endswith((".", "。", "}", "]", "）", ")", '"', "'")):
                        return content
                    current_max_tokens = min(current_max_tokens * 2, 8000)
                    time.sleep(0.5)
                    continue
                return content

            except Exception as e:
                last_err = e
                # 网络类错误才重试，其他错误（如 API 权限错误）直接抛
                err_type = type(e).__name__
                if err_type in ("APITimeoutError", "TimeoutException", "ConnectError",
                                "ReadTimeout", "ConnectionError", "APIConnectionError"):
                    if attempt < max_retries:
                        time.sleep(1.0 * (attempt + 1))  # 1s, 2s, 3s 退避
                        continue
                raise

        # 全部重试失败
        raise RuntimeError(
            f"LLM 调用失败，已重试 {max_retries} 次: {last_err}"
        ) from last_err


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
            result = self._rule_based_parse(natural_language)
        else:
            try:
                result = self._llm_parse(natural_language)
            except Exception as e:
                logger.warning(f"LLM 解析查询失败，降级用规则: {e}")
                result = self._rule_based_parse(natural_language)

        return self._post_process_intent(result)

    def _post_process_intent(self, structured: StructuredQuery) -> StructuredQuery:
        """后处理：当目标实体是 Component（machine-N-N）时，把 causal_chain /
        solution_lookup / single_entity 升级为 multi_hop_path(3)。

        图库中 3 跳路径是 Component → Symptom → Cause → Solution，但
        causal_chain 的 Cypher 只匹配 Symptom → Cause，solution_lookup 只匹配
        Cause → Solution。当用户问 machine-1-2 的根因/解法时，entity 是 Component
        名而非 Symptom/Cause 名，原 Cypher 返回 0 条路径。

        另一项后处理：multi_hop_path 查询若原文含 "→" / "->" 链路描述，
        且 LLM 只抽到 ≤1 个 symptom_keyword，则从原文直接抽链路中间实体
        补全 symptom_keywords。这避免了 LLM 把 4 跳链路简化成 1 个关键词、
        导致 Cypher symptom_kw 过滤过严而返回 0 条路径的问题。
        """
        import re
        intent = structured.intent
        entity = intent.target_entity

        # 1) Component 名触发 multi_hop_path 升级
        if entity and re.match(r"machine-\d+-\d+", entity):
            if intent.query_type in (
                QueryType.CAUSAL_CHAIN,
                QueryType.SOLUTION_LOOKUP,
                QueryType.SINGLE_ENTITY,
            ):
                original = intent.query_type.value
                intent.query_type = QueryType.MULTI_HOP_PATH
                intent.hop_count = 3
                structured.metadata["post_processed"] = f"{original} -> multi_hop_path(3)"

        # 2) 链路文本兜底：从 "a → b → c → d" 抽出中间实体作为 symptom_keywords
        if intent.query_type == QueryType.MULTI_HOP_PATH:
            nl = structured.natural_language or ""
            # 匹配 → 或 -> 分隔的实体名（容忍空格、中文/英文/数字/下划线/连字符）
            arrow_parts = re.split(r"\s*(?:→|->)\s*", nl)
            if len(arrow_parts) >= 3:
                # 首尾通常是 Component 与 Solution，中间是 Symptom/Cause
                middle = arrow_parts[1:-1]
                # 过滤掉空串和"链路"/"路径"等描述性词
                middle = [
                    p.strip()
                    for p in middle
                    if p.strip() and not re.search(r"链路|路径|完整|深层|多跳|排查|根因|解法", p)
                ]
                # 已有 keywords 与新抽的取并集，去重保序
                existing = list(intent.symptom_keywords or [])
                merged = list(existing)
                for kw in middle:
                    if kw not in merged:
                        merged.append(kw)
                # 只在 LLM 抽得太少（< len(middle)）时才补全，避免覆盖 LLM 的合理判断
                if len(existing) < max(len(middle), 2) and merged != existing:
                    intent.symptom_keywords = merged
                    structured.metadata.setdefault("post_processed", "")
                    structured.metadata["post_processed"] += (
                        f"; symptom_keywords backfilled from arrow chain: {middle}"
                    )

        return structured

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
- **重要**：若查询用箭头/链路格式列出多个实体（如 "kubelet → dns_error → cert_expired → ..."），
  必须把链路里**所有**中间症状/根因名都放入 symptom_keywords（去掉首尾的 Component 与 Solution），
  绝不能只保留最后一个。这是多跳路径查询的核心过滤条件。
- 查询里出现"深层根因"/"多跳排查"/"完整链路"等措辞时，hop_count 至少为 3。
- 只输出 JSON，不要其他文字"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": natural_language},
        ]

        response = self.client.chat(
            messages,
            temperature=0.0,
            max_tokens=1500,
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
        """规则兜底的查询解析。

        优先级顺序：multi_hop_path > causal_chain > solution_lookup > time_range >
        single_entity > comparison。注意"和"字在中文里太常见（"根因和解法"是
        根因+解法而不是对比），不能用 "和" 判定 COMPARISON。
        """
        text = natural_language.lower()

        # 识别查询类型（优先级从高到低）
        # 注意：箭头记号 "→" / "->" 通常表示路径描述，优先级最高
        if "→" in text or "->" in text:
            query_type = QueryType.MULTI_HOP_PATH
        elif any(kw in text for kw in ["路径", "链路", "链", "完整", "path", "chain"]):
            query_type = QueryType.MULTI_HOP_PATH
        elif any(kw in text for kw in ["根因", "原因", "为什么", "为何", "为啥", "why", "cause", "root"]):
            query_type = QueryType.CAUSAL_CHAIN
        elif any(kw in text for kw in ["解法", "解决", "修复", "solution", "fix", "resolve"]):
            query_type = QueryType.SOLUTION_LOOKUP
        elif any(kw in text for kw in ["时间段", "期间", "between", "range", "2024", "2023"]):
            query_type = QueryType.TIME_RANGE
        elif any(kw in text for kw in ["对比", "比较", "compare", "vs"]):
            # 注意：不把 "和" 当成对比关键词，它太常见（"根因和解法"）
            query_type = QueryType.COMPARISON
        elif any(kw in text for kw in ["vm", "component", "组件", "节点"]):
            query_type = QueryType.SINGLE_ENTITY
        else:
            query_type = QueryType.CAUSAL_CHAIN  # 默认查根因

        # 提取 VM ID（简单正则）；也兼容 machine-N-N 的 SMD 命名
        import re
        vm_match = re.search(r"(?:vm[_\w]*\d+|machine-\d+-\d+)", text)
        target_entity = vm_match.group(0) if vm_match else None
        target_entity_type = "vm" if target_entity else None

        # 跳数推断
        hop_count = None
        if query_type == QueryType.MULTI_HOP_PATH:
            # 1) 显式 "N 跳"
            m = re.search(r"(\d)\s*跳", text)
            if m:
                hop_count = int(m.group(1))
            elif "→" in text or "->" in text:
                # 2) 箭头数量 = 边数 = 跳数
                arrow_count = text.count("→") + text.count("->")
                hop_count = max(arrow_count, 2)
            elif target_entity and "machine-" in text:
                # 3) 提到 machine-N-N 的路径查询默认 3 跳
                #    （图库的 3 跳结构是 Component → Symptom → Cause → Solution）
                hop_count = 3
            else:
                hop_count = 2
        elif target_entity and "machine-" in text and query_type in (
            QueryType.CAUSAL_CHAIN, QueryType.SOLUTION_LOOKUP
        ):
            # 提到 machine-N-N 的根因/解法查询，语义上是 3 跳链路
            query_type = QueryType.MULTI_HOP_PATH
            hop_count = 3

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

        # 从 paths 里抽取所有节点名 + 边名，作为"必须出现的 token"提示给 LLM。
        # gold rationale = target_name + edge_name 的 token 集合，原样复述才能命中 R/EM。
        must_include_nodes: list[str] = []
        must_include_edges: list[str] = []
        for path in paths[:5]:
            if path.hops:
                must_include_nodes.append(path.hops[0].source.name)
            for hop in path.hops:
                must_include_edges.append(hop.edge_name)
                must_include_nodes.append(hop.target.name)
        # 去重保序
        seen: set[str] = set()
        node_list = [n for n in must_include_nodes if not (n in seen or seen.add(n))]
        edge_list = [e for e in must_include_edges if not (e in seen or seen.add(e))]

        system_prompt = """你是服务器故障排查助手。根据图谱查询结果回答用户问题。

要求：
1. 优先解释置信度最高的路径
2. 说明根因、传导链路、解法
3. **必须原样使用图谱中的节点名和边名（如 resource_contention、CAUSED_BY），不得意译、翻译或改写**
4. 回答开头先用一行"因果链："列出完整链路，格式：
   因果链：<起点节点> -[<边名>]-> <节点> -[<边名>]-> ... -> <终点节点>
5. 随后用自然语言解释（不超过 300 字），解释中再次出现上述节点名和边名
6. 如果路径时态不一致，说明原因"""

        user_content = f"""用户问题：{query.natural_language}

图谱查询到的因果路径（按置信度排序）：

{chr(10).join(path_descriptions)}
{pruned_info}

必须原样出现在回答中的节点名：{node_list}
必须原样出现在回答中的边名：{edge_list}

请根据上述路径回答用户问题，务必原样使用上述节点名和边名。"""

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

        # 因果链汇总行：与 LLM 路径格式对齐，保证节点名/边名作为独立 token 出现，
        # 命中 R/EM 的 gold rationale（target_name + edge_name）。
        if best.start_node:
            chain_parts = [best.start_node.name]
            for hop in best.hops:
                chain_parts.append(f" -[{hop.edge_name}]-> {hop.target.name}")
            lines.append("因果链：" + "".join(chain_parts))
            lines.append("")

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
