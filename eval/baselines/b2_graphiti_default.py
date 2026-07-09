"""B2: Graphiti default baseline —— Graphiti 内置 search + LLM 回答。

隔离「自研控制器」对答案质量的边际价值。
B4 - B2 ≈ 自研推理控制器的净收益。
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from eval.baselines.common import BaselineResult, BaselineRunner
from reasoning.hallucination_verifier import HallucinationVerifier
from reasoning.llm_interpreter import LLMClient
from reasoning.result_models import PathHop

logger = logging.getLogger(__name__)


# Graphiti search 检索结果 → 自然语言答案
GRAPHITI_DEFAULT_PROMPT = """你是一个云原生运维专家。基于以下知识图谱检索结果，回答用户问题。

检索结果：
{context}

问题：{query}

要求：简洁回答（3-5 句），每条事实用 [1][2][3] 编号。"""


class B2GraphitiDefault(BaselineRunner):
    """B2 — Graphiti 内置混合检索（COMBINED_HYBRID_SEARCH_CROSS_ENCODER）。"""

    name = "B2_GraphitiDefault"

    def __init__(
        self,
        graphiti_client: Any | None = None,
        llm_client: LLMClient | None = None,
    ):
        """
        Parameters
        ----------
        graphiti_client : Graphiti | None
            Graphiti 客户端；None=该 baseline 不可用
        llm_client : LLMClient | None
            LLM 客户端（用于把检索结果合成为答案）
        """
        self.graphiti = graphiti_client
        self.llm_client = llm_client
        self.verifier = HallucinationVerifier(client=llm_client)
        # 关键：Graphiti 内部的 Neo4j async driver 绑定到创建它的事件循环。
        # 每次 predict 都 new_event_loop 会触发
        # "got Future attached to a different loop" 错误。
        # 解决：所有 predict 共用一个后台事件循环线程，async task 提交到那里执行。
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: Any = None
        if self.graphiti is not None:
            self._start_loop()

    def _start_loop(self) -> None:
        """启动一个专用的事件循环线程，供所有 predict 共用。

        关键点：Graphiti 内部的 Neo4j async driver 绑定到创建它的事件循环，
        所以 _loop 必须在持有 graphiti client 的线程里创建，而不是随便 new 一个。
        """
        import threading
        ready = threading.Event()
        self._loop_ref: dict[str, Any] = {}

        def _runner():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop_ref["loop"] = loop
            ready.set()
            loop.run_forever()
            loop.close()

        self._thread = threading.Thread(target=_runner, daemon=True)
        self._thread.start()
        ready.wait(timeout=10)
        self._loop = self._loop_ref.get("loop")

    def _run_async(self, coro: Any) -> Any:
        """在专用事件循环线程里跑协程。"""
        if self._loop is None:
            raise RuntimeError("B2 事件循环未启动")
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result()

    async def _async_search(self, query: str, group_ids: list[str] | None = None) -> Any:
        """调用 Graphiti 内置 search（异步）。

        graphiti-core 0.29.x 用 search_() 接受 SearchConfig，search() 简化为混合搜索。
        这里显式构造 SearchConfig 用 cosine + bm25 + bfs 三路召回（RRF 融合）。
        """
        if self.graphiti is None:
            return None
        from graphiti_core.search.search_config import (
            SearchConfig,
            EdgeSearchConfig,
            EdgeSearchMethod,
            EdgeReranker,
        )
        config = SearchConfig(
            edge_config=EdgeSearchConfig(
                search_methods=[
                    EdgeSearchMethod.cosine_similarity,
                    EdgeSearchMethod.bm25,
                    EdgeSearchMethod.bfs,
                ],
                reranker=EdgeReranker.rrf,
                bfs_max_depth=3,
            ),
            limit=10,
        )
        if group_ids:
            from graphiti_core.search.search_filters import SearchFilters
            sf = SearchFilters(group_ids=group_ids)
            return await self.graphiti.search_(
                query=query, config=config, search_filter=sf
            )
        return await self.graphiti.search_(query=query, config=config)

    def predict(self, case) -> BaselineResult:
        t0 = time.time()
        try:
            if self.graphiti is None or self.llm_client is None:
                return BaselineResult(
                    case_id=case.case_id,
                    baseline_name=self.name,
                    answer="(Graphiti client not configured)",
                    predicted_hops=[],
                    verified_claims=[],
                    elapsed_seconds=time.time() - t0,
                )

            # 1. Graphiti 内置 search（用专用事件循环线程）
            try:
                search_result = self._run_async(self._async_search(case.query))
            except Exception as e:
                logger.debug(f"Graphiti search 失败: {e}")
                search_result = None

            # 2. 把 search 结果转成 predicted_hops（如果 result 是 Edge 列表）
            predicted_hops: list[PathHop] = []
            context_text = "(no results)"
            if search_result is not None:
                edges = getattr(search_result, "edges", None) or []
                facts = []
                for edge in edges[:10]:
                    facts.append(
                        f"- {getattr(edge, 'source_node_name', '?')} → "
                        f"{getattr(edge, 'target_node_name', '?')}: "
                        f"{getattr(edge, 'fact', '?')}"
                    )
                    # 把 edge 包装成单跳 PathHop
                    try:
                        from reasoning.path_extractor import (
                            parse_neo4j_node,
                            parse_neo4j_relationship,
                        )
                        source_info = parse_neo4j_node({
                            "name": getattr(edge, "source_node_name", ""),
                            "_labels": [getattr(edge.source_node, "label", "Component")] if hasattr(edge, "source_node") else [],
                        })
                        target_info = parse_neo4j_node({
                            "name": getattr(edge, "target_node_name", ""),
                            "_labels": [],
                        })
                        rel_info = parse_neo4j_relationship({
                            "name": getattr(edge, "name", ""),
                            "valid_at": getattr(edge, "valid_at", None),
                            "invalid_at": getattr(edge, "invalid_at", None),
                            "attributes": {"fact": getattr(edge, "fact", "")},
                        })
                        predicted_hops.append(PathHop(
                            edge_name=rel_info["name"] or "RELATES_TO",
                            source=source_info,
                            target=target_info,
                            valid_at=rel_info["valid_at"],
                            invalid_at=rel_info["invalid_at"],
                            attributes=rel_info["attributes"],
                        ))
                    except Exception as e:
                        logger.debug(f"边转 PathHop 失败: {e}")

                context_text = "\n".join(facts) if facts else "(no facts)"

            # 3. LLM 合成答案
            answer = self.llm_client.chat(
                messages=[
                    {"role": "system", "content": "你是云原生运维专家。"},
                    {"role": "user", "content": GRAPHITI_DEFAULT_PROMPT.format(
                        context=context_text, query=case.query
                    )},
                ],
                temperature=0.1,
            )

            # 4. 拆解 + 核验
            from reasoning.claim_decomposer import ClaimDecomposer
            decomposer = ClaimDecomposer(client=None)  # 用规则拆解
            decomp = decomposer.decompose(answer, [])
            # 用空的 predicted_hops 核验，视为"无路径支撑"
            report = self.verifier.verify(decomp, [])

            return BaselineResult(
                case_id=case.case_id,
                baseline_name=self.name,
                answer=answer,
                predicted_hops=predicted_hops,
                verified_claims=report.verified,
                elapsed_seconds=time.time() - t0,
                metadata={"edge_count": len(predicted_hops)},
            )
        except Exception as e:
            logger.exception(f"B2 推理失败: {case.case_id}")
            return BaselineResult(
                case_id=case.case_id,
                baseline_name=self.name,
                error=str(e),
                elapsed_seconds=time.time() - t0,
            )
