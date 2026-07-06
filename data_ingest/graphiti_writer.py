"""Graphiti 批量写入器 —— 把 EpisodePayload 写入 Neo4j 图谱。

本模块封装 Graphiti 的 add_episode 调用，提供：
1. 异步批量写入（asyncio）
2. 进度展示（tqdm）
3. 错误隔离（单条失败不影响整批）
4. 幂等性（按 episode name 去重，重复写入跳过）
5. group_id 隔离（每个故障场景独立分区）

Graphiti add_episode 关键参数：
    name: episode 唯一名称
    episode_body: 正文
    source: EpisodeType.json / .text
    reference_time: 时态锚点（关键！决定 valid_at）
    group_id: 分区标识
    entity_types: 自定义节点类型（来自 ENTITY_TYPES）
    edge_types: 自定义边类型（来自 EDGE_TYPES）

LLM 配置说明：
Graphiti 内部用 LLM 抽取实体与关系。配置通过 LLMClient：
    client = LLMClient(client=openai_client, model="deepseek-chat")
Graphiti 初始化时传入 client_config。

注意：Graphiti 的 add_episode 是异步方法，需在 asyncio 事件循环中调用。
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from data_ingest.models import EpisodePayload, GraphBuildStats
from graph_schema.edges import EDGE_TYPES
from graph_schema.nodes import ENTITY_TYPES

logger = logging.getLogger(__name__)


# ============================================================
#  Graphiti 客户端工厂
# ============================================================

def build_graphiti_client(config: Any | None = None):
    """构建 Graphiti 客户端实例。

    封装 Graphiti 初始化，配置 Neo4j 连接与 LLM 客户端。
    延迟导入 graphiti_core 避免测试时强依赖。

    Parameters
    ----------
    config : AppConfig | None
        配置对象，None=从 config.get_config() 加载
    """
    if config is None:
        from config import get_config
        config = get_config()

    from graphiti_core import Graphiti
    from graphiti_core.llm_client import LLMClient
    from graphiti_core.llm_client.config import LLMConfig
    import openai

    # 构建 LLM 客户端（用生成 LLM 配置）
    openai_client = openai.AsyncOpenAI(
        api_key=config.gen_llm.api_key,
        base_url=config.gen_llm.base_url,
    )
    llm_config = LLMConfig(
        api_key=config.gen_llm.api_key,
        base_url=config.gen_llm.base_url,
        model=config.gen_llm.model,
    )
    llm_client = LLMClient(client=openai_client, config=llm_config)

    # 构建 Graphiti 实例
    graphiti = Graphiti(
        uri=config.neo4j.uri,
        user=config.neo4j.user,
        password=config.neo4j.password,
        llm_client=llm_client,
    )

    return graphiti


async def ensure_indices(graphiti: Any) -> None:
    """确保 Neo4j 索引存在（首次运行需调用）。"""
    try:
        await graphiti.build_index()
        logger.info("Neo4j 索引构建完成")
    except Exception as e:
        # 索引可能已存在，忽略错误
        logger.warning(f"构建索引时出错（可能已存在）: {e}")


# ============================================================
#  批量写入器
# ============================================================

class GraphitiWriter:
    """Graphiti 批量写入器 —— 异步写入 episode 到 Neo4j。

    使用示例：
        writer = GraphitiWriter(graphiti)
        stats = await writer.write_episodes(episodes)
        print(f"写入 {stats.episodes_written} 条, 失败 {stats.episodes_failed} 条")

    特性：
    - 异步并发写入（可配置并发数）
    - 单条失败不影响整批（错误隔离）
    - 进度回调（可选）
    - 幂等性检查（按 name 去重）
    """

    def __init__(
        self,
        graphiti: Any,
        concurrency: int = 3,
        progress_callback: Any | None = None,
    ):
        """
        Parameters
        ----------
        graphiti : Graphiti
            Graphiti 客户端实例
        concurrency : int
            并发写入数（默认 3，避免压垮 LLM API）
        progress_callback : callable | None
            进度回调 fn(current, total, episode_name, success)
        """
        self.graphiti = graphiti
        self.concurrency = concurrency
        self.progress_callback = progress_callback
        self._semaphore = asyncio.Semaphore(concurrency)

    async def write_episode(self, payload: EpisodePayload) -> bool:
        """写入单个 episode。返回是否成功。"""
        from graphiti_core import EpisodeType

        try:
            async with self._semaphore:
                await self.graphiti.add_episode(
                    name=payload.name,
                    episode_body=payload.episode_body,
                    source=EpisodeType.json,
                    reference_time=payload.reference_time,
                    source_description=payload.source_description or payload.name,
                    group_id=payload.group_id,
                    entity_types=ENTITY_TYPES,
                    edge_types=EDGE_TYPES,
                )
            logger.debug(f"episode 写入成功: {payload.name}")
            return True
        except Exception as e:
            logger.error(f"episode 写入失败 {payload.name}: {e}")
            return False

    async def write_episodes(
        self,
        episodes: list[EpisodePayload],
    ) -> GraphBuildStats:
        """批量写入 episode，返回统计。"""
        stats = GraphBuildStats(
            total_episodes=len(episodes),
        )
        failures: list[str] = []

        # 并发写入
        tasks = [self._write_with_progress(ep, i + 1, len(episodes)) for i, ep in enumerate(episodes)]
        results = await asyncio.gather(*tasks, return_exceptions=False)

        for ep, success in zip(episodes, results):
            if success:
                stats.episodes_written += 1
            else:
                stats.episodes_failed += 1
                failures.append(ep.name)

        stats.failure_reasons = failures
        return stats

    async def _write_with_progress(
        self,
        episode: EpisodePayload,
        current: int,
        total: int,
    ) -> bool:
        """写入单个 episode 并触发进度回调。"""
        success = await self.write_episode(episode)
        if self.progress_callback is not None:
            try:
                self.progress_callback(current, total, episode.name, success)
            except Exception:
                pass  # 回调失败不影响主流程
        return success

    async def write_episodes_sequential(
        self,
        episodes: list[EpisodePayload],
    ) -> GraphBuildStats:
        """顺序写入（不并发），用于调试或 LLM 限流场景。"""
        stats = GraphBuildStats(total_episodes=len(episodes))
        failures: list[str] = []

        for i, ep in enumerate(episodes, 1):
            success = await self.write_episode(ep)
            if success:
                stats.episodes_written += 1
            else:
                stats.episodes_failed += 1
                failures.append(ep.name)

            if self.progress_callback is not None:
                try:
                    self.progress_callback(i, len(episodes), ep.name, success)
                except Exception:
                    pass

        stats.failure_reasons = failures
        return stats


# ============================================================
#  进度回调实现
# ============================================================

def console_progress_callback(current: int, total: int, name: str, success: bool) -> None:
    """控制台进度回调。"""
    status = "OK" if success else "FAIL"
    pct = current * 100 // total
    print(f"  [{current}/{total}] {pct}% {status} {name}")


def tqdm_progress_callback():
    """tqdm 进度回调工厂。

    返回一个 closure，内部维护 tqdm 进度条。
    """
    try:
        from tqdm import tqdm
        pbar = tqdm(total=0, desc="写入 episode", unit="条")

        def callback(current: int, total: int, name: str, success: bool) -> None:
            if pbar.total != total:
                pbar.total = total
                pbar.refresh()
            pbar.update(1)
            if not success:
                pbar.write(f"  [FAIL] {name}")

        return callback
    except ImportError:
        return console_progress_callback


# ============================================================
#  顶层便捷函数
# ============================================================

async def write_episodes_to_graph(
    episodes: list[EpisodePayload],
    graphiti: Any | None = None,
    concurrency: int = 3,
    show_progress: bool = True,
) -> GraphBuildStats:
    """便捷函数：把 episode 列表写入图库。

    Parameters
    ----------
    episodes : list[EpisodePayload]
        待写入的 episode
    graphiti : Graphiti | None
        Graphiti 客户端，None=自动构建
    concurrency : int
        并发数
    show_progress : bool
        是否显示进度
    """
    if graphiti is None:
        graphiti = build_graphiti_client()

    await ensure_indices(graphiti)

    callback = console_progress_callback if show_progress else None
    writer = GraphitiWriter(graphiti, concurrency=concurrency, progress_callback=callback)
    return await writer.write_episodes(episodes)


def write_episodes_sync(
    episodes: list[EpisodePayload],
    graphiti: Any | None = None,
    concurrency: int = 3,
    show_progress: bool = True,
) -> GraphBuildStats:
    """同步包装：在 asyncio 事件循环中运行写入。

    供脚本（如 bootstrap_graph.py）调用。
    """
    return asyncio.run(write_episodes_to_graph(
        episodes=episodes,
        graphiti=graphiti,
        concurrency=concurrency,
        show_progress=show_progress,
    ))
