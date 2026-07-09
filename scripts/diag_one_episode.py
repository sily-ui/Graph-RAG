"""诊断单条 episode 写入失败原因 — 绕过 try/except 直出 traceback。"""
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from data_ingest.graphiti_writer import build_graphiti_client
from data_ingest.models import EpisodePayload
from graphiti_core.nodes import EpisodeType
from graph_schema.edges import EDGE_TYPES
from graph_schema.nodes import ENTITY_TYPES
from datetime import datetime, timezone


async def main():
    print("--- 构建 Graphiti 客户端 ---")
    graphiti = build_graphiti_client()

    payload = EpisodePayload(
        name="diag_test_001",
        episode_body=(
            '{"event_id":"diag_test_001","type":"cpu_spike",'
            '"machine":"machine-1-1","metric":"cpu_user_rate",'
            '"value":0.95,"baseline":0.4,"threshold":0.85}'
        ),
        reference_time=datetime(2026, 7, 7, 10, 0, 0, tzinfo=timezone.utc),
        group_id="diag_cluster",
        source_description="diag",
    )

    print("--- 调用 add_episode（直接 await，不包 try/except） ---")
    try:
        await graphiti.add_episode(
            name=payload.name,
            episode_body=payload.episode_body,
            source=EpisodeType.json,
            reference_time=payload.reference_time,
            source_description=payload.source_description,
            group_id=payload.group_id,
            entity_types=ENTITY_TYPES,
            edge_types=EDGE_TYPES,
        )
        print("OK: 写入成功")
    except Exception as e:
        import traceback
        print("FAILED with full traceback:")
        traceback.print_exc()
        print(f"\n异常类型: {type(e).__name__}")
        print(f"异常消息: {e}")


if __name__ == "__main__":
    asyncio.run(main())
