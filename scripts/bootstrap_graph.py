r"""图谱构建脚本 —— 端到端把合成数据/真实数据灌入 Neo4j。

运行方式（在项目根目录）：
    python scripts/bootstrap_graph.py                       # 默认合成数据 50 VM
    python scripts/bootstrap_graph.py --vms 100             # 指定 VM 数量
    python scripts/bootstrap_graph.py --csv path/to.csv    # 用真实 Azure V2 CSV
    python scripts/bootstrap_graph.py --dry-run             # 只构建 episode 不写入图库

流程：
    1. 加载 VM 时序（合成或真实 CSV）
    2. 异常检测（IQR 法）
    3. 故障事件抽取
    4. 因果骨架匹配
    5. episode 构建
    6. Graphiti 写入 Neo4j
    7. 验证图谱统计（节点/关系数）

验证项（对应实现方案 Verification Steps #2）：
    - 实体数 ≥ 50
    - 关系数 ≥ 100
    - 覆盖四层概念
    - 时态字段正确填充
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

# 确保能导入项目根目录的模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_ingest.azure_trace_loader import AzureTraceLoader
from data_ingest.episode_builder import EpisodeBuilder
from data_ingest.fault_event_extractor import FaultEventExtractor
from data_ingest.models import FaultEvent, GraphBuildStats
from data_ingest.synthetic_data import generate_vm_batch


def build_episodes_offline(
    vms: list,
    fault_extractor: FaultEventExtractor,
    episode_builder: EpisodeBuilder,
    include_skeletons: bool = True,
    progress: bool = True,
) -> tuple[list, list[FaultEvent]]:
    """离线构建 episode（不写入图库）。"""
    all_events: list[FaultEvent] = []
    for i, ts in enumerate(vms):
        if progress:
            print(f"  [{i + 1}/{len(vms)}] VM={ts.vm_id} readings={ts.reading_count}"
                  f" deleted={ts.is_deleted}")
        # 异常检测
        loader = AzureTraceLoader(cluster_id=ts.cluster_id, detection_method="iqr")
        anomalies = loader.detect_anomalies(ts)
        if not anomalies:
            continue
        # 事件抽取
        events = fault_extractor.extract(ts, anomalies)
        all_events.extend(events)

    if progress:
        print(f"  共抽取 {len(all_events)} 个故障事件")

    # 构建 episode
    episodes = episode_builder.build_from_fault_events(all_events)
    if include_skeletons:
        skeletons = episode_builder.build_skeleton_episodes()
        episodes = skeletons + episodes

    if progress:
        print(f"  共构建 {len(episodes)} 个 episode (含 {len(episodes) - len(all_events)} 骨架)")

    return episodes, all_events


async def write_episodes_async(episodes: list, concurrency: int = 3) -> GraphBuildStats:
    """异步写入 episode 到图库。"""
    from data_ingest.graphiti_writer import (
        GraphitiWriter,
        build_graphiti_client,
        console_progress_callback,
        ensure_indices,
    )

    print("\n--- 初始化 Graphiti ---")
    graphiti = build_graphiti_client()
    await ensure_indices(graphiti)

    print(f"\n--- 写入 {len(episodes)} 个 episode (并发={concurrency}) ---")
    writer = GraphitiWriter(
        graphiti=graphiti,
        concurrency=concurrency,
        progress_callback=console_progress_callback,
    )
    return await writer.write_episodes(episodes)


def verify_graph_stats() -> dict:
    """查询图库统计，验证图谱构建结果。"""
    from neo4j import GraphDatabase
    from config import get_config

    cfg = get_config()
    driver = GraphDatabase.driver(
        cfg.neo4j.uri,
        auth=(cfg.neo4j.user, cfg.neo4j.password),
    )
    stats: dict = {}
    with driver.session() as session:
        # 节点总数
        stats["nodes"] = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        # 关系总数
        stats["relationships"] = session.run(
            "MATCH ()-[r]->() RETURN count(r) AS c"
        ).single()["c"]
        # 按 label 分组
        result = session.run(
            "MATCH (n) UNWIND labels(n) AS label "
            "RETURN label, count(*) AS c ORDER BY c DESC"
        )
        stats["by_label"] = {r["label"]: r["c"] for r in result}
        # 按 relationship name 分组
        result = session.run(
            "MATCH ()-[r:RELATES_TO]->() RETURN r.name AS name, count(*) AS c "
            "ORDER BY c DESC"
        )
        stats["by_edge_name"] = {r["name"]: r["c"] for r in result}
        # episode 数
        stats["episodes"] = session.run(
            "MATCH (e:Episode) RETURN count(e) AS c"
        ).single()["c"]
    driver.close()
    return stats


def print_stats(episodes: list, events: list[FaultEvent], stats: GraphBuildStats, elapsed: float) -> None:
    """打印构建结果统计。"""
    print("\n" + "=" * 60)
    print("  图谱构建完成")
    print("=" * 60)
    print(f"  故障事件数: {len(events)}")
    print(f"  episode 总数: {stats.total_episodes}")
    print(f"  写入成功: {stats.episodes_written}")
    print(f"  写入失败: {stats.episodes_failed}")
    print(f"  耗时: {elapsed:.1f}s")
    if stats.failure_reasons:
        print(f"  失败列表（前 5）: {stats.failure_reasons[:5]}")
    print("=" * 60)

    # 故障事件类型分布
    print("\n故障事件类型分布:")
    type_count: dict[str, int] = {}
    for ev in events:
        type_count[ev.event_type.value] = type_count.get(ev.event_type.value, 0) + 1
    for t, c in sorted(type_count.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c}")


def main():
    parser = argparse.ArgumentParser(
        description="Graph-RAG 图谱构建脚本（合成数据 / Azure V2 CSV）",
    )
    parser.add_argument(
        "--vms", type=int, default=50,
        help="合成 VM 数量（默认 50）",
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Azure V2 CSV 文件路径（不指定则用合成数据）",
    )
    parser.add_argument(
        "--cluster", type=str, default="cluster_A",
        help="集群标识（默认 cluster_A）",
    )
    parser.add_argument(
        "--observation-days", type=int, default=7,
        help="合成数据观察期天数（默认 7）",
    )
    parser.add_argument(
        "--fault-rate", type=float, default=0.2,
        help="合成数据故障率（默认 0.2）",
    )
    parser.add_argument(
        "--concurrency", type=int, default=3,
        help="Graphiti 写入并发数（默认 3）",
    )
    parser.add_argument(
        "--no-skeletons", action="store_true",
        help="不写入因果骨架 episode",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只构建 episode 不写入图库",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="写入后查询图库统计验证",
    )
    args = parser.parse_args()

    start_time = time.time()

    print("=" * 60)
    print("  Graph-RAG 图谱构建")
    print("=" * 60)

    # ---- 1. 加载 VM 时序 ----
    print("\n--- 1. 加载 VM 时序 ---")
    if args.csv:
        csv_path = Path(args.csv)
        if not csv_path.exists():
            print(f"[ERROR] CSV 文件不存在: {csv_path}")
            return 1
        loader = AzureTraceLoader(cluster_id=args.cluster, detection_method="iqr")
        # 用真实 Azure V2 长格式加载
        vms = list(loader.load_long(csv_path, max_vms=args.vms))
        print(f"  从长格式 CSV 加载 {len(vms)} 个 VM")
    else:
        print(f"  生成 {args.vms} 个合成 VM (cluster={args.cluster}, "
              f"days={args.observation_days}, fault_rate={args.fault_rate})")
        vms = generate_vm_batch(
            num_vms=args.vms,
            cluster_id=args.cluster,
            observation_days=args.observation_days,
            fault_rate=args.fault_rate,
            seed=42,
        )
        print(f"  合成完成，其中 {sum(1 for v in vms if v.is_deleted)} 个 VM 被删除")

    # ---- 2. 离线构建 episode ----
    print("\n--- 2. 异常检测 + 事件抽取 + episode 构建 ---")
    fault_extractor = FaultEventExtractor()
    episode_builder = EpisodeBuilder()
    episodes, events = build_episodes_offline(
        vms=vms,
        fault_extractor=fault_extractor,
        episode_builder=episode_builder,
        include_skeletons=not args.no_skeletons,
        progress=True,
    )

    if not episodes:
        print("\n[WARN] 未构建任何 episode（可能无故障事件）")
        return 0

    if args.dry_run:
        print("\n[dry-run] 跳过图库写入")
        elapsed = time.time() - start_time
        print_stats(episodes, events, GraphBuildStats(
            total_episodes=len(episodes),
            total_fault_events=len(events),
        ), elapsed)
        # 打印第一个 episode 示例
        print("\n首 episode 示例:")
        print("-" * 60)
        print(f"name: {episodes[0].name}")
        print(f"group_id: {episodes[0].group_id}")
        print(f"reference_time: {episodes[0].reference_time}")
        print(f"body (前 500 字符):\n{episodes[0].episode_body[:500]}...")
        return 0

    # ---- 3. 写入图库 ----
    print("\n--- 3. 写入 Neo4j 图库 ---")
    stats = asyncio.run(write_episodes_async(episodes, concurrency=args.concurrency))
    stats.total_fault_events = len(events)

    elapsed = time.time() - start_time
    print_stats(episodes, events, stats, elapsed)

    # ---- 4. 验证 ----
    if args.verify:
        print("\n--- 4. 验证图库统计 ---")
        try:
            graph_stats = verify_graph_stats()
            print(f"  节点总数: {graph_stats['nodes']}")
            print(f"  关系总数: {graph_stats['relationships']}")
            print(f"  episode 数: {graph_stats['episodes']}")
            print(f"  按 label:")
            for label, c in graph_stats["by_label"].items():
                print(f"    {label}: {c}")
            print(f"  按边名称:")
            for name, c in graph_stats["by_edge_name"].items():
                print(f"    {name}: {c}")

            # 验证条件
            print("\n验证条件:")
            ok_nodes = graph_stats["nodes"] >= 50
            ok_rels = graph_stats["relationships"] >= 100
            print(f"  [{'OK' if ok_nodes else 'FAIL'}] 节点数 ≥ 50: {graph_stats['nodes']}")
            print(f"  [{'OK' if ok_rels else 'FAIL'}] 关系数 ≥ 100: {graph_stats['relationships']}")
        except Exception as e:
            print(f"  [ERROR] 验证失败: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
