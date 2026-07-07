r"""图谱构建脚本 —— 端到端把合成数据/真实数据灌入 Neo4j。

支持四种数据源（``--source``）：
    - synthetic : 合成 VM 时序（默认，SyntheticDataGenerator）
    - azure     : Azure V2 真实 CSV（AzureTraceLoader，需配合 --csv）
    - smd       : SMD Server Machine Dataset（SMDLoader，需配合 --smd-dir）
    - gaia      : GAIA MicroSS 故障注入数据集（GAIALoader，需配合 --gaia-dir）

运行方式（在项目根目录）：
    # 合成数据（默认）
    python scripts/bootstrap_graph.py --vms 100 --dry-run

    # Azure V2 真实 CSV
    python scripts/bootstrap_graph.py --source azure --csv dataset/vmtablev2_000000000000.csv --vms 50
    python scripts/bootstrap_graph.py --csv path/to.csv    # 旧用法，缺省 --source 自动推断为 azure

    # SMD 真实数据
    python scripts/bootstrap_graph.py --source smd --smd-dir dataset/ServerMachineDataset --vms 5 --dry-run

    # GAIA 真实数据
    python scripts/bootstrap_graph.py --source gaia --gaia-dir dataset/GAIA-DataSet --dry-run

    python scripts/bootstrap_graph.py --verify             # 写入后查询图库统计

流程：
    1. 加载数据源（合成时序 / Azure CSV / SMD 多维时序 / GAIA 故障注入）
    2. 故障事件抽取（合成/Azure 走 IQR 异常检测；SMD/GAIA 由 loader 直接给出）
    3. 因果骨架匹配
    4. episode 构建
    5. Graphiti 写入 Neo4j
    6. 验证图谱统计（节点/关系数）

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
from data_ingest.gaia_loader import GAIALoader, download_gaia
from data_ingest.models import FaultEvent, GraphBuildStats
from data_ingest.smd_loader import SMDLoader, download_smd
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


def build_episodes_from_events(
    events: list[FaultEvent],
    episode_builder: EpisodeBuilder,
    include_skeletons: bool = True,
    progress: bool = True,
) -> tuple[list, list[FaultEvent]]:
    """从已有的 FaultEvent 列表构建 episode（用于 SMD/GAIA 等直出事件的数据源）。

    与 build_episodes_offline 的区别：跳过 VM 时序加载与 IQR 异常检测，
    因为 SMD 用 interpretation_label 抽窗口、GAIA 直接给故障注入记录，
    两者在 loader 内部已生成项目统一的 FaultEvent。

    Parameters
    ----------
    events : list[FaultEvent]
        loader 已抽取的故障事件列表
    episode_builder : EpisodeBuilder
        episode 构建器
    include_skeletons : bool
        是否在结果前追加因果骨架 episode
    progress : bool
        是否打印进度
    """
    if progress:
        print(f"  共 {len(events)} 个故障事件（由 loader 直接抽取）")

    episodes = episode_builder.build_from_fault_events(events)
    if include_skeletons:
        skeletons = episode_builder.build_skeleton_episodes()
        episodes = skeletons + episodes

    if progress:
        skeleton_count = len(episodes) - len(events)
        print(f"  共构建 {len(episodes)} 个 episode (含 {skeleton_count} 骨架)")

    return episodes, events


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


def print_stats(
    episodes: list,
    events: list[FaultEvent],
    stats: GraphBuildStats,
    elapsed: float,
    source_dataset: str = "synthetic",
) -> None:
    """打印构建结果统计。

    Parameters
    ----------
    source_dataset : str
        数据来源标识（synthetic/azure_v2/smd/gaia），从首个事件的 source_dataset
        字段或 --source 参数推导得到
    """
    print("\n" + "=" * 60)
    print("  图谱构建完成")
    print("=" * 60)
    print(f"  数据来源: {source_dataset}")
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
        description="Graph-RAG 图谱构建脚本（支持 synthetic / azure / smd / gaia 四种数据源）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ==================== 数据源选择 ====================
    src_group = parser.add_argument_group("数据源选择")
    src_group.add_argument(
        "--source", choices=["azure", "smd", "gaia", "synthetic"], default=None,
        help="数据源类型。不指定时按 --csv 是否存在自动推断（有 --csv → azure，否则 synthetic）",
    )
    src_group.add_argument(
        "--csv", type=str, default=None,
        help="Azure V2 长格式 CSV 路径（仅 --source azure 使用）",
    )
    src_group.add_argument(
        "--smd-dir", type=str, default=None,
        help="SMD 解压根目录（含 train/test/test_label/interpretation_label 子目录，仅 --source smd 使用）",
    )
    src_group.add_argument(
        "--gaia-dir", type=str, default=None,
        help="GAIA-DataSet 解压根目录（含 MicroSS/ 子目录，仅 --source gaia 使用）",
    )

    # ==================== 通用参数 ====================
    common_group = parser.add_argument_group("通用参数")
    common_group.add_argument(
        "--vms", type=int, default=50,
        help="合成 VM 数量（synthetic）/ Azure CSV 抽样上限 / SMD 加载机器数",
    )
    common_group.add_argument(
        "--cluster", type=str, default="cluster_A",
        help="集群标识（仅 synthetic / azure 生效）",
    )
    common_group.add_argument(
        "--observation-days", type=int, default=7,
        help="合成数据观察期天数（仅 synthetic 生效）",
    )
    common_group.add_argument(
        "--fault-rate", type=float, default=0.2,
        help="合成数据故障率（仅 synthetic 生效）",
    )
    common_group.add_argument(
        "--concurrency", type=int, default=3,
        help="Graphiti 写入并发数",
    )
    common_group.add_argument(
        "--no-skeletons", action="store_true",
        help="不写入因果骨架 episode",
    )
    common_group.add_argument(
        "--dry-run", action="store_true",
        help="只构建 episode 不写入图库",
    )
    common_group.add_argument(
        "--verify", action="store_true",
        help="写入后查询图库统计验证",
    )
    args = parser.parse_args()

    # ---- 向后兼容：缺省 --source 时按 --csv 推断 ----
    if args.source is None:
        args.source = "azure" if args.csv else "synthetic"

    # ---- 数据源参数校验 ----
    if args.source == "azure" and not args.csv:
        print("[ERROR] --source azure 必须配合 --csv <path> 使用")
        return 1
    if args.source == "smd" and not args.smd_dir:
        print("[ERROR] --source smd 必须配合 --smd-dir <path> 使用")
        print("  SMD 数据集获取方式：")
        print("    git clone https://github.com/NetManAIOps/OmniAnomaly.git tmp_omni")
        print("    mv tmp_omni/ServerMachineDataset dataset/ServerMachineDataset")
        print("    rm -rf tmp_omni")
        return 1
    if args.source == "gaia" and not args.gaia_dir:
        print("[ERROR] --source gaia 必须配合 --gaia-dir <path> 使用")
        print("  GAIA 数据集获取方式：")
        print("    git clone https://github.com/CloudWise-OpenSource/GAIA-DataSet.git dataset/GAIA-DataSet")
        return 1

    start_time = time.time()

    print("=" * 60)
    print("  Graph-RAG 图谱构建")
    print("=" * 60)
    print(f"  数据源: {args.source}")

    # ---- 1. 加载数据源 ----
    print(f"\n--- 1. 加载数据源 ({args.source}) ---")
    # 标识本次构建的数据来源，用于 stats 输出
    source_dataset: str = args.source
    # 走 VM 时序管线的数据源（synthetic / azure），返回 vms 列表
    vms: list | None = None
    # 直出 FaultEvent 的数据源（smd / gaia），返回 events 列表
    prebuilt_events: list[FaultEvent] | None = None

    if args.source == "azure":
        csv_path = Path(args.csv)  # type: ignore[arg-type]
        if not csv_path.exists():
            print(f"[ERROR] CSV 文件不存在: {csv_path}")
            return 1
        loader = AzureTraceLoader(cluster_id=args.cluster, detection_method="iqr")
        vms = list(loader.load_long(csv_path, max_vms=args.vms))
        print(f"  从长格式 CSV 加载 {len(vms)} 个 VM (cluster={args.cluster})")
        source_dataset = "azure_v2"

    elif args.source == "synthetic":
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
        source_dataset = "synthetic"

    elif args.source == "smd":
        smd_dir = Path(args.smd_dir)  # type: ignore[arg-type]
        if not smd_dir.exists():
            print(f"[ERROR] SMD 数据目录不存在: {smd_dir}")
            download_smd(str(smd_dir))
            return 1
        smd_loader = SMDLoader(str(smd_dir))
        machines = smd_loader.list_machines()
        if not machines:
            print(f"[ERROR] SMD 数据目录 {smd_dir} 下未发现任何机器（train/ 子目录为空）")
            download_smd(str(smd_dir))
            return 1
        print(f"  发现 {len(machines)} 台机器，按 --vms={args.vms} 限制加载数量")
        prebuilt_events = []
        loaded_count = 0
        for entity in smd_loader.load_all(max_machines=args.vms):
            loaded_count += 1
            windows = len(entity.anomaly_windows)
            print(f"  [{loaded_count}/{min(args.vms, len(machines))}] "
                  f"machine={entity.machine_id} group={entity.group_id} "
                  f"anomaly_windows={windows}")
            events = smd_loader.extract_fault_events(entity)
            prebuilt_events.extend(events)
        print(f"  共加载 {loaded_count} 台机器，抽取 {len(prebuilt_events)} 个故障事件")
        source_dataset = "smd"

    elif args.source == "gaia":
        gaia_dir = Path(args.gaia_dir)  # type: ignore[arg-type]
        if not gaia_dir.exists():
            print(f"[ERROR] GAIA 数据目录不存在: {gaia_dir}")
            download_gaia(str(gaia_dir))
            return 1
        gaia_loader = GAIALoader(str(gaia_dir))
        if not gaia_loader.run_dir.is_dir():
            print(f"[ERROR] 未找到 MicroSS/run 子目录：{gaia_loader.run_dir}")
            download_gaia(str(gaia_dir))
            return 1
        print(f"  加载故障注入记录 (run_dir={gaia_loader.run_dir})")
        injections = gaia_loader.load_fault_injections()
        print(f"  共 {len(injections)} 条故障注入记录")
        if not injections:
            print("[WARN] 未加载到任何 GAIA 故障注入记录（run 目录可能为空或非 MicroSS 结构）")
            download_gaia(str(gaia_dir))
            return 0
        prebuilt_events = gaia_loader.extract_fault_events(injections)
        print(f"  转换为 {len(prebuilt_events)} 个 FaultEvent")
        source_dataset = "gaia"

    else:  # 理论上不可达（argparse choices 已限制）
        print(f"[ERROR] 未知数据源: {args.source}")
        return 1

    # ---- 2. episode 构建 ----
    print("\n--- 2. episode 构建 ---")
    episode_builder = EpisodeBuilder()
    if vms is not None:
        # synthetic / azure：走 VM 时序 + IQR 异常检测 + 事件抽取
        fault_extractor = FaultEventExtractor()
        episodes, events = build_episodes_offline(
            vms=vms,
            fault_extractor=fault_extractor,
            episode_builder=episode_builder,
            include_skeletons=not args.no_skeletons,
            progress=True,
        )
    else:
        # smd / gaia：loader 已直出 FaultEvent，跳过异常检测
        assert prebuilt_events is not None
        episodes, events = build_episodes_from_events(
            events=prebuilt_events,
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
        print_stats(
            episodes, events,
            GraphBuildStats(
                total_episodes=len(episodes),
                total_fault_events=len(events),
            ),
            elapsed,
            source_dataset=source_dataset,
        )
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
    print_stats(episodes, events, stats, elapsed, source_dataset=source_dataset)

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
