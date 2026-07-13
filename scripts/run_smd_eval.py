"""SMD 专用评估脚本 —— 端到端建图 + 测试集构造 + 4 baseline 评估，支持断点续跑。

使用方式：
    # 1. 先建图（SMD 真实数据写入 Neo4j）
    PYTHONPATH=. python scripts/run_smd_eval.py --step build_graph --vms 28

    # 2. 构造测试集（从 Neo4j 查路径反向构造）
    PYTHONPATH=. python scripts/run_smd_eval.py --step build_testset

    # 3. 跑评估（4 baseline × 测试集）
    PYTHONPATH=. python scripts/run_smd_eval.py --step run_eval

    # 4. 一键跑完（建图 + 测试集 + 评估）
    PYTHONPATH=. python scripts/run_smd_eval.py --step all --vms 28

后台运行（推荐）：
    nohup python scripts/run_smd_eval.py --step all --vms 28 > logs/smd_eval.log 2>&1 &

断点续跑：
    - 默认开启：检查 eval/reports_smd/ 下已完成的 case，跳过重跑
    - 强制重跑：加 --no-resume
    - 建图阶段：检查 Neo4j 是否已有 SMD episode，有则跳过

每个子任务完成后立即保存：
    - 建图：bootstrap_graph.py 本身有 verify + 统计输出
    - 测试集：构造完立即写 JSONL
    - 评估：每个 case 跑完立即追加一行到 .jsonl（write + flush + fsync）
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

# 确保能导入项目根目录的模块
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import get_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_smd_eval")

# SMD 专用输出目录（与默认 eval/reports/ 隔离）
SMD_OUTPUT_DIR = PROJECT_ROOT / "eval" / "reports_smd"
SMD_TESTSET_PATH = SMD_OUTPUT_DIR / "testset.jsonl"


def check_smd_data(smd_dir: Path) -> bool:
    """检查 SMD 数据是否完整（四件套齐全）。"""
    required_dirs = ["train", "test", "test_label", "interpretation_label"]
    for subdir in required_dirs:
        if not (smd_dir / subdir).is_dir():
            logger.error(f"SMD 数据不完整：缺失 {smd_dir / subdir}")
            return False
    # 检查是否有数据文件
    train_files = list((smd_dir / "train").glob("machine-*.txt"))
    if not train_files:
        logger.error(f"SMD 数据不完整：{smd_dir / 'train'} 下无数据文件")
        return False
    logger.info(f"SMD 数据检查通过：{len(train_files)} 台机器")
    return True


def check_neo4j_has_smd_data() -> bool:
    """检查 Neo4j 是否已有 SMD 来源的 episode。"""
    try:
        from neo4j import GraphDatabase
        cfg = get_config()
        driver = GraphDatabase.driver(cfg.neo4j.uri, auth=(cfg.neo4j.user, cfg.neo4j.password))
        with driver.session() as session:
            result = session.run(
                """
                MATCH (e:Episodic)
                WHERE e.source_description CONTAINS 'smd'
                   OR e.source_description CONTAINS 'SMD'
                   OR e.source_description CONTAINS 'ServerMachineDataset'
                RETURN count(e) AS cnt
                """
            )
            cnt = result.single()["cnt"]
        driver.close()
        return cnt > 0
    except Exception as e:
        logger.warning(f"检查 Neo4j SMD 数据失败：{e}")
        return False


def step_build_graph(vms: int, dry_run: bool = False) -> bool:
    """步骤 1：用 SMD 真实数据建图。"""
    smd_dir = PROJECT_ROOT / "dataset" / "ServerMachineDataset"
    if not check_smd_data(smd_dir):
        logger.error("SMD 数据不完整，请先下载数据集")
        return False

    # 检查是否已有 SMD 数据
    if check_neo4j_has_smd_data():
        logger.info("Neo4j 中已有 SMD 数据，跳过建图步骤（如需重建，请先清空 Neo4j）")
        return True

    logger.info(f"开始 SMD 建图：{vms} 台机器")
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "bootstrap_graph.py"),
        "--source", "smd",
        "--smd-dir", str(smd_dir),
        "--vms", str(vms),
        "--verify",
    ]
    if dry_run:
        cmd.append("--dry-run")

    logger.info(f"执行命令：{' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=False, text=True)

    if result.returncode != 0:
        logger.error(f"建图失败：returncode={result.returncode}")
        return False

    logger.info("建图完成")
    return True


def step_build_testset(per_hop: dict[int, int] | None = None, output_dir: Path | None = None) -> bool:
    """步骤 2：从 Neo4j 构造 SMD 测试集。"""
    if per_hop is None:
        per_hop = {2: 50, 3: 50, 4: 50}
    if output_dir is None:
        output_dir = SMD_OUTPUT_DIR

    testset_path = output_dir / "testset.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 如果测试集已存在，询问是否覆盖
    if testset_path.exists():
        logger.info(f"测试集已存在：{testset_path}，将直接使用（如需重建请删除该文件）")
        return True

    logger.info(f"开始构造 SMD 测试集：{per_hop}")
    from eval.testset_builder import build_testset

    testset = build_testset(
        n_per_hop=per_hop,
        output_path=testset_path,
    )
    logger.info(f"测试集构造完成：{len(testset)} 条")
    return True


def step_run_eval(
    baselines: list[str] | None = None,
    per_hop: list[int] | None = None,
    max_cases: int | None = None,
    resume: bool = True,
    output_dir: Path | None = None,
    testset_path: Path | None = None,
) -> bool:
    """步骤 3：跑 SMD 评估。"""
    if baselines is None:
        baselines = ["B1_NaiveRAG", "B2_GraphitiDefault", "B3_NoTemporal", "B4_Full_GraphRAG"]
    if per_hop is None:
        per_hop = [2, 3, 4]

    if output_dir is None:
        output_dir = SMD_OUTPUT_DIR
    if testset_path is None:
        testset_path = output_dir / "testset.jsonl"

    # 检查测试集
    if not testset_path.exists():
        logger.error(f"测试集不存在：{testset_path}，请先运行 --step build_testset")
        return False

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_eval.py"),
        "--testset", str(testset_path),
        "--output", str(output_dir),
        "--per-hop",
    ] + [str(h) for h in per_hop]

    if max_cases is not None:
        cmd.extend(["--max-cases", str(max_cases)])

    if not resume:
        cmd.append("--no-resume")
    else:
        cmd.append("--resume")

    cmd.extend(["--baselines"] + baselines)

    logger.info(f"执行评估命令：{' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=False, text=True)

    if result.returncode != 0:
        logger.error(f"评估失败：returncode={result.returncode}")
        return False

    logger.info(f"评估完成，结果保存在：{SMD_OUTPUT_DIR}")
    return True


def print_progress(output_dir: Path) -> None:
    """打印当前进度（已完成的 case 数）。"""
    from eval.checkpoint import load_completed_case_ids

    for baseline in ["B1_NaiveRAG", "B2_GraphitiDefault", "B3_NoTemporal", "B4_Full_GraphRAG"]:
        completed = load_completed_case_ids(output_dir, baseline)
        logger.info(f"  {baseline}: 已完成 {len(completed)} 条")


def main():
    parser = argparse.ArgumentParser(
        description="SMD 专用评估脚本 —— 端到端建图 + 测试集 + 评估，支持断点续跑",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
    # 一键跑完（推荐）
    python scripts/run_smd_eval.py --step all --vms 28

    # 后台运行
    nohup python scripts/run_smd_eval.py --step all --vms 28 > logs/smd_eval.log 2>&1 &

    # 查看进度
    python scripts/run_smd_eval.py --step progress

    # 强制重跑（清空 checkpoint）
    python scripts/run_smd_eval.py --step run_eval --no-resume
""",
    )
    parser.add_argument(
        "--step",
        choices=["build_graph", "build_testset", "run_eval", "progress", "all"],
        default="all",
        help="执行步骤（默认 all）",
    )
    parser.add_argument("--vms", type=int, default=28, help="SMD 加载机器数（默认 28=全量）")
    parser.add_argument("--per-hop", type=int, nargs="+", default=[2, 3, 4], help="跳数列表")
    parser.add_argument("--per-case", type=int, default=50, help="每跳目标用例数")
    parser.add_argument("--max-cases", type=int, default=None, help="最多跑多少条")
    parser.add_argument("--baselines", type=str, nargs="+", default=None, help="指定 baseline")
    parser.add_argument("--no-resume", dest="resume", action="store_false", default=True, help="强制重跑")
    parser.add_argument("--dry-run", action="store_true", help="建图阶段只构建不写入")
    parser.add_argument("--output", type=str, default=str(SMD_OUTPUT_DIR), help="输出目录")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 确保 logs 目录存在
    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"  SMD 专用评估脚本")
    logger.info(f"  输出目录：{output_dir}")
    logger.info(f"  数据目录：{PROJECT_ROOT / 'dataset' / 'ServerMachineDataset'}")
    logger.info("=" * 60)

    if args.step == "progress":
        print_progress(output_dir)
        return 0

    success = True

    # 步骤 1：建图
    if args.step in ("build_graph", "all"):
        logger.info("\n>>> 步骤 1/3：SMD 建图")
        t0 = time.time()
        if not step_build_graph(vms=args.vms, dry_run=args.dry_run):
            logger.error("建图失败，终止")
            return 1
        logger.info(f"建图耗时：{time.time() - t0:.1f}s")

    # 步骤 2：构造测试集
    if args.step in ("build_testset", "all"):
        logger.info("\n>>> 步骤 2/3：构造 SMD 测试集")
        t0 = time.time()
        if not step_build_testset(per_hop={h: args.per_case for h in args.per_hop}, output_dir=output_dir):
            logger.error("测试集构造失败，终止")
            return 1
        logger.info(f"测试集构造耗时：{time.time() - t0:.1f}s")

    # 步骤 3：跑评估
    if args.step in ("run_eval", "all"):
        logger.info("\n>>> 步骤 3/3：跑评估")
        t0 = time.time()
        if not step_run_eval(
            baselines=args.baselines,
            per_hop=args.per_hop,
            max_cases=args.max_cases,
            resume=args.resume,
            output_dir=output_dir,
        ):
            logger.error("评估失败，终止")
            return 1
        logger.info(f"评估总耗时：{time.time() - t0:.1f}s")

    logger.info("\n" + "=" * 60)
    logger.info("  全部完成！")
    logger.info(f"  结果目录：{output_dir}")
    logger.info("=" * 60)

    # 打印最终进度
    print_progress(output_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
