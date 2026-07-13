#!/usr/bin/env python3
"""SMD 评估看门狗 —— 监控主进程，异常退出后自动重启。

使用方式：
    python scripts/run_smd_eval_watchdog.py --step all --vms 28

特性：
    - 进程崩溃/ killed / OOM 后自动重启
    - 每次重启在同一 checkpoint 继续（不丢进度）
    - 最多重启 10 次，防止死循环
    - 日志写入 logs/smd_eval_watchdog.log
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "logs" / "smd_eval_watchdog.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("watchdog")


def build_command(args: argparse.Namespace) -> list[str]:
    """构造被监控的主命令。"""
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "run_smd_eval.py")]
    cmd.extend(["--step", args.step])
    cmd.extend(["--vms", str(args.vms)])
    cmd.extend(["--per-hop"] + [str(h) for h in args.per_hop])
    cmd.extend(["--per-case", str(args.per_case)])
    cmd.extend(["--output", str(args.output)])

    if args.max_cases is not None:
        cmd.extend(["--max-cases", str(args.max_cases)])
    if args.baselines:
        cmd.extend(["--baselines"] + args.baselines)
    if not args.resume:
        cmd.append("--no-resume")
    if args.dry_run:
        cmd.append("--dry-run")

    return cmd


def main():
    parser = argparse.ArgumentParser(description="SMD 评估看门狗")
    parser.add_argument("--step", default="all", choices=["build_graph", "build_testset", "run_eval", "all"])
    parser.add_argument("--vms", type=int, default=28)
    parser.add_argument("--per-hop", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--per-case", type=int, default=50)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--baselines", type=str, nargs="+", default=None)
    parser.add_argument("--no-resume", dest="resume", action="store_false", default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=str, default=str(PROJECT_ROOT / "eval" / "reports_smd"))
    parser.add_argument("--max-retries", type=int, default=10, help="最大重启次数")
    args = parser.parse_args()

    cmd = build_command(args)
    logger.info(f"监控命令：{' '.join(cmd)}")

    retries = 0
    while retries < args.max_retries:
        logger.info(f"启动主进程（第 {retries + 1}/{args.max_retries} 次）")
        start_time = time.time()

        # 用 Popen 启动子进程，实时转发 stdout/stderr
        proc = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        # 实时输出子进程日志
        for line in proc.stdout:
            print(line, end="", flush=True)

        proc.wait()
        exit_code = proc.returncode
        elapsed = time.time() - start_time

        if exit_code == 0:
            logger.info(f"主进程正常退出，耗时 {elapsed:.1f}s")
            return 0

        retries += 1
        logger.warning(
            f"主进程异常退出：exit_code={exit_code}，"
            f"耗时 {elapsed:.1f}s，{retries}/{args.max_retries} 次重启"
        )

        # 指数退避
        sleep_sec = min(30, 5 * (2 ** (retries - 1)))
        logger.info(f"等待 {sleep_sec}s 后重启...")
        time.sleep(sleep_sec)

    logger.error(f"达到最大重启次数 {args.max_retries}，终止")
    return 1


if __name__ == "__main__":
    sys.exit(main())
