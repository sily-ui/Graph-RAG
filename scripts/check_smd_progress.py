#!/usr/bin/env python3
"""SMD 评估进度查看器 —— 快速检查每个 baseline 的完成情况。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SMD_OUTPUT_DIR = PROJECT_ROOT / "eval" / "reports_smd"


def main():
    import argparse
    parser = argparse.ArgumentParser(description="查看 SMD 评估进度")
    parser.add_argument("--output", type=str, default=str(SMD_OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output)
    if not output_dir.exists():
        print(f"输出目录不存在：{output_dir}")
        return

    from eval.checkpoint import load_completed_case_ids

    baselines = ["B1_NaiveRAG", "B2_GraphitiDefault", "B3_NoTemporal", "B4_Full_GraphRAG"]
    print(f"\n进度目录：{output_dir}")
    print("=" * 60)

    total_completed = 0
    for baseline in baselines:
        completed = load_completed_case_ids(output_dir, baseline)
        total_completed += len(completed)

        # 读取 detail.json（如果存在）显示汇总指标
        detail_file = output_dir / f"{baseline}_detail.json"
        if detail_file.exists():
            try:
                with open(detail_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                overall = data.get("overall", {})
                print(
                    f"{baseline}: {len(completed)} 条 | "
                    f"PathErr={overall.get('path_error_rate', 0):.3f} | "
                    f"Hallu={overall.get('hallucination_rate_overall', 0):.3f} | "
                    f"PipeF1={overall.get('pipeline_f1', 0):.3f}"
                )
            except (json.JSONDecodeError, OSError):
                print(f"{baseline}: {len(completed)} 条（detail.json 解析失败）")
        else:
            print(f"{baseline}: {len(completed)} 条（尚未生成 summary）")

    print("=" * 60)
    print(f"总计完成：{total_completed} 条 case")

    # 显示测试集信息
    testset_path = output_dir / "testset.jsonl"
    if testset_path.exists():
        with open(testset_path, "r", encoding="utf-8") as f:
            lines = [l for l in f if l.strip()]
        print(f"测试集：{len(lines)} 条")


if __name__ == "__main__":
    main()
