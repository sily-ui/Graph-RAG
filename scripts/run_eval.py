"""评估运行器 —— 统一调度 B1/B2/B3/B4 × 2/3/4 跳 × N 条测试集，输出指标报告。

用法：
    PYTHONPATH=. python scripts/run_eval.py
    PYTHONPATH=. python scripts/run_eval.py --per-hop 2 3 --per-case 3 --output eval/reports/run1.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import get_config
from eval.baselines.b1_naive_rag import B1NaiveRAG
from eval.baselines.b2_graphiti_default import B2GraphitiDefault
from eval.baselines.b3_no_temporal import B3NoTemporal
from eval.baselines.b4_full import B4FullGraphRAG
from eval.checkpoint import (
    CheckpointWriter,
    checkpoint_path,
    clear_checkpoint,
    detail_path,
    load_all_completed_cases,
    load_completed_case_ids,
    migrate_legacy_detail_if_needed,
)
from eval.metrics import (
    MetricsReport,
    aggregate_metrics,
    evaluate_case,
    report_to_markdown,
)
from eval.testset_builder import TestCase, build_testset
from reasoning.claim_decomposer import ClaimDecomposer
from reasoning.controller import ReasoningController
from reasoning.hallucination_verifier import HallucinationVerifier
from reasoning.llm_interpreter import LLMClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_eval")


def build_neo4j_driver(uri: str, user: str, password: str):
    from neo4j import GraphDatabase
    return GraphDatabase.driver(uri, auth=(user, password))


def build_baselines(
    config,
    driver,
    gen_client: LLMClient,
    verify_client: LLMClient,
) -> dict:
    """构造 4 个 baseline。"""
    # B3 / B4 共享 controller 实例（共用 path_extractor 状态）
    controller = ReasoningController.from_config(
        neo4j_driver=driver,
        llm_api_key=config.gen_llm.api_key,
        llm_base_url=config.gen_llm.base_url,
        llm_model=config.gen_llm.model,
    )
    decomposer = ClaimDecomposer(client=gen_client)
    verifier = HallucinationVerifier(client=verify_client)

    baselines: dict = {}
    baselines["B1_NaiveRAG"] = B1NaiveRAG(llm_client=gen_client)

    # B2: 尝试构造 Graphiti 客户端（可能失败）
    try:
        from data_ingest.graphiti_writer import build_graphiti_client
        graphiti = build_graphiti_client(config=config)
        baselines["B2_GraphitiDefault"] = B2GraphitiDefault(
            graphiti_client=graphiti, llm_client=gen_client
        )
        logger.info("B2 Graphiti 客户端构造成功")
    except Exception as e:
        logger.warning(f"B2 Graphiti 不可用: {e}")
        baselines["B2_GraphitiDefault"] = B2GraphitiDefault(
            graphiti_client=None, llm_client=gen_client
        )

    baselines["B3_NoTemporal"] = B3NoTemporal(
        controller=controller,
        decomposer=decomposer,
        verifier=verifier,
    )
    baselines["B4_Full_GraphRAG"] = B4FullGraphRAG(
        controller=controller,
        decomposer=decomposer,
        verifier=verifier,
    )
    return baselines


def run_eval(
    baselines: dict,
    testset: list[TestCase],
    output_dir: Path,
    enabled_baselines: list[str] | None = None,
    *,
    resume: bool = True,
) -> dict[str, MetricsReport]:
    """跑所有 baseline × testset，输出 reports dict。

    支持断点续跑：
    - resume=True（默认）：检查 <output>/<baseline>.jsonl，跳过已跑 case_id
    - 每个 case 跑完立即 write+flush+fsync 一行到 .jsonl（单行原子）
    - 跑完该 baseline 后从 .jsonl 重新聚合写 *_detail.json
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    enabled = enabled_baselines or list(baselines.keys())

    reports: dict[str, MetricsReport] = {}
    for name in enabled:
        if name not in baselines:
            logger.warning(f"跳过未知 baseline: {name}")
            continue
        baseline = baselines[name]

        # 加载已完成 case_id
        completed = load_completed_case_ids(output_dir, name) if resume else set()
        to_run = [c for c in testset if c.case_id not in completed]
        logger.info(
            f"\n{'='*60}\n  跑 baseline: {name}\n"
            f"  resume={resume}, 已完成 {len(completed)}, "
            f"待跑 {len(to_run)}/{len(testset)}\n{'='*60}"
        )

        with CheckpointWriter(checkpoint_path(output_dir, name)) as writer:
            for i, case in enumerate(to_run, 1):
                t_case = time.time()
                try:
                    result = baseline.predict(case)
                    err = result.error
                except Exception as e:
                    logger.error(f"[{case.case_id}] 预测异常: {e}")
                    from eval.baselines.common import BaselineResult
                    result = BaselineResult(
                        case_id=case.case_id,
                        baseline_name=name,
                        predicted_hops=[],
                        verified_claims=[],
                        error=str(e),
                    )
                    err = str(e)
                elapsed = time.time() - t_case

                metrics = evaluate_case(
                    case=case,
                    predicted_hops=result.predicted_hops,
                    verified_claims=result.verified_claims,
                    answer=result.answer or "",
                )
                metrics["elapsed_sec"] = elapsed
                metrics["error"] = err
                writer.append_case(metrics)  # 立即落盘

                logger.info(
                    f"  [{i}/{len(to_run)} {case.case_id}] {elapsed:.1f}s "
                    f"hops={len(result.predicted_hops)} "
                    f"claims={len(result.verified_claims)} "
                    f"err={err}"
                )

        # 跑完该 baseline：从 .jsonl 重新聚合，再写一次 _detail.json
        all_metrics = load_all_completed_cases(output_dir, name)
        report = aggregate_metrics(name, all_metrics)
        reports[name] = report
        logger.info(f"\n{report_to_markdown(report)}\n")
        detail_path(output_dir, name).write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    return reports


def render_summary(reports: dict[str, MetricsReport]) -> str:
    """渲染 4 baseline × 2/3/4 跳的横向对比表（含 GraphRAG-Bench 7 项新指标）。"""
    lines = ["# 评估汇总", ""]
    # 15 列：7 旧指标 + 4 图构建指标 + 3 推理质量 + 1 case 数
    lines.append(
        "## Overall 指标对比（4 baseline × 2/3/4 跳，含 GraphRAG-Bench 7 项新指标）"
    )
    lines.append("")
    lines.append(
        "| Baseline | N | PathErr↓ | Hallu↓ | Hallu(h)↓ | Recall↑ | Prec↑ | TempAcc↑ | "
        "Prov↑ | EntityR↑ | EntityP↑ | RelR↑ | PipeF1↑ | R↑ | AR↑ | EM↑ |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, r in reports.items():
        o = r.overall
        lines.append(
            f"| {name} | {o.case_count} | {o.path_error_rate:.3f} | "
            f"{o.hallucination_rate_overall:.3f} | {o.hallucination_rate_per_hop:.3f} | "
            f"{o.recall:.3f} | {o.precision:.3f} | {o.temporal_accuracy:.3f} | "
            f"{o.provenance_completeness:.3f} | {o.entity_recall:.3f} | "
            f"{o.entity_precision:.3f} | {o.relation_recall:.3f} | {o.pipeline_f1:.3f} | "
            f"{o.r_score:.3f} | {o.ar_score:.3f} | {o.em:.3f} |"
        )

    lines.append("\n## 按跳数细分")
    for hop in [2, 3, 4]:
        lines.append(f"\n### {hop} 跳")
        lines.append(
            "| Baseline | N | PathErr↓ | Hallu↓ | Hallu(h)↓ | Recall↑ | Prec↑ | TempAcc↑ | "
            "Prov↑ | EntityR↑ | EntityP↑ | RelR↑ | PipeF1↑ | R↑ | AR↑ | EM↑ |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for name, r in reports.items():
            if hop not in r.per_hop:
                continue
            s = r.per_hop[hop]
            lines.append(
                f"| {name} | {s.case_count} | {s.path_error_rate:.3f} | "
                f"{s.hallucination_rate_overall:.3f} | {s.hallucination_rate_per_hop:.3f} | "
                f"{s.recall:.3f} | {s.precision:.3f} | {s.temporal_accuracy:.3f} | "
                f"{s.provenance_completeness:.3f} | {s.entity_recall:.3f} | "
                f"{s.entity_precision:.3f} | {s.relation_recall:.3f} | {s.pipeline_f1:.3f} | "
                f"{s.r_score:.3f} | {s.ar_score:.3f} | {s.em:.3f} |"
            )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Graph-RAG 评估运行器")
    parser.add_argument("--per-hop", type=int, nargs="+", default=[2, 3, 4],
                        help="跳数列表（默认 2 3 4）")
    parser.add_argument("--per-case", type=int, default=3,
                        help="每个跳数取多少用例（默认 3，--testset 存在时忽略）")
    parser.add_argument("--max-cases", type=int, default=None,
                        help="最多跑多少条（截断测试集，默认不限制）")
    parser.add_argument("--baselines", type=str, nargs="+",
                        default=["B1_NaiveRAG", "B2_GraphitiDefault", "B3_NoTemporal", "B4_Full_GraphRAG"])
    parser.add_argument("--output", type=str, default="eval/reports",
                        help="输出目录")
    parser.add_argument("--testset", type=str, default="eval/testset.jsonl",
                        help="测试集路径（不存在则现场构造）")
    parser.add_argument(
        "--resume", dest="resume", action="store_true", default=True,
        help="断点续跑，跳过已完成的 case（默认开启）"
    )
    parser.add_argument(
        "--no-resume", dest="resume", action="store_false",
        help="强制重跑：先清空 checkpoint 再跑"
    )
    args = parser.parse_args()

    config = get_config()
    print(f"Neo4j: {config.neo4j.uri}")
    print(f"GEN LLM: {config.gen_llm.model} @ {config.gen_llm.base_url}")
    print(f"VERIFY LLM: {config.verify_llm.model} @ {config.verify_llm.base_url}")

    # 1. 构造或加载测试集
    testset_path = Path(args.testset)
    if testset_path.exists():
        logger.info(f"加载测试集: {testset_path}")
        with testset_path.open("r", encoding="utf-8") as f:
            testset_dicts = [json.loads(line) for line in f]
        testset = [_dict_to_testcase(d) for d in testset_dicts]
    else:
        logger.info("现场构造测试集...")
        n_per_hop = {h: args.per_case for h in args.per_hop}
        testset = build_testset(n_per_hop=n_per_hop, output_path=testset_path)
    logger.info(f"测试集: {len(testset)} 条")
    # 按跳数过滤（只在 testset 存在时生效）
    if testset_path.exists() and args.per_hop:
        before = len(testset)
        testset = [c for c in testset if c.hop_count in args.per_hop]
        logger.info(f"按跳数 {args.per_hop} 过滤: {before} → {len(testset)} 条")
    if args.max_cases is not None and len(testset) > args.max_cases:
        testset = testset[: args.max_cases]
        logger.info(f"截断到前 {args.max_cases} 条")

    # 2. 构造 LLM 客户端
    try:
        gen_client = LLMClient(
            api_key=config.gen_llm.api_key,
            base_url=config.gen_llm.base_url,
            model=config.gen_llm.model,
        )
    except ValueError as e:
        print(f"\n❌ 生成 LLM 配置不可用: {e}\n")
        sys.exit(1)
    try:
        verify_client = LLMClient(
            api_key=config.verify_llm.api_key,
            base_url=config.verify_llm.base_url,
            model=config.verify_llm.model,
        )
    except ValueError as e:
        logger.warning(f"核验 LLM 配置不可用，回退 gen: {e}")
        verify_client = gen_client

    # 3. 构造 driver + baselines
    driver = build_neo4j_driver(config.neo4j.uri, config.neo4j.user, config.neo4j.password)
    baselines = build_baselines(config, driver, gen_client, verify_client)
    # 只跑请求的 baselines
    baselines = {k: v for k, v in baselines.items() if k in args.baselines}

    # 4. 处理历史 checkpoint（--resume / --no-resume）
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        for bname in args.baselines:
            clear_checkpoint(output_dir, bname)
    else:
        for bname in args.baselines:
            n = migrate_legacy_detail_if_needed(output_dir, bname)
            if n > 0:
                logger.info(f"[{bname}] 从 {bname}_detail.json 迁移 {n} 条 case 到 .jsonl")

    # 5. 跑评估
    reports = run_eval(
        baselines, testset, output_dir,
        enabled_baselines=args.baselines, resume=args.resume,
    )

    # 6. 输出汇总
    summary = render_summary(reports)
    summary_path = output_dir / "summary.md"
    summary_path.write_text(summary, encoding="utf-8")
    print(f"\n{'='*60}\n{summary}\n{'='*60}")
    print(f"\n汇总报告: {summary_path}")

    # 7. 清理
    driver.close()


def _dict_to_testcase(d: dict) -> TestCase:
    """dict → TestCase（用于加载已有 testset.jsonl）。"""
    from eval.testset_builder import ExpectedHop, SupportingFact, TestCase
    return TestCase(
        case_id=d["case_id"],
        domain=d["domain"],
        hop_count=d["hop_count"],
        query=d["query"],
        expected_path=[ExpectedHop(**h) for h in d["expected_path"]],
        supporting_facts=[SupportingFact(**f) for f in d["supporting_facts"]],
        query_time=d["query_time"],
        ground_truth_free_text=d["ground_truth_free_text"],
        metadata=d.get("metadata", {}),
    )


if __name__ == "__main__":
    main()
