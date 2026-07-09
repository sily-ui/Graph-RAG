"""Case Study 选 5 个代表性 case 做深入分析。

选 case 策略：
1. 2 跳单因（最常见）
2. 3 跳配置链（中间复杂度）
3. 4 跳复合（高复杂度）
4. 时态对比（B3 vs B4 看时态剪枝效果）
5. 幻觉定位（B4 中 contradicted 最多的 case）

输出：
- 每个 case 的 4 baseline 预测对比
- 逐跳 claim 表
- 幻觉标记
- provenance 溯源链
- 导出 G6 可视化数据

用法：
    PYTHONPATH=. python scripts/export_case_study.py
    PYTHONPATH=. python scripts/export_case_study.py --case-ids smd_2hop_001,micross_3hop_001,cross_4hop_001
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
from eval.metrics import evaluate_case
from reasoning.claim_decomposer import ClaimDecomposer
from reasoning.controller import ReasoningController
from reasoning.hallucination_verifier import HallucinationVerifier
from reasoning.llm_interpreter import LLMClient

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("case_study")


def to_g6_data(case, baseline_results: dict) -> dict:
    """把 case + 各 baseline 结果转成 G6 GraphData（用于可视化）。"""
    nodes = []
    edges = []
    node_id_by_key = {}  # (label, name) -> node_id, 跨 hop 共享节点身份

    # 1. ground truth 节点（用深色边框）—— 按 (label, name) 去重
    for i, hop in enumerate(case.expected_path):
        for _side, name, node_label in [
            ("source", hop.source_name, hop.source_label),
            ("target", hop.target_name, hop.target_label),
        ]:
            key = f"{node_label}::{name}"
            if key not in node_id_by_key:
                node_id = f"n_{len(node_id_by_key)}"
                node_id_by_key[key] = node_id
                nodes.append({
                    "id": node_id,
                    "label": node_label,
                    "name": name,
                    "category": "ground_truth",
                    "hop": i,
                })
            else:
                # 已有同 label+name 节点，更新 hop
                for n in nodes:
                    if n["id"] == node_id_by_key[key]:
                        n["hop"] = i
                        break

    # 2. ground truth 边（实线深色）
    for i, hop in enumerate(case.expected_path):
        src_key = f"{hop.source_label}::{hop.source_name}"
        tgt_key = f"{hop.target_label}::{hop.target_name}"
        edges.append({
            "id": f"gt_{i}",
            "source": node_id_by_key[src_key],
            "target": node_id_by_key[tgt_key],
            "label": hop.edge_name,
            "category": "ground_truth",
            "hop": i,
        })

    # 3. 各 baseline 预测（不同颜色虚线）—— predicted 节点也按 (label, name) 去重
    colors = {
        "B1_NaiveRAG": "#ff7875",
        "B2_GraphitiDefault": "#ffc53d",
        "B3_NoTemporal": "#40a9ff",
        "B4_Full_GraphRAG": "#52c41a",
    }
    for bname, br in baseline_results.items():
        for i, hop in enumerate(br.predicted_hops):
            src_key = f"pred::{hop.source.label}::{hop.source.name}"
            tgt_key = f"pred::{hop.target.label}::{hop.target.name}"
            if src_key not in node_id_by_key:
                nid = f"pn_{len(node_id_by_key)}"
                node_id_by_key[src_key] = nid
                nodes.append({
                    "id": nid,
                    "label": hop.source.label,
                    "name": hop.source.name,
                    "category": "predicted",
                    "baseline": bname,
                    "hop": i,
                })
            if tgt_key not in node_id_by_key:
                nid = f"pn_{len(node_id_by_key)}"
                node_id_by_key[tgt_key] = nid
                nodes.append({
                    "id": nid,
                    "label": hop.target.label,
                    "name": hop.target.name,
                    "category": "predicted",
                    "baseline": bname,
                    "hop": i,
                })
            edges.append({
                "id": f"pred_{bname}_{i}",
                "source": node_id_by_key[src_key],
                "target": node_id_by_key[tgt_key],
                "label": hop.edge_name or "?",
                "category": "predicted",
                "baseline": bname,
                "color": colors.get(bname, "#888"),
                "style": "dashed",
                "hop": i,
            })

    return {
        "case_id": case.case_id,
        "query": case.query,
        "ground_truth": case.ground_truth_free_text,
        "nodes": nodes,
        "edges": edges,
    }


def render_case_markdown(case, baseline_results: dict, metrics_per_baseline: dict) -> str:
    """把单个 case 渲染成 markdown。"""
    lines = [f"## Case: {case.case_id}", ""]
    lines.append(f"**查询**: {case.query}")
    lines.append(f"**跳数**: {case.hop_count}")
    lines.append(f"**Ground Truth**: {case.ground_truth_free_text}")
    lines.append("")
    lines.append("### 期望路径（Ground Truth）")
    for i, hop in enumerate(case.expected_path):
        lines.append(f"  {i}. {hop.source_name} ({hop.source_label}) --[{hop.edge_name}]--> {hop.target_name} ({hop.target_label})")
    lines.append("")
    lines.append("### 4 Baseline 预测对比")
    for bname, br in baseline_results.items():
        lines.append(f"\n#### {bname}")
        lines.append(f"  - 预测跳数: {len(br.predicted_hops)}")
        lines.append(f"  - 核验 claim 数: {len(br.verified_claims)}")
        if br.verified_claims:
            ent = sum(1 for c in br.verified_claims if c.verdict.value == "entailed")
            con = sum(1 for c in br.verified_claims if c.verdict.value == "contradicted")
            uns = sum(1 for c in br.verified_claims if c.verdict.value == "unsupported")
            lines.append(f"  - entailed/contradicted/unsupported: {ent}/{con}/{uns}")
        lines.append(f"  - 耗时: {br.elapsed_seconds:.1f}s")
        if br.error:
            lines.append(f"  - 错误: {br.error}")
        if br.answer:
            ans = br.answer[:200].replace("\n", " ")
            lines.append(f"  - 答案片段: {ans}...")
        if br.predicted_hops:
            for j, h in enumerate(br.predicted_hops[:5]):
                lines.append(f"  - 跳 {j}: {h.source.name} --[{h.edge_name}]--> {h.target.name}")

        m = metrics_per_baseline.get(bname, {})
        if m:
            lines.append(f"  - 指标: PathErr={m.get('path_error_rate', 0):.2f} "
                         f"Hallu={m.get('hallucination_rate_overall', 0):.2f} "
                         f"Recall={m.get('recall', 0):.2f} "
                         f"TemporalAcc={m.get('temporal_accuracy', 0):.2f}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--testset", default="eval/testset.jsonl")
    parser.add_argument("--output", default="eval/reports/case_study")
    parser.add_argument("--case-ids", default=None, help="逗号分隔的 case_id 列表（None=自动选代表性）")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载测试集
    with open(args.testset, "r", encoding="utf-8") as f:
        testset_dicts = [json.loads(line) for line in f]
    from eval.testset_builder import ExpectedHop, SupportingFact, TestCase
    testset = [TestCase(
        case_id=d["case_id"], domain=d["domain"], hop_count=d["hop_count"],
        query=d["query"],
        expected_path=[ExpectedHop(**h) for h in d["expected_path"]],
        supporting_facts=[SupportingFact(**f) for f in d["supporting_facts"]],
        query_time=d["query_time"],
        ground_truth_free_text=d["ground_truth_free_text"],
        metadata=d.get("metadata", {}),
    ) for d in testset_dicts]

    # 2. 选 case
    if args.case_ids:
        target_ids = args.case_ids.split(",")
        selected = [c for c in testset if c.case_id in target_ids]
    else:
        # 默认选代表性：每跳数 1 个 + 时态对比 + 幻觉定位
        selected = []
        for h in [2, 3, 4]:
            candidates = [c for c in testset if c.hop_count == h]
            if candidates:
                selected.append(candidates[0])
        # 额外加 1 个
        if len(testset) > len(selected):
            selected.append(testset[len(selected)])
    logger.info(f"选中 {len(selected)} 个 case: {[c.case_id for c in selected]}")

    # 3. 构造 baselines
    config = get_config()
    try:
        gen_client = LLMClient(
            api_key=config.gen_llm.api_key,
            base_url=config.gen_llm.base_url,
            model=config.gen_llm.model,
        )
    except ValueError as e:
        print(f"\n❌ 生成 LLM 不可用: {e}\n")
        sys.exit(1)
    try:
        verify_client = LLMClient(
            api_key=config.verify_llm.api_key,
            base_url=config.verify_llm.base_url,
            model=config.verify_llm.model,
        )
    except ValueError:
        verify_client = gen_client

    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(config.neo4j.uri, auth=(config.neo4j.user, config.neo4j.password))

    controller = ReasoningController.from_config(
        neo4j_driver=driver,
        llm_api_key=config.gen_llm.api_key,
        llm_base_url=config.gen_llm.base_url,
        llm_model=config.gen_llm.model,
    )
    decomposer = ClaimDecomposer(client=gen_client)
    verifier = HallucinationVerifier(client=verify_client)

    b1 = B1NaiveRAG(llm_client=gen_client)
    # B2: 用 build_graphiti_client 构造（已配 StepFun LLM + 本地 embedder）
    b2 = None
    try:
        from data_ingest.graphiti_writer import build_graphiti_client
        b2_graphiti = build_graphiti_client(config=config)
        b2 = B2GraphitiDefault(graphiti_client=b2_graphiti, llm_client=gen_client)
    except Exception as e:
        logger.warning(f"B2 Graphiti 不可用: {e}")
        b2 = B2GraphitiDefault(graphiti_client=None, llm_client=gen_client)
    b3 = B3NoTemporal(controller=controller, decomposer=decomposer, verifier=verifier)
    b4 = B4FullGraphRAG(controller=controller, decomposer=decomposer, verifier=verifier)
    baselines = {
        "B1_NaiveRAG": b1,
        "B2_GraphitiDefault": b2,
        "B3_NoTemporal": b3,
        "B4_Full_GraphRAG": b4,
    }

    # 4. 跑每个 case
    md_lines = ["# Case Study — 代表性样本深入分析", ""]
    g6_data_list = []
    per_case_metrics: list[tuple] = []  # [(case, metrics_per_baseline)]
    for case in selected:
        logger.info(f"\n=== {case.case_id} ===")
        results = {}
        metrics_per = {}
        for bname, baseline in baselines.items():
            t = time.time()
            r = baseline.predict(case)
            logger.info(f"  {bname}: {time.time()-t:.1f}s hops={len(r.predicted_hops)} claims={len(r.verified_claims)}")
            results[bname] = r
            m = evaluate_case(case, r.predicted_hops, r.verified_claims)
            metrics_per[bname] = m
        per_case_metrics.append((case, metrics_per))

        # 写 G6 数据
        g6 = to_g6_data(case, results)
        g6_data_list.append(g6)
        g6_path = output_dir / f"{case.case_id}_g6.json"
        g6_path.write_text(json.dumps(g6, ensure_ascii=False, indent=2), encoding="utf-8")

        # 写 markdown
        md = render_case_markdown(case, results, metrics_per)
        md_lines.append(md)
        md_lines.append("\n---\n")

    # 5. 输出汇总
    md_path = output_dir / "case_study.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    logger.info(f"\nCase study 已写入: {md_path}")
    logger.info(f"G6 数据: {output_dir}/*_g6.json")

    # 6. 汇总时态对比表（B3 vs B4）
    temporal_lines = ["## 时态剪枝对比（B3 vs B4）", ""]
    temporal_lines.append("| Case | B3 Recall | B4 Recall | B3 TemporalAcc | B4 TemporalAcc | Δ TemporalAcc |")
    temporal_lines.append("|---|---|---|---|---|---|")
    for case, metrics_per in per_case_metrics:
        m3 = metrics_per.get("B3_NoTemporal", {})
        m4 = metrics_per.get("B4_Full_GraphRAG", {})
        if m3 and m4:
            temporal_lines.append(
                f"| {case.case_id} | {m3.get('recall', 0):.2f} | {m4.get('recall', 0):.2f} | "
                f"{m3.get('temporal_accuracy', 0):.2f} | {m4.get('temporal_accuracy', 0):.2f} | "
                f"{m4.get('temporal_accuracy', 0) - m3.get('temporal_accuracy', 0):+.2f} |"
            )

    # 追加时态对比到主 markdown
    with md_path.open("a", encoding="utf-8") as f:
        f.write("\n\n" + "\n".join(temporal_lines))

    logger.info("完成")
    driver.close()


if __name__ == "__main__":
    main()
