"""Graph-RAG 自研推理端到端 demo。

完整链路：
    1. NLQ → LLM 解析查询意图（StructuredQuery）
    2. StructuredQuery → Cypher 生成
    3. Neo4j 执行 → 多跳路径抽取
    4. 时态剪枝（valid_at/invalid_at/lag_seconds）
    5. 路径排序（几何平均置信度）
    6. LLM 解释 → 自然语言答案
    7. ClaimDecomposer → 拆解成 atomic claims
    8. HallucinationVerifier → 逐条核验 + 幻觉率

该脚本是模块 3（自研 LLM 控制器）的可执行验收。
论文核心创新点（时态剪枝 + 幻觉可追溯）在此集中体现。

运行：
    python scripts/end_to_end_demo.py
    python scripts/end_to_end_demo.py --query "machine-1-1 资源争抢的根因"
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# 项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import get_config
from reasoning.claim_decomposer import ClaimDecomposer
from reasoning.controller import ReasoningController
from reasoning.hallucination_verifier import HallucinationVerifier
from reasoning.llm_interpreter import LLMClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("e2e_demo")


DEFAULT_QUERIES = [
    "cpu 飙升的根因是什么？",                                  # symptom 关键词路径
    "resource_contention 类型故障有哪些解法？",                  # solution_lookup
    "latency_ms 相关的 3 跳因果链路是什么？",                  # multi_hop_path
]


def build_neo4j_driver(uri: str, user: str, password: str):
    """构造 Neo4j 同步驱动（path_extractor 用 sync API）。"""
    try:
        from neo4j import GraphDatabase as _GDB
    except ImportError as e:
        raise ImportError("需要 neo4j 库：pip install neo4j") from e
    return _GDB.driver(uri, auth=(user, password))


def print_section(title: str, char: str = "=") -> None:
    """打印带分隔符的小节标题。"""
    print(f"\n{char * 60}")
    print(f"  {title}")
    print(char * 60)


def print_structured_query(sq) -> None:
    """打印结构化查询。"""
    intent = sq.intent
    print(f"  query_type:    {intent.query_type.value}")
    print(f"  target_entity: {intent.target_entity}")
    print(f"  entity_type:   {intent.target_entity_type}")
    print(f"  symptom_kw:    {intent.symptom_keywords}")
    print(f"  cause_kw:      {intent.cause_keywords}")
    print(f"  hop_count:     {intent.hop_count}")
    print(f"  severity:      {intent.severity_filter}")
    print(f"  limit:         {intent.limit}")
    print(f"  parser:        {sq.metadata.get('parser', '?')}")


def print_cypher(cypher: str) -> None:
    """打印 Cypher（多行）。"""
    for line in cypher.split("\n"):
        print(f"  | {line}")


def print_paths(paths, label: str, max_show: int = 3) -> None:
    """打印路径列表。"""
    print(f"  [{label}] 共 {len(paths)} 条")
    for i, path in enumerate(paths[:max_show], 1):
        print(f"  路径 {i}（置信度 {path.path_confidence:.3f}，{path.hop_count} 跳）：")
        print(f"    起点: {path.start_node.name} ({path.start_node.label})")
        for j, hop in enumerate(path.hops):
            print(
                f"    跳 {j}: --[{hop.edge_name} lag={hop.lag_seconds}s]--> "
                f"{hop.target.name} ({hop.target.label})"
            )
        if path.end_node and path.end_node.label == "Solution":
            print(f"    终点: {path.end_node.name} ({path.end_node.label})")
    if len(paths) > max_show:
        print(f"  ... 另有 {len(paths) - max_show} 条省略")


def print_claims(claims) -> None:
    """打印 atomic claim 列表。"""
    for c in claims:
        nodes = f"nodes={c.source_nodes}" if c.source_nodes else ""
        edges = f"edges={c.source_edges}" if c.source_edges else ""
        meta = ", ".join(filter(None, [nodes, edges]))
        print(f"  [{c.claim_id}] hop={c.hop_index} ({meta})")
        print(f"      {c.claim_text}")


def print_verified(report) -> None:
    """打印幻觉核验报告。"""
    print(f"  parser:             {report.parser}")
    print(f"  total_claims:       {report.total_claims}")
    print(f"  entailed:           {report.entailed_count}")
    print(f"  contradicted:       {report.contradicted_count}  ← 幻觉")
    print(f"  unsupported:        {report.unsupported_count}")
    print(f"  hallucination_rate: {report.hallucination_rate:.1%}")
    print()
    print("  逐跳统计：")
    for hop, stats in sorted(report.per_hop.items(), key=lambda x: int(x[0])):
        print(
            f"    跳 {hop}: total={stats['total']} "
            f"entailed={stats['entailed']} "
            f"contradicted={stats['contradicted']} "
            f"unsupported={stats['unsupported']}"
        )
    print()
    print("  逐条核验明细：")
    for v in report.verified:
        marker = "🔴 幻觉" if v.is_hallucination else (
            "✓ 蕴含" if v.verdict.value == "entailed" else "? 未支撑"
        )
        print(
            f"    [{v.claim_id}] hop={v.hop_index} "
            f"{marker} (conf={v.confidence:.2f})"
        )
        print(f"        {v.claim_text[:80]}{'...' if len(v.claim_text) > 80 else ''}")
        if v.evidence:
            print(f"        → {v.evidence[:100]}")


def run_one_query(controller, decomposer, verifier, query: str, use_llm: bool = True) -> dict:
    """跑单条查询，返回结构化结果。"""
    print_section(f"查询: {query}")
    result = {
        "query": query,
        "steps": {},
    }

    # Step 1-6: ReasoningController.ask() 或 .query()（跳 LLM）
    print_section("Step 1-6: ReasoningController 推理", "-")
    if use_llm:
        reasoning = controller.ask(query)
        sq = controller.interpreter.parse_query(query)
    else:
        from reasoning.query_types import QueryIntent, QueryType, StructuredQuery
        sq = StructuredQuery(
            natural_language=query,
            intent=QueryIntent(
                query_type=QueryType.MULTI_HOP_PATH,
                target_entity=None,
                target_entity_type=None,
                symptom_keywords=["cpu"],
                cause_keywords=[],
                hop_count=2,
                severity_filter=None,
                limit=10,
            ),
            metadata={"parser": "manual"},
        )
        reasoning = controller.query(sq)

    # Step 1
    print(f"\n[Step 1] {'LLM 解析' if use_llm else '手动构造'}查询意图 → StructuredQuery")
    print_structured_query(sq)
    result["steps"]["intent"] = sq.model_dump()

    # Step 2
    print("\n[Step 2] StructuredQuery → Cypher")
    cypher, params = controller.cypher_generator.generate(sq)
    print_cypher(cypher)
    print(f"  params: {params}")
    result["steps"]["cypher"] = cypher
    result["steps"]["params"] = {k: str(v) for k, v in params.items()}

    # Step 3 + 4 + 5
    print(f"\n[Step 3-5] 路径抽取 + 时态剪枝 + 排序")
    print_paths(reasoning.paths, "保留")
    print_paths(reasoning.pruned_paths, "剪枝")
    result["steps"]["path_count"] = reasoning.path_count
    result["steps"]["pruned_count"] = reasoning.pruned_count

    # Step 6
    print(f"\n[Step 6] LLM 解释 → 答案")
    print(f"  {reasoning.answer}")
    print(f"  综合置信度: {reasoning.confidence:.3f}")
    print(f"  耗时: {reasoning.elapsed_seconds:.2f}s")
    result["steps"]["answer"] = reasoning.answer
    result["steps"]["confidence"] = reasoning.confidence

    # Step 7
    print_section("Step 7: ClaimDecomposer 拆解成 atomic claims", "-")
    decomposition = decomposer.decompose(reasoning.answer, reasoning.paths)
    print(f"  parser: {decomposition.parser}")
    print(f"  共拆出 {len(decomposition.claims)} 条 claim：")
    print_claims(decomposition.claims)
    result["steps"]["decomposition"] = {
        "parser": decomposition.parser,
        "claim_count": len(decomposition.claims),
    }

    # Step 8
    print_section("Step 8: HallucinationVerifier 逐条核验", "-")
    report = verifier.verify(decomposition, reasoning.paths)
    print_verified(report)
    result["steps"]["hallucination"] = {
        "total": report.total_claims,
        "entailed": report.entailed_count,
        "contradicted": report.contradicted_count,
        "unsupported": report.unsupported_count,
        "rate": report.hallucination_rate,
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="Graph-RAG 端到端推理 demo")
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="单条查询；不传则用默认 3 条 demo",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="跳过 LLM 查询解析，用 hardcoded StructuredQuery（验证管线）",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="把结果保存到 JSON 文件",
    )
    args = parser.parse_args()

    config = get_config()
    print(f"Neo4j:  {config.neo4j.uri}")
    print(f"Gen LLM:  {config.gen_llm.model} @ {config.gen_llm.base_url}")
    print(f"Verify LLM: {config.verify_llm.model} @ {config.verify_llm.base_url}")

    # 构造 LLM 客户端
    # --no-llm 时直接走规则模式，跳过 LLMClient 构造（占位 key 也不会触发校验）
    if not args.no_llm:
        try:
            gen_client = LLMClient(
                api_key=config.gen_llm.api_key,
                base_url=config.gen_llm.base_url,
                model=config.gen_llm.model,
            )
        except ValueError as e:
            # gen_client 配置不可用（占位 key / 空 key / 非 ASCII），
            # demo 必须中止，否则无意义跑下去
            print(f"\n❌ 生成 LLM 配置不可用，请先在 .env 中填好真实 API Key：\n   {e}\n")
            sys.exit(1)
        try:
            verify_client = LLMClient(
                api_key=config.verify_llm.api_key,
                base_url=config.verify_llm.base_url,
                model=config.verify_llm.model,
            )
            logger.info(
                f"核验 LLM 独立配置: {verify_client.model} @ {verify_client.base_url}"
            )
        except ValueError as e:
            # verify 配置不可用（占位 key / 空 key），回退到 gen_client
            # 这样自评偏置无法完全避免，但至少链路能跑通
            logger.warning(f"核验 LLM 配置不可用，回退到 gen_client: {e}")
            verify_client = gen_client

    # 构造 Neo4j 驱动
    driver = build_neo4j_driver(
        config.neo4j.uri,
        config.neo4j.user,
        config.neo4j.password,
    )

    # 构造推理控制器
    # --no-llm 时传空 key，强制 controller 走规则降级（不调 LLM）
    if args.no_llm:
        controller = ReasoningController.from_config(
            neo4j_driver=driver,
            llm_api_key="",          # 走 client=None 模式
            llm_base_url="",
            llm_model="",
        )
        decomposer = ClaimDecomposer(client=None)
        verifier = HallucinationVerifier(client=None)
    else:
        controller = ReasoningController.from_config(
            neo4j_driver=driver,
            llm_api_key=config.gen_llm.api_key,
            llm_base_url=config.gen_llm.base_url,
            llm_model=config.gen_llm.model,
        )
        decomposer = ClaimDecomposer(client=gen_client)
        verifier = HallucinationVerifier(client=verify_client)

    # 跑查询
    queries = [args.query] if args.query else DEFAULT_QUERIES
    all_results = []
    for q in queries:
        try:
            r = run_one_query(controller, decomposer, verifier, q, use_llm=not args.no_llm)
            all_results.append(r)
        except Exception as e:
            logger.exception(f"查询失败: {q}")
            print(f"❌ 查询失败: {e}")

    # 汇总
    print_section("多查询汇总")
    for r in all_results:
        h = r["steps"].get("hallucination", {})
        if h:
            print(
                f"  '{r['query'][:50]}': "
                f"路径={r['steps']['path_count']}, "
                f"置信={r['steps']['confidence']:.3f}, "
                f"幻觉率={h['rate']:.1%}"
            )

    # 保存
    if args.save:
        save_path = Path(args.save)
        save_path.write_text(
            json.dumps(all_results, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"\n结果已保存到: {save_path}")

    # 关闭 driver
    driver.close()


if __name__ == "__main__":
    main()
