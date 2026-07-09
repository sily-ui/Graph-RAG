"""生成 eval/testset_multitype.jsonl —— GraphRAG-Bench §3.1 借鉴。

从 eval/testset.jsonl 已有 150 条 OE case 派生 MC / TF 变体：
- 每个 hop 取前 5 条 → 5 MC + 5 TF = 30 条
- 总共 30 条新 case，存到 eval/testset_multitype.jsonl
- 题目分发到原文件不污染

References
----------
- arXiv:2506.02404 (GraphRAG-Bench, Xiao et al. 2025) §3.1

用法：
    PYTHONPATH=. python scripts/build_multitype_testset.py
"""
from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval.question_type_scorer import build_mc_variants, build_tf_variants
from eval.testset_builder import (
    ExpectedHop,
    SupportingFact,
    TestCase,
    load_testset,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("build_multitype")


def main() -> None:
    src = PROJECT_ROOT / "eval" / "testset.jsonl"
    out = PROJECT_ROOT / "eval" / "testset_multitype.jsonl"

    logger.info(f"读 {src}")
    raw = load_testset(str(src))
    logger.info(f"原测试集 {len(raw)} 条")

    rng = random.Random(20260709)
    new_cases: list[TestCase] = []
    # 按 hop 分桶，每 hop 取前 5 条
    by_hop: dict[int, list[dict]] = {2: [], 3: [], 4: []}
    for c in raw:
        h = c.get("hop_count", 0)
        if h in by_hop and len(by_hop[h]) < 5:
            by_hop[h].append(c)

    for hop, cases in sorted(by_hop.items()):
        for c in cases:
            # 转 TestCase
            tc = TestCase(
                case_id=c["case_id"],
                domain=c["domain"],
                hop_count=c["hop_count"],
                query=c["query"],
                expected_path=[ExpectedHop(**h) for h in c["expected_path"]],
                supporting_facts=[SupportingFact(**f) for f in c["supporting_facts"]],
                query_time=c["query_time"],
                ground_truth_free_text=c["ground_truth_free_text"],
                metadata=c.get("metadata", {}),
                question_type=c.get("question_type", "OE"),
                task_level=c.get("task_level", "Complex_Reasoning"),
            )
            # 派生 MC
            for mc in build_mc_variants(tc, rng):
                mc.case_id = f"mc_{hop}hop_{len([x for x in new_cases if x.question_type=='MC'])+1:03d}"
                new_cases.append(mc)
            # 派生 TF
            for tf in build_tf_variants(tc, rng):
                tf.case_id = f"tf_{hop}hop_{len([x for x in new_cases if x.question_type=='TF'])+1:03d}"
                new_cases.append(tf)

    logger.info(f"生成 {len(new_cases)} 条多题型 case（MC + TF）")
    # 写 JSONL
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for c in new_cases:
            row = {
                "case_id": c.case_id,
                "domain": c.domain,
                "hop_count": c.hop_count,
                "query": c.query,
                "expected_path": [
                    {
                        "edge_name": h.edge_name,
                        "source_name": h.source_name,
                        "source_label": h.source_label,
                        "target_name": h.target_name,
                        "target_label": h.target_label,
                        "valid_at": h.valid_at,
                        "invalid_at": h.invalid_at,
                        "lag_seconds": h.lag_seconds,
                    }
                    for h in c.expected_path
                ],
                "supporting_facts": [
                    {
                        "hop_index": f.hop_index,
                        "source": f.source,
                        "text": f.text,
                        "reference": f.reference,
                    }
                    for f in c.supporting_facts
                ],
                "query_time": c.query_time,
                "ground_truth_free_text": c.ground_truth_free_text,
                "metadata": c.metadata,
                "question_type": c.question_type,
                "task_level": c.task_level,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info(f"已写入 {out}（{len(new_cases)} 行）")
    # 统计
    by_type: dict[str, int] = {}
    for c in new_cases:
        by_type[c.question_type] = by_type.get(c.question_type, 0) + 1
    logger.info(f"题型分布: {by_type}")


if __name__ == "__main__":
    main()
