#!/bin/bash
# 后台跑完整 300 条评估
# 用 nohup + disown 避免关闭终端后被 SIGHUP 终止

set -e
cd /root/Graph-RAG

# 1. 重新生成 300 条测试集
echo "=== 生成 300 条测试集 ==="
PYTHONPATH=. python3 -c "
from eval.testset_builder import build_testset
from pathlib import Path
import logging
logging.basicConfig(level=logging.WARNING)
cases = build_testset(
    n_per_hop={2: 100, 3: 100, 4: 100},
    output_path=Path('eval/testset.jsonl'),
)
print(f'测试集: {len(cases)} 条')
"

# 2. 跑评估（4 小时超时）
echo "=== 开始评估 ==="
PYTHONPATH=. timeout 14400 python3 scripts/run_eval.py \
    --per-hop 2 3 4 \
    --baselines B1_NaiveRAG B3_NoTemporal B4_Full_GraphRAG \
    > /tmp/eval_full.log 2>&1

echo "=== 评估完成 ==="
echo "查看结果: cat /root/Graph-RAG/eval/reports/summary.md"
