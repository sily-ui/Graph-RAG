#!/bin/bash
# 跑 150 条评估（约 1 小时）
set -e
cd /root/Graph-RAG

# 1. 生成 150 条测试集
echo "=== 生成 150 条测试集 ==="
PYTHONPATH=. python3 << 'PYEOF'
from eval.testset_builder import build_testset
from pathlib import Path
import logging
logging.basicConfig(level=logging.WARNING)
cases = build_testset(
    n_per_hop={2: 50, 3: 50, 4: 50},
    output_path=Path('eval/testset.jsonl'),
)
print(f'测试集: {len(cases)} 条')
PYEOF

# 2. 启动后台评估
echo "=== 启动评估 ==="
setsid nohup python3 scripts/run_eval.py \
    --per-hop 2 3 4 \
    --baselines B1_NaiveRAG B3_NoTemporal B4_Full_GraphRAG \
    > /tmp/eval_150.log 2>&1 < /dev/null &

EVAL_PID=$!
echo "评估 PID: $EVAL_PID"
echo "查看日志: tail -f /tmp/eval_150.log"
