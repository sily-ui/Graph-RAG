#!/bin/bash
# 真正脱离会话的后台评估脚本
# 用 setsid + nohup + disown 三重保险

set -e
cd /root/Graph-RAG

# 1. 重新生成 300 条测试集（如果需要）
if [ ! -f eval/testset.jsonl ] || [ $(wc -l < eval/testset.jsonl) -lt 300 ]; then
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
fi

# 2. 启动评估（用 setsid 启动新 session，nohup 防 HUP）
echo "=== 启动评估 ==="
setsid nohup bash -c '
    cd /root/Graph-RAG
    PYTHONPATH=. timeout 18000 python3 scripts/run_eval.py \
        --per-hop 2 3 4 \
        --baselines B1_NaiveRAG B3_NoTemporal B4_Full_GraphRAG \
        > /tmp/eval_v6.log 2>&1
    echo "评估完成时间: $(date)" >> /tmp/eval_v6.log
' > /dev/null 2>&1 < /dev/null &

echo "PID: $!"
