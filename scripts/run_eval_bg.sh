#!/bin/bash
# 后台评估启动脚本 —— nohup + setsid，完全脱离当前终端会话。
#
# 保证：
# - 断开 SSH / 关掉本地电脑 → 服务器进程继续跑（nohup 免疫 SIGHUP，setsid 新会话脱离控制终端）
# - 服务器重启/进程被杀 → 已完成 case 数据已 fsync 落盘不丢，重新跑本脚本 --resume 即可续跑
# - 每个 case 跑完立即写一行到 .jsonl（write+flush+fsync），中断最多丢 1 条
#
# 用法：
#   bash scripts/run_eval_bg.sh                      # 全量 4 baseline，断点续跑
#   bash scripts/run_eval_bg.sh B4_Full_GraphRAG     # 只跑指定 baseline
#   bash scripts/run_eval_bg.sh --restart            # 清空 checkpoint 重跑
#
# 监控：
#   tail -f eval/reports_full_v3/run.log             # 实时日志
#   tail -f eval/reports_full_v3/B4_Full_GraphRAG.jsonl | jq -c '{c:.case_id,r:.r_score,em:.em}'  # 实时指标
#   pgrep -af 'scripts/run_eval.py'                  # 查进程
#   wc -l eval/reports_full_v3/*.jsonl               # 已完成 case 数
#
# 停止：
#   kill $(cat eval/reports_full_v3/run.pid)

set -e

cd /root/Graph-RAG

OUTPUT_DIR=eval/reports_full_v3
LOG=$OUTPUT_DIR/run.log
PIDFILE=$OUTPUT_DIR/run.pid

# 解析参数
RESTART=0
BASELINES=""
for arg in "$@"; do
    case $arg in
        --restart)
            RESTART=1
            shift
            ;;
        *)
            BASELINES="$BASELINES $arg"
            shift
            ;;
    esac
done

# 默认全量 4 baseline
if [ -z "$BASELINES" ]; then
    BASELINES="B1_NaiveRAG B2_GraphitiDefault B3_NoTemporal B4_Full_GraphRAG"
fi

mkdir -p $OUTPUT_DIR

# 防止重复启动
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "已有评估任务在跑 (PID $(cat "$PIDFILE"))。如需重启：kill $(cat "$PIDFILE") 后再运行本脚本。"
    exit 1
fi

# --restart：清空 checkpoint
if [ "$RESTART" = "1" ]; then
    echo "清空 $OUTPUT_DIR 下的 checkpoint..."
    rm -f $OUTPUT_DIR/*.jsonl $OUTPUT_DIR/*.json $OUTPUT_DIR/run.log $OUTPUT_DIR/run.pid
fi

RESUME_FLAG="--resume"
if [ "$RESTART" = "1" ]; then
    RESUME_FLAG="--no-resume"
fi

# 构造命令
CMD="PYTHONPATH=. python3 scripts/run_eval.py \
    --baselines $BASELINES \
    --output $OUTPUT_DIR \
    $RESUME_FLAG"

echo "启动命令: $CMD"
echo "输出目录: $OUTPUT_DIR"
echo "日志文件: $LOG"
echo ""

# setsid: 新会话，脱离控制终端（防进程组被杀）
# nohup: 免疫 SIGHUP（终端关闭信号）
# < /dev/null: 断开 stdin
# > $LOG 2>&1: 合并 stdout/stderr 到日志
setsid bash -c "$CMD" > "$LOG" 2>&1 < /dev/null &

# setsid 后进程被 init 收养，$! 不可靠，用 pgrep 找真实 PID
sleep 2
PID=$(pgrep -f 'scripts/run_eval.py' | head -1)

if [ -z "$PID" ]; then
    echo "❌ 启动失败，查看日志: tail -20 $LOG"
    exit 1
fi

echo $PID > "$PIDFILE"
echo "✅ 评估任务已后台启动"
echo "   PID: $PID (已写入 $PIDFILE)"
echo "   Baselines: $BASELINES"
echo ""
echo "监控:"
echo "   tail -f $LOG"
echo "   wc -l $OUTPUT_DIR/*.jsonl"
echo "停止:"
echo "   kill $PID"
