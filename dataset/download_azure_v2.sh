#!/bin/bash
# Azure Public Dataset V2 下载脚本
# 用法：bash download_azure_v2.sh [分片数量]
# 示例：bash download_azure_v2.sh 2  # 下载前 2 个分片

set -e

# 分片数量（默认 2 个，每个约 1.2GB）
NUM_FILES=${1:-2}

# 数据目录（脚本所在目录）
DATASET_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DATASET_DIR"

echo "=== Azure V2 数据集下载脚本 ==="
echo "下载数量: $NUM_FILES 个分片"
echo "下载目录: $DATASET_DIR"
echo ""

# 步骤 1：获取链接清单
LINKS_FILE="azure_v2_links.txt"
if [ ! -f "$LINKS_FILE" ]; then
    echo "[1/3] 下载链接清单..."
    # 尝试多个源，保证成功
    SOURCES=(
        "https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/AzurePublicDatasetV2.txt"
        "https://ghproxy.com/https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/AzurePublicDatasetV2.txt"
        "https://mirror.ghproxy.com/https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/AzurePublicDatasetV2.txt"
    )
    for src in "${SOURCES[@]}"; do
        echo "  尝试: $src"
        if curl -L -s -o "$LINKS_FILE" "$src" && [ -s "$LINKS_FILE" ]; then
            echo "  [成功] 清单已下载"
            break
        fi
    done
    if [ ! -s "$LINKS_FILE" ]; then
        echo "  [失败] 无法下载链接清单，请检查网络"
        exit 1
    fi
else
    echo "[1/3] 链接清单已存在，跳过"
fi

LINK_COUNT=$(wc -l < "$LINKS_FILE")
echo "  清单含 $LINK_COUNT 个文件链接"

# 步骤 2：下载指定数量的分片
echo ""
echo "[2/3] 开始下载前 $NUM_FILES 个分片（每个约 1.2GB）..."

# 提取前 N 个链接
head -n "$NUM_FILES" "$LINKS_FILE" > /tmp/azure_v2_to_download.txt

# 逐个下载（带断点续传 + 重试）
i=1
total=$(wc -l < /tmp/azure_v2_to_download.txt)
while IFS= read -r url; do
    filename=$(basename "$url")
    echo ""
    echo "  [$i/$total] 下载: $filename"

    # 断点续传，最多重试 3 次
    for retry in 1 2 3; do
        if curl -L -C - -o "$filename" --connect-timeout 30 --max-time 3600 "$url"; then
            # 校验文件大小（应 > 500MB）
            size=$(stat -c%s "$filename" 2>/dev/null || stat -f%z "$filename")
            if [ "$size" -gt 524288000 ]; then
                echo "  [成功] $filename ($(numfmt --to=iec $size))"
                break
            else
                echo "  [警告] 文件大小异常: $size bytes，重试 $retry/3"
                rm -f "$filename"
            fi
        else
            echo "  [失败] 重试 $retry/3"
            sleep 5
        fi
    done

    i=$((i + 1))
done < /tmp/azure_v2_to_download.txt

# 步骤 3：校验
echo ""
echo "[3/3] 校验下载结果..."
echo ""
echo "已下载文件:"
ls -lh vmtablev2_*.csv 2>/dev/null || echo "  无 CSV 文件"
echo ""
echo "总大小:"
du -sh . 2>/dev/null
echo ""

# 验证首行格式（应为 20 列长格式 CSV）
FIRST_CSV=$(ls vmtablev2_*.csv 2>/dev/null | head -n 1)
if [ -n "$FIRST_CSV" ]; then
    cols=$(head -n 1 "$FIRST_CSV" | awk -F',' '{print NF}')
    echo "首行字段数: $cols (预期 20)"
    echo "首行预览:"
    head -n 1 "$FIRST_CSV" | cut -c1-200
    echo ""
fi

echo "=== 下载完成 ==="
echo ""
echo "下一步："
echo "  cd ~/Graph-RAG"
echo "  python scripts/bootstrap_graph.py --csv dataset/$FIRST_CSV --vms 50 --dry-run"
