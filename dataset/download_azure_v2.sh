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

# 检查已有文件：必须非空才跳过，否则删除重建
if [ -s "$LINKS_FILE" ]; then
    echo "[1/3] 链接清单已存在（$(wc -l < $LINKS_FILE) 行），跳过"
else
    # 空文件/不存在 → 删除后重新下载
    rm -f "$LINKS_FILE"

    echo "[1/3] 下载链接清单..."
    # 尝试多个源，**不再静默吞错**，显示每个源的 HTTP 状态码
    SOURCES=(
        "https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/AzurePublicDatasetV2.txt"
        "https://ghproxy.com/https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/AzurePublicDatasetV2.txt"
        "https://mirror.ghproxy.com/https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/AzurePublicDatasetV2.txt"
    )
    success=0
    for src in "${SOURCES[@]}"; do
        echo "  尝试: $src"
        # -w 输出 HTTP 状态码到 stderr
        http_code=$(curl -L -o "$LINKS_FILE" -s -w "%{http_code}" --connect-timeout 15 --max-time 60 "$src" 2>&1 | tail -n 1)
        # 实际是合并输出，状态码在最后一行
        http_code=$(curl -L -o "$LINKS_FILE" -s -w "\n%{http_code}" --connect-timeout 15 --max-time 60 "$src" 2>/dev/null | tail -n 1)
        size=$(stat -c%s "$LINKS_FILE" 2>/dev/null || echo 0)
        echo "    HTTP 状态: $http_code, 文件大小: $size bytes"
        if [ "$size" -gt 1000 ] && [ "$http_code" = "200" ]; then
            echo "    [成功] 清单已下载"
            success=1
            break
        fi
    done

    # 兜底方案：直接用硬编码的 URL 构造清单（Azure Blob 链接是稳定公开的）
    if [ "$success" -eq 0 ]; then
        echo "  [警告] 所有源失败，使用硬编码 URL 兜底"
        cat > "$LINKS_FILE" << 'EOF'
https://azurepublicdatasetv2.blob.core.windows.net/vmtablev2/vmtablev2_000000000000.csv
https://azurepublicdatasetv2.blob.core.windows.net/vmtablev2/vmtablev2_000000000001.csv
https://azurepublicdatasetv2.blob.core.windows.net/vmtablev2/vmtablev2_000000000002.csv
https://azurepublicdatasetv2.blob.core.windows.net/vmtablev2/vmtablev2_000000000003.csv
https://azurepublicdatasetv2.blob.core.windows.net/vmtablev2/vmtablev2_000000000004.csv
https://azurepublicdatasetv2.blob.core.windows.net/vmtablev2/vmtablev2_000000000005.csv
https://azurepublicdatasetv2.blob.core.windows.net/vmtablev2/vmtablev2_000000000006.csv
https://azurepublicdatasetv2.blob.core.windows.net/vmtablev2/vmtablev2_000000000007.csv
https://azurepublicdatasetv2.blob.core.windows.net/vmtablev2/vmtablev2_000000000008.csv
https://azurepublicdatasetv2.blob.core.windows.net/vmtablev2/vmtablev2_000000000009.csv
EOF
        echo "  [兜底] 已写入 10 个硬编码分片 URL"
    fi
fi

LINK_COUNT=$(wc -l < "$LINKS_FILE")
echo "  清单含 $LINK_COUNT 个文件链接"

if [ "$LINK_COUNT" -eq 0 ]; then
    echo "  [错误] 链接清单为空，无法继续"
    exit 1
fi

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
    echo "    URL: $url"

    downloaded=0
    for retry in 1 2 3; do
        # 用 HTTP 状态码判断成功
        http_code=$(curl -L -C - -o "$filename" -s -w "%{http_code}" --connect-timeout 30 --max-time 3600 "$url" 2>&1 | tail -n 1)
        size=$(stat -c%s "$filename" 2>/dev/null || stat -f%z "$filename" 2>/dev/null || echo 0)
        echo "    尝试 $retry/3: HTTP=$http_code, 大小=$size bytes"

        if [ "$http_code" = "200" ] && [ "$size" -gt 524288000 ]; then
            echo "  [成功] $filename ($(numfmt --to=iec $size 2>/dev/null || echo ${size} bytes))"
            downloaded=1
            break
        else
            echo "  [失败] HTTP=$http_code, 大小=$size"
            sleep 5
        fi
    done

    if [ "$downloaded" -eq 0 ]; then
        echo "  [错误] 跳过此分片（已重试 3 次）"
    fi

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
