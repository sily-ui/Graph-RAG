"""诊断 graphiti-core 的 Graphiti 初始化签名 + 当前 graphiti_writer.py 的 embedder 传参情况。"""
import inspect
from graphiti_core import Graphiti

print("=== Graphiti.__init__ 参数 ===")
sig = inspect.signature(Graphiti.__init__)
for name, param in sig.parameters.items():
    print(f"  {name}: default={param.default}")
print()

# 检查 graphiti_writer.py 中 embedder 的实际传参
print("=== graphiti_writer.py build_graphiti_client 中 embedder 相关代码 ===")
import os
writer_path = os.path.abspath("data_ingest/graphiti_writer.py")
print(f"文件: {writer_path}")
print(f"存在: {os.path.exists(writer_path)}")
if os.path.exists(writer_path):
    with open(writer_path, encoding="utf-8") as f:
        content = f.read()
    # 找到 build_graphiti_client 函数
    idx = content.find("def build_graphiti_client")
    if idx >= 0:
        end = content.find("\ndef ", idx + 10)
        if end < 0:
            end = len(content)
        print(content[idx:end])
print()

# 检查 local_embedder 类的 __bool__ 实现
print("=== SentenceTransformerEmbedder 类的布尔值判断 ===")
try:
    from data_ingest.local_embedder import SentenceTransformerEmbedder
    e = SentenceTransformerEmbedder()
    print(f"  bool(e) = {bool(e)}")
    print(f"  type(e) = {type(e).__name__}")
except Exception as ex:
    print(f"  import 失败: {ex}")

