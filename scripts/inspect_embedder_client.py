"""查看 graphiti-core 0.29.2 的 EmbedderClient 基类。"""
import inspect

from graphiti_core.embedder import client as embedder_client_module

print("=== graphiti_core.embedder 模块 ===")
import graphiti_core.embedder
print(dir(graphiti_core.embedder))
print()

# 找 EmbedderClient 基类
for name in dir(embedder_client_module):
    obj = getattr(embedder_client_module, name)
    if inspect.isclass(obj) and "Client" in name:
        print(f"=== {name} ===")
        print(f"  文件: {inspect.getfile(obj)}")
        print(f"  父类: {[c.__name__ for c in obj.__mro__]}")
        src = inspect.getsource(obj)
        print(f"  源码:\n{src}")
        print()
