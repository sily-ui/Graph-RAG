r"""Neo4j 连接与环境验证脚本。

运行方式（在项目根目录）：
    python scripts/test_neo4j_connection.py

验证项：
1. .env 配置是否完整
2. Neo4j 数据库是否可连接
3. Graphiti 是否能初始化
4. LLM API Key 是否非空
"""
from __future__ import annotations

import sys
from pathlib import Path

# 确保能导入项目根目录的模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neo4j import GraphDatabase

from config import get_config


def check_env() -> bool:
    """检查 .env 配置完整性。"""
    cfg = get_config()
    issues = []

    if not cfg.neo4j.password or cfg.neo4j.password == "请改成你设置的密码":
        issues.append("NEO4J_PASSWORD 未设置，请编辑 .env 填入你修改后的 Neo4j 密码")

    if not cfg.gen_llm.api_key or cfg.gen_llm.api_key == "请填入你的API Key":
        issues.append("GEN_LLM_API_KEY 未设置，请编辑 .env 填入你的生成 LLM API Key")

    if not cfg.verify_llm.api_key or cfg.verify_llm.api_key == "请填入另一个API Key":
        issues.append("VERIFY_LLM_API_KEY 未设置，请编辑 .env 填入你的核验 LLM API Key")

    warnings = []

    if cfg.gen_llm.api_key and cfg.verify_llm.api_key:
        if cfg.gen_llm.api_key == cfg.verify_llm.api_key:
            warnings.append("生成 LLM 与核验 LLM 使用了相同的 API Key（当前过渡期可接受，正式评估实验前需换成不同 provider）")

    if issues:
        print("[FAIL] 配置检查发现以下问题：")
        for issue in issues:
            print(f"  - {issue}")
        return False

    print("[OK] .env 配置完整")
    print(f"  Neo4j: {cfg.neo4j.uri} (user={cfg.neo4j.user})")
    print(f"  生成 LLM: {cfg.gen_llm.model} @ {cfg.gen_llm.base_url}")
    print(f"  核验 LLM: {cfg.verify_llm.model} @ {cfg.verify_llm.base_url}")

    if warnings:
        print("[WARN] 提醒：")
        for w in warnings:
            print(f"  - {w}")

    return True


def check_neo4j() -> bool:
    """验证 Neo4j 数据库连接。"""
    cfg = get_config()
    try:
        driver = GraphDatabase.driver(
            cfg.neo4j.uri,
            auth=(cfg.neo4j.user, cfg.neo4j.password),
        )
        # 验证连接
        driver.verify_connectivity()
        print("[OK] Neo4j 连接成功")

        # 查询数据库基本信息
        with driver.session() as session:
            result = session.run("RETURN 1 AS test")
            record = result.single()
            assert record["test"] == 1

            # 查询现有节点和关系数
            node_count = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            rel_count = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
            print(f"  当前图库: {node_count} 个节点, {rel_count} 条关系")

        driver.close()
        return True

    except Exception as e:
        print(f"[FAIL] Neo4j 连接失败: {e}")
        return False


def check_graphiti_import() -> bool:
    """验证 Graphiti 能正常导入和初始化（不实际连接）。"""
    try:
        from graphiti_core import Graphiti
        print("[OK] graphiti_core 导入成功")

        # 验证关键类可导入
        from graphiti_core.nodes import EntityNode
        from graphiti_core.edges import EntityEdge
        from graphiti_core.driver.driver import GraphDriver
        print("[OK] Graphiti 核心类导入成功 (EntityNode, EntityEdge, GraphDriver)")
        return True

    except Exception as e:
        print(f"[FAIL] Graphiti 导入失败: {e}")
        return False


def main():
    print("=" * 60)
    print("  Graph-RAG 环境验证")
    print("=" * 60)
    print()

    all_ok = True

    print("--- 1. 配置文件检查 ---")
    all_ok &= check_env()
    print()

    print("--- 2. Neo4j 连接检查 ---")
    all_ok &= check_neo4j()
    print()

    print("--- 3. Graphiti 依赖检查 ---")
    all_ok &= check_graphiti_import()
    print()

    print("=" * 60)
    if all_ok:
        print("  所有检查通过！环境就绪，可以开始开发。")
    else:
        print("  存在未通过项，请按提示修复后重试。")
    print("=" * 60)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
