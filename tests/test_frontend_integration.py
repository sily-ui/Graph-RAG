"""G6 前端集成测试 —— 验证后端 API 能正确支持前端可视化。

测试矩阵：
1. / 路由返回 index.html（前端入口）
2. OpenAPI 文档可访问
3-5. /api/health, /api/testset, /api/case-study 路由已注册
6. G6 数据格式校验（节点有 id/label/name/category，边有 source/target/edge_name）
7-8. 前端 HTML 包含 G6 CDN + 3 个 API 调用 + render 函数

所有测试用 FastAPI TestClient（不实际启动 uvicorn）。
"""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 直接用 app 构造测试 client（不跑 lifespan，避免需要真实 Neo4j/LLM）
from api.server import app  # noqa: E402

# 用 lifespan 但不依赖 Neo4j（lifespan 在 Neo4j 失败时仍能完成 state 初始化）
# TestClient 用 with 触发 lifespan
# 由于 lifespan 可能因为 Neo4j 失败而抛异常，我们用 mock 跳过依赖

# 简单做法：测试路由是否被注册，不实际调用需要 state 的端点
client = TestClient(app)


# ============================================================
#  1. / 路由返回 index.html
# ============================================================
def test_root_returns_frontend_html():
    r = client.get("/")
    assert r.status_code == 200
    # FileResponse 媒体类型是 text/html
    assert "text/html" in r.headers.get("content-type", "")
    body = r.text
    assert "Graph-RAG" in body
    assert "g6" in body.lower() or "antv" in body.lower()  # G6 CDN
    print(f"✅ / 返回 HTML，长度 {len(body)} 字节")


# ============================================================
#  2. OpenAPI 文档可访问
# ============================================================
def test_openapi_docs_accessible():
    r = client.get("/openapi.json")
    assert r.status_code == 200
    spec = r.json()
    paths = spec.get("paths", {})
    # 关键端点都在
    assert "/api/health" in paths
    assert "/api/ask" in paths
    assert "/api/case-study/{case_id}" in paths
    assert "/api/testset" in paths
    print(f"✅ OpenAPI 规范包含 {len(paths)} 条路径")


# ============================================================
#  3. /api/health 路由已注册（不需要 state）
# ============================================================
def test_health_endpoint_registered():
    """健康端点的路径在 OpenAPI 里，不需要 state。"""
    r = client.get("/openapi.json")
    spec = r.json()
    assert "/api/health" in spec["paths"]
    assert spec["paths"]["/api/health"]["get"] is not None
    print(f"✅ /api/health 路由已注册")


# ============================================================
#  4. /api/testset 路由已注册
# ============================================================
def test_testset_endpoint_registered():
    r = client.get("/openapi.json")
    spec = r.json()
    assert "/api/testset" in spec["paths"]
    print(f"✅ /api/testset 路由已注册")


# ============================================================
#  5. /api/case-study 路由已注册
# ============================================================
def test_case_study_endpoint_registered():
    r = client.get("/openapi.json")
    spec = r.json()
    assert "/api/case-study/{case_id}" in spec["paths"]
    print(f"✅ /api/case-study 路由已注册")


# ============================================================
#  6. G6 数据格式校验（不通过 HTTP，直接调 to_g6_data）
# ============================================================
def test_g6_data_format_valid():
    """直接验证 to_g6_data 返回的 G6 格式。"""
    from eval.testset_builder import ExpectedHop, TestCase
    from scripts.export_case_study import to_g6_data

    case = TestCase(
        case_id="test_001",
        domain="graph",
        hop_count=2,
        query="vm_001 的 cpu_spike 根因",
        expected_path=[
            ExpectedHop(
                source_label="Component",
                source_name="vm_001",
                edge_name="HAS_SYMPTOM",
                target_label="Symptom",
                target_name="cpu_spike",
            ),
            ExpectedHop(
                source_label="Symptom",
                source_name="cpu_spike",
                edge_name="CAUSED_BY",
                target_label="Cause",
                target_name="high_load",
            ),
        ],
        supporting_facts=[],
        query_time="2026-07-09T00:00:00Z",
        ground_truth_free_text="vm_001 因高负载导致 cpu_spike",
        metadata={},
    )

    # 模拟一个 baseline 预测
    class MockHop:
        def __init__(self, source, target, edge_name):
            self.source = type("S", (), {"name": source, "label": "Component"})()
            self.target = type("T", (), {"name": target, "label": "Symptom"})()
            self.edge_name = edge_name

    class MockResult:
        def __init__(self, hops):
            self.predicted_hops = hops
            self.verified_claims = []
            self.answer = "test"
            self.elapsed_seconds = 1.0
            self.error = None

    # 构造一个 baseline 预测（命中 1 跳，错过 1 跳）
    baseline_results = {
        "B4_Full_GraphRAG": MockResult([
            MockHop("vm_001", "cpu_spike", "HAS_SYMPTOM"),  # 命中
            # 故意漏掉第 2 跳
        ]),
    }

    g6 = to_g6_data(case, baseline_results)

    # 验证字段
    assert g6["case_id"] == "test_001"
    assert g6["query"] == "vm_001 的 cpu_spike 根因"
    assert isinstance(g6["nodes"], list)
    assert isinstance(g6["edges"], list)
    assert len(g6["nodes"]) > 0
    assert len(g6["edges"]) > 0

    # 节点字段校验
    for n in g6["nodes"]:
        assert "id" in n
        assert "label" in n
        assert "name" in n
        assert "category" in n
        assert n["category"] in ("ground_truth", "predicted")

    # 边字段校验
    for e in g6["edges"]:
        assert "id" in e
        assert "source" in e
        assert "target" in e
        assert "label" in e
        assert "category" in e

    # ground truth 节点数：2 跳共享中间节点 = 3 个（Comp, Symptom, Cause）
    # 注意：之前的旧版按 source/target 分别建节点会得到 4 个，但 2 跳应该只有 3 个不同节点
    gt_nodes = [n for n in g6["nodes"] if n["category"] == "ground_truth"]
    assert len(gt_nodes) == 3, f"期望 3 个 ground_truth 节点（2 跳共享中间），实际 {len(gt_nodes)}"

    # ground truth 边数 = 2 跳
    gt_edges = [e for e in g6["edges"] if e["category"] == "ground_truth"]
    assert len(gt_edges) == 2, f"期望 2 条 ground_truth 边，实际 {len(gt_edges)}"

    # 验证边引用都正确（关键：旧版有 source:/target: 不同 id 断链问题）
    node_ids = {n["id"] for n in g6["nodes"]}
    for e in g6["edges"]:
        assert e["source"] in node_ids, f"边 {e['id']} 引用不存在的 source: {e['source']}"
        assert e["target"] in node_ids, f"边 {e['id']} 引用不存在的 target: {e['target']}"

    # predicted 节点应该有（baseline 命中 1 跳 → 2 个节点 1 条边）
    pred_nodes = [n for n in g6["nodes"] if n["category"] == "predicted"]
    assert len(pred_nodes) == 2

    print(f"✅ G6 数据格式校验通过：{len(g6['nodes'])} 节点, {len(g6['edges'])} 边（无断链）")


# ============================================================
#  7. 静态文件目录存在
# ============================================================
def test_frontend_file_exists():
    index_path = ROOT / "api" / "frontend" / "index.html"
    assert index_path.exists(), f"前端文件不存在: {index_path}"
    content = index_path.read_text(encoding="utf-8")
    # 关键元素
    assert "Graph-RAG" in content
    assert "g6" in content.lower() or "antv" in content.lower()
    # API 调用
    assert "/api/case-study" in content
    assert "/api/health" in content
    assert "/api/testset" in content
    print(f"✅ 前端 HTML 完整（含 G6 CDN + 3 个 API 调用）")


# ============================================================
#  8. 前端 JS 关键函数存在
# ============================================================
def test_frontend_has_render_logic():
    index_path = ROOT / "api" / "frontend" / "index.html"
    content = index_path.read_text(encoding="utf-8")
    # 关键函数
    assert "renderGraph" in content or "render" in content, "缺少 render 函数"
    assert "G6" in content or "Graph" in content, "缺少 G6 实例化"
    # 错误处理
    assert "catch" in content, "缺少错误处理"
    print(f"✅ 前端 JS 渲染逻辑完整")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
