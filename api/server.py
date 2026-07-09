"""Graph-RAG FastAPI 服务 —— 模块 5。

提供 HTTP API 把推理能力开放给前端 / 第三方调用。

端点：
  GET  /                健康检查
  GET  /api/health      服务状态 + 模型配置
  POST /api/ask         单条自然语言问答
  POST /api/ask/stream  流式问答（SSE）
  POST /api/eval        跑评估（4 baseline × 测试集）
  GET  /api/testset     查看测试集
  GET  /api/testset/{case_id}  看单条 case
  GET  /api/case-study  导出 G6 可视化数据
  GET  /docs            OpenAPI 交互式文档

启动：
  PYTHONPATH=. python -m api.server --port 8000
  PYTHONPATH=. uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import get_config
from eval.testset_builder import TestCase, build_testset, load_testset
from reasoning.claim_decomposer import ClaimDecomposer
from reasoning.controller import ReasoningController
from reasoning.hallucination_verifier import HallucinationVerifier
from reasoning.llm_interpreter import LLMClient

logger = logging.getLogger("api.server")

# ============================================================
#  Pydantic 数据模型（API 入参/出参）
# ============================================================

class AskRequest(BaseModel):
    """POST /api/ask 请求体。"""
    query: str = Field(..., description="自然语言问题", min_length=2, max_length=2000)
    use_llm: bool = Field(default=True, description="是否使用 LLM（False=规则降级）")
    time_window_hours: int | None = Field(default=None, description="时态窗口（小时）")


class HopInfo(BaseModel):
    """单跳信息（API 出参）。"""
    edge_name: str
    source_name: str
    source_label: str
    target_name: str
    target_label: str
    valid_at: str | None
    invalid_at: str | None
    lag_seconds: int
    attributes: dict[str, Any] = Field(default_factory=dict)


class PathInfo(BaseModel):
    """单条路径信息（API 出参）。"""
    path_id: str
    start_node: str
    end_node: str
    hop_count: int
    path_confidence: float
    is_temporally_consistent: bool
    hops: list[HopInfo]
    pruned_reason: str | None = None


class ClaimInfo(BaseModel):
    """单条原子声明（API 出参）。"""
    claim_id: str
    claim_text: str
    hop_index: int
    supporting_path_index: int
    source_nodes: list[str]
    source_edges: list[str]
    confidence: float
    is_verified: bool = False
    verification_confidence: float = 0.0


class AskResponse(BaseModel):
    """POST /api/ask 响应体。"""
    query: str
    answer: str
    confidence: float
    elapsed_seconds: float
    path_count: int
    paths: list[PathInfo]
    claims: list[ClaimInfo]
    metadata: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """GET /api/health 响应体。"""
    status: str
    gen_llm_model: str
    gen_llm_base_url: str
    verify_llm_model: str
    verify_llm_base_url: str
    neo4j_uri: str
    testset_size: int
    version: str = "0.1.0"


class EvalRequest(BaseModel):
    """POST /api/eval 请求体。"""
    max_cases: int = Field(default=5, ge=1, le=50, description="最多跑多少条 case")
    hops: list[int] = Field(default=[2, 3, 4], description="跳数列表")
    baselines: list[str] = Field(
        default=["B1_NaiveRAG", "B2_GraphitiDefault", "B3_NoTemporal", "B4_Full_GraphRAG"],
        description="要跑的 baseline",
    )


class EvalResponse(BaseModel):
    """POST /api/eval 响应体。"""
    total_cases: int
    results: dict[str, dict[str, float]]  # baseline_name -> { metric: value }
    summary: str
    elapsed_seconds: float


# ============================================================
#  状态管理（应用启动时初始化，进程内单例）
# ============================================================

class AppState:
    """应用全局状态。"""
    config: Any = None
    gen_client: LLMClient | None = None
    verify_client: LLMClient | None = None
    controller: ReasoningController | None = None
    decomposer: Any = None
    verifier: Any = None
    neo4j_driver: Any = None
    testset_cache: list[dict] = []
    testset_path: str = "eval/testset.jsonl"


def _to_path_info(p, idx: int = 0) -> PathInfo:
    """把 CausalPath 转 API 友好的 PathInfo。"""
    hops: list[HopInfo] = []
    for h in p.hops:
        hops.append(HopInfo(
            edge_name=h.edge_name,
            source_name=h.source.name,
            source_label=h.source.label,
            target_name=h.target.name,
            target_label=h.target.label,
            valid_at=h.valid_at.isoformat() if h.valid_at else None,
            invalid_at=h.invalid_at.isoformat() if h.invalid_at else None,
            lag_seconds=int(h.attributes.get("lag_seconds", 0)),
            attributes=h.attributes or {},
        ))
    return PathInfo(
        path_id=f"path_{idx}",
        start_node=p.start_node.name if p.start_node else "",
        end_node=p.end_node.name if p.end_node else "",
        hop_count=p.hop_count,
        path_confidence=p.path_confidence,
        is_temporally_consistent=p.is_temporally_consistent,
        hops=hops,
        pruned_reason=p.pruned_reason,
    )


def _to_claim_info(c, verified: list) -> ClaimInfo:
    """把 AtomicClaim 转 API 友好的 ClaimInfo。"""
    # 找对应的核验结果
    v_conf = 0.0
    is_verified = False
    for vc in verified:
        if vc.get("claim_id") == c.claim_id:
            v_conf = float(vc.get("confidence", 0.0))
            # verdict 可能是 enum 字符串或对象，统一用 .value 取
            verdict = vc.get("verdict")
            if hasattr(verdict, "value"):
                verdict = verdict.value
            is_verified = (verdict == "entailed")
            break
    return ClaimInfo(
        claim_id=c.claim_id,
        claim_text=c.claim_text,
        hop_index=c.hop_index,
        supporting_path_index=c.supporting_path_index,
        source_nodes=c.source_nodes,
        source_edges=c.source_edges,
        confidence=c.confidence,
        is_verified=is_verified,
        verification_confidence=v_conf,
    )


# ============================================================
#  应用入口
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时加载模型/客户端，关闭时清理。"""
    state = AppState()
    app.state.s = state

    state.config = get_config()
    # 加载测试集
    testset_path = PROJECT_ROOT / state.testset_path
    if testset_path.exists():
        state.testset_cache = load_testset(state.testset_path)
    else:
        # 现场构造
        testset = build_testset(
            n_per_hop={2: 100, 3: 100, 4: 100},
            output_path=testset_path,
        )
        state.testset_cache = [json.loads(json.dumps(t.__dict__, default=str))
                              for t in testset]

    # 构造 LLM 客户端
    try:
        state.gen_client = LLMClient(
            api_key=state.config.gen_llm.api_key,
            base_url=state.config.gen_llm.base_url,
            model=state.config.gen_llm.model,
        )
    except Exception as e:
        logger.error(f"Gen LLM 客户端构造失败: {e}")
        state.gen_client = None
    try:
        state.verify_client = LLMClient(
            api_key=state.config.verify_llm.api_key,
            base_url=state.config.verify_llm.base_url,
            model=state.config.verify_llm.model,
        )
    except Exception:
        state.verify_client = state.gen_client

    # 构造 controller + decomposer + verifier
    try:
        from neo4j import GraphDatabase
        state.neo4j_driver = GraphDatabase.driver(
            state.config.neo4j.uri,
            auth=(state.config.neo4j.user, state.config.neo4j.password),
        )
    except Exception as e:
        logger.error(f"Neo4j 连接失败: {e}")
        state.neo4j_driver = None

    if state.gen_client and state.neo4j_driver:
        state.controller = ReasoningController.from_config(
            neo4j_driver=state.neo4j_driver,
            llm_api_key=state.config.gen_llm.api_key,
            llm_base_url=state.config.gen_llm.base_url,
            llm_model=state.config.gen_llm.model,
        )
        state.decomposer = ClaimDecomposer(client=state.gen_client)
        state.verifier = HallucinationVerifier(client=state.verify_client)
    logger.info(
        f"API 启动: gen_client={state.gen_client is not None}, "
        f"neo4j={state.neo4j_driver is not None}, "
        f"testset={len(state.testset_cache)} 条"
    )
    yield
    # 关闭时清理
    if state.neo4j_driver is not None:
        state.neo4j_driver.close()


app = FastAPI(
    title="Graph-RAG API",
    description="Graph-RAG 推理引擎 HTTP 接口 —— 自然语言问答主答案可追溯",
    version="0.1.0",
    lifespan=lifespan,
)

# 允许跨域（前端 G6 页面调用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


FRONTEND_DIR = PROJECT_ROOT / "api" / "frontend"


@app.get("/", response_class=FileResponse, include_in_schema=False)
async def root():
    """根路径：返回 G6 可视化前端页面。

    如果前端文件不存在则降级返回 JSON 端点清单。
    """
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path, media_type="text/html")
    return JSONResponse({
        "service": "Graph-RAG API",
        "version": "0.1.0",
        "endpoints": [
            "GET  /api/health",
            "POST /api/ask",
            "POST /api/eval",
            "GET  /api/testset",
            "GET  /api/case-study/{case_id}",
            "GET  /docs",
        ],
    })


# ============================================================
#  路由
# ============================================================


@app.get("/api/health", response_model=HealthResponse)
async def health():
    """健康检查：返回模型/数据库/测试集状态。"""
    s = app.state.s
    return HealthResponse(
        status="ok" if s.gen_client and s.controller else "degraded",
        gen_llm_model=s.config.gen_llm.model if s.config else "",
        gen_llm_base_url=s.config.gen_llm.base_url if s.config else "",
        verify_llm_model=s.config.verify_llm.model if s.config else "",
        verify_llm_base_url=s.config.verify_llm.base_url if s.config else "",
        neo4j_uri=s.config.neo4j.uri if s.config else "",
        testset_size=len(s.testset_cache),
    )


@app.post("/api/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    """单条问答：自然语言 → 答案 + 路径 + claim 核验。"""
    s = app.state.s
    if s.controller is None:
        raise HTTPException(503, "推理控制器未就绪（Neo4j 或 LLM 不可用）")

    t0 = time.time()
    try:
        result = s.controller.ask(req.query)
        # 拆解 + 核验
        if s.decomposer and result.paths:
            decomp = s.decomposer.decompose(result.answer, result.paths)
            if s.verifier:
                report = s.verifier.verify(decomp, result.paths)
                verified_list = [
                    {
                        "claim_id": c.claim_id,
                        "confidence": c.confidence,
                        "verdict": c.verdict.value if hasattr(c.verdict, "value") else c.verdict,
                        "is_hallucination": c.is_hallucination,
                    }
                    for c in report.verified
                ]
            else:
                verified_list = []
            claims = [_to_claim_info(c, verified_list) for c in decomp.claims]
        else:
            claims = []

        paths = [_to_path_info(p, i) for i, p in enumerate(result.paths)]
        return AskResponse(
            query=req.query,
            answer=result.answer,
            confidence=result.confidence,
            elapsed_seconds=time.time() - t0,
            path_count=len(result.paths),
            paths=paths,
            claims=claims,
            metadata={
                "pruned_count": len(result.pruned_paths),
                "model": s.config.gen_llm.model,
            },
        )
    except Exception as e:
        logger.exception("ask 失败")
        raise HTTPException(500, f"推理失败: {e}")


@app.get("/api/testset")
async def list_testset(
    domain: str | None = Query(None, description="过滤域 smd/micross/cross"),
    hop: int | None = Query(None, description="过滤跳数 2/3/4"),
    limit: int = Query(50, ge=1, le=500, description="最多返回多少条"),
    offset: int = Query(0, ge=0, description="跳过多少条"),
):
    """分页查看测试集。"""
    s = app.state.s
    items = s.testset_cache
    if domain:
        items = [x for x in items if x.get("domain") == domain]
    if hop:
        items = [x for x in items if x.get("hop_count") == hop]
    total = len(items)
    items = items[offset:offset + limit]
    return {"total": total, "offset": offset, "limit": limit, "items": items}


@app.get("/api/testset/{case_id}")
async def get_case(case_id: str):
    """单条 case 详情。"""
    s = app.state.s
    for c in s.testset_cache:
        if c.get("case_id") == case_id:
            return c
    raise HTTPException(404, f"Case 不存在: {case_id}")


@app.get("/api/case-study/{case_id}")
async def case_study(case_id: str, baseline: str = Query("B4_Full_GraphRAG")):
    """导出 G6 可视化数据：单 case 的 ground truth + 选定 baseline 预测。

    返回格式与 scripts/export_case_study.py 一致，便于前端 G6 直接消费。
    """
    s = app.state.s
    case_dict = next((c for c in s.testset_cache if c.get("case_id") == case_id), None)
    if case_dict is None:
        raise HTTPException(404, f"Case 不存在: {case_id}")
    # 调 export_case_study 的 helper
    try:
        from scripts.export_case_study import to_g6_data
    except ImportError:
        # 兼容：脚本作为子模块
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "export_case_study", PROJECT_ROOT / "scripts" / "export_case_study.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore
        to_g6_data = mod.to_g6_data

    # 构造 TestCase
    from eval.testset_builder import ExpectedHop, SupportingFact
    case = TestCase(
        case_id=case_dict["case_id"],
        domain=case_dict["domain"],
        hop_count=case_dict["hop_count"],
        query=case_dict["query"],
        expected_path=[ExpectedHop(**h) for h in case_dict["expected_path"]],
        supporting_facts=[SupportingFact(**f) for f in case_dict["supporting_facts"]],
        query_time=case_dict["query_time"],
        ground_truth_free_text=case_dict["ground_truth_free_text"],
        metadata=case_dict.get("metadata", {}),
    )

    # 跑指定 baseline
    baseline_results: dict[str, Any] = {}
    try:
        if baseline == "B4_Full_GraphRAG" and s.controller is not None:
            from eval.baselines.b4_full import B4FullGraphRAG
            runner = B4FullGraphRAG(
                controller=s.controller,
                decomposer=s.decomposer,
                verifier=s.verifier,
            )
            br = runner.predict(case)
            baseline_results[baseline] = br
        elif baseline == "B3_NoTemporal" and s.controller is not None:
            from eval.baselines.b3_no_temporal import B3NoTemporal
            runner = B3NoTemporal(
                controller=s.controller,
                decomposer=s.decomposer,
                verifier=s.verifier,
            )
            br = runner.predict(case)
            baseline_results[baseline] = br
    except Exception as e:
        logger.exception(f"case_study 跑 {baseline} 失败")
        raise HTTPException(500, f"baseline 跑失败: {e}")

    g6 = to_g6_data(case, baseline_results)
    g6["query"] = case.query
    g6["query_time"] = case.query_time
    g6["ground_truth"] = case.ground_truth_free_text
    return g6


@app.post("/api/eval", response_model=EvalResponse)
async def eval_run(req: EvalRequest):
    """跑评估：4 baseline × 测试集子集。"""
    s = app.state.s
    t0 = time.time()
    if s.controller is None or s.gen_client is None:
        raise HTTPException(503, "推理组件未就绪")

    # 选测试集
    cases = s.testset_cache
    if req.hops:
        cases = [c for c in cases if c.get("hop_count") in req.hops]
    cases = cases[: req.max_cases]
    if not cases:
        raise HTTPException(400, "过滤后无 case")

    # 转 TestCase 对象
    from eval.testset_builder import ExpectedHop, SupportingFact
    testset = [
        TestCase(
            case_id=c["case_id"],
            domain=c["domain"],
            hop_count=c["hop_count"],
            query=c["query"],
            expected_path=[ExpectedHop(**h) for h in c["expected_path"]],
            supporting_facts=[SupportingFact(**f) for f in c["supporting_facts"]],
            query_time=c["query_time"],
            ground_truth_free_text=c["ground_truth_free_text"],
            metadata=c.get("metadata", {}),
        )
        for c in cases
    ]

    # 构造 baselines
    baselines: dict[str, Any] = {}
    if "B1_NaiveRAG" in req.baselines:
        from eval.baselines.b1_naive_rag import B1NaiveRAG
        baselines["B1_NaiveRAG"] = B1NaiveRAG(llm_client=s.gen_client)
    if "B2_GraphitiDefault" in req.baselines:
        from eval.baselines.b2_graphiti_default import B2GraphitiDefault
        try:
            from data_ingest.graphiti_writer import build_graphiti_client
            g = build_graphiti_client(config=s.config)
            baselines["B2_GraphitiDefault"] = B2GraphitiDefault(
                graphiti_client=g, llm_client=s.gen_client,
            )
        except Exception as e:
            logger.warning(f"B2 Graphiti 不可用: {e}")
            baselines["B2_GraphitiDefault"] = B2GraphitiDefault(
                graphiti_client=None, llm_client=s.gen_client,
            )
    if "B3_NoTemporal" in req.baselines and s.controller is not None:
        from eval.baselines.b3_no_temporal import B3NoTemporal
        baselines["B3_NoTemporal"] = B3NoTemporal(
            controller=s.controller, decomposer=s.decomposer, verifier=s.verifier,
        )
    if "B4_Full_GraphRAG" in req.baselines and s.controller is not None:
        from eval.baselines.b4_full import B4FullGraphRAG
        baselines["B4_Full_GraphRAG"] = B4FullGraphRAG(
            controller=s.controller, decomposer=s.decomposer, verifier=s.verifier,
        )

    # 跑评估
    from eval.metrics import aggregate_metrics, evaluate_case, report_to_markdown
    results: dict[str, dict[str, float]] = {}
    report_objs: dict[str, Any] = {}
    for name, baseline in baselines.items():
        logger.info(f"[api/eval] 跑 {name} × {len(testset)} 条")
        case_results = []
        for case in testset:
            try:
                r = baseline.predict(case)
                m = evaluate_case(
                    case=case,
                    predicted_hops=r.predicted_hops,
                    verified_claims=r.verified_claims,
                )
                case_results.append(m)
            except Exception as e:
                logger.warning(f"  {case.case_id} 失败: {e}")
        report = aggregate_metrics(name, case_results)
        report_objs[name] = report
        overall = report.overall
        results[name] = {
            "path_error_rate": round(overall.path_error_rate, 3),
            "hallucination_rate_overall": round(overall.hallucination_rate_overall, 3),
            "hallucination_rate_per_hop": round(overall.hallucination_rate_per_hop, 3),
            "recall": round(overall.recall, 3),
            "precision": round(overall.precision, 3),
            "temporal_accuracy": round(overall.temporal_accuracy, 3),
            "provenance_completeness": round(overall.provenance_completeness, 3),
        }

    # 生成 markdown summary
    summary_md = ""
    for name, r in report_objs.items():
        summary_md += report_to_markdown(r) + "\n\n"

    return EvalResponse(
        total_cases=len(testset),
        results=results,
        summary=summary_md,
        elapsed_seconds=time.time() - t0,
    )


# ============================================================
#  CLI 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Graph-RAG API 服务")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="开发模式自动重载")
    args = parser.parse_args()

    import uvicorn
    uvicorn.run(
        "api.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()
