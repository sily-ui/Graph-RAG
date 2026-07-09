"""全局配置加载模块。

从 .env 文件读取 Neo4j 连接信息与 LLM 配置，
所有模块通过 config.get_config() 获取统一配置对象。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# 项目根目录（config.py 位于项目根目录下）
PROJECT_ROOT = Path(__file__).resolve().parent

# 加载 .env
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Neo4jConfig:
    uri: str
    user: str
    password: str


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    api_key: str
    base_url: str
    model: str


@dataclass(frozen=True)
class AppConfig:
    neo4j: Neo4jConfig
    gen_llm: LLMConfig          # 生成 LLM（查询规划/Cypher生成/答案合成）
    verify_llm: LLMConfig       # 核验 LLM（幻觉检测，必须与生成不同 provider）
    group_id_prefix: str


def _build_llm_config(prefix: str) -> LLMConfig:
    return LLMConfig(
        provider=os.environ.get(f"{prefix}_PROVIDER", "openai"),
        api_key=os.environ.get(f"{prefix}_API_KEY", ""),
        base_url=os.environ.get(f"{prefix}_BASE_URL", "https://api.openai.com/v1"),
        model=os.environ.get(f"{prefix}_MODEL", "gpt-4o"),
    )


def get_config() -> AppConfig:
    """从 .env 加载配置。

    VERIFY_LLM_* 任何字段缺失时，回退到 GEN_LLM_* 对应字段。
    这样 .env 只需配 GEN_LLM_*，核验 LLM 自动复用同一个 client。
    """
    gen = _build_llm_config("GEN_LLM")
    ver = _build_llm_config("VERIFY_LLM")

    # verify 配置回退：空字符串 → 用 gen 的
    ver_api_key = ver.api_key or gen.api_key
    ver_base_url = ver.base_url or gen.base_url
    ver_model = ver.model or gen.model
    ver_provider = ver.provider or gen.provider

    return AppConfig(
        neo4j=Neo4jConfig(
            uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            user=os.environ.get("NEO4J_USER", "neo4j"),
            password=os.environ.get("NEO4J_PASSWORD", ""),
        ),
        gen_llm=gen,
        verify_llm=LLMConfig(
            provider=ver_provider,
            api_key=ver_api_key,
            base_url=ver_base_url,
            model=ver_model,
        ),
        group_id_prefix=os.environ.get("GROUP_ID_PREFIX", "cluster"),
    )
