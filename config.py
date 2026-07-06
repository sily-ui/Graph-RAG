"""全局配置加载模块。

从 .env 文件读取 Neo4j 连接信息与 LLM 配置，
所有模块通过 config.get_config() 获取统一配置对象。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
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
    return AppConfig(
        neo4j=Neo4jConfig(
            uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            user=os.environ.get("NEO4J_USER", "neo4j"),
            password=os.environ.get("NEO4J_PASSWORD", ""),
        ),
        gen_llm=_build_llm_config("GEN_LLM"),
        verify_llm=_build_llm_config("VERIFY_LLM"),
        group_id_prefix=os.environ.get("GROUP_ID_PREFIX", "cluster"),
    )
