"""本地 sentence-transformers Embedder 适配器 —— graphiti-core embedder 接口。

为什么需要这个适配器：
    graphiti-core 内置的 OpenAIEmbedder 需要 API Key + 外网，
    对演示场景不稳定。本适配器让 graphiti-core 用本地 sentence-transformers 模型，
    首次运行从魔搭社区（ModelScope）下载，缓存到本地后离线可用。

魔搭社区模型 ID 约定：
    BAAI/bge-base-zh-v1.5 → ModelScope 上的 ID 是 "BAAI/bge-base-zh-v1.5"（同名映射）
    魔搭社区搜索页：https://www.modelscope.cn/models/BAAI/bge-base-zh-v1.5

支持的环境变量：
    EMBEDDING_MODEL    默认 "BAAI/bge-base-zh-v1.5"
    EMBEDDING_DEVICE   默认 "cpu"，可选 "cuda"
    EMBEDDING_DIM      默认 768（bge-base-zh-v1.5 的输出维度）
    EMBEDDING_CACHE_DIR  默认 ~/.cache/graph-rag/embeddings
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Iterable

logger = logging.getLogger(__name__)


# ============================================================
#  魔搭社区下载辅助
# ============================================================

def _ensure_modelscope_installed() -> None:
    """确保 modelscope 库已安装，否则抛错。"""
    try:
        import modelscope  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "未安装 modelscope 库。\n"
            "请先执行：pip install modelscope\n"
            "（用于从魔搭社区下载 embedding 模型）"
        ) from e


def download_from_modelscope(model_id: str, cache_dir: str) -> str:
    """从魔搭社区下载模型到本地，返回本地缓存目录。

    魔搭社区下载逻辑：
        - 用 snapshot_download 把整个模型仓库下到 cache_dir
        - 模型文件结构与 HuggingFace 兼容，sentence-transformers 可直接加载
        - 首次下载约 400MB（bge-base-zh-v1.5），后续自动复用

    Parameters
    ----------
    model_id : str
        魔搭社区模型 ID，如 "BAAI/bge-base-zh-v1.5"
    cache_dir : str
        本地缓存根目录
    """
    _ensure_modelscope_installed()
    from modelscope import snapshot_download

    logger.info(f"开始从魔搭社区下载模型: {model_id}")
    logger.info(f"缓存目录: {cache_dir}")

    local_dir = snapshot_download(
        model_id=model_id,
        cache_dir=cache_dir,
        revision="master",
    )
    logger.info(f"模型下载完成: {local_dir}")
    return local_dir


# ============================================================
#  本地 SentenceTransformer Embedder（继承 graphiti-core EmbedderClient）
# ============================================================

def _import_embedder_client():
    """延迟导入 graphiti-core 的 EmbedderClient 基类。

    graphiti-core 0.29.x 要求 embedder 必须是 EmbedderClient 子类，
    且 pydantic 强校验（GraphitiClients.embedder 字段）。本类继承它。
    """
    from graphiti_core.embedder.client import EmbedderClient
    return EmbedderClient


class SentenceTransformerEmbedder(_import_embedder_client()):
    """基于 sentence-transformers 的本地 Embedder，继承 graphiti-core EmbedderClient。

    继承 EmbedderClient 后才能通过 GraphitiClients 的 pydantic 强校验。

    用法：
        embedder = SentenceTransformerEmbedder(
            model_name="BAAI/bge-base-zh-v1.5",
            cache_dir="~/.cache/graph-rag/embeddings",
        )
        # 注入到 graphiti
        graphiti = Graphiti(..., embedder=embedder)
    """

    def __init__(
        self,
        model_name: str | None = None,
        cache_dir: str | None = None,
        device: str | None = None,
        embedding_dim: int | None = None,
    ):
        """
        Parameters
        ----------
        model_name : str | None
            模型 ID，默认从环境变量 EMBEDDING_MODEL 读取
        cache_dir : str | None
            模型缓存目录，默认 ~/.cache/graph-rag/embeddings
        device : str | None
            推理设备，"cpu" / "cuda"
        embedding_dim : int | None
            输出维度（用于 graphiti 内部向量索引维度匹配）
        """
        self.model_name = model_name or os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-zh-v1.5")
        self.cache_dir = cache_dir or os.path.expanduser(
            os.environ.get("EMBEDDING_CACHE_DIR", "~/.cache/graph-rag/embeddings")
        )
        self.device = device or os.environ.get("EMBEDDING_DEVICE", "cpu")
        self.embedding_dim = embedding_dim or int(os.environ.get("EMBEDDING_DIM", "768"))
        self._model: Any = None

    def __bool__(self) -> bool:
        """让 graphiti-core 的 `if embedder:` 判断永远为 True。"""
        return True

    def _ensure_model(self) -> Any:
        """懒加载模型。首次调用时从魔搭社区下载。"""
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "未安装 sentence-transformers。\n"
                "请先执行：pip install sentence-transformers"
            ) from e

        # 1) 先尝试从魔搭社区下载（如果本地没缓存）
        os.makedirs(self.cache_dir, exist_ok=True)
        local_path = download_from_modelscope(self.model_name, self.cache_dir)

        # 2) sentence-transformers 加载本地路径
        logger.info(f"加载 sentence-transformers 模型: {local_path}, device={self.device}")
        self._model = SentenceTransformer(local_path, device=self.device)
        logger.info("模型加载完成")
        return self._model

    def _encode_sync(self, texts: list[str]) -> list[list[float]]:
        """同步推理：把文本列表转成向量列表。"""
        model = self._ensure_model()
        vectors = model.encode(
            texts,
            normalize_embeddings=True,  # 归一化，便于 cosine 相似度
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vectors.tolist()

    async def create(
        self,
        input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]],
    ) -> list[float]:
        """graphiti-core 调用的入口（EmbedderClient 抽象方法签名）。

        graphiti 0.29.x 期望签名是 create(input_data) -> list[float]，
        但允许 input_data 是单个字符串或字符串列表。
        对于字符串列表我们取第一条；字符串则直接编码。
        """
        # 统一成字符串列表
        if isinstance(input_data, str):
            texts = [input_data]
        elif isinstance(input_data, list) and all(isinstance(t, str) for t in input_data):
            texts = list(input_data)
        else:
            # tokens 等其他类型暂不支持，转字符串占位（实际上 graphiti 不会传这种）
            texts = [str(input_data)]

        # 单条直接返回第一条向量；多条也只返回第一条（按 EmbedderClient 抽象方法签名）
        loop = asyncio.get_event_loop()
        vectors = await loop.run_in_executor(None, self._encode_sync, texts)
        return vectors[0]

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        """批量编码（EmbedderClient 可选方法）。

        真正的批量实现，比逐条调用 create 高效很多（一次 GPU/CPU forward）。
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._encode_sync, list(input_data_list))


# ============================================================
#  工厂函数
# ============================================================

def build_local_embedder(
    model_name: str | None = None,
    cache_dir: str | None = None,
) -> SentenceTransformerEmbedder:
    """构建本地 embedder 工厂函数。

    供 graphiti_writer 调用：
        embedder = build_local_embedder()
    """
    return SentenceTransformerEmbedder(
        model_name=model_name,
        cache_dir=cache_dir,
    )
