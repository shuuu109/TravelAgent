"""
RAG 检索基类 RAGBaseAgent
========================
提供共用的 ChromaDB 初始化和 search_knowledge() 方法，
供 RAGExperienceAgent 和 RAGRiskAgent 继承复用，避免重复代码。

子类只需实现 run(input_data: dict) -> dict 方法，
通过 self.search_knowledge() 完成向量检索。
"""
import logging
from typing import Optional, List, Dict
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"RAG dependencies not available: {e}. Install: pip install chromadb sentence-transformers")
    DEPENDENCIES_AVAILABLE = False


class RAGBaseAgent:
    """
    RAG 检索基类。

    封装 ChromaDB 连接初始化和 search_knowledge() 语义检索，
    不含任何业务逻辑（查询构造、结构化抽取等由子类实现）。

    知识库路径默认复用 ask-question 技能的 ChromaDB，
    确保所有 RAG 节点共享同一份向量化语料。
    """

    def __init__(
        self,
        name: str,
        model=None,
        knowledge_base_path: Optional[str] = None,
        collection_name: str = "travel_knowledge",
        embedding_model: str = "BAAI/bge-small-zh-v1.5",
        top_k: int = 5,
        **kwargs
    ):
        self.name = name
        self.model = model
        self.top_k = top_k

        # 默认复用 ask-question 技能的知识库，共享同一份 ChromaDB 数据
        if knowledge_base_path is None:
            skills_root = Path(__file__).parent.parent / ".claude" / "skills"
            knowledge_base_path = str(skills_root / "ask-question" / "data" / "rag_knowledge")

        self.knowledge_base_path = Path(knowledge_base_path)

        if not DEPENDENCIES_AVAILABLE:
            logger.error(f"[{self.name}] RAG dependencies not installed.")
            self.initialized = False
            return

        # embedding model：优先读 config，支持本地路径避免联网
        try:
            from config import RAG_CONFIG
            embedding_model = RAG_CONFIG.get("embedding_model", embedding_model)
        except Exception:
            pass

        path_obj = Path(embedding_model).expanduser()
        if not path_obj.is_absolute():
            path_obj = Path.cwd() / path_obj
        model_path_or_id = str(path_obj.resolve()) if path_obj.exists() else embedding_model

        logger.info(f"[{self.name}] Loading embedding model: {model_path_or_id}")
        self.embedding_model = SentenceTransformer(model_path_or_id)

        # ChromaDB 本地文件存储，复用已有 collection
        chroma_db_path = str(self.knowledge_base_path / "chroma_db")
        self.knowledge_base_path.mkdir(parents=True, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=chroma_db_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.initialized = True
        logger.info(f"[{self.name}] ChromaDB ready, docs: {self.collection.count()}")

    def search_knowledge(
        self,
        query: str,
        top_k: Optional[int] = None,
        city_filter: Optional[str] = None,
        section_filter: Optional[str] = None,
    ) -> List[Dict]:
        """
        向量语义检索。

        Args:
            query:          检索 query 文本
            top_k:          返回文档数，默认使用 self.top_k
            city_filter:    按 metadata.city 过滤，精准命中城市语料；
                            若过滤后无结果，调用方应主动降级为不带过滤的全局检索
            section_filter: 按 metadata.section 过滤，只召回特定章节（如"避坑指南"、
                            "住宿指南"）；与 city_filter 同时存在时使用 $and 组合过滤。
                            传入 None 表示不做 section 限制。

        Returns:
            检索结果列表，每项含 id / content / metadata / distance
        """
        if not self.initialized:
            return []

        k = top_k if top_k is not None else self.top_k
        try:
            query_embedding = self.embedding_model.encode(query).tolist()
            query_kwargs: dict = dict(
                query_embeddings=[query_embedding],
                n_results=min(k, max(self.collection.count(), 1)),
                include=["documents", "metadatas", "distances"]
            )

            # 构建 metadata 过滤条件
            # city_filter 只匹配 section 类 chunk（排除不带 section 的 city_overview）
            if city_filter and section_filter:
                # $and 组合：同时满足城市 + 章节
                query_kwargs["where"] = {
                    "$and": [
                        {"city": {"$eq": city_filter}},
                        {"section": {"$eq": section_filter}},
                    ]
                }
            elif city_filter:
                query_kwargs["where"] = {"city": city_filter}
            elif section_filter:
                query_kwargs["where"] = {"section": section_filter}

            results = self.collection.query(**query_kwargs)

            retrieved: List[Dict] = []
            if results and results.get("ids"):
                for doc_id, content, meta, dist in zip(
                    results["ids"][0],
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0]
                ):
                    retrieved.append({
                        "id": doc_id,
                        "content": content,
                        "metadata": meta,
                        "distance": dist
                    })

            logger.info(f"[{self.name}] Retrieved {len(retrieved)} docs for: {query[:50]!r}")
            return retrieved

        except Exception as e:
            logger.error(f"[{self.name}] search_knowledge error: {e}")
            return []
