"""
RAG知识库智能体 RAGKnowledgeAgent
职责：基于向量数据库的知识检索与问答

核心功能：
1. 知识库构建：将商旅相关文档向量化并存储到ChromaDB
2. 语义检索：根据用户查询检索最相关的知识片段
3. 知识问答：结合检索到的知识和LLM生成准确答案
4. 知识管理：支持添加、更新、删除知识库内容

技术栈：
- ChromaDB: 轻量级向量数据库（本地存储，支持Windows）
- sentence-transformers: 文本向量化模型
- LLM: 用户配置的豆包模型用于生成答案

安装：
pip install chromadb sentence-transformers
"""
from typing import Optional, List, Dict
import json
import logging
import os
from pathlib import Path

# Add project root to sys.path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

logger = logging.getLogger(__name__)

try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"RAG dependencies not available: {e}")
    logger.warning("Install with: pip install chromadb sentence-transformers")
    DEPENDENCIES_AVAILABLE = False


class RAGKnowledgeAgent:
    """RAG知识库智能体"""

    def __init__(
        self,
        name: str = "RAGKnowledgeAgent",
        model=None,
        knowledge_base_path: str = None,
        collection_name: str = "business_travel_knowledge",
        embedding_model: str = "BAAI/bge-small-zh-v1.5",
        top_k: int = 3,
        **kwargs
    ):
        self.name = name
        self.model = model

        if knowledge_base_path is None:
            # Default to local data directory in skill folder
            current_dir = Path(__file__).parent.parent
            knowledge_base_path = str(current_dir / "data" / "rag_knowledge")

        self.knowledge_base_path = Path(knowledge_base_path)
        self.collection_name = collection_name
        self.top_k = top_k
        from utils.skill_loader import SkillLoader
        self.skill_loader = SkillLoader()

        if not DEPENDENCIES_AVAILABLE:
            logger.error("RAG dependencies not installed. Install with: pip install chromadb sentence-transformers")
            self.initialized = False
            return

        # 优先使用 config 中的配置（支持本地路径，避免连 HuggingFace）
        try:
            from config import RAG_CONFIG
            embedding_model = RAG_CONFIG.get("embedding_model", embedding_model)
        except Exception:
            pass

        # 若配置的是本地路径且存在，则从本地加载，否则按模型 ID 使用（会联网）
        model_path_or_id = embedding_model
        path_obj = Path(embedding_model).expanduser()
        if not path_obj.is_absolute():
            path_obj = Path.cwd() / path_obj
        if path_obj.exists():
            model_path_or_id = str(path_obj.resolve())
            logger.info(f"Using local embedding model: {model_path_or_id}")
        else:
            if "/" in embedding_model or "\\" in embedding_model or embedding_model.startswith("."):
                logger.warning(
                    f"Configured embedding path does not exist: {embedding_model}，将使用 BAAI/bge-small-zh-v1.5 并尝试联网下载。"
                )
                model_path_or_id = "BAAI/bge-small-zh-v1.5"
        logger.info(f"Loading embedding model: {model_path_or_id}")
        self.embedding_model = SentenceTransformer(model_path_or_id)

        # 初始化 ChromaDB（本地文件存储，完全支持 Windows）
        chroma_db_path = str(self.knowledge_base_path / "chroma_db")
        self.knowledge_base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initializing ChromaDB at: {chroma_db_path}")

        self.chroma_client = chromadb.PersistentClient(path=chroma_db_path)

        # 获取或创建 collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # 余弦相似度
        )
        logger.info(f"ChromaDB collection '{collection_name}' ready, "
                    f"current doc count: {self.collection.count()}")

        self.initialized = True
        logger.info("RAG Knowledge Agent (ChromaDB) initialized successfully")

    def _ensure_connection(self):
        """ChromaDB 是本地文件，无需重连，保留此方法保持接口兼容"""
        pass

    def add_documents(self, documents: List[Dict[str, str]]) -> Dict:
        """
        添加文档到知识库

        Args:
            documents: 文档列表，每个文档包含 {'content': '内容', 'metadata': {...}}

        Returns:
            添加结果统计
        """
        if not self.initialized:
            return {"status": "error", "message": "RAG Agent not initialized"}

        try:
            current_count = self.collection.count()

            ids = []
            embeddings = []
            contents = []
            metadatas = []

            for i, doc in enumerate(documents):
                doc_id = str(current_count + i + 1)
                content = doc['content']
                metadata = doc.get('metadata', {})
                # ChromaDB 的 metadata 值只支持 str/int/float/bool，需将嵌套结构序列化
                flat_metadata = {k: (v if isinstance(v, (str, int, float, bool)) else json.dumps(v, ensure_ascii=False))
                                 for k, v in metadata.items()}

                embedding = self.embedding_model.encode(content).tolist()

                ids.append(doc_id)
                embeddings.append(embedding)
                contents.append(content)
                metadatas.append(flat_metadata)

            # 批量插入
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=contents,
                metadatas=metadatas
            )

            total_count = self.collection.count()
            logger.info(f"Successfully added {len(documents)} documents to knowledge base")
            return {
                "status": "success",
                "added_count": len(documents),
                "total_count": total_count
            }

        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return {"status": "error", "message": str(e)}

    def search_knowledge(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """
        检索知识库

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            检索结果列表
        """
        if not self.initialized:
            return []

        k = top_k if top_k is not None else self.top_k

        try:
            # 生成查询向量
            query_embedding = self.embedding_model.encode(query).tolist()

            # 在 ChromaDB 中检索
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(k, self.collection.count()) if self.collection.count() > 0 else 1,
                include=["documents", "metadatas", "distances"]
            )

            # 格式化结果
            retrieved_docs = []
            if results and results.get("ids") and len(results["ids"]) > 0:
                ids = results["ids"][0]
                docs = results["documents"][0]
                metas = results["metadatas"][0]
                distances = results["distances"][0]

                for doc_id, content, metadata, distance in zip(ids, docs, metas, distances):
                    retrieved_docs.append({
                        'id': doc_id,
                        'content': content,
                        'metadata': metadata,
                        'distance': distance
                    })

            logger.info(f"Retrieved {len(retrieved_docs)} documents for query: {query[:50]}")
            return retrieved_docs

        except Exception as e:
            logger.error(f"Error searching knowledge: {e}")
            return []

    async def run(self, input_data: dict) -> dict:
        """
        RAG问答主流程
        1. 接收用户查询
        2. 检索相关知识
        3. 结合知识生成答案
        """
        if not self.initialized:
            return {
                "status": "error",
                "message": "RAG Agent not initialized. Please install dependencies: pip install chromadb sentence-transformers"
            }

        # 获取用户查询
        user_query = ""
        if "context" in input_data and isinstance(input_data["context"], dict):
            user_query = input_data["context"].get("rewritten_query", "")
        elif "rewritten_query" in input_data:
            user_query = input_data.get("rewritten_query", "")

        # 检索相关知识
        retrieved_docs = self.search_knowledge(user_query)

        if not retrieved_docs:
            return {
                "status": "no_knowledge",
                "query": user_query,
                "answer": "抱歉，我在知识库中没有找到相关信息。",
                "retrieved_documents": []
            }

        # 构建知识上下文
        knowledge_context = "\n\n".join([
            f"【知识片段{i+1}】\n{doc['content']}"
            for i, doc in enumerate(retrieved_docs)
        ])

        # 如果有LLM，使用LLM生成答案
        if self.model:
            skill_instruction = self.skill_loader.get_skill_content("ask-question")
            if not skill_instruction:
                skill_instruction = "请基于知识库中的信息回答用户的问题。"

            prompt = f"""你是一个旅游知识专家。请严格基于以下知识库中的信息回答用户的问题。

【用户问题】
{user_query}

【知识库信息】
{knowledge_context}

【任务说明】
{skill_instruction}

【重要约束】
1. 如果【知识库信息】中没有包含回答用户问题所需的信息，请直接回答"抱歉，知识库中没有找到相关信息"，不要尝试根据你自己的知识编造答案。
2. 即使问题很基础，如果知识库里没写，就说不知道。
3. 请以专业、客观的语气回答。
"""

            try:
                messages = [
                    {"role": "system", "content": "你是一个旅游知识专家。"},
                    {"role": "user", "content": prompt}
                ]
                response = await self.model.ainvoke(messages)
                answer = response.content

                if not answer:
                    answer = "无法生成答案"

                answer_str = answer.strip()
                if answer_str.startswith("{") and answer_str.endswith("}"):
                    try:
                        json_obj = json.loads(answer_str)
                        if isinstance(json_obj, dict):
                            answer = json_obj.get("answer") or json_obj.get("content") or answer
                    except Exception:
                        pass

            except Exception as e:
                logger.error(f"Error generating answer with LLM: {e}")
                answer = f"知识库中找到相关信息，但生成答案时出错：{str(e)}"
        else:
            answer = "以下是知识库中的相关信息：\n\n" + knowledge_context

        result = {
            "status": "success",
            "query": user_query,
            "answer": answer,
            "retrieved_documents": [
                {
                    "content": doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content'],
                    "metadata": doc['metadata']
                }
                for doc in retrieved_docs
            ]
        }

        return result

    def get_stats(self) -> Dict:
        """获取知识库统计信息"""
        if not self.initialized:
            return {"status": "error", "message": "Not initialized"}

        try:
            total = self.collection.count()
            return {
                "status": "success",
                "collection_name": self.collection_name,
                "total_documents": total,
                "knowledge_base_path": str(self.knowledge_base_path)
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def close(self):
        """ChromaDB 无需显式关闭连接"""
        pass

    def __del__(self):
        """析构函数"""
        self.close()
