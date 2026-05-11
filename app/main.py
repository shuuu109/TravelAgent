"""
Aligo Travel Agent FastAPI 入口。

启动方式：
    uvicorn app.main:app --reload --port 8000

设计说明：
  - 每个 (user_id, session_id) 维护独立的 MemoryManager + 编译后的 graph
    （memory_manager 通过闭包绑入节点工厂，所以无法跨用户复用 graph 实例）
  - 全局共享一个 AsyncSqliteSaver 作为 LangGraph checkpointer：进程重启后仍能续上历史
  - sessions 元数据由 Database 维护（data/aligo.sqlite，与 checkpointer 同文件不同连接）
  - CORS 仅允许 Vite dev 端口，正式部署需改为前端域名
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, Tuple

import aiosqlite
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.db import DEFAULT_DB_PATH, Database
from app.sessions_routes import router as sessions_router
from app.sse_routes import router as chat_router
from context.memory_manager import MemoryManager
from graph.workflow import build_graph

logger = logging.getLogger(__name__)


class SessionCache:
    """
    进程内的 graph / memory 缓存。

    checkpointer 由外部注入并跨 session 共享（持久化到同一 sqlite 文件），
    本类只负责按 (user_id, session_id) 缓存 build_graph 的产物。
    """

    def __init__(self, checkpointer: AsyncSqliteSaver) -> None:
        self._checkpointer = checkpointer
        self._memories: Dict[Tuple[str, str], MemoryManager] = {}
        self._graphs: Dict[Tuple[str, str], Any] = {}

    @property
    def checkpointer(self) -> AsyncSqliteSaver:
        return self._checkpointer

    def get_memory(self, user_id: str, session_id: str) -> MemoryManager:
        key = (user_id, session_id)
        if key not in self._memories:
            self._memories[key] = MemoryManager(
                user_id=user_id,
                session_id=session_id,
                llm_model=None,
            )
        return self._memories[key]

    def get_graph(self, user_id: str, session_id: str) -> Any:
        key = (user_id, session_id)
        if key not in self._graphs:
            mm = self.get_memory(user_id, session_id)
            self._graphs[key] = build_graph(memory_manager=mm, checkpointer=self._checkpointer)
            logger.info("graph built for session user=%s session=%s", user_id, session_id)
        return self._graphs[key]

    def evict(self, user_id: str, session_id: str) -> None:
        """删除 session 时一并清掉 graph / memory 的内存缓存。"""
        key = (user_id, session_id)
        self._graphs.pop(key, None)
        self._memories.pop(key, None)


@asynccontextmanager
async def lifespan(app: FastAPI):
    db = Database()
    await db.open()

    # checkpointer 专用连接，与 sessions 表分开，避免事务互相阻塞
    cp_conn = await aiosqlite.connect(DEFAULT_DB_PATH)
    checkpointer = AsyncSqliteSaver(cp_conn)
    await checkpointer.setup()  # 幂等，首次创建 checkpoints/writes 表

    app.state.db = db
    app.state.checkpointer = checkpointer
    app.state.cp_conn = cp_conn  # 仅供 lifespan 关闭时使用
    app.state.session_cache = SessionCache(checkpointer)
    logger.info("Aligo backend started (sqlite checkpointer at %s)", DEFAULT_DB_PATH)
    try:
        yield
    finally:
        await cp_conn.close()
        await db.close()
        logger.info("Aligo backend shutting down")


app = FastAPI(title="Aligo Travel Agent API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],  # Vite dev：localhost 与 127.0.0.1 是两个不同 Origin，都要放行
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(sessions_router)


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}
