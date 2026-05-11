"""
/api/sessions
=============

会话元数据 CRUD。和 checkpointer 解耦：checkpoints 表是 LangGraph 自管，
本路由只操作 Database.sessions 表，再调 checkpointer.adelete_thread 清理 graph state。

约定：
  - 创建 session 时 title 默认 '新对话'，sse_routes 在首轮对话完成后改写为消息前 30 字
  - 删除 session 会同步清除 checkpointer 中该 thread_id 的所有 checkpoint，
    以及 SessionCache 内存中已编译的 graph 实例
"""
from __future__ import annotations

import logging
import uuid
from typing import List

from fastapi import APIRouter, HTTPException, Request

from app.schemas import CreateSessionRequest, SessionInfo

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.get("", response_model=List[SessionInfo])
async def list_sessions(user_id: str, request: Request) -> List[SessionInfo]:
    db = request.app.state.db
    rows = await db.list_sessions(user_id)
    return [SessionInfo(**r) for r in rows]


@router.post("", response_model=SessionInfo)
async def create_session(payload: CreateSessionRequest, request: Request) -> SessionInfo:
    db = request.app.state.db
    session_id = uuid.uuid4().hex
    row = await db.create_session(session_id, payload.user_id)
    logger.info("session created user=%s session=%s", payload.user_id, session_id)
    return SessionInfo(**row, has_final=False)


@router.delete("/{session_id}", status_code=204)
async def delete_session(session_id: str, request: Request) -> None:
    db = request.app.state.db
    cache = request.app.state.session_cache
    checkpointer = request.app.state.checkpointer

    info = await db.get_session(session_id)
    if info is None:
        raise HTTPException(status_code=404, detail="session not found")

    # 1) 清 checkpointer 里的 graph state
    await checkpointer.adelete_thread(session_id)
    # 2) 清内存缓存里的 graph / memory
    cache.evict(info["user_id"], session_id)
    # 3) 删 sessions 表行
    await db.delete_session(session_id)
    logger.info("session deleted user=%s session=%s", info["user_id"], session_id)
