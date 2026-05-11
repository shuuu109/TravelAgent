"""
SQLite 持久化层。

统一管理 data/aligo.sqlite，承载两类数据：
  1. sessions 元数据表（本文件维护：标题、时间、last_final_data）
  2. LangGraph checkpointer（AsyncSqliteSaver，A2 阶段接入；和本表共用文件、独立连接）

设计要点：
  - 进程内单实例 Database，FastAPI lifespan 启动时 open()、关闭时 close()
  - 同一个 sqlite 文件，两条连接（sessions / checkpointer），避免 saver 内部事务和 UI 查询互相阻塞
  - 表结构变更目前不做迁移脚本（按约定：dev 阶段直接删 .sqlite 重建）
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiosqlite

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "data/aligo.sqlite"

_SESSIONS_DDL = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id      TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL,
    title           TEXT NOT NULL DEFAULT '新对话',
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL,
    last_final_data TEXT
);
"""

_SESSIONS_INDEX = """
CREATE INDEX IF NOT EXISTS idx_sessions_user_updated
    ON sessions (user_id, updated_at DESC);
"""


class Database:
    """单实例 SQLite 网关，仅负责 sessions 表的 CRUD。

    checkpointer 在 A2 接入后由本类持有，但 schema 由 AsyncSqliteSaver.setup() 自建。
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        self.db_path = db_path
        self._conn: Optional[aiosqlite.Connection] = None

    async def open(self) -> None:
        Path(os.path.dirname(self.db_path) or ".").mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self.db_path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL;")
        await self._conn.execute(_SESSIONS_DDL)
        await self._conn.execute(_SESSIONS_INDEX)
        await self._conn.commit()
        logger.info("sqlite opened at %s", self.db_path)

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None
            logger.info("sqlite closed")

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("Database is not opened; call open() in lifespan first")
        return self._conn

    # ── sessions CRUD ───────────────────────────────────────────────────────

    async def create_session(self, session_id: str, user_id: str) -> Dict[str, Any]:
        """插入一条新 session，title 默认为 '新对话'，首轮对话后由 sse_routes 改写。"""
        now = time.time()
        await self.conn.execute(
            "INSERT INTO sessions (session_id, user_id, title, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, user_id, "新对话", now, now),
        )
        await self.conn.commit()
        return {
            "session_id": session_id,
            "user_id": user_id,
            "title": "新对话",
            "created_at": now,
            "updated_at": now,
            "last_final_data": None,
        }

    async def list_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        cursor = await self.conn.execute(
            "SELECT session_id, user_id, title, created_at, updated_at, "
            "       (last_final_data IS NOT NULL) AS has_final "
            "FROM sessions WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return [dict(r) for r in rows]

    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        cursor = await self.conn.execute(
            "SELECT session_id, user_id, title, created_at, updated_at, last_final_data "
            "FROM sessions WHERE session_id = ?",
            (session_id,),
        )
        row = await cursor.fetchone()
        await cursor.close()
        if row is None:
            return None
        d = dict(row)
        if d.get("last_final_data"):
            try:
                d["last_final_data"] = json.loads(d["last_final_data"])
            except json.JSONDecodeError:
                logger.warning("malformed last_final_data for session %s", session_id)
                d["last_final_data"] = None
        return d

    async def touch_session(
        self,
        session_id: str,
        *,
        title: Optional[str] = None,
        last_final_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """更新 updated_at；可选同时更新 title / last_final_data。

        title=None 表示不动；last_final_data=None 同理（要清空请显式传 {}，调用方按需）。
        """
        fields = ["updated_at = ?"]
        params: List[Any] = [time.time()]
        if title is not None:
            fields.append("title = ?")
            params.append(title)
        if last_final_data is not None:
            fields.append("last_final_data = ?")
            params.append(json.dumps(last_final_data, ensure_ascii=False))
        params.append(session_id)
        await self.conn.execute(
            f"UPDATE sessions SET {', '.join(fields)} WHERE session_id = ?",
            params,
        )
        await self.conn.commit()

    async def delete_session(self, session_id: str) -> bool:
        cursor = await self.conn.execute(
            "DELETE FROM sessions WHERE session_id = ?", (session_id,)
        )
        await self.conn.commit()
        deleted = cursor.rowcount > 0
        await cursor.close()
        return deleted
