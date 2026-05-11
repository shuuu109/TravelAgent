"""
HTTP 请求与 SSE 事件信封的 Pydantic 模型。

前后端契约都收敛在此文件，前端 TS 类型可对照本文件手写或脚本生成。
"""
from __future__ import annotations

import time
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """POST /api/chat/stream 请求体"""

    user_id: str = Field(..., description="用户 ID，对应 MemoryManager.user_id")
    session_id: str = Field(..., description="会话 ID，对应 LangGraph thread_id")
    message: str = Field(..., min_length=1, description="用户输入文本")


class SessionInfo(BaseModel):
    """GET /api/sessions 列表项 / POST 返回体"""

    session_id: str
    user_id: str
    title: str
    created_at: float
    updated_at: float
    has_final: bool = Field(default=False, description="是否已有最终规划结果可恢复右栏")


class CreateSessionRequest(BaseModel):
    user_id: str = Field(..., min_length=1)


class HistoryMessage(BaseModel):
    """GET /api/chat/history 返回的消息项（仅 user / agent 文本，不含进度块）"""

    role: Literal["user", "agent"]
    text: str


class ChatHistoryResponse(BaseModel):
    session_id: str
    messages: list[HistoryMessage]
    last_final_data: Optional[Dict[str, Any]] = None


# SSE event 类型枚举
EventKind = Literal[
    "node_start",      # 节点开始执行
    "node_complete",   # 节点执行完成（携带 data）
    "node_retry",      # 触发回环（如 itinerary_review / budget_check）
    "needs_input",     # 流程暂停，等用户补充信息（negotiate）
    "final",           # 流结束信号，data 含完整结果
    "error",           # 异常终止
]


class SSEEnvelope(BaseModel):
    """
    所有 SSE 事件的统一 JSON 信封。

    前端处理逻辑：
      - label  -> 聊天框上方的过程流文字
      - data   -> 累积到右侧结构化结果面板
      - event=final 收到后关闭 EventSource
    """

    event: EventKind
    node: Optional[str] = Field(default=None, description="节点名，final/error 时可省")
    phase: Optional[str] = Field(default=None, description="阶段标签：P1/P2/P3/P3.5/P3.6/P4/P4.5/P4.6/P5")
    label: Optional[str] = Field(default=None, description="用户可见的中文状态文字")
    data: Optional[Dict[str, Any]] = Field(default=None, description="节点结构化输出")
    ts: float = Field(default_factory=time.time, description="服务端时间戳（秒）")
