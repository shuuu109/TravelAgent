"""
POST /api/chat/stream
=====================

将 LangGraph workflow 包成 Server-Sent Events 推送给前端。

事件流构成：
  1. 节点级 node_start / node_complete（来自 graph.astream_events）
  2. P3 内部子步骤事件（由 itinerary_planning_node 通过 progress_cb 主动推）
  3. final 事件 - 一次性下发完整结果数据，前端收到后关闭流

非 planning 分支（preference_only / memory_only / info_only）静默处理：
  - 抑制中间过程事件
  - 仅推一个 final 事件（result_type=text_only），data 只含 final_response
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator, Dict, Optional

from fastapi import APIRouter, HTTPException, Request
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from sse_starlette.sse import EventSourceResponse

from app.event_mapper import NODE_LABELS, SILENT_NODES, extract_node_data
from app.schemas import ChatHistoryResponse, ChatRequest, HistoryMessage, SSEEnvelope
from utils.memory_summary import build_long_term_summary

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chat", tags=["chat"])


def _envelope(**kwargs: Any) -> Dict[str, Any]:
    """构造 SSEEnvelope 并序列化为 dict（含默认 ts）"""
    return SSEEnvelope(**kwargs).model_dump(exclude_none=True)


def _to_sse_message(payload: Dict[str, Any]) -> Dict[str, Any]:
    """sse-starlette 接收的格式：{event, data: <str>}"""
    return {"event": "message", "data": json.dumps(payload, ensure_ascii=False)}


@router.post("/stream")
async def chat_stream(req: ChatRequest, request: Request) -> EventSourceResponse:
    """
    SSE endpoint。前端通过 fetch + ReadableStream 消费（EventSource 不支持 POST）。
    """
    cache = request.app.state.session_cache
    db = request.app.state.db
    graph = cache.get_graph(req.user_id, req.session_id)
    memory_manager = cache.get_memory(req.user_id, req.session_id)

    # 兜底：旧前端可能直接发不存在的 session_id（如写死的 'demo'）；
    # 这里自动落一条 sessions 记录，避免后续 history / list 找不到。
    existing = await db.get_session(req.session_id)
    if existing is None:
        await db.create_session(req.session_id, req.user_id)

    queue: asyncio.Queue = asyncio.Queue()

    # ── progress_cb：供 itinerary_planning_node 推送 P3 子步骤 ───────────────
    # 节点端通过 RunnableConfig.configurable.progress_cb 取出并 await 调用：
    #   cb(sub_node_name, label, data=...)
    async def progress_cb(node: str, label: str, data: Optional[Dict] = None) -> None:
        await queue.put(_envelope(
            event="node_complete",
            node=node,
            phase="P3",
            label=label,
            data=data,
        ))

    config = {
        "configurable": {
            "thread_id": req.session_id,
            "progress_cb": progress_cb,
        }
    }

    # 构造本轮新增消息。checkpointer 已通过 thread_id 持久化历史 messages，
    # 无需再手动拼接短期上下文。长期记忆摘要使用固定 id 让 add_messages reducer
    # 按 id 去重：首轮追加，后续原地替换，避免多轮堆积同一条 summary。
    new_messages: list = []
    long_term_summary = await build_long_term_summary(memory_manager, req.message)
    if long_term_summary:
        new_messages.append(
            SystemMessage(content=long_term_summary, id="long_term_summary")
        )
    new_messages.append(HumanMessage(content=req.message))

    async def run_graph() -> None:
        """后台跑 graph，把节点事件转译成 SSE envelope 投递到 queue。"""
        # 用单元素 dict 而不是普通变量，便于在嵌套作用域中改写
        intent_type_holder: Dict[str, Optional[str]] = {"value": None}

        try:
            async for ev in graph.astream_events(
                {"messages": new_messages},
                config=config,
                version="v2",
            ):
                kind = ev.get("event")
                name = ev.get("name")

                # 只关心已注册节点；忽略 graph 顶层、内部子 chain、LLM/tool 事件
                if name not in NODE_LABELS:
                    continue
                if name in SILENT_NODES:
                    continue

                # 静默判定：intent_type 已知且非 planning -> 抑制中间事件
                silent = (
                    intent_type_holder["value"] is not None
                    and intent_type_holder["value"] != "planning"
                )

                if kind == "on_chain_start":
                    if silent:
                        continue
                    phase, label = NODE_LABELS[name]
                    await queue.put(_envelope(
                        event="node_start", node=name, phase=phase, label=label,
                    ))

                elif kind == "on_chain_end":
                    output = (ev.get("data") or {}).get("output") or {}

                    # intent_node 完成后立即识别 intent_type，决定后续是否静默
                    if name == "intent":
                        intent_type_holder["value"] = output.get("intent_type")

                    if silent:
                        continue

                    phase, label = NODE_LABELS[name]
                    data = extract_node_data(name, output)
                    await queue.put(_envelope(
                        event="node_complete",
                        node=name, phase=phase, label=label, data=data,
                    ))

                    # negotiate 是终止节点（缺信息时）-> 通知前端等待用户输入
                    if name == "negotiate":
                        await queue.put(_envelope(
                            event="needs_input",
                            data={
                                "question": output.get("final_response"),
                                "missing_info": output.get("missing_info") or [],
                            },
                        ))

            # ── 流结束：读 checkpointer 拿最终 state，组装 final 事件 ─────────
            snapshot = await graph.aget_state(config)
            values: Dict[str, Any] = snapshot.values if snapshot else {}
            intent_type = values.get("intent_type") or intent_type_holder["value"]

            if intent_type == "planning":
                rag_ctx = values.get("rag_context")
                if hasattr(rag_ctx, "model_dump"):
                    rag_ctx = rag_ctx.model_dump()

                final_data: Dict[str, Any] = {
                    "result_type": "planning",
                    "final_response": values.get("final_response"),
                    "chat_summary": values.get("chat_summary"),
                    "current_plan": values.get("current_plan"),
                    "daily_routes": values.get("daily_routes"),
                    "daily_restaurants": values.get("daily_restaurants"),
                    "daily_options_by_tier": values.get("daily_options_by_tier"),
                    "poi_descriptions": values.get("poi_descriptions"),
                    "poi_photos": values.get("poi_photos"),
                    "rag_context": rag_ctx,
                    "transport_options": values.get("transport_options"),
                    "transport_return_options": values.get("transport_return_options"),
                    "budget_fit_message": values.get("budget_fit_message"),
                }
            else:
                final_data = {
                    "result_type": "text_only",
                    "final_response": values.get("final_response"),
                }

            await queue.put(_envelope(event="final", data=final_data))

            # ── 写回 sessions 表：刷新 updated_at + 落 last_final_data + 首轮起标题
            try:
                title_update = None
                if existing is None or (existing and existing.get("title") == "新对话"):
                    title_update = (req.message or "").strip()[:30] or "新对话"
                await db.touch_session(
                    req.session_id,
                    title=title_update,
                    last_final_data=final_data,
                )
            except Exception:  # noqa: BLE001
                logger.exception("touch_session failed (non-fatal)")

        except asyncio.CancelledError:
            logger.info("graph stream cancelled (client disconnected)")
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("graph stream failed")
            await queue.put(_envelope(event="error", data={"message": str(exc)}))
        finally:
            await queue.put(None)  # sentinel：结束 SSE 循环

    runner_task = asyncio.create_task(run_graph())

    async def sse_generator() -> AsyncIterator[Dict[str, Any]]:
        """从 queue 拉事件并推给客户端。检测断连时取消 graph 任务。"""
        try:
            while True:
                if await request.is_disconnected():
                    runner_task.cancel()
                    break
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                if item is None:  # sentinel
                    break
                yield _to_sse_message(item)
        finally:
            if not runner_task.done():
                runner_task.cancel()
            try:
                await runner_task
            except (asyncio.CancelledError, Exception):
                pass

    return EventSourceResponse(sse_generator())


@router.get("/history", response_model=ChatHistoryResponse)
async def chat_history(session_id: str, request: Request) -> ChatHistoryResponse:
    """
    回填指定 session 的对话历史 + 上次的右栏结构化结果。

    数据来源：
      - messages   ← checkpointer 中 graph state 的 messages 字段
      - last_final ← sessions 表 last_final_data 列

    注意：进度块、needs_input 这类临时消息不会回填——历史里只有静态的 user / agent text。
    """
    db = request.app.state.db
    checkpointer = request.app.state.checkpointer

    session = await db.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")

    # 直接调 checkpointer.aget_tuple；不经过 SessionCache 是为了避免给纯查询场景
    # build_graph 一份新 graph（成本不低，且会污染缓存）
    config = {"configurable": {"thread_id": session_id}}
    snapshot = await checkpointer.aget_tuple(config)
    raw_messages = []
    if snapshot is not None:
        channel_values = snapshot.checkpoint.get("channel_values", {})
        raw_messages = channel_values.get("messages", []) or []

    messages: list[HistoryMessage] = []
    for m in raw_messages:
        # 过滤掉给 LLM 看的系统提示（包括 long_term_summary）
        if isinstance(m, SystemMessage):
            continue
        text = m.content if isinstance(m.content, str) else str(m.content)
        if not text:
            continue
        if isinstance(m, HumanMessage):
            messages.append(HistoryMessage(role="user", text=text))
        elif isinstance(m, AIMessage):
            messages.append(HistoryMessage(role="agent", text=text))
        # 其他类型（ToolMessage 等）不回填前端

    return ChatHistoryResponse(
        session_id=session_id,
        messages=messages,
        last_final_data=session.get("last_final_data"),
    )
