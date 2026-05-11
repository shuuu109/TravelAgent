"""
长期记忆摘要构造器（CLI / Web SSE 路由共用）。

把 cli.py::AligoCLI._get_long_term_summary 的逻辑抽出来，
让 FastAPI 入口也能复用同一份摘要拼接规则。

输出格式（拼接为单个字符串，作为 SystemMessage 注入对话）：
    【用户背景信息】（来自长期记忆，可用于推断缺失信息）
    • home_location: 上海
    • hotel_brands: 汉庭, 如家
    ...
    【历史会话总结】
    <LLM 生成的多轮聊天浓缩>
    【历史行程】
    1. ✦ 上海 → 杭州 (2025-12-01) - 周末游
    ...
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


async def build_long_term_summary(memory_manager: Any, user_input: str = "") -> str:
    """
    构造长期记忆摘要。

    Args:
        memory_manager: MemoryManager 实例（含 long_term 与 get_long_term_summary_async）
        user_input:     当前轮用户输入。用于按地点筛选相关历史行程，
                        让 SystemMessage 优先包含与本次提问相关的轨迹。

    Returns:
        拼接好的摘要字符串；若三类信息都为空则返回空串（调用方应跳过注入）。
    """
    if memory_manager is None:
        return ""

    summary_parts: list[str] = []

    # 1. 用户偏好（始终全量加载）
    try:
        prefs = memory_manager.long_term.get_preference()
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"long_term.get_preference 失败: {exc}")
        prefs = {}

    if prefs:
        pref_lines = ["【用户背景信息】（来自长期记忆，可用于推断缺失信息）"]
        for pref_key, pref_value in prefs.items():
            if not pref_value:
                continue
            if isinstance(pref_value, list):
                pref_lines.append(f"• {pref_key}: {', '.join(pref_value)}")
            else:
                pref_lines.append(f"• {pref_key}: {pref_value}")
        if len(pref_lines) > 1:
            summary_parts.extend(pref_lines)

    # 2. LLM 总结的历史会话
    try:
        chat_summary = await memory_manager.get_long_term_summary_async(max_messages=50)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"get_long_term_summary_async 失败: {exc}")
        chat_summary = ""

    if chat_summary:
        summary_parts.append("\n【历史会话总结】")
        summary_parts.append(chat_summary)

    # 3. 历史行程：与 user_input 地点匹配的优先，最多 2 条相关 + 1 条最近
    try:
        all_trips = memory_manager.long_term.get_trip_history(limit=None) or []
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"get_trip_history 失败: {exc}")
        all_trips = []

    if all_trips:
        relevant_trips: list[dict] = []
        other_trips: list[dict] = []
        for trip in all_trips:
            origin = trip.get("origin") or ""
            destination = trip.get("destination") or ""
            if (origin and origin in user_input) or (destination and destination in user_input):
                relevant_trips.append(trip)
            else:
                other_trips.append(trip)

        trips_to_show = relevant_trips[:2] + other_trips[:1]
        if trips_to_show:
            summary_parts.append("\n【历史行程】")
            for i, trip in enumerate(trips_to_show[:3], 1):
                origin = trip.get("origin", "未知")
                destination = trip.get("destination", "未知")
                start_date = trip.get("start_date", "")
                purpose = trip.get("purpose", "")
                relevance_mark = "* " if trip in relevant_trips else ""
                summary_parts.append(
                    f"{i}. {relevance_mark}{origin} → {destination} ({start_date}) - {purpose}"
                )

    return "\n".join(summary_parts) if summary_parts else ""
