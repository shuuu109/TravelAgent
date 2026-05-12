"""天气简报 + LLM 穿衣建议（chat_summary.weather）。"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from graph.state import TravelGraphState, ensure_hard_constraints

logger = logging.getLogger(__name__)


def _extract_weather_summary(state: TravelGraphState) -> str:
    """从 state.transport_outbound 提取天气简报字符串，缺失返回空串。"""
    outbound = state.get("transport_outbound") or {}
    if not isinstance(outbound, dict):
        return ""
    summary = outbound.get("weather_summary")
    if isinstance(summary, str):
        return summary.strip()
    return ""


async def _build_weather(state: TravelGraphState, llm) -> Optional[Dict[str, Any]]:
    """
    构造 chat_summary.weather。

    数据源:
      - state.transport_outbound.weather_summary（TransportAgent 通过高德 maps_weather 写入）

    返回:
      - 天气简报为空 -> None（前端不渲染该区块）
      - 正常 -> {"summary": "晴 18-27℃", "advice": "..."}

    LLM 失败时 advice 兜底为空串，仍返回 summary，让前端至少能展示天气本身。
    """
    summary = _extract_weather_summary(state)
    if not summary:
        return None

    hc = ensure_hard_constraints(state.get("hard_constraints"))
    destination = hc.destination or ""
    start_date = hc.start_date or ""
    travel_style = state.get("travel_style") or "普通"

    prompt = (
        "你是一位贴心的旅行小助手。请根据下面的天气简报，"
        "用亲切、口语化的中文，给用户写一句穿衣建议。\n\n"
        f"目的地：{destination or '目的地'}\n"
        f"出发日期：{start_date or '近期'}\n"
        f"旅行风格：{travel_style}\n"
        f"天气简报：{summary}\n\n"
        "要求：\n"
        "1. 30-60 字，1-2 句。\n"
        "2. 口语化、亲切，可以使用「啦」「哦」「呢」等语气词，但不要过度。\n"
        "3. 直接给穿搭建议，必要时提醒早晚温差、防晒、雨具等。\n"
        "4. 不要复述天气数值，不要重复目的地名，不要用 markdown。\n"
        "5. 直接输出建议正文，不要前缀、不要引号。"
    )

    advice = ""
    try:
        response = await llm.ainvoke([{"role": "user", "content": prompt}])
        advice = (response.content or "").strip().strip('"').strip("「」")
    except Exception as e:
        logger.warning(f"_build_weather: LLM 生成穿衣建议失败（降级为空串）: {e}")

    return {
        "summary": summary,
        "advice": advice,
    }
