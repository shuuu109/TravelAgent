"""chat_summary 子包入口：装配 ChatPanel 用结构化摘要。"""
import logging
from typing import Any, Dict, List, Optional

from graph.state import TravelGraphState, ensure_hard_constraints
from graph.nodes._respond.chat_summary.budget import _build_budget, _select_food_tier
from graph.nodes._respond.chat_summary.timeline import _build_timeline

logger = logging.getLogger(__name__)

def _build_chat_summary(
    state: TravelGraphState,
    refined_tips: List[str],
    refined_risks: List[str],
    weather: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    装配面向前端 ChatPanel 的结构化摘要（state.chat_summary）。

    约定:
      - refined_tips / refined_risks 已是"去序号、去 markdown 标题"的纯文本列表，
        调用方负责把 _llm_refine_tips_risks 的输出拆成 list[str]。
      - weather 由调用方（respond_node）通过 _build_weather 异步生成后传入；
        本函数保持同步，不在内部触发 LLM 调用。
      - 任意子步失败兜底为空值/默认值，绝不抛出异常——它在生成回复后才执行，
        不能让一个字段拖垮整次回复。

    返回结构:
      {
        "headline": {origin, destination, start_date, end_date, travel_days, pax},
        "timeline": [...],   # _build_timeline 输出
        "budget":   {...},   # _build_budget 输出
        "tips":     [str],   # 透传 refined_tips
        "risks":    [str],   # 透传 refined_risks
        "weather":  {summary, advice} | None,
      }
    """
    # ---- Step 1: 硬约束 + 旅行天数兜底 ----
    hc = ensure_hard_constraints(state.get("hard_constraints"))

    travel_days: int = state.get("travel_days") or 0
    if travel_days <= 0 and hc.start_date and hc.end_date:
        try:
            from datetime import datetime
            sd = datetime.strptime(hc.start_date, "%Y-%m-%d")
            ed = datetime.strptime(hc.end_date, "%Y-%m-%d")
            travel_days = (ed - sd).days + 1
        except Exception:
            travel_days = 0
    if travel_days <= 0:
        travel_days = len(state.get("daily_routes") or []) or 1

    # ---- Step 2: headline ----
    headline: Dict[str, Any] = {
        "origin":      hc.origin or "",
        "destination": hc.destination or "",
        "start_date":  hc.start_date or "",
        "end_date":    hc.end_date or "",
        "travel_days": travel_days,
        "pax":         hc.pax or 1,
    }

    # ---- Step 3: timeline ----
    try:
        timeline = _build_timeline(state)
    except Exception as e:
        logger.warning(f"_build_chat_summary: timeline 构建失败: {e}")
        timeline = []

    # ---- Step 4: budget ----
    try:
        food_tier = _select_food_tier(state)
        budget = _build_budget(state, food_tier)
    except Exception as e:
        logger.warning(f"_build_chat_summary: budget 构建失败: {e}")
        budget = {
            "currency": "¥",
            "total":    0,
            "limit":    None,
            "fit":      "unknown",
            "items":    [],
        }

    # ---- Step 5 & 6: 透传 tips/risks + weather + 组装返回 ----
    return {
        "headline": headline,
        "timeline": timeline,
        "budget":   budget,
        "tips":     list(refined_tips or []),
        "risks":    list(refined_risks or []),
        "weather":  weather,
    }
