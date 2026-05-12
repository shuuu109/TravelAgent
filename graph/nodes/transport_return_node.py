"""
返程交通节点 transport_return_node (P2 fan-out)

职责：查询返程 (destination → origin) 的交通方案，写入 state.transport_return（完整 plan）
      + state.transport_return_options（校验后选项）+ skill_results（agent_name="transport_return"）。

返程日期推断（优先级从高到低）：
  1. hard_constraints.end_date            — 用户显式给了返程
  2. start_date + travel_days             — 用 P1.4 计算的天数（itinerary 总天数）
  3. start_date + 1                       — 默认次日返程兜底

触发条件与 transport_outbound_node 相同。
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from graph.state import TravelGraphState, ensure_hard_constraints
from graph.nodes._transport_helper import (
    make_skill_result,
    query_one_date,
    validate_options,
)

logger = logging.getLogger(__name__)


def _infer_return_date(
    start_date: str,
    end_date: Optional[str],
    travel_days: int,
) -> Optional[str]:
    """返程日期推断。返回 YYYY-MM-DD 字符串或 None（无法推断时）。"""
    if end_date:
        return end_date[:10]
    if not start_date:
        return None
    try:
        start = datetime.strptime(start_date[:10], "%Y-%m-%d")
    except (ValueError, TypeError):
        return None
    # travel_days 含起讫两端：N 天行程的返程在 start + (N-1)
    offset = max(travel_days - 1, 1)
    return (start + timedelta(days=offset)).strftime("%Y-%m-%d")


def create_transport_return_node(registry):
    async def transport_return_node(state: TravelGraphState) -> dict:
        if state.get("intent_type") != "planning":
            return {}

        hard_constraints = ensure_hard_constraints(state.get("hard_constraints"))
        origin: str = hard_constraints.origin or ""
        destination: str = hard_constraints.destination or ""

        if not origin or not destination or origin == destination:
            logger.info(
                f"[transport_return_node] 跳过 "
                f"(origin={origin!r}, destination={destination!r})"
            )
            return {}

        return_date = _infer_return_date(
            start_date=hard_constraints.start_date or "",
            end_date=hard_constraints.end_date,
            travel_days=state.get("travel_days") or 0,
        )
        if not return_date:
            logger.info("[transport_return_node] 无法推断返程日期，跳过")
            return {}

        try:
            agent = registry["transport_query"]
        except KeyError:
            logger.warning("[transport_return_node] transport_query agent 未注册")
            return {}

        # 返程是 destination → origin
        logger.info(f"[transport_return_node] 查询返程 {destination} → {origin} ({return_date})")
        plan, error = await query_one_date(
            agent=agent,
            origin=destination,
            destination=origin,
            date=return_date,
            intent_data=state.get("intent_data") or {},
        )

        if error:
            return {
                "skill_results": [make_skill_result(
                    "transport_return", "error", {"error": error}, message=error,
                )],
            }

        updates: Dict[str, Any] = {
            "transport_return": plan,
            "skill_results": [make_skill_result(
                "transport_return", "success", {"transport_plan": plan},
            )],
        }

        validated = validate_options(plan.get("options") or [])
        if validated:
            updates["transport_return_options"] = validated

        return updates

    return transport_return_node
