"""
去程交通节点 transport_outbound_node (P2 fan-out)

职责：查询用户在 start_date 当天从 origin → destination 的真实交通方案，
      写入 state.transport_outbound（完整 plan）+ state.transport_options（校验后选项）
      + skill_results（agent_name="transport_outbound"）。

与 transport_return_node 并行，二者共享 _transport_helper.query_one_date。

触发条件（满足所有）：
  1. intent_type == "planning"
  2. origin / destination / start_date 均非空（跨城出行）
  3. registry 中已注册 transport_query

注：保留 state.transport_options 写入是为了让下游 itinerary_planning_node 与
    accommodation_node 继续按"去程"语义读取最低交通费 / 到达枢纽，无需改动它们。
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from graph.state import TravelGraphState, ensure_hard_constraints
from graph.nodes._transport_helper import (
    make_skill_result,
    query_one_date,
    validate_options,
)

logger = logging.getLogger(__name__)


def create_transport_outbound_node(registry):
    async def transport_outbound_node(state: TravelGraphState) -> dict:
        if state.get("intent_type") != "planning":
            return {}

        hard_constraints = ensure_hard_constraints(state.get("hard_constraints"))
        origin: str = hard_constraints.origin or ""
        destination: str = hard_constraints.destination or ""
        start_date: str = hard_constraints.start_date or ""

        if not origin or not destination or origin == destination:
            logger.info(
                f"[transport_outbound_node] 跳过 "
                f"(origin={origin!r}, destination={destination!r})"
            )
            return {}

        try:
            agent = registry["transport_query"]
        except KeyError:
            logger.warning("[transport_outbound_node] transport_query agent 未注册")
            return {}

        logger.info(f"[transport_outbound_node] 查询去程 {origin} → {destination} ({start_date})")
        plan, error = await query_one_date(
            agent=agent,
            origin=origin,
            destination=destination,
            date=start_date,
            intent_data=state.get("intent_data") or {},
        )

        if error:
            return {
                "skill_results": [make_skill_result(
                    "transport_outbound", "error", {"error": error}, message=error,
                )],
            }

        updates: Dict[str, Any] = {
            "transport_outbound": plan,
            "skill_results": [make_skill_result(
                "transport_outbound", "success", {"transport_plan": plan},
            )],
        }

        # 写入 transport_options（供 itinerary_planning / accommodation 复用，
        # 它们关心的是"去程"语义下的最低价 + 到达枢纽）
        validated = validate_options(plan.get("options") or [])
        if validated:
            updates["transport_options"] = validated

        return updates

    return transport_outbound_node
