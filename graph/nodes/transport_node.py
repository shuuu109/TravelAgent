"""
交通节点 transport_node (P2 fan-out)
职责：当用户存在跨城出行需求时，调用 TransportAgent 查询真实车次/航班，
      将结果写入 transport_options（经 TravelOption 校验后），并追加到 skill_results。

触发条件（满足所有）：
  1. intent_type == "planning"
  2. origin 与 destination 均非空（跨城出行）
  3. registry 中已注册 transport_query

不触发场景：
  - 同城游 / preference_only / memory_only / info_only
  - origin 缺失（由 extract_constraints_node 添加 missing_info='出发地'，
    走 negotiate 分支，不应到达本节点；此处仍兜底跳过）

注：origin 已在 extract_constraints_node 由 home_location 偏好回填，
    此处仅校验当前 state 中 hard_constraints.origin。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from graph.state import (
    TravelGraphState,
    TravelOption,
    ensure_hard_constraints,
)

logger = logging.getLogger(__name__)


def create_transport_node(registry):
    """
    工厂函数：将 agent registry 通过闭包注入。

    Args:
        registry: dict[str, agent]，需提供 transport_query

    Returns:
        async 节点函数 transport_node(state) -> dict
    """

    async def transport_node(state: TravelGraphState) -> dict:
        if state.get("intent_type") != "planning":
            return {}

        hard_constraints = ensure_hard_constraints(state.get("hard_constraints"))
        origin: str = hard_constraints.origin or ""
        destination: str = hard_constraints.destination or ""

        if not origin or not destination:
            logger.info(
                f"[transport_node] origin/destination 缺失 "
                f"(origin={origin!r}, destination={destination!r})，跳过"
            )
            return {}
        if origin == destination:
            logger.info("[transport_node] 同城出行，跳过交通查询")
            return {}

        try:
            agent = registry["transport_query"]
        except KeyError:
            logger.warning("[transport_node] transport_query agent 未注册")
            return {}

        # 构造 input_data：context.key_entities 含 origin/destination/date，与旧 orchestrate 字段一致
        intent_data: Dict[str, Any] = state.get("intent_data") or {}
        key_entities: Dict[str, Any] = dict(intent_data.get("key_entities") or {})
        # 用归一后的 hard_constraints 覆盖 key_entities 字段，避免相对日期未解析等差异
        key_entities["origin"] = origin
        key_entities["destination"] = destination
        if hard_constraints.start_date:
            key_entities["date"] = hard_constraints.start_date

        context: Dict[str, Any] = {
            "key_entities": key_entities,
            "rewritten_query": intent_data.get("rewritten_query", ""),
            "travel_style": intent_data.get("travel_style", "普通"),
        }
        input_data = {"context": context}

        logger.info(f"[transport_node] 查询 {origin} → {destination}")
        try:
            result = await agent.run(input_data)
        except Exception as e:
            logger.error(f"[transport_node] TransportAgent 执行失败: {e}")
            return {"skill_results": [_skill_result(
                "transport_query", "error",
                {"error": str(e)}, message=str(e),
            )]}

        if isinstance(result, dict) and "error" in result:
            return {"skill_results": [_skill_result(
                "transport_query", "error", result, message=result.get("error", ""),
            )]}

        flat = _skill_result("transport_query", "success", result)
        updates: Dict[str, Any] = {"skill_results": [flat]}

        # 校验并写入 transport_options
        raw_options = (result.get("transport_plan") or {}).get("options", [])
        validated: List[Dict] = []
        for i, opt in enumerate(raw_options):
            try:
                validated.append(TravelOption(**opt).model_dump())
            except Exception as e:
                logger.warning(f"[transport_node] TravelOption[{i}] 校验失败，跳过: {e}")
        if validated:
            updates["transport_options"] = validated

        return updates

    return transport_node


def _skill_result(
    agent_name: str,
    status: str,
    data: Dict[str, Any],
    message: str = "",
) -> Dict[str, Any]:
    out = {"agent_name": agent_name, "status": status, "data": data}
    if message:
        out["message"] = message
    return out
