"""
POI 获取节点 poi_fetch_node (P2 fan-out)
职责：调用 POIFetchAgent 在目的地搜索景点候选，写入 poi_candidates，
      供 P3 itinerary_planning_node 消费（producer-consumer 关系）。

触发条件（满足所有）：
  1. intent_type == "planning"
  2. key_entities.destination 非空
  3. registry 中已注册 poi_fetch
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from graph.state import TravelGraphState

logger = logging.getLogger(__name__)


def create_poi_fetch_node(registry):
    """
    工厂函数：将 agent registry 通过闭包注入。

    Args:
        registry: dict[str, agent]，需提供 poi_fetch

    Returns:
        async 节点函数 poi_fetch_node(state) -> dict
    """

    async def poi_fetch_node(state: TravelGraphState) -> dict:
        if state.get("intent_type") != "planning":
            return {}

        intent_data: Dict[str, Any] = state.get("intent_data") or {}
        destination: str = (intent_data.get("key_entities") or {}).get("destination", "") or ""
        if not destination:
            logger.info("[poi_fetch_node] destination 为空，跳过 POI 抓取")
            return {}

        try:
            agent = registry["poi_fetch"]
        except KeyError:
            logger.warning("[poi_fetch_node] poi_fetch agent 未注册")
            return {}

        context: Dict[str, Any] = {
            "key_entities": intent_data.get("key_entities", {}),
            "travel_style": intent_data.get("travel_style", "普通"),
            "attraction_hints": state.get("attraction_hints") or intent_data.get("attraction_hints", []),
            # llm_seed_extract_node 在 P2 中段抽取的 RAG+KB 共识地标，
            # poi_agent 路径②b 用于精准搜索（trust_kb=True）
            "llm_seed_pois": state.get("llm_seed_pois") or [],
        }
        input_data = {"context": context}

        logger.info(f"[poi_fetch_node] 抓取 destination={destination!r}")
        try:
            result = await agent.run(input_data)
        except Exception as e:
            logger.error(f"[poi_fetch_node] POIFetchAgent 执行失败: {e}")
            return {"skill_results": [_skill_result(
                "poi_fetch", "error",
                {"error": str(e)}, message=str(e),
            )]}

        if isinstance(result, dict) and "error" in result:
            return {"skill_results": [_skill_result(
                "poi_fetch", "error", result, message=result.get("error", ""),
            )]}

        flat = _skill_result("poi_fetch", "success", result)
        updates: Dict[str, Any] = {"skill_results": [flat]}

        pois = (result.get("result") or {}).get("pois", [])
        if pois:
            updates["poi_candidates"] = pois
            logger.info(f"[poi_fetch_node] 写入 {len(pois)} 条 POI 候选")
        return updates

    return poi_fetch_node


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
