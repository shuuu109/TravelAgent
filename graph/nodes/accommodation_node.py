"""
住宿节点 accommodation_node
职责：在 itinerary_planning 之后（或直接在 orchestrate 之后）作为独立节点执行，
      利用 daily_routes 的地理重心（经纬度均值）计算最优住宿中心坐标，
      再调用 AccommodationAgent 搜索并推荐酒店。

地理感知降级顺序：
  1. daily_routes 有效 POI 坐标均值 → 精确坐标传给 MCP
  2. transport_options[0].arrival_hub → 枢纽名称传给 Agent 提示 LLM
  3. hard_constraints.destination → 目的地城市（最后兜底）
"""
import logging
from statistics import mean

from graph.state import TravelGraphState
from utils.skill_loader import SkillLoader

logger = logging.getLogger(__name__)

_skill_loader = SkillLoader()


def create_accommodation_node(model, memory_manager=None):
    """
    工厂函数：将 model 和 memory_manager 通过闭包注入。

    Args:
        model: LangChain ChatModel 实例，传给 AccommodationAgent
        memory_manager: MemoryManager 实例（可选），用于读取用户偏好

    Returns:
        async 节点函数 accommodation_node(state) -> dict
    """

    async def accommodation_node(state: TravelGraphState) -> dict:
        # 只在 intent_schedule 包含 accommodation_query 时执行
        intent_schedule = state.get("intent_schedule", [])
        if not any(t.get("agent_name") == "accommodation_query" for t in intent_schedule):
            return {}

        # ── Step 1：计算地理重心 ──────────────────────────────────
        daily_routes = state.get("daily_routes", [])
        location_hint: str = ""

        if daily_routes:
            all_pois = [
                poi
                for day in daily_routes
                for poi in day.get("ordered_pois", [])
            ]
            valid_pois = [
                p for p in all_pois
                if p.get("lng") is not None and p.get("lat") is not None
            ]
            if valid_pois:
                center_lng = mean(float(p["lng"]) for p in valid_pois)
                center_lat = mean(float(p["lat"]) for p in valid_pois)
                location_hint = f"{center_lng},{center_lat}"
                logger.info(
                    f"AccommodationNode: geo center {location_hint} "
                    f"from {len(valid_pois)} POIs across {len(daily_routes)} days"
                )

        # ── Step 2：降级到 arrival_hub ────────────────────────────
        if not location_hint:
            transport_options = state.get("transport_options", [])
            if transport_options and isinstance(transport_options, list):
                first = transport_options[0]
                arrival_hub = (
                    first.get("arrival_hub", "")
                    or first.get("arrival_station", "")
                )
                if arrival_hub:
                    location_hint = arrival_hub
                    logger.info(f"AccommodationNode: using arrival_hub '{arrival_hub}' as location_hint")

        # ── Step 3：降级到 hard_constraints.destination ──────────
        if not location_hint:
            hard_constraints = state.get("hard_constraints") or {}
            if hasattr(hard_constraints, "destination"):
                location_hint = hard_constraints.destination or ""
            elif isinstance(hard_constraints, dict):
                location_hint = hard_constraints.get("destination", "")
            if location_hint:
                logger.info(f"AccommodationNode: falling back to hard_constraints destination '{location_hint}'")

        # ── Step 4：降级到 intent_data.key_entities.destination ──
        if not location_hint:
            intent_data = state.get("intent_data", {})
            kv_dest = (intent_data.get("key_entities") or {}).get("destination", "")
            if kv_dest:
                location_hint = kv_dest
                logger.info(f"AccommodationNode: falling back to intent_data destination '{location_hint}'")

        # ── 构建 context（与 orchestrate_node._prepare_context 一致）─
        intent_data = state.get("intent_data", {})
        context: dict = {
            "reasoning": intent_data.get("reasoning", ""),
            "intents": intent_data.get("intents", []),
            "key_entities": intent_data.get("key_entities", {}),
            "rewritten_query": intent_data.get("rewritten_query", ""),
            "travel_style": intent_data.get("travel_style", "普通"),
        }
        if memory_manager:
            context["user_preferences"] = memory_manager.long_term.get_preference()

        skill_guide = _skill_loader.get_skill_content("accommodation-query")
        if skill_guide:
            context["skill_guide"] = skill_guide

        # 从已有 skill_results 中取 transport_query 结果，供 Agent 提取到达枢纽
        skill_results = state.get("skill_results", [])
        previous_results = [
            {
                "agent_name": r["agent_name"],
                "result": {"data": r.get("data", {})},
            }
            for r in skill_results
            if r.get("agent_name") == "transport_query"
        ]

        # ── 调用 AccommodationAgent ───────────────────────────────
        from agents.accommodation_agent import AccommodationAgent

        agent = AccommodationAgent(name="AccommodationAgent", model=model)
        input_data = {
            "context": context,
            "previous_results": previous_results,
            "location_hint": location_hint,
        }

        try:
            result = await agent.run(input_data)
            if "error" in result:
                flat = {
                    "agent_name": "accommodation_query",
                    "status": "error",
                    "data": result,
                    "message": result["error"],
                }
            else:
                flat = {
                    "agent_name": "accommodation_query",
                    "status": "success",
                    "data": result,
                }
        except Exception as e:
            logger.error(f"AccommodationNode failed: {e}")
            flat = {
                "agent_name": "accommodation_query",
                "status": "error",
                "data": {"error": str(e)},
                "message": str(e),
            }

        return {"skill_results": [flat]}

    return accommodation_node
