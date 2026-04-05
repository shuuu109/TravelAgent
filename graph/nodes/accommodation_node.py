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
from utils.knowledge_parser import CityKnowledgeDB

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

        # ── Step 1：按天计算各天地理重心 ─────────────────────────────
        # 不再取全程 POI 的均值坐标（会落在无意义的折中位置），
        # 而是为每天单独计算重心，供 AccommodationAgent 分天搜索酒店。
        daily_routes = state.get("daily_routes", [])
        daily_centers: list[dict] = []   # [{day, lng, lat, poi_count}, ...]
        location_hint: str = ""           # 单坐标兜底值（降级链用）

        if daily_routes:
            for day_route in daily_routes:
                pois = day_route.get("ordered_pois", [])
                valid = [
                    p for p in pois
                    if p.get("lng") is not None and p.get("lat") is not None
                ]
                if valid:
                    daily_centers.append({
                        "day": day_route["day"],
                        "lng": round(mean(float(p["lng"]) for p in valid), 6),
                        "lat": round(mean(float(p["lat"]) for p in valid), 6),
                        "poi_count": len(valid),
                    })

            if daily_centers:
                logger.info(
                    f"AccommodationNode: {len(daily_centers)} daily centers computed "
                    f"(total {sum(d['poi_count'] for d in daily_centers)} POIs)"
                )
                # 取 Day 1 重心作为单坐标降级值，供后续降级链使用
                location_hint = f"{daily_centers[0]['lng']},{daily_centers[0]['lat']}"

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
        # 从知识库获取该城市的结构化住宿建议，供 Agent 直接参考
        # 例如：["上城区湖滨/龙翔桥：紧邻西湖...", "西湖区青芝坞：民宿极具设计感..."]
        knowledge_db = CityKnowledgeDB.get_instance()
        kb_city = location_hint if location_hint and not ',' in location_hint else ""
        if not kb_city:
            # location_hint 是坐标时，从 intent_data 取城市名
            intent_data_local: dict = state.get("intent_data") or {}
            kb_city = (
                (intent_data_local.get("key_entities") or {}).get("destination", "")
                or intent_data_local.get("destination", "")
            )
        knowledge_accommodation = knowledge_db.get_accommodation(kb_city) if kb_city else []

        input_data = {
            "context": context,
            "previous_results": previous_results,
            "location_hint": location_hint,           # 兜底：单坐标或枢纽名
            "daily_centers": daily_centers,           # 主路径：按天重心列表
            "knowledge_accommodation": knowledge_accommodation,  # 知识库住宿建议
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
