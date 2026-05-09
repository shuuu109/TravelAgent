"""
餐厅推荐节点 restaurant_node
============================
位置：itinerary_review 通过后，accommodation_node 之前。

职责：
  对已通过 P4.5 自检（或用尽重试）的 daily_routes，逐天计算景点地理重心，
  调用高德 search_restaurants_nearby 获取每天周边餐厅推荐 5 家，
  写入 state["daily_restaurants"]。

为什么独立成节点（从 itinerary_planning_node 迁出）：
  P3 (itinerary_planning) → P3.5 (poi_enrich) → P4.5 (itinerary_review) 形成
  retry 回环。餐厅搜索若放在 P3 内，每次回环重规划都会重复消耗高德 API 配额。
  迁到 review 通过出口的下一站，仅在行程稳定后查一次餐厅。

失败容忍：
  MCP session 整体失败 / 单天搜索失败时，对应天 restaurants 列表为空，
  不阻断主流程；respond_node 已对空餐厅做兜底渲染。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from graph.state import TravelGraphState, ensure_hard_constraints
from mcp_clients.amap_client import amap_mcp_session, search_restaurants_nearby

logger = logging.getLogger(__name__)


def create_restaurant_node():
    """
    工厂函数。当前实现无外部依赖（不需要 LLM），保留工厂壳便于未来注入。
    """

    async def restaurant_node(state: TravelGraphState) -> dict:
        daily_routes: List[Dict] = state.get("daily_routes") or []
        if not daily_routes:
            logger.warning("[restaurant] daily_routes 为空，跳过餐厅推荐")
            return {"daily_restaurants": []}

        city = _resolve_city(state)
        if not city:
            logger.warning("[restaurant] 未解析到目的地城市，餐厅搜索可能不准")

        daily_restaurants: List[Dict] = []
        try:
            async with amap_mcp_session() as session:
                for day_route in daily_routes:
                    day_pois = day_route.get("ordered_pois") or []
                    restaurants = await _fetch_day_restaurants(
                        day_pois=day_pois,
                        session=session,
                        city=city,
                    )
                    daily_restaurants.append({
                        "day": day_route.get("day"),
                        "restaurants": restaurants,
                    })
            logger.info(
                f"[restaurant] {len(daily_restaurants)} 天餐厅推荐完成"
            )
        except Exception as e:
            logger.error(f"[restaurant] MCP session 失败: {e}，所有天返回空餐厅列表")
            daily_restaurants = [
                {"day": day_route.get("day"), "restaurants": []}
                for day_route in daily_routes
            ]

        return {"daily_restaurants": daily_restaurants}

    return restaurant_node


# =============================================================================
# 内部辅助
# =============================================================================

def _resolve_city(state: TravelGraphState) -> str:
    """
    与 itinerary_planning_node 一致的城市解析回退链：
    hard_constraints → intent_data → skill_results(event_collection)。
    """
    hard_constraints = ensure_hard_constraints(state.get("hard_constraints"))
    city = hard_constraints.destination or ""
    if city:
        return city

    intent_data: dict = state.get("intent_data") or {}
    city = (
        intent_data.get("key_entities", {}).get("destination", "")
        or intent_data.get("destination", "")
        or ""
    )
    if city:
        return city

    for sr in state.get("skill_results", []):
        if sr.get("agent_name") == "event_collection":
            city = sr.get("data", {}).get("destination", "") or ""
            if city:
                return city
    return ""


async def _fetch_day_restaurants(
    day_pois: List[Dict],
    session: Any,
    city: str,
    radius: int = 3000,
    count: int = 5,
) -> List[Dict]:
    """
    计算当天所有景点的地理重心，以此为中心调用高德周边搜索。

    选择重心而非第一个景点：覆盖更均匀，避免"只推荐第一个景点附近"的偏差。
    """
    if not day_pois:
        return []

    valid_pois = [p for p in day_pois if p.get("lng") and p.get("lat")]
    if not valid_pois:
        return []

    centroid_lng = sum(p["lng"] for p in valid_pois) / len(valid_pois)
    centroid_lat = sum(p["lat"] for p in valid_pois) / len(valid_pois)
    centroid_location = f"{centroid_lng:.6f},{centroid_lat:.6f}"

    try:
        restaurants = await search_restaurants_nearby(
            session=session,
            location=centroid_location,
            radius=radius,
            city=city,
            count=count,
        )
        logger.info(
            f"[restaurant] 重心={centroid_location}, 搜索到 {len(restaurants)} 家餐厅"
        )
        return restaurants
    except Exception as e:
        logger.warning(f"[restaurant] 单天餐厅搜索失败: {e}")
        return []
