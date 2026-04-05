"""
POI 搜索智能体 POIFetchAgent
职责：根据目的地城市，搜索景区候选 POI，输出标准化列表，供后续 TSP 路线规划使用。

核心逻辑：
1. 从上下文中读取目的地城市、旅行风格
2. 只搜索"景点"类别（餐厅由 itinerary_planning_node 在路线确定后按天搜索周边）
3. 特种兵模式下多返回一些 POI
4. 过滤掉缺少经纬度坐标的 POI（TSP 无法处理）
5. 输出标准化列表，与 state.poi_candidates 字段对应
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from mcp_clients.amap_client import amap_mcp_session, search_pois

logger = logging.getLogger(__name__)

# 每类默认返回数量
_DEFAULT_TOP_N = 20
_SPECIAL_FORCES_TOP_N = 30  # 特种兵模式多搜一些

# 景点搜索关键词模板（当无 poi_search_hints 时的兜底）
# 注意：餐厅不在此处搜索，改为在行程规划后按每天景点重心搜索周边餐厅
_CATEGORY_KEYWORDS = [
    ("景点", "{city}景点"),
]

# 用于从 hint 关键词中过滤掉餐厅/体验类，保证 poi_candidates 全为景点
_RESTAURANT_KW = frozenset(["餐", "美食", "小吃", "食", "饭", "菜", "吃", "火锅", "茶"])
_EXPERIENCE_KW = frozenset(["体验", "活动", "游乐", "娱乐", "演出", "表演", "民宿"])


def _infer_category(keyword: str) -> str:
    """
    根据关键词文本推断 POI 类别，供 hints 模式下过滤非景点词条。
    餐厅和体验类关键词在此统一归为景点（后续会通过周边搜索专门获取），
    确保 poi_candidates 中只含景区类 POI。
    """
    return "景点"


def _is_restaurant_or_experience_hint(keyword: str) -> bool:
    """判断 hint 关键词是否属于餐厅/体验类，用于从 poi_search_hints 中跳过这些词条。"""
    return any(w in keyword for w in _RESTAURANT_KW) or any(w in keyword for w in _EXPERIENCE_KW)


def _parse_location(location: str) -> Optional[tuple[float, float]]:
    """
    将高德 "lng,lat" 字符串解析为 (lng, lat) 浮点元组。
    格式不合法或坐标为零时返回 None（过滤掉无效 POI）。
    """
    if not location:
        return None
    parts = location.split(",")
    if len(parts) != 2:
        return None
    try:
        lng, lat = float(parts[0].strip()), float(parts[1].strip())
    except ValueError:
        return None
    # 经纬度为零视为无效
    if lng == 0.0 and lat == 0.0:
        return None
    return lng, lat


def _normalize_rating(raw_rating: Any) -> float:
    """将各种评分格式统一转为 float，无法解析时返回 0.0。"""
    if isinstance(raw_rating, (int, float)):
        return float(raw_rating)
    try:
        return float(str(raw_rating).strip())
    except (ValueError, TypeError):
        return 0.0


def _normalize_pois(raw_pois: List[Dict], category: str, top_n: int) -> List[Dict]:
    """
    将 search_pois() 的原始结果转换为标准 POI 格式，同时：
    - 过滤掉没有有效坐标的条目
    - 截取前 top_n 条
    """
    result: List[Dict] = []
    for item in raw_pois:
        coords = _parse_location(item.get("location", ""))
        if coords is None:
            continue  # 无坐标，丢弃
        lng, lat = coords
        result.append({
            "name": item.get("name", ""),
            "lng": lng,
            "lat": lat,
            "category": category,
            "rating": _normalize_rating(item.get("rating", 0.0)),
            "address": item.get("address", ""),
            # 记录在本类别搜索结果中的排名（1-based），高德按相关性/热度排序，
            # 越靠前的 POI 通常越知名，供 rating=0 时作为评分代理指标
            "search_rank": len(result) + 1,
        })
        if len(result) >= top_n:
            break
    return result


class POIFetchAgent:
    def __init__(self, name: str = "POIFetchAgent", model=None, **kwargs):
        self.name = name
        self.model = model  # 保留接口一致性，当前实现不需要 LLM

    async def run(self, input_data: dict) -> dict:
        context = input_data.get("context", {})
        key_entities = context.get("key_entities", {})

        city: str = (
            key_entities.get("destination", "")
            or context.get("destination", "")
        )
        if not city:
            return {"agent": "poi_fetch", "error": "缺少目的地城市，无法搜索 POI"}

        travel_style: str = context.get("travel_style", "普通")
        top_n = _SPECIAL_FORCES_TOP_N if travel_style == "特种兵" else _DEFAULT_TOP_N

        # ── 构建搜索目标列表 ──────────────────────────────────────────────────
        # 优先使用 intent_node LLM 生成的语义提示词；缺失时退回静态模板
        # 只保留景点类 hint，跳过餐厅/体验类（它们由路线规划后的周边搜索覆盖）
        poi_hints: list[str] = context.get("poi_search_hints", [])
        if poi_hints:
            attraction_hints = [h for h in poi_hints if not _is_restaurant_or_experience_hint(h)]
            # 若 hints 全部被过滤（全是餐厅/体验词），降级为静态模板
            if attraction_hints:
                search_targets = [("景点", hint) for hint in attraction_hints]
                logger.info(
                    f"POIFetchAgent: city={city}, style={travel_style}, top_n={top_n}, "
                    f"hints(景点)={attraction_hints} (跳过非景点: {[h for h in poi_hints if h not in attraction_hints]})"
                )
            else:
                search_targets = [
                    (cat, tmpl.format(city=city))
                    for cat, tmpl in _CATEGORY_KEYWORDS
                ]
                logger.info(
                    f"POIFetchAgent: city={city}, style={travel_style}, top_n={top_n}, "
                    f"hints全为餐厅/体验，降级为静态模板"
                )
        else:
            search_targets = [
                (cat, tmpl.format(city=city))
                for cat, tmpl in _CATEGORY_KEYWORDS
            ]
            logger.info(
                f"POIFetchAgent: city={city}, style={travel_style}, top_n={top_n}, "
                f"hints=[] → fallback to static templates"
            )
        # ─────────────────────────────────────────────────────────────────────

        all_pois: List[Dict] = []

        # 每个类别独立建连：规避高德 MCP SSE 连接在类别搜索间隙空闲超时
        # 导致 post_writer ConnectTimeout 的问题（Windows 代理环境下更易触发）
        for category, keyword in search_targets:
            try:
                async with amap_mcp_session() as session:
                    raw = await search_pois(session, city=city, keywords=keyword)
                normalized = _normalize_pois(raw, category=category, top_n=top_n)
                all_pois.extend(normalized)
                logger.info(
                    f"POIFetchAgent: [{category}] 搜索 '{keyword}' → "
                    f"原始 {len(raw)} 条，有效 {len(normalized)} 条"
                )
            except Exception as e:
                logger.warning(f"POIFetchAgent: [{category}] 搜索失败: {e}")

        logger.info(f"POIFetchAgent: 共获得有效 POI {len(all_pois)} 条")
        return {
            "agent": "poi_fetch",
            "result": {"pois": all_pois},
        }
