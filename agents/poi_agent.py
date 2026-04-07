"""
POI 搜索智能体 POIFetchAgent
职责：根据目的地城市，搜索景区候选 POI，输出标准化列表，供后续 TSP 路线规划使用。

搜索策略（分 KB 城市 / 非 KB 城市两条路径）：

【KB 城市路径（城市在 CityKnowledgeDB 中）】
  高德只做"地理数据提供者"——由知识库决定推荐哪些景点，高德只负责返回坐标/评分。
  ① must_visit 精准查询（top_n=2）：保证所有必去景点进入候选池，供 Phase-1 锚定
  ② route_combo 额外子景点精准查询（top_n=2）：断桥/法喜寺/龙井村等细粒度坐标，
     支持 TSP 精确路由和 Phase-2 combo_boost 评分
  ③ LLM poi_search_hints 补充（top_n=3）：捕捉用户特定兴趣（如"特别想去大熊猫基地"）
  ✗ 不做 "{city}景点" 泛搜：避免引入大量低质量噪声 POI

【非 KB 城市路径（城市不在 CityKnowledgeDB 中）】
  ③ LLM poi_search_hints 精准查询（top_n=5）
  ④ 兜底 "{city}景点" 泛搜（top_n=10），仅当 hints 全部为餐厅/体验类或为空时触发

全局去重：按 POI 名称去重，优先保留先搜到的条目（KB 路径 > LLM hints > 泛搜）。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from mcp_clients.amap_client import amap_mcp_session, search_pois
from utils.knowledge_parser import CityKnowledgeDB

logger = logging.getLogger(__name__)

# 非KB城市泛搜的默认 top_n（结果质量有限，不宜过多）
_FALLBACK_TOP_N = 10
# 特种兵模式下各路径 top_n 乘数（搜更多候选以满足高密度行程需求）
_SPECIAL_FORCES_MULTIPLIER = 2

# 景点泛搜兜底模板（非KB城市且无 LLM hints 时使用）
_FALLBACK_KEYWORD_TMPL = "{city}景点"

# 用于从 hint 关键词中过滤掉餐厅/体验类，保证 poi_candidates 全为景点
_RESTAURANT_KW = frozenset(["餐", "美食", "小吃", "食", "饭", "菜", "吃", "火锅", "茶"])
_EXPERIENCE_KW = frozenset(["体验", "活动", "游乐", "娱乐", "演出", "表演", "民宿"])


# =============================================================================
# 模块级辅助函数
# =============================================================================

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

    字段说明：
      amap_type : 高德 typecode（6 位字符串，如 "110104"）。
                  amap_client.search_pois 已将原始 typecode 写入 item["type"]，
                  此处直接透传，供 itinerary_review_node Check 4 大类判断使用。
                  place/text 接口返回的均为真实 POI，typecode 字段必然存在。
    """
    result: List[Dict] = []
    for item in raw_pois:
        coords = _parse_location(item.get("location", ""))
        if coords is None:
            continue
        lng, lat = coords
        result.append({
            "name": item.get("name", ""),
            "lng": lng,
            "lat": lat,
            "category": category,
            "rating": _normalize_rating(item.get("rating", 0.0)),
            "address": item.get("address", ""),
            # 记录在本次搜索结果中的排名（1-based），高德按相关性/热度排序，
            # 越靠前的 POI 通常越知名，供 rating=0 时作为评分代理指标
            "search_rank": len(result) + 1,
            # 高德 typecode：amap_client.search_pois 已将 typecode 写入 item["type"]
            "amap_type": item.get("type", ""),
        })
        if len(result) >= top_n:
            break
    return result


async def _search_single(city: str, keyword: str, top_n: int) -> List[Dict]:
    """
    对单个关键词发起一次高德 MCP 搜索，返回标准化 POI 列表。

    每次独立建连：规避高德 MCP SSE 连接在搜索间隙空闲超时
    导致 post_writer ConnectTimeout 的问题（Windows 代理环境下更易触发）。
    """
    try:
        async with amap_mcp_session() as session:
            raw = await search_pois(session, city=city, keywords=keyword)
        normalized = _normalize_pois(raw, category="景点", top_n=top_n)
        logger.info(
            f"POIFetchAgent: 搜索 '{keyword}' (top_n={top_n}) → "
            f"原始 {len(raw)} 条，有效 {len(normalized)} 条"
        )
        return normalized
    except Exception as e:
        logger.warning(f"POIFetchAgent: 搜索 '{keyword}' 失败: {e}")
        return []


def _extend_deduped(
    all_pois: List[Dict],
    new_pois: List[Dict],
    seen_names: set,
) -> None:
    """
    将 new_pois 中名称未出现过的条目追加到 all_pois，同时更新 seen_names。
    保证全局名称去重：先搜到的条目优先保留（KB路径 > LLM hints > 泛搜）。
    """
    for poi in new_pois:
        name = poi.get("name", "").strip()
        if name and name not in seen_names:
            seen_names.add(name)
            all_pois.append(poi)


# =============================================================================
# POIFetchAgent
# =============================================================================

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
        # 特种兵模式下各路径 top_n 翻倍，以满足高密度行程的候选需求
        style_multiplier = _SPECIAL_FORCES_MULTIPLIER if travel_style == "特种兵" else 1

        # 过滤 LLM 生成的搜索提示词，去掉餐厅/体验类（由周边搜索覆盖）
        poi_hints: List[str] = context.get("poi_search_hints", [])
        attraction_hints: List[str] = [
            h for h in poi_hints if not _is_restaurant_or_experience_hint(h)
        ]

        all_pois: List[Dict] = []
        seen_names: set = set()  # 全局去重集合

        knowledge_db = CityKnowledgeDB.get_instance()

        if city and knowledge_db.has_city(city):
            # ═══════════════════════════════════════════════════════════════
            # KB 城市路径：知识库决定"去哪些景点"，高德只提供坐标/评分
            # ═══════════════════════════════════════════════════════════════

            # ── 路径①：must_visit 精准查询 ───────────────────────────────
            # top_n=2：名称明确，高德 top1 基本就是目标，取 2 作为安全冗余
            kb_must_visit = knowledge_db.get_must_visit_names(city)
            top_n_kb = 2 * style_multiplier
            for name in kb_must_visit:
                pois = await _search_single(city, f"{city} {name}", top_n=top_n_kb)
                _extend_deduped(all_pois, pois, seen_names)

            logger.info(
                f"POIFetchAgent [KB路径①-must_visit]: city={city}, "
                f"搜索 {len(kb_must_visit)} 个必去景点, 累计 {len(all_pois)} 个去重POI"
            )

            # ── 路径②：route_combo 额外子景点精准查询 ────────────────────
            # 只搜索不被 must_visit 覆盖的新增子景点（如断桥、法喜寺、龙井村等）
            # 用途：为 TSP 路由提供细粒度坐标；Phase-2 combo_boost 评分来源
            kb_extra = knowledge_db.get_extra_combo_spots(city)
            for name in kb_extra:
                pois = await _search_single(city, f"{city} {name}", top_n=top_n_kb)
                _extend_deduped(all_pois, pois, seen_names)

            logger.info(
                f"POIFetchAgent [KB路径②-combo额外子景点]: "
                f"搜索 {len(kb_extra)} 个子景点 {kb_extra}, 累计 {len(all_pois)} 个去重POI"
            )

            # ── 路径③：LLM hints 补充用户特定兴趣 ───────────────────────
            # top_n=3：hint 有一定模糊性，多取几条；名称与KB重叠的会被去重过滤
            top_n_hint = 3 * style_multiplier
            for hint in attraction_hints:
                pois = await _search_single(city, hint, top_n=top_n_hint)
                _extend_deduped(all_pois, pois, seen_names)

            if attraction_hints:
                logger.info(
                    f"POIFetchAgent [KB路径③-LLM hints]: "
                    f"hints={attraction_hints}, 累计 {len(all_pois)} 个去重POI"
                )

        else:
            # ═══════════════════════════════════════════════════════════════
            # 非 KB 城市路径：LLM hints 为主，泛搜兜底
            # 结果质量有限，接受为降级行为
            # ═══════════════════════════════════════════════════════════════
            top_n_hint = 5 * style_multiplier

            if attraction_hints:
                for hint in attraction_hints:
                    pois = await _search_single(city, hint, top_n=top_n_hint)
                    _extend_deduped(all_pois, pois, seen_names)
                logger.info(
                    f"POIFetchAgent [非KB路径-LLM hints]: city={city}, "
                    f"hints={attraction_hints}, 累计 {len(all_pois)} 个去重POI"
                )
            else:
                # 兜底泛搜：hints 为空或全部被过滤时触发
                keyword = _FALLBACK_KEYWORD_TMPL.format(city=city)
                pois = await _search_single(
                    city, keyword, top_n=_FALLBACK_TOP_N * style_multiplier
                )
                _extend_deduped(all_pois, pois, seen_names)
                logger.info(
                    f"POIFetchAgent [非KB路径-泛搜兜底]: city={city}, "
                    f"keyword='{keyword}', 累计 {len(all_pois)} 个去重POI"
                )

        logger.info(f"POIFetchAgent: 最终 poi_candidates 共 {len(all_pois)} 条（已全局去重）")
        return {
            "agent": "poi_fetch",
            "result": {"pois": all_pois},
        }
