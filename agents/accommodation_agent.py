"""
住宿专家智能体 AccommodationAgent
职责：根据目的地、每天行程的地理重心和用户偏好，通过两阶段搜索获取真实酒店数据，
      再由 LLM 对结果进行分析和个性化推荐。

两阶段搜索流程（每天重心独立执行）：
  Phase 1 — 高德 maps_around_search（地理发现）
            基于坐标 + 半径拉取周边酒店 POI，获得精确 distance_m（距当天重心的米数）。
  Phase 2 — RollingGo searchHotels（价格增强）
            复用单一 stdio session，对 Phase 1 每家酒店按名称+坐标查询真实价格/可用性。
  merge   — _merge_hotel_data 合并两份数据，LLM 拿到兼具"地理精度"和"真实价格"的融合视图。

降级链（任一阶段失败时透明切换）：
  Amap Phase 1 失败 → RollingGo-only 单次搜索（原有逻辑，用 location 坐标传入）
  RollingGo Phase 2 失败 → 纯 Amap 数据（有距离信息，标注"价格待查"）
  两者均失败 → 纯 LLM 知识推荐
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# 两阶段搜索配置常量
_AMAP_HOTEL_RADIUS_M = 2000    # Phase 1 搜索半径（米）
_AMAP_HOTEL_MAX_COUNT = 10     # Phase 1 每天最多拉取的酒店数量


def _normalize_date(date_str: str) -> str | None:
    """
    将各种日期格式统一转换为 YYYY-MM-DD，供 MCP API 使用。
    支持：'2026-04-06'、'2026年4月6日'、'2026/4/6'、含括号说明文字等。
    无法解析时返回 None。
    """
    import re

    if not date_str:
        return None

    # 已是标准格式
    if re.match(r"^\d{4}-\d{2}-\d{2}$", date_str.strip()):
        return date_str.strip()

    # 提取数字部分，尝试解析中文/斜杠格式
    m = re.search(r"(\d{4})[年/\-](\d{1,2})[月/\-](\d{1,2})", date_str)
    if m:
        y, mo, d = m.group(1), m.group(2).zfill(2), m.group(3).zfill(2)
        return f"{y}-{mo}-{d}"

    logger.warning(f"AccommodationAgent: 无法解析日期格式 '{date_str}'，跳过入住日期")
    return None


class AccommodationAgent:
    def __init__(self, name: str = "AccommodationAgent", model=None, **kwargs):
        self.name = name
        self.model = model

    # ══════════════════════════════════════════════════════════════════
    # 内部辅助：从前序 transport_query 结果提取到达枢纽信息
    # ══════════════════════════════════════════════════════════════════

    def _extract_transport_info(self, previous_results: List[Dict]) -> Dict[str, str]:
        """从前序智能体结果中提取交通信息（到达车站等）"""
        transport_info: Dict[str, str] = {}
        for result in previous_results:
            if result.get("agent_name") != "transport_query":
                continue
            data = result.get("result", {}).get("data", {})
            transport_plan = data.get("transport_plan", {})
            recommendation = transport_plan.get("recommendation", {})
            if recommendation:
                transport_info["arrival_station"] = (
                    recommendation.get("arrival_hub", "")
                    or recommendation.get("arrival_station", "")
                )
                transport_info["best_choice"] = recommendation.get("best_choice", "")
            query_info = transport_plan.get("query_info", {})
            if query_info:
                transport_info["destination"] = query_info.get("destination", "")
                transport_info["date"] = query_info.get("date", "")
            break
        return transport_info

    # ══════════════════════════════════════════════════════════════════
    # Phase 1：高德 maps_around_search — 地理发现
    # ══════════════════════════════════════════════════════════════════

    async def _search_amap_nearby_hotels(
        self,
        location: str,
        radius: int = _AMAP_HOTEL_RADIUS_M,
        city: str = "",
        count: int = _AMAP_HOTEL_MAX_COUNT,
    ) -> List[Dict]:
        """
        调用高德 maps_around_search 获取坐标周边酒店 POI。

        返回已含 distance_m 字段（米）的酒店列表，按距离升序排列。
        失败时返回空列表，触发调用方的降级路径。
        """
        try:
            from mcp_clients.amap_client import amap_mcp_session, search_hotels_nearby

            async with amap_mcp_session() as session:
                hotels = await search_hotels_nearby(
                    session=session,
                    location=location,
                    radius=radius,
                    city=city,
                    count=count,
                )
            logger.info(
                f"AccommodationAgent Amap Phase1: location={location} radius={radius}m "
                f"→ {len(hotels)} 家酒店"
            )
            return hotels

        except Exception as e:
            logger.warning(f"AccommodationAgent Amap Phase1 失败，将降级: {e}")
            return []

    # ══════════════════════════════════════════════════════════════════
    # Phase 2：RollingGo searchHotels — 价格增强
    # ══════════════════════════════════════════════════════════════════

    async def _query_rollinggo_single(
        self,
        session: Any,
        hotel: Dict,
        check_in_date: str | None,
        stay_nights: int,
        adults: int,
        price_min: float | None,
        price_max: float | None,
    ) -> Dict | None:
        """
        在已有 RollingGo session 内，对单家酒店查询价格/可用性。

        使用 placeType="landmark" + 酒店名 + Amap 坐标三者组合定位，
        最大程度提高与 Amap 结果的匹配精度。
        失败时返回 None，由调用方合并为"价格待查"状态。
        """
        try:
            arguments: dict = {
                "originQuery": hotel["name"],
                "place":       hotel["name"],
                "placeType":   "landmark",   # 按地标/酒店名精确搜索
                "size":        1,            # 只取排名最高的 1 条
            }
            if check_in_date:
                arguments["checkInParam"] = {
                    "checkInDate": check_in_date,
                    "stayNights":  stay_nights,
                    "adults":      adults,
                }
            hotel_tags: dict = {}
            if price_min is not None:
                hotel_tags["priceMin"] = price_min
            if price_max is not None:
                hotel_tags["priceMax"] = price_max
            if hotel_tags:
                arguments["hotelTags"] = hotel_tags
            # 传入 Amap 坐标辅助 RollingGo 做地理范围过滤
            if hotel.get("location"):
                arguments["location"] = hotel["location"]

            raw = await session.call_tool("searchHotels", arguments=arguments)

            if not (hasattr(raw, "content") and raw.content):
                return None
            text = (
                raw.content[0].text
                if hasattr(raw.content[0], "text")
                else str(raw.content[0])
            )
            parsed = json.loads(text)

            hotels_list: List[Dict] = []
            if isinstance(parsed, list):
                hotels_list = parsed
            elif isinstance(parsed, dict):
                for key in ("hotelInformationList", "hotels", "data"):
                    if key in parsed and isinstance(parsed[key], list) and parsed[key]:
                        hotels_list = parsed[key]
                        break

            return hotels_list[0] if hotels_list else None

        except Exception as e:
            logger.debug(f"RollingGo 查询 '{hotel['name']}' 失败: {e}")
            return None

    def _merge_hotel_data(self, amap_hotel: Dict, rg_hotel: Dict | None) -> Dict:
        """
        将 Amap POI 数据与 RollingGo 价格数据合并为统一结构。

        Amap 提供：距日程重心的精确距离、地址、评分、坐标。
        RollingGo 提供：真实价格、可用性、酒店ID（供后续详情查询）。
        rg_hotel 为 None 时，标注 price_note="价格待查"，地理数据仍保留。
        """
        distance_m = amap_hotel.get("distance_m", 0)
        distance_str = (
            f"{distance_m}m" if distance_m < 1000 else f"{distance_m / 1000:.1f}km"
        )

        merged: Dict = {
            "name":             amap_hotel.get("name", ""),
            "distance_to_center": distance_str,
            "distance_m":       distance_m,
            "address":          amap_hotel.get("address", ""),
            "amap_rating":      amap_hotel.get("amap_rating", ""),
            "location":         amap_hotel.get("location", ""),
            "data_sources":     ["Amap"],
        }

        if rg_hotel:
            # 防御性取价格字段（不同版本 RollingGo 字段名略有差异）
            price = (
                rg_hotel.get("price")
                or rg_hotel.get("minPrice")
                or rg_hotel.get("lowestPrice")
                or rg_hotel.get("minRoomPrice")
            )
            merged.update({
                "hotel_id":       rg_hotel.get("hotelId") or rg_hotel.get("id"),
                "price_per_night": price,
                "star":           rg_hotel.get("star") or rg_hotel.get("starLevel", ""),
                "rg_rating":      rg_hotel.get("score") or rg_hotel.get("rating", ""),
                # RollingGo 确认的名称（可能与 Amap 名称略有差异）
                "rg_name":        rg_hotel.get("hotelName") or rg_hotel.get("name", ""),
                "availability":   True,
                "data_sources":   ["Amap", "RollingGo"],
            })
        else:
            merged["price_per_night"] = None
            merged["price_note"]      = "价格待查，请自行查询"
            merged["availability"]    = None

        return merged

    async def _enrich_hotels_with_rollinggo(
        self,
        hotels: List[Dict],
        destination: str,
        check_in_date: str | None,
        stay_nights: int,
        adults: int,
        budget_level: str,
    ) -> List[Dict]:
        """
        Phase 2 入口：在单一 RollingGo stdio session 内顺序增强每家酒店的价格数据。

        【为什么顺序而非并发】
        RollingGo MCP 通过 stdio 运行，单进程内并发调用存在流交错风险。
        10 家酒店的顺序查询耗时通常 < 5s（每次约 0.3-0.5s），
        远优于并发启动 10 个 stdio 子进程的方案。

        降级：
          - RollingGo Key 未配置 → 返回纯 Amap 数据（带距离，无价格）
          - session 启动异常    → 同上
        """
        budget_map = {
            "经济":   (None, 300), "经济型": (None, 300),
            "舒适":   (200, 600),  "舒适型": (200, 600),
            "高端":   (500, None), "高端型": (500, None),
            "豪华":   (1000, None),
        }
        price_min, price_max = budget_map.get(budget_level, (None, None))

        try:
            from mcp_clients.hotel_client import hotel_mcp_session

            async with hotel_mcp_session() as session:
                enriched: List[Dict] = []
                for hotel in hotels:
                    rg_hotel = await self._query_rollinggo_single(
                        session=session,
                        hotel=hotel,
                        check_in_date=check_in_date,
                        stay_nights=stay_nights,
                        adults=adults,
                        price_min=price_min,
                        price_max=price_max,
                    )
                    enriched.append(self._merge_hotel_data(hotel, rg_hotel))

            rg_hit = sum(1 for h in enriched if "RollingGo" in h.get("data_sources", []))
            logger.info(
                f"AccommodationAgent Phase2: {rg_hit}/{len(enriched)} 家酒店获取到 RollingGo 价格"
            )
            return sorted(enriched, key=lambda h: h.get("distance_m", 9999))

        except ValueError:
            logger.warning("RollingGo Key 未配置，Phase 2 跳过，使用纯 Amap 数据")
            return [self._merge_hotel_data(h, None) for h in hotels]
        except Exception as e:
            logger.warning(f"RollingGo Phase 2 session 异常，降级为纯 Amap 数据: {e}")
            return [self._merge_hotel_data(h, None) for h in hotels]

    # ══════════════════════════════════════════════════════════════════
    # 降级路径：RollingGo-only 单次搜索（保留原有逻辑）
    # ══════════════════════════════════════════════════════════════════

    async def _search_hotels_via_mcp(
        self,
        destination: str,
        check_in_date: str | None,
        stay_nights: int,
        adults: int,
        hotel_brands: List[str],
        budget_level: str,
        location: str | None = None,
    ) -> List[Dict]:
        """
        调用 RollingGo searchHotels（城市级搜索，Phase 1 失败时的降级路径）。
        返回空列表时，后续 LLM 将基于自身知识推荐。
        """
        try:
            from mcp_clients.hotel_client import search_hotels
            from config import ROLLINGGO_MCP_CONFIG

            budget_map = {
                "经济":   (None, 300), "经济型": (None, 300),
                "舒适":   (200, 600),  "舒适型": (200, 600),
                "高端":   (500, None), "高端型": (500, None),
                "豪华":   (1000, None),
            }
            price_min, price_max = budget_map.get(budget_level, (None, None))

            raw = await search_hotels(
                origin_query=f"{destination} 酒店",
                place=destination,
                place_type="city",
                check_in_date=check_in_date,
                stay_nights=stay_nights,
                adults=adults,
                price_min=price_min,
                price_max=price_max,
                hotel_brands=hotel_brands or None,
                size=ROLLINGGO_MCP_CONFIG.get("default_size", 5),
                location=location or None,
            )

            if hasattr(raw, "content") and raw.content:
                text = (
                    raw.content[0].text
                    if hasattr(raw.content[0], "text")
                    else str(raw.content[0])
                )
                logger.info(f"RollingGo MCP raw (首300字符): {text[:300]}")
                hotels = json.loads(text)
                if isinstance(hotels, list):
                    return hotels
                if isinstance(hotels, dict):
                    for key in ("hotelInformationList", "hotels", "data"):
                        if key in hotels:
                            val = hotels[key]
                            if isinstance(val, list):
                                return val
                            if val:
                                return [val]
                            return []
                    logger.warning(f"RollingGo 返回未知结构，keys: {list(hotels.keys())}")
                    return []
            return []

        except ValueError as e:
            logger.warning(f"RollingGo Key 未配置，降级为纯 LLM 推荐: {e}")
            return []
        except Exception as e:
            logger.warning(f"RollingGo MCP 调用失败，降级为纯 LLM 推荐: {e}")
            return []

    # ══════════════════════════════════════════════════════════════════
    # 主入口
    # ══════════════════════════════════════════════════════════════════

    async def run(self, input_data: dict) -> dict:
        import re

        context = input_data.get("context", {})
        key_entities = context.get("key_entities", {})
        previous_results = input_data.get("previous_results", [])

        # daily_centers：按天重心列表，来自 accommodation_node 的分天计算
        # 格式：[{day: 1, lng: 116.39, lat: 39.92, poi_count: 3}, ...]
        daily_centers: List[Dict] = input_data.get("daily_centers", [])

        # location_hint 来自 accommodation_node 的降级链（单坐标或枢纽名）
        raw_location_hint: str = input_data.get("location_hint", "") or ""
        _is_coord = bool(re.match(r"^[\d.]+,[\d.]+$", raw_location_hint.strip()))
        coord_location: str | None = raw_location_hint.strip() if _is_coord else None
        hub_from_hint: str = (
            raw_location_hint.strip() if raw_location_hint and not _is_coord else ""
        )

        # ── 基础信息提取 ──────────────────────────────────────────────
        destination = key_entities.get("destination", "")
        date        = key_entities.get("date", "")
        duration    = key_entities.get("duration", "")
        adults      = int(key_entities.get("adults", 1))

        stay_nights = 1
        if duration:
            try:
                stay_nights = int("".join(filter(str.isdigit, str(duration)))) or 1
            except Exception:
                stay_nights = 1

        transport_info   = self._extract_transport_info(previous_results)
        arrival_station  = transport_info.get("arrival_station", "")

        # 多级兜底：从 context 各处取 arrival_station
        if not arrival_station:
            recommendation = context.get("transport_recommendation", {})
            arrival_station = (
                recommendation.get("arrival_hub", "")
                or recommendation.get("arrival_station", "")
            )
        if not arrival_station:
            transport_options = context.get("transport_options", [])
            if transport_options and isinstance(transport_options, list):
                arrival_station = (
                    transport_options[0].get("arrival_hub", "")
                    or transport_options[0].get("arrival_station", "")
                )
        if not arrival_station and hub_from_hint:
            arrival_station = hub_from_hint
        if not destination:
            destination = transport_info.get("destination", "")
        if not date:
            date = transport_info.get("date", "")

        if not destination:
            return {"error": "缺少目的地信息，无法推荐住宿"}

        # ── 用户偏好 ──────────────────────────────────────────────────
        user_preferences = context.get("user_preferences", {})
        raw_brands = user_preferences.get("hotel_brands", [])
        if isinstance(raw_brands, str):
            hotel_brands: List[str] = [
                b.strip()
                for b in raw_brands.replace("和", ",").replace("、", ",").split(",")
                if b.strip()
            ]
        else:
            hotel_brands = raw_brands or []
        budget_level: str = user_preferences.get("budget_level", "")
        other_prefs: Dict  = user_preferences.get("other_preferences", {})

        # ══════════════════════════════════════════════════════════════
        # Step A：两阶段酒店搜索（每天重心独立执行）
        # ══════════════════════════════════════════════════════════════
        check_in_date = _normalize_date(date) if date else None

        per_day_results: List[Dict] = []    # [{day, center, hotels, search_mode}, ...]
        all_hotel_results: List[Dict] = []  # 全部酒店合并（供 LLM 计数参考）

        if daily_centers:
            for dc in daily_centers:
                coord = f"{dc['lng']},{dc['lat']}"

                # ── Phase 1：Amap 地理发现 ──────────────────────────
                amap_hotels = await self._search_amap_nearby_hotels(
                    location=coord,
                    radius=_AMAP_HOTEL_RADIUS_M,
                    city=destination,
                    count=_AMAP_HOTEL_MAX_COUNT,
                )

                if amap_hotels:
                    # ── Phase 2：RollingGo 价格增强 ─────────────────
                    enriched_hotels = await self._enrich_hotels_with_rollinggo(
                        hotels=amap_hotels,
                        destination=destination,
                        check_in_date=check_in_date,
                        stay_nights=stay_nights,
                        adults=adults,
                        budget_level=budget_level,
                    )
                    per_day_results.append({
                        "day":         dc["day"],
                        "center":      coord,
                        "hotels":      enriched_hotels,
                        "search_mode": "two_stage",
                    })
                    all_hotel_results.extend(enriched_hotels)
                    rg_count = sum(
                        1 for h in enriched_hotels
                        if "RollingGo" in h.get("data_sources", [])
                    )
                    logger.info(
                        f"AccommodationAgent: Day {dc['day']} 两阶段完成："
                        f"{len(amap_hotels)} Amap → {rg_count}/{len(enriched_hotels)} RollingGo增强"
                    )
                else:
                    # ── Phase 1 失败：降级到 RollingGo-only ─────────
                    day_hotels = await self._search_hotels_via_mcp(
                        destination=destination,
                        check_in_date=check_in_date,
                        stay_nights=stay_nights,
                        adults=adults,
                        hotel_brands=hotel_brands,
                        budget_level=budget_level,
                        location=coord,
                    )
                    per_day_results.append({
                        "day":         dc["day"],
                        "center":      coord,
                        "hotels":      day_hotels,
                        "search_mode": "rollinggo_only",
                    })
                    all_hotel_results.extend(day_hotels)
                    logger.info(
                        f"AccommodationAgent: Day {dc['day']} 降级 RollingGo-only"
                        f"→ {len(day_hotels)} 家酒店"
                    )
        else:
            # 无 daily_centers：退回原始单次城市级搜索
            fallback_hotels = await self._search_hotels_via_mcp(
                destination=destination,
                check_in_date=check_in_date,
                stay_nights=stay_nights,
                adults=adults,
                hotel_brands=hotel_brands,
                budget_level=budget_level,
                location=coord_location,
            )
            all_hotel_results = fallback_hotels
            logger.info(
                f"AccommodationAgent: 无 daily_centers，单次兜底搜索"
                f"→ {len(fallback_hotels)} 家酒店"
            )

        hotel_results = all_hotel_results   # 兼容后续变量名

        # ══════════════════════════════════════════════════════════════
        # Step B：构建 mcp_data_section（区分两阶段 / 降级路径）
        # ══════════════════════════════════════════════════════════════
        mcp_data_section = ""

        if per_day_results and any(d["hotels"] for d in per_day_results):
            try:
                day_blocks: List[str] = []
                for d in per_day_results:
                    mode = d.get("search_mode", "unknown")
                    h_list = d["hotels"]

                    if mode == "two_stage":
                        # 两阶段结果：额外输出每家酒店的距离 + 价格摘要行，方便 LLM 快速扫描
                        summary_lines = []
                        for h in h_list:
                            sources  = "+".join(h.get("data_sources", ["未知"]))
                            price    = h.get("price_per_night")
                            price_str = f"¥{price}/晚" if price else h.get("price_note", "价格未知")
                            summary_lines.append(
                                f"  · {h['name']} | 距重心 {h.get('distance_to_center','?')} "
                                f"| {price_str} | 高德评分 {h.get('amap_rating', '?')} "
                                f"| 来源: {sources}"
                            )
                        block = (
                            f"【第 {d['day']} 天】活动重心 {d['center']}（两阶段搜索）"
                            f"共 {len(h_list)} 家附近酒店：\n"
                            + "\n".join(summary_lines)
                            + f"\n\n详细字段：\n{json.dumps(h_list, ensure_ascii=False, indent=2)}"
                        )
                    else:
                        block = (
                            f"【第 {d['day']} 天】活动重心 {d['center']}（RollingGo单阶段）"
                            f"共 {len(h_list)} 家酒店：\n"
                            f"{json.dumps(h_list, ensure_ascii=False, indent=2)}"
                        )
                    day_blocks.append(block)

                mcp_data_section = (
                    "【酒店数据：高德地理发现 + RollingGo 价格增强（两阶段搜索）】\n\n"
                    + "\n\n".join(day_blocks)
                    + "\n\n"
                    "【数据字段说明】\n"
                    "- distance_to_center: 该酒店距当天景点活动重心的距离（越小通勤越短）\n"
                    "- data_sources 含 RollingGo：已获取真实价格，price_per_night 可直接引用\n"
                    "- price_note='价格待查'：地理位置已确认，但价格未从 RollingGo 获取，请在推荐时注明\n"
                    "- 请勿虚构任何酒店名称、价格或距离数字\n"
                    "- 推荐时请优先选择 distance_to_center 较小且有真实价格的酒店\n"
                )
            except Exception as e:
                logger.warning(f"构建 mcp_data_section 失败: {e}")
                mcp_data_section = ""

        elif hotel_results:
            # 无 per_day_results（无 daily_centers 的兜底路径）
            try:
                mcp_data_section = (
                    f"【真实酒店数据（来自 RollingGo MCP，共 {len(hotel_results)} 条）】\n"
                    f"{json.dumps(hotel_results, ensure_ascii=False, indent=2)}\n\n"
                    "请基于以上真实数据进行分析和推荐，优先使用这些真实酒店，不要虚构酒店名称或价格。"
                )
            except Exception:
                mcp_data_section = ""

        if not mcp_data_section:
            mcp_data_section = (
                "【注意】当前无真实酒店数据，请基于你的知识给出合理推荐，并注明价格为估算。"
            )

        # ══════════════════════════════════════════════════════════════
        # Step C：构建 LLM Prompt 并生成结构化推荐
        # ══════════════════════════════════════════════════════════════
        location_hint = ""
        if daily_centers:
            day_coords_str = "、".join(
                f"第{d['day']}天({d['lng']},{d['lat']})" for d in daily_centers
            )
            location_hint = (
                f"\n【各天活动重心坐标（lng,lat）】{day_coords_str}\n"
                "请优先为每天推荐位于当天活动重心附近的酒店，以减少通勤时间。\n"
                "评估相邻两天重心距离：若 <3 km 可建议连住同一酒店；"
                "若某天重心明显偏离（>8 km）则建议当天换住更近的酒店。"
            )
        elif coord_location:
            location_hint = (
                f"\n【行程地理重心】景点地理重心坐标 {coord_location}（lng,lat），"
                "请优先推荐此坐标附近的酒店以减少每日通勤。"
            )
        elif arrival_station:
            location_hint = f"\n【到达交通枢纽】用户将抵达 {arrival_station}，请优先推荐该枢纽附近酒店。"

        brand_hint  = f"\n用户偏好品牌: {'、'.join(hotel_brands)}" if hotel_brands else ""
        budget_hint = f"\n用户预算等级: {budget_level}" if budget_level else ""
        other_hint  = ""
        if other_prefs:
            lines = [f"  - {k}: {v}" for k, v in other_prefs.items() if v]
            if lines:
                other_hint = "\n其他偏好:\n" + "\n".join(lines)

        skill_guide: str = context.get("skill_guide", "")

        # ── 知识库住宿建议：来自 CityKnowledgeDB.get_accommodation()──────
        # 仅注入到 analysis 字段的参考上下文，不得作为酒店白名单（options 仍只取 MCP 真实数据）
        knowledge_accommodation: List[str] = input_data.get("knowledge_accommodation", [])
        kb_hint = ""
        if knowledge_accommodation:
            kb_lines = "\n".join(f"  - {item}" for item in knowledge_accommodation)
            kb_hint = (
                "\n【本地住宿区域参考（来自旅游知识库，仅供 analysis 字段区域分析参考）】\n"
                + kb_lines
                + "\n重要：以上为知识库静态建议，不代表真实酒店；"
                "options 列表的 hotel_name 必须来自 MCP 白名单，不得从本段推断或虚构。\n"
            )

        # ── 酒店名单约束：只允许 LLM 从 MCP 返回的酒店中选择 ──────────
        # 提前从 per_day_results 提取所有酒店名，写入 prompt 防止 LLM 虚构
        mcp_hotel_names: List[str] = []
        for d in per_day_results:
            for h in d.get("hotels", []):
                name = h.get("name", "").strip()
                if name and name not in mcp_hotel_names:
                    mcp_hotel_names.append(name)

        hotel_name_constraint = ""
        if mcp_hotel_names:
            hotel_name_constraint = (
                "\n【酒店白名单（严格约束）】\n"
                "options 中的 hotel_name 必须且只能来自以下列表，不得虚构任何不在列表中的酒店：\n"
                + "\n".join(f"  - {n}" for n in mcp_hotel_names)
                + "\n"
            )

        prompt = f"""你是一个专业的住宿推荐专家（AccommodationAgent）。
请为用户在【{destination}】的住宿提供分析和推荐。

【入住信息】
- 目的地: {destination}
- 入住日期: {date or '未指定'}
- 行程时长: {duration or '未指定'}（约 {stay_nights} 晚）
- 成人人数: {adults}
{location_hint}{brand_hint}{budget_hint}{other_hint}
{kb_hint}
{mcp_data_section}
{hotel_name_constraint}
【字段填写铁律 — 必须严格遵守】
1. 所有字段值必须有实际依据：hotel_name、price_range、distance_info 均须来自上方 MCP 数据。
2. 若某字段在 MCP 数据中未提供（如 star、highlights 等），JSON 中必须填写 null，
   绝对禁止填写"无"、"暂无"、"数据未提及"、"未知"等字符串。
3. data_source 字段：若酒店数据含 RollingGo 字样则填 "mcp_two_stage"，否则填 "mcp_amap_only"，
   若无任何 MCP 数据则填 "llm_inferred"。
4. analysis 字段中可自由说明推荐区域逻辑（如哪些区域适合哪类旅客），
   但 options 列表只允许出现 MCP 数据中真实存在的酒店。

【输出格式要求】
请严格输出以下JSON格式，不要包含任何其他文本：
{{
    "destination": "{destination}",
    "arrival_station": "{arrival_station or '未知'}",
    "mcp_data_used": {"true" if hotel_results else "false"},
    "analysis": "住宿选址分析：结合到达枢纽、当天景点重心位置、用户偏好，说明推荐区域逻辑及整体住宿策略",
    "options": [
        {{
            "tier": "档次（经济型/舒适型/高端型）",
            "hotel_name": "酒店名称（必须来自 MCP 白名单）",
            "hotel_id": "酒店ID（RollingGo数据提供时填写，否则填null）",
            "area": "所在区域",
            "price_range": "每晚价格，格式'XXX元/晚'（必须来自MCP，无真实价格则填null）",
            "star": "星级（MCP提供则填，否则填null）",
            "highlights": "真实亮点（仅填MCP数据可佐证的内容，无则填null）",
            "distance_info": "距当天活动重心距离（来自 distance_to_center 字段，无则填null）",
            "data_source": "mcp_two_stage 或 mcp_amap_only 或 llm_inferred"
        }}
    ],
    "daily_suggestions": [
        {{
            "day": 1,
            "center_coord": "当天活动重心坐标",
            "suggested_hotel": "推荐酒店名称（必须来自 options 列表）",
            "reason": "推荐理由（引用 distance_to_center 和真实价格，不得编造）",
            "stay_strategy": "连住 或 换酒店"
        }}
    ],
    "recommendation": {{
        "best_choice": "综合最推荐的酒店（全程连住时）",
        "reason": "推荐理由（引用真实数据支撑）",
        "booking_tips": "预订建议"
    }}
}}
""" + (f"\n【住宿规划指南】请严格遵循以下选址原则：\n{skill_guide}\n" if skill_guide else "")

        try:
            messages = [
                {"role": "system", "content": "你是一个住宿推荐专家。只输出JSON，不含任何额外文本。"},
                {"role": "user",   "content": prompt},
            ]
            response = await self.model.ainvoke(messages)
            text = response.content

            # 清洗 Markdown 代码块
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            result = json.loads(text)
            return {
                "accommodation_plan":   result,
                "mcp_hotels_count":     len(hotel_results),
                "daily_centers_used":   len(daily_centers),
                "two_stage_days":       sum(
                    1 for d in per_day_results if d.get("search_mode") == "two_stage"
                ),
            }

        except Exception as e:
            logger.error(f"AccommodationAgent LLM failed: {e}")
            if hotel_results:
                return {
                    "accommodation_plan": {
                        "destination":   destination,
                        "mcp_data_used": True,
                        "raw_hotels":    hotel_results,
                        "analysis":      "LLM 分析失败，以下为 MCP 原始酒店数据",
                    },
                    "mcp_hotels_count":  len(hotel_results),
                }
            return {"error": str(e)}
