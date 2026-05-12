"""
住宿专家智能体 AccommodationAgent
职责：根据目的地、每天行程的地理重心和用户偏好，融合高德地理数据与途牛真实价格数据，
      再由 LLM 对结果进行分析和个性化推荐。

双源数据流（每天重心独立 Amap 发现，Tuniu 按 spread 路由共享/分段池）：
  Amap maps_around_search   — 地理发现
        基于坐标 + 半径拉取周边酒店 POI，获得精确 distance_m（距当天重心的米数）。
  Tuniu hotel_search        — 价格增强
        城市级搜索拿到 normalize_hotel 输出（lowest_price / star_name / business …），
        与 Amap POI 做"去后缀名称双向包含 + 商圈 token 重叠"模糊匹配。
  merge — _merge_hotel_data 合并两份数据，LLM 拿到兼具"地理精度"和"真实价格"的融合视图。

Tuniu 路由（_enrich_days_via_tuniu，受 50 RPD 预算约束）：
  spread <= 5km                  整段单查（1 次 tuniu，池子复用到各天）
  spread > 5km 且 N 天 <= 6      分段每晚查（N 次 tuniu，每天 keyword 锚定商圈）
  spread > 5km 且 N 天 > 6       强制降级整段单查 + warning

降级路径：
  Amap 某天空结果           该天 hotels=[]，交由 LLM 兜底（不重复消耗 tuniu RPD）
  Tuniu 匹配不上的 Amap     保留 Amap 条目，price_note="价格待查"
  无 daily_centers          tuniu 城市级单次搜索作为唯一数据源
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Dict, List, Optional

from utils.date_resolver import normalize_date

logger = logging.getLogger(__name__)

# Amap 周边酒店发现配置
_AMAP_HOTEL_RADIUS_M = 2000    # 搜索半径（米）
_AMAP_HOTEL_MAX_COUNT = 5      # 每天最多拉取的酒店数量（3 档位备选已足够）

# budget_level -> (price_min, price_max) 元/晚；与 intent_node 输出对齐。
# intent_node.type 仅产出 {"连锁","经济","豪华","民宿",""}；long_term_memory
# 历史 budget_level 同口径，故无需 "经济型"/"舒适型" 之类别名。
_BUDGET_MAP: Dict[str, tuple[Optional[float], Optional[float]]] = {
    "经济": (None, 300),
    "舒适": (200, 600),
    "高端": (500, None),
    "豪华": (1000, None),
}


def _resolve_price_range(
    intent_min: Optional[float],
    intent_max: Optional[float],
    budget_level: str,
) -> tuple[Optional[float], Optional[float]]:
    """intent.price_range 优先；缺失则 fallback 到 budget_level 映射。"""
    if intent_min is not None or intent_max is not None:
        return intent_min, intent_max
    return _BUDGET_MAP.get(budget_level, (None, None))


def _parse_price_range(s: str) -> tuple[Optional[float], Optional[float]]:
    """
    解析住宿价格区间字符串为 (min, max) 浮点元组。

    支持示例：
      "300-500元/晚"   -> (300, 500)
      "200~600/晚"      -> (200, 600)
      "500元以上"       -> (500, None)
      "不超过800元"     -> (None, 800)
      ""/None/无效串    -> (None, None)
    """
    import re

    if not s or not isinstance(s, str):
        return (None, None)

    # 范围格式：300-500 / 200~600
    m = re.search(r"(\d+(?:\.\d+)?)\s*[-~到至]\s*(\d+(?:\.\d+)?)", s)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        return (lo, hi) if lo <= hi else (hi, lo)

    # 下限格式：500以上 / 500+ / 大于500
    m = re.search(r"(?:大于|超过|>=?)?\s*(\d+(?:\.\d+)?)\s*(?:元)?\s*(?:以上|\+|起)", s)
    if m:
        return (float(m.group(1)), None)

    # 上限格式：不超过800 / <=800 / 800以下
    m = re.search(r"(?:不超过|不高于|<=?)\s*(\d+(?:\.\d+)?)|(\d+(?:\.\d+)?)\s*(?:元)?\s*以下", s)
    if m:
        val = m.group(1) or m.group(2)
        return (None, float(val))

    return (None, None)


class AccommodationAgent:
    def __init__(self, name: str = "AccommodationAgent", model=None, **kwargs):
        self.name = name
        self.model = model

    # ══════════════════════════════════════════════════════════════════
    # 内部辅助：从前序去程查询结果提取到达枢纽信息
    # ══════════════════════════════════════════════════════════════════

    def _extract_transport_info(self, previous_results: List[Dict]) -> Dict[str, str]:
        """从前序智能体结果中提取交通信息（到达车站等）。
        兼容旧 agent_name 'transport_query' 与新的 'transport_outbound'。"""
        transport_info: Dict[str, str] = {}
        for result in previous_results:
            if result.get("agent_name") not in ("transport_query", "transport_outbound"):
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
    # 路由：每日重心地理分散度（米），决定整段单查 vs 分段每晚查
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def _compute_max_daycenter_distance(daily_centers: List[Dict]) -> float:
        """
        计算 daily_centers 列表内任意两个重心之间的最大球面距离（米）。

        用于路由决策：spread <= 5km 走整段单查；> 5km 走分段每晚查。
        - 输入为空 / 单点：返回 0.0
        - 坐标缺失或非法：跳过该点
        - 使用 Haversine 公式（地球半径 6371000m），WGS-84/GCJ-02 偏差对 5km
          决策阈值影响 < 0.5%，可忽略
        """
        import math

        pts: List[tuple[float, float]] = []
        for dc in daily_centers:
            try:
                lng = float(dc.get("lng"))
                lat = float(dc.get("lat"))
            except (TypeError, ValueError):
                continue
            pts.append((lng, lat))

        if len(pts) < 2:
            return 0.0

        R = 6371000.0
        max_dist = 0.0
        for i in range(len(pts)):
            lng1, lat1 = pts[i]
            phi1 = math.radians(lat1)
            for j in range(i + 1, len(pts)):
                lng2, lat2 = pts[j]
                phi2 = math.radians(lat2)
                dphi = math.radians(lat2 - lat1)
                dlmb = math.radians(lng2 - lng1)
                a = (
                    math.sin(dphi / 2) ** 2
                    + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
                )
                d = 2 * R * math.asin(math.sqrt(a))
                if d > max_dist:
                    max_dist = d
        return max_dist

    @staticmethod
    def _extract_business_keyword(amap_hotels: List[Dict]) -> Optional[str]:
        """
        从一天的 amap 酒店 POI 列表中，按 business_area 频率投票取 top-1。

        用于分段每晚查（spread > 5km）模式下，给 tuniu hotel_search 传 keyword，
        把搜索结果锚定到当天活动重心所在的商圈。

        - 空列表 / 全部 business_area 为空：返回 None（调用方据此不传 keyword）
        - 同票时取最先出现的（amap 已按 distance 升序，等价于"距离最近优先"）
        """
        if not amap_hotels:
            return None

        counts: Dict[str, int] = {}
        order: Dict[str, int] = {}     # 记录首次出现位序，用于同票 tiebreak
        for idx, h in enumerate(amap_hotels):
            ba = (h.get("business_area") or "").strip()
            if not ba:
                continue
            counts[ba] = counts.get(ba, 0) + 1
            order.setdefault(ba, idx)

        if not counts:
            return None

        # 排序：频率降序，位序升序（出现早的优先）
        best = sorted(counts.items(), key=lambda kv: (-kv[1], order[kv[0]]))[0][0]
        return best

    # ══════════════════════════════════════════════════════════════════
    # Fuzzy match：把 amap 酒店与 tuniu 池子一一对齐
    # ══════════════════════════════════════════════════════════════════

    # 常见酒店名后缀；按长度降序匹配，避免 "酒店" 抢先吃掉 "大酒店"
    _HOTEL_SUFFIXES: tuple[str, ...] = (
        "国际大酒店", "度假酒店", "国际酒店", "连锁酒店", "大酒店",
        "度假村", "酒店", "宾馆", "饭店", "客栈", "公寓", "民宿", "旅馆", "青年旅舍", "招待所"
    )

    @classmethod
    def _strip_hotel_suffix(cls, name: str) -> str:
        """去掉常见酒店尾缀；长后缀优先。空串返回空串。"""
        s = (name or "").replace(" ", "").strip()
        for suf in cls._HOTEL_SUFFIXES:
            if s.endswith(suf) and len(s) > len(suf):
                return s[:-len(suf)]
        return s

    @staticmethod
    def _chinese_tokens(text: str, min_len: int = 2) -> set[str]:
        """从字符串抽取长度 >= min_len 的连续汉字片段集合。用于地址/商圈重叠判定。"""
        import re as _re
        if not text:
            return set()
        return set(_re.findall(r"[一-鿿]{" + str(min_len) + ",}", text))

    @classmethod
    def _match_amap_to_tuniu_pool(
        cls,
        amap_hotels: List[Dict],
        tuniu_pool: List[Dict],
    ) -> List[Optional[Dict]]:
        """
        把 amap 酒店列表与 tuniu 池子做模糊匹配，返回与 amap_hotels 等长的对齐列表。

        匹配规则：
          1. 名称去后缀后双向包含（短串长度 >= 2，避免 "如家" 命中所有汉庭如歌系列）
          2. 多候选时用 business_area / address 的 2+ 字汉字 token 重叠数做 tiebreak
          3. 全部无命中 → 该位置 None（调用方走 price_note="价格待查"）

        tuniu_pool 元素允许被多次匹配（连锁多分店场景）。
        """
        if not tuniu_pool:
            return [None] * len(amap_hotels)

        # 预处理 tuniu 池子：缓存 stripped name 与地理 token 集合
        tuniu_meta: List[tuple[str, set[str]]] = []
        for t in tuniu_pool:
            t_name = cls._strip_hotel_suffix(t.get("hotel_name", ""))
            t_geo  = cls._chinese_tokens(
                f"{t.get('business', '') or ''} {t.get('address', '') or ''}"
            )
            tuniu_meta.append((t_name, t_geo))

        results: List[Optional[Dict]] = []
        for a in amap_hotels:
            a_name = cls._strip_hotel_suffix(a.get("name", ""))
            if not a_name or len(a_name) < 2:
                results.append(None)
                continue

            # Step 1：收集所有名称双向包含的候选
            candidates: List[tuple[int, Dict]] = []  # (idx, tuniu_dict)
            for idx, (t_name, _) in enumerate(tuniu_meta):
                if not t_name or len(t_name) < 2:
                    continue
                if a_name in t_name or t_name in a_name:
                    candidates.append((idx, tuniu_pool[idx]))

            if not candidates:
                results.append(None)
                continue

            if len(candidates) == 1:
                results.append(candidates[0][1])
                continue

            # Step 2：多候选 → 用地理 token 重叠数 tiebreak
            a_geo = cls._chinese_tokens(
                f"{a.get('business_area', '') or ''} {a.get('address', '') or ''}"
            )
            best_idx, best_t = candidates[0]
            best_score = len(a_geo & tuniu_meta[best_idx][1])
            for idx, t in candidates[1:]:
                score = len(a_geo & tuniu_meta[idx][1])
                if score > best_score:
                    best_idx, best_t, best_score = idx, t, score
            results.append(best_t)

        return results

    # ══════════════════════════════════════════════════════════════════
    # 高德 maps_around_search — 地理发现
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
                f"AccommodationAgent Amap nearby: location={location} radius={radius}m "
                f"-> {len(hotels)} 家酒店"
            )
            return hotels

        except Exception as e:
            logger.warning(f"AccommodationAgent Amap nearby 失败，将降级: {e}")
            return []

    # ══════════════════════════════════════════════════════════════════
    # 合并：Amap POI + Tuniu normalize_hotel → 统一结构
    # ══════════════════════════════════════════════════════════════════

    def _merge_hotel_data(self, amap_hotel: Dict, tuniu_hotel: Dict | None) -> Dict:
        """
        将 Amap POI 数据与 tuniu normalize_hotel 输出合并为统一结构。

        Amap 提供：距日程重心的精确距离、地址、评分、坐标。
        tuniu 提供：真实价格、酒店ID、星级、评分、商圈/品牌。
        tuniu_hotel 为 None 时，标注 price_note="价格待查"，地理数据仍保留。
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

        if tuniu_hotel:
            # tuniu normalize_hotel 已扁平化：lowest_price 直接为 int/None
            merged.update({
                "hotel_id":        tuniu_hotel.get("hotel_id"),
                "price_per_night": tuniu_hotel.get("lowest_price"),
                "star":            tuniu_hotel.get("star_name"),
                "tuniu_rating":    tuniu_hotel.get("score"),
                # tuniu 确认的名称（可能与 Amap 名称略有差异）
                "tuniu_name":      tuniu_hotel.get("hotel_name", ""),
                "business":        tuniu_hotel.get("business"),
                "brand":           tuniu_hotel.get("brand"),
                "availability":    True,
                "data_sources":    ["Amap", "Tuniu"],
            })
        else:
            merged["price_per_night"] = None
            merged["price_note"]      = "价格待查，请自行查询"
            merged["availability"]    = None

        return merged

    # ══════════════════════════════════════════════════════════════════
    # 路由 + 集成：每天 Amap 池子 + tuniu 价格池子合并
    # ══════════════════════════════════════════════════════════════════

    # 路由阈值常量
    _TUNIU_SPREAD_THRESHOLD_M = 5000   # 重心最大间距 <= 此值走整段单查
    _TUNIU_SEGMENTED_MAX_DAYS = 6      # 分段每晚查模式的天数上限，超过强制整段

    async def _enrich_days_via_tuniu(
        self,
        daily_centers: List[Dict],
        amap_results: List[List[Dict]],
        destination: str,
        check_in_date: str | None,
        stay_nights: int,
        price_min: float | None,
        price_max: float | None,
    ) -> Dict[int, List[Dict]]:
        """
        用 tuniu 池子增强 amap 酒店价格，按 daily_centers 地理分散度路由：

          spread <= 5km            -> 整段单查（1 次 tuniu，池子复用到各天）
          spread > 5km & N <= 6    -> 分段每晚查（N 次 tuniu，每天独立 keyword）
          spread > 5km & N > 6     -> 强制整段单查 + warning（避免过多 API 调用）

        返回 {day: [merged_hotel_dicts]}，契约与旧 _enrich_all_days_rollinggo 一致，
        每条 merged dict 已经过 _merge_hotel_data 标准化。
        """
        from datetime import date, timedelta

        n_days = len(daily_centers)
        spread = self._compute_max_daycenter_distance(daily_centers)

        # 路由判定
        if spread <= self._TUNIU_SPREAD_THRESHOLD_M:
            mode = "batch"
        elif n_days > self._TUNIU_SEGMENTED_MAX_DAYS:
            mode = "batch"
            logger.warning(
                f"AccommodationAgent: 重心分散 spread={spread:.0f}m 但 N={n_days} > "
                f"{self._TUNIU_SEGMENTED_MAX_DAYS}，强制降级为整段单查"
            )
        else:
            mode = "segmented"

        logger.info(
            f"AccommodationAgent tuniu route: mode={mode} spread={spread:.0f}m N={n_days}"
        )

        # ── 整段单查 ──────────────────────────────────────────────────
        if mode == "batch":
            pool = await self._search_hotels_via_tuniu(
                city_name=destination,
                check_in_date=check_in_date,
                stay_nights=stay_nights,
                price_min=price_min,
                price_max=price_max,
                keyword=None,
            )
            result_map: Dict[int, List[Dict]] = {}
            for dc, amap_hotels in zip(daily_centers, amap_results):
                if not amap_hotels:
                    result_map[dc["day"]] = []
                    continue
                matched = self._match_amap_to_tuniu_pool(amap_hotels, pool)
                merged = [
                    self._merge_hotel_data(a, t) for a, t in zip(amap_hotels, matched)
                ]
                merged.sort(key=lambda h: h.get("distance_m", 9999))
                hit = sum(1 for h in merged if "Tuniu" in h.get("data_sources", []))
                logger.info(
                    f"AccommodationAgent tuniu batch: Day {dc['day']} "
                    f"{hit}/{len(merged)} 家匹配到 tuniu 价格"
                )
                result_map[dc["day"]] = merged
            return result_map

        # ── 分段每晚查 ────────────────────────────────────────────────
        # 各天 check_in 偏移 (day_index)，stay_nights=1 锚定该晚
        ci0: date | None = None
        if check_in_date:
            try:
                ci0 = date.fromisoformat(check_in_date)
            except ValueError:
                ci0 = None

        async def _one_day(dc: Dict, amap_hotels: List[Dict]) -> tuple[int, List[Dict]]:
            day = dc["day"]
            if not amap_hotels:
                return day, []

            keyword = self._extract_business_keyword(amap_hotels)
            day_check_in: str | None = None
            if ci0 is not None:
                # daily_centers 的 day 从 1 开始，第 1 天 = 入住首晚
                day_check_in = (ci0 + timedelta(days=max(day - 1, 0))).isoformat()

            pool = await self._search_hotels_via_tuniu(
                city_name=destination,
                check_in_date=day_check_in,
                stay_nights=1,
                price_min=price_min,
                price_max=price_max,
                keyword=keyword,
            )
            matched = self._match_amap_to_tuniu_pool(amap_hotels, pool)
            merged = [
                self._merge_hotel_data(a, t) for a, t in zip(amap_hotels, matched)
            ]
            merged.sort(key=lambda h: h.get("distance_m", 9999))
            hit = sum(1 for h in merged if "Tuniu" in h.get("data_sources", []))
            logger.info(
                f"AccommodationAgent tuniu segmented: Day {day} "
                f"check_in={day_check_in} keyword={keyword} "
                f"{hit}/{len(merged)} 家匹配到 tuniu 价格"
            )
            return day, merged

        try:
            pairs = await asyncio.wait_for(
                asyncio.gather(*[
                    _one_day(dc, hotels) for dc, hotels in zip(daily_centers, amap_results)
                ]),
                timeout=35.0,
            )
            return dict(pairs)
        except asyncio.TimeoutError:
            logger.warning(
                "AccommodationAgent tuniu segmented 整体超时(35s)，降级为纯 Amap 数据"
            )
            return {
                dc["day"]: [self._merge_hotel_data(h, None) for h in hotels]
                for dc, hotels in zip(daily_centers, amap_results)
            }

    # ══════════════════════════════════════════════════════════════════
    # 途牛 hotel_search 城市级搜索（新主路径）
    # ══════════════════════════════════════════════════════════════════

    async def _search_hotels_via_tuniu(
        self,
        city_name: str,
        check_in_date: str | None,
        stay_nights: int,
        price_min: float | None,
        price_max: float | None,
        keyword: str | None = None,
    ) -> List[Dict]:
        """
        途牛 hotel_search 城市级搜索，返回 normalize_hotel 输出格式的酒店列表。

        - prices 仅在两端都有值时拼成 "min-max" 传入；单边/None 不传，由 LLM 后处理过滤
        - check_out = check_in + stay_nights 天；check_in 为空则不传日期（tuniu 默认今天起）
        - keyword 可选，用于聚焦商圈/地标（如 "三里屯"）
        - 失败（TuniuCallError / TuniuBudgetExceeded）一律返回 []，由调用方降级
        """
        from datetime import date, timedelta

        try:
            from mcp_clients.tuniu_client import (
                hotel_search,
                unwrap_mcp_content,
                iter_hotels,
                normalize_hotel,
                TuniuCallError,
            )
            from utils.tuniu_budget import TuniuBudgetExceeded
        except ImportError as e:
            logger.warning(f"tuniu_client 导入失败: {e}")
            return []

        prices: str | None = None
        if price_min is not None and price_max is not None:
            prices = f"{int(price_min)}-{int(price_max)}"

        check_out_date: str | None = None
        if check_in_date:
            try:
                ci = date.fromisoformat(check_in_date)
                check_out_date = (ci + timedelta(days=max(stay_nights, 1))).isoformat()
            except ValueError:
                logger.warning(
                    f"_search_hotels_via_tuniu: 无效 check_in_date '{check_in_date}'，跳过日期"
                )

        try:
            raw = await hotel_search(
                city_name=city_name,
                check_in=check_in_date,
                check_out=check_out_date,
                keyword=keyword,
                prices=prices,
            )
            unwrapped = unwrap_mcp_content(raw)
            hotels = [normalize_hotel(h) for h in iter_hotels(unwrapped)]
            logger.info(
                f"AccommodationAgent tuniu hotel_search: city={city_name} "
                f"check_in={check_in_date} keyword={keyword} prices={prices} "
                f"-> {len(hotels)} 家酒店"
            )
            return hotels
        except TuniuBudgetExceeded as e:
            logger.warning(f"tuniu 预算耗尽，hotel_search 降级: {e}")
            return []
        except TuniuCallError as e:
            logger.warning(f"tuniu hotel_search 失败 [{e.type}]: {e.message}")
            return []
        except Exception as e:
            logger.warning(f"tuniu hotel_search 未预期异常: {e}")
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

        # location_hint 来自 accommodation_node 的降级链。当存在 daily_centers 时，
        # node 会用首天坐标作为 hint（详见 accommodation_node._build_input_data）；
        # 此分支下 prompt 已直接使用 daily_centers 列表，故坐标 hint 在此处冗余。
        # 仅在 daily_centers 为空时，hint 才会是枢纽/城市名，用作 arrival_station 兜底。
        raw_location_hint: str = input_data.get("location_hint", "") or ""
        _is_coord = bool(re.match(r"^[\d.]+,[\d.]+$", raw_location_hint.strip()))
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

        # ── 用户偏好（历史，来自 MemoryManager.long_term）──────────────
        # long_term_memory 始终以 list 形式存储 hotel_brands（参见 long_term_memory.py），
        # 故不再做 str 兼容解析。
        user_preferences = context.get("user_preferences", {})
        history_brands: List[str] = list(user_preferences.get("hotel_brands") or [])
        budget_level: str = user_preferences.get("budget_level", "")
        other_prefs: Dict = user_preferences.get("other_preferences", {})

        # ── 当前 query 的住宿意图（P1 intent_node 提取，优先级高于历史偏好）─
        acc_prefs: Dict = context.get("accommodation_prefs", {}) or {}
        intent_brands = [
            b for b in acc_prefs.get("brand_keywords", [])
            if isinstance(b, str) and b.strip()
        ]
        intent_type = acc_prefs.get("type", "") or ""
        # "连锁" / "民宿" 作品牌词；"经济" / "豪华" 仅作为 budget 兜底
        if intent_type in ("连锁", "民宿") and intent_type not in intent_brands:
            intent_brands.append(intent_type)
        # 合并：intent 在前、历史偏好在后，去重保序
        hotel_brands = list(dict.fromkeys(intent_brands + history_brands))

        # type 兜底 budget_level：intent_node 的 type 只可能是 {连锁/经济/豪华/民宿/空}
        if not budget_level and intent_type in ("经济", "豪华"):
            budget_level = intent_type

        # price_range 解析（用于覆盖 budget_map 价格区间，及 LLM prompt 提示）
        intent_price_range_str: str = acc_prefs.get("price_range") or ""
        intent_price_min, intent_price_max = _parse_price_range(intent_price_range_str)

        if intent_brands or intent_type or intent_price_range_str:
            logger.info(
                f"AccommodationAgent: 注入 intent prefs - "
                f"brands={intent_brands}, type='{intent_type}', "
                f"price_range='{intent_price_range_str}' -> "
                f"merged_brands={hotel_brands}, budget_level='{budget_level}', "
                f"price=({intent_price_min}, {intent_price_max})"
            )

        # ══════════════════════════════════════════════════════════════
        # Step A：Amap 地理发现 + Tuniu 价格增强（每天重心独立执行）
        # ══════════════════════════════════════════════════════════════
        check_in_date = normalize_date(date) if date else None
        if date and not check_in_date:
            logger.warning(f"AccommodationAgent: 无法解析日期格式 '{date}'，跳过入住日期")

        per_day_results: List[Dict] = []    # [{day, center, hotels}, ...]（仅含非空天）
        hotel_results: List[Dict] = []  # 全部酒店合并（供 LLM 计数参考）

        if daily_centers:
            # 并行 Amap 地理发现（各天独立，asyncio.gather）
            amap_results: List[List[Dict]] = list(await asyncio.gather(*[
                self._search_amap_nearby_hotels(
                    location=f"{dc['lng']},{dc['lat']}",
                    radius=_AMAP_HOTEL_RADIUS_M,
                    city=destination,
                    count=_AMAP_HOTEL_MAX_COUNT,
                )
                for dc in daily_centers
            ]))
            logger.info(
                f"AccommodationAgent Amap parallel: {len(daily_centers)} 天，"
                f"各天酒店数: {[len(r) for r in amap_results]}"
            )

            # tuniu 价格增强：按 spread 路由（整段单查 / 分段每晚查）
            tuniu_price_min, tuniu_price_max = _resolve_price_range(
                intent_price_min, intent_price_max, budget_level
            )

            enriched_day_map: Dict[int, List[Dict]] = {}
            if any(amap_results):
                enriched_day_map = await self._enrich_days_via_tuniu(
                    daily_centers=daily_centers,
                    amap_results=amap_results,
                    destination=destination,
                    check_in_date=check_in_date,
                    stay_nights=stay_nights,
                    price_min=tuniu_price_min,
                    price_max=tuniu_price_max,
                )

            # 按原始天序组装 per_day_results；Amap 空结果的天跳过 tuniu 单查
            # （避免重复消耗 RPD），不进入 per_day_results，下游交由 LLM 兜底。
            for dc, amap_hotels in zip(daily_centers, amap_results):
                if not amap_hotels:
                    logger.info(
                        f"AccommodationAgent: Day {dc['day']} Amap 空结果，"
                        f"跳过 tuniu 单查，交由 LLM 兜底"
                    )
                    continue
                day      = dc["day"]
                enriched = enriched_day_map.get(day, [])
                per_day_results.append({
                    "day":    day,
                    "center": f"{dc['lng']},{dc['lat']}",
                    "hotels": enriched,
                })
                hotel_results.extend(enriched)
                tuniu_count = sum(
                    1 for h in enriched if "Tuniu" in h.get("data_sources", [])
                )
                logger.info(
                    f"AccommodationAgent: Day {day} Amap+Tuniu 双源完成："
                    f"{len(amap_hotels)} Amap -> {tuniu_count}/{len(enriched)} Tuniu 增强"
                )
        else:
            # 无 daily_centers：直接调 tuniu 城市级单次搜索作为唯一数据源
            # 此处不带 keyword（无活动重心信息）
            fb_price_min, fb_price_max = _resolve_price_range(
                intent_price_min, intent_price_max, budget_level
            )

            fallback_hotels = await self._search_hotels_via_tuniu(
                city_name=destination,
                check_in_date=check_in_date,
                stay_nights=stay_nights,
                price_min=fb_price_min,
                price_max=fb_price_max,
                keyword=None,
            )
            hotel_results = fallback_hotels
            logger.info(
                f"AccommodationAgent: 无 daily_centers，tuniu 城市级单次兜底"
                f"→ {len(fallback_hotels)} 家酒店"
            )

        # ══════════════════════════════════════════════════════════════
        # Step B：构建 mcp_data_section（per_day_results 已过滤空天，只剩
        # Amap+Tuniu 双源数据；无 daily_centers 时走 hotel_results 兜底）
        # ══════════════════════════════════════════════════════════════
        mcp_data_section = ""

        if per_day_results:
            try:
                day_blocks: List[str] = []
                for d in per_day_results:
                    h_list = d["hotels"]
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
                        f"【第 {d['day']} 天】活动重心 {d['center']}"
                        f"（Amap+Tuniu 双源）共 {len(h_list)} 家附近酒店：\n"
                        + "\n".join(summary_lines)
                        + f"\n\n详细字段：\n{json.dumps(h_list, ensure_ascii=False, indent=2)}"
                    )
                    day_blocks.append(block)

                mcp_data_section = (
                    "【酒店数据：Amap 地理发现 + Tuniu 价格增强（双源融合）】\n\n"
                    + "\n\n".join(day_blocks)
                    + "\n\n"
                    "【数据字段说明】\n"
                    "- distance_to_center: 该酒店距当天景点活动重心的距离（越小通勤越短）\n"
                    "- data_sources 含 Tuniu：已获取真实价格，price_per_night 可直接引用\n"
                    "- price_note='价格待查'：地理位置已确认，但 Tuniu 未匹配到价格，请在推荐时注明\n"
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
                    f"【真实酒店数据（来自 Tuniu 城市级搜索，共 {len(hotel_results)} 条）】\n"
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
        elif arrival_station:
            location_hint = f"\n【到达交通枢纽】用户将抵达 {arrival_station}，请优先推荐该枢纽附近酒店。"

        brand_hint  = f"\n用户偏好品牌: {'、'.join(hotel_brands)}" if hotel_brands else ""
        budget_hint = f"\n用户预算等级: {budget_level}" if budget_level else ""
        # 当前 query 明确价位（来自 P1 accommodation_prefs.price_range），
        # 优先级高于 budget_level，需 LLM 过滤掉超出区间的酒店
        price_range_hint = (
            f"\n用户本次明确要求住宿价位: {intent_price_range_str}"
            "（请在 options 与 daily_suggestions 中只保留落在此区间内的酒店；超出区间的请放在 analysis 中说明并排除）"
            if intent_price_range_str else ""
        )
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
        # 单次遍历 per_day_results 同时收集：
        #   1) mcp_hotel_names: prompt 白名单（去重保序）
        #   2) day_closest:     每天距重心最近的一家（hotels 已按 distance_m 升序，
        #                       取首条），供 LLM 漏天时后处理补充
        mcp_hotel_names: List[str] = []
        seen_names: set[str] = set()
        day_closest: List[tuple[Dict, Dict]] = []  # [(per_day_entry, hotel), ...]
        for d in per_day_results:
            hotels = d.get("hotels") or []
            if not hotels:
                continue
            for h in hotels:
                name = (h.get("name") or "").strip()
                if name and name not in seen_names:
                    seen_names.add(name)
                    mcp_hotel_names.append(name)
            day_closest.append((d, hotels[0]))

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
{location_hint}{brand_hint}{budget_hint}{price_range_hint}{other_hint}
{kb_hint}
{mcp_data_section}
{hotel_name_constraint}
【字段填写铁律 — 必须严格遵守】
1. 所有字段值必须有实际依据：hotel_name、price_range、distance_info 均须来自上方 MCP 数据。
2. 若某字段在 MCP 数据中未提供（如 star、highlights 等），JSON 中必须填写 null，
   绝对禁止填写"无"、"暂无"、"数据未提及"、"未知"等字符串。
3. data_source 字段：若酒店数据含 Tuniu 字样则填 "mcp_two_stage"，否则填 "mcp_amap_only"，
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
            "hotel_id": "酒店ID（Tuniu 数据提供时填写，否则填null）",
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
            "price_per_night": 680,
            "reason": "推荐理由（引用 distance_to_center 和真实价格，不得编造）",
            "stay_strategy": "连住 或 换酒店"
        }}
    ],
    "daily_tier_options": [
        {{
            "day": 1,
            "high": {{"hotel_name": "该天高价选项（来自 options，无真实价格则填 null）", "price_per_night": 800, "area": "区域名"}},
            "mid":  {{"hotel_name": "该天中档选项（来自 options，无真实价格则填 null）", "price_per_night": 400, "area": "区域名"}},
            "low":  {{"hotel_name": "该天低价选项（来自 options，无真实价格则填 null）", "price_per_night": null, "area": "区域名"}}
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

            # 后处理：确保每天至少有一条 Amap 酒店进入 options
            # 当某天 Tuniu 全部未匹配（0/N）时，LLM 可能因无真实价格而跳过该天
            # 使用上一步预计算的 day_closest（无需再次遍历 per_day_results）
            if day_closest:
                options_list = result.get("options") or []
                option_names = {
                    (o.get("hotel_name") or "").strip() for o in options_list
                }
                for d, closest in day_closest:
                    day_hotel_names = {
                        (h.get("name") or "").strip() for h in d["hotels"]
                    }
                    if day_hotel_names & option_names:
                        continue  # 该天已有酒店在 options 中
                    name = (closest.get("name") or "").strip()
                    if not name:
                        continue
                    options_list.append({
                        "tier": "经济型",
                        "hotel_name": name,
                        "hotel_id": None,
                        "area": closest.get("address", ""),
                        "price_range": None,
                        "star": None,
                        "highlights": None,
                        "distance_info": closest.get("distance_to_center"),
                        "data_source": "mcp_amap_only",
                    })
                    option_names.add(name)
                    logger.info(f"[AccommodationAgent] 补充兜底酒店: Day {d['day']} -> '{name}'")
                result["options"] = options_list

            # 构建酒店名 → {address, location} 索引（供 timeline / 地图兜底使用）
            # 数据源：per_day_results 的 merged hotel（Amap 提供 address + location）；
            # 城市级兜底走 hotel_results。同名酒店以首次出现为准。
            address_by_hotel: Dict[str, Dict[str, Any]] = {}
            _hotel_pool: List[Dict] = []
            if per_day_results:
                for d in per_day_results:
                    _hotel_pool.extend(d.get("hotels") or [])
            elif hotel_results:
                _hotel_pool.extend(hotel_results)
            for h in _hotel_pool:
                name = (h.get("name") or h.get("tuniu_name") or "").strip()
                if not name or name in address_by_hotel:
                    continue
                entry: Dict[str, Any] = {}
                addr = (h.get("address") or "").strip()
                if addr:
                    entry["address"] = addr
                loc = (h.get("location") or "").strip()
                if loc:
                    entry["location"] = loc  # "lng,lat"
                if entry:
                    address_by_hotel[name] = entry
            if address_by_hotel:
                result["address_by_hotel"] = address_by_hotel

            # 汇总住宿总成本：各天 daily_suggestions 的 price_per_night 之和
            daily_sugg = result.get("daily_suggestions") or []
            estimated_accommodation_total: float | None = None
            prices = [
                s["price_per_night"]
                for s in daily_sugg
                if isinstance(s.get("price_per_night"), (int, float))
            ]
            if prices:
                estimated_accommodation_total = float(sum(prices))

            return {
                "accommodation_plan":            result,
                "mcp_hotels_count":              len(hotel_results),
                "daily_centers_used":            len(daily_centers),
                # 双源数据覆盖的天数（= per_day_results 长度，已过滤 Amap 空天）
                "dual_source_days":              len(per_day_results),
                "daily_tier_options":            result.get("daily_tier_options", []),
                "estimated_accommodation_total": estimated_accommodation_total,
                "downgrade_level":               input_data.get("downgrade_level", 0),
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
