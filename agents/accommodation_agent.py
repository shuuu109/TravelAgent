"""
住宿专家智能体 AccommodationAgent
职责：根据目的地、到达交通枢纽和用户偏好，调用 RollingGo MCP 查询真实酒店数据，
      再由 LLM 对结果进行分析和个性化推荐。

核心逻辑：
1. 从 Orchestrator 传入的上下文中提取目的地、到达车站/机场、入住日期等信息
2. 如果前序有 transport_query 结果，优先使用其 arrival_station 作为住宿选址的锚点
3. 结合用户偏好（酒店品牌、预算等级）调用 MCP searchHotels 获取真实数据
4. LLM 对搜索结果进行二次分析，输出结构化的住宿方案 JSON
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class AccommodationAgent:
    def __init__(self, name: str = "AccommodationAgent", model=None, **kwargs):
        self.name = name
        self.model = model

    # ──────────────────────────────────────────────────────────
    # 内部辅助：从前序 transport_query 结果提取到达枢纽信息
    # ──────────────────────────────────────────────────────────

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

    # ──────────────────────────────────────────────────────────
    # 内部辅助：调用 RollingGo MCP 搜索真实酒店
    # ──────────────────────────────────────────────────────────

    async def _search_hotels_via_mcp(
        self,
        destination: str,
        check_in_date: str | None,
        stay_nights: int,
        adults: int,
        hotel_brands: List[str],
        budget_level: str,
    ) -> List[Dict]:
        """
        调用 mcp_clients.hotel_client.search_hotels 获取真实酒店数据。
        如果 MCP 不可用，返回空列表（后续降级为纯 LLM 推荐）。
        """
        try:
            from mcp_clients.hotel_client import search_hotels
            from config import ROLLINGGO_MCP_CONFIG

            # 预算等级 → 价格区间映射
            budget_map = {
                "经济": (None, 300),
                "经济型": (None, 300),
                "舒适": (200, 600),
                "舒适型": (200, 600),
                "高端": (500, None),
                "高端型": (500, None),
                "豪华": (1000, None),
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
            )

            # MCP 返回结构为 CallToolResult，取 content[0].text
            if hasattr(raw, "content") and raw.content:
                text = raw.content[0].text if hasattr(raw.content[0], "text") else str(raw.content[0])
                logger.info(f"RollingGo MCP raw response (first 300 chars): {text[:300]}")
                hotels = json.loads(text)
                if isinstance(hotels, list):
                    return hotels
                if isinstance(hotels, dict):
                    candidates = (
                        hotels.get("hotelInformationList")
                        or hotels.get("hotels")
                        or hotels.get("data")
                    )
                    if candidates is not None:
                        return candidates if isinstance(candidates, list) else [candidates]
                    # 若整个 dict 不含已知的酒店列表字段，说明是错误/元数据响应，不包装为假酒店
                    logger.warning(f"RollingGo MCP returned unexpected dict structure, keys: {list(hotels.keys())}")
                    return []
            return []

        except ValueError as e:
            # API Key 未配置
            logger.warning(f"RollingGo MCP Key 未配置，降级为纯 LLM 推荐: {e}")
            return []
        except Exception as e:
            logger.warning(f"RollingGo MCP 调用失败，降级为纯 LLM 推荐: {e}")
            return []

    # ──────────────────────────────────────────────────────────
    # 主入口
    # ──────────────────────────────────────────────────────────

    async def run(self, input_data: dict) -> dict:
        context = input_data.get("context", {})
        key_entities = context.get("key_entities", {})
        previous_results = input_data.get("previous_results", [])

        # ── 基础信息提取 ──
        destination = key_entities.get("destination", "")
        date = key_entities.get("date", "")
        duration = key_entities.get("duration", "")
        adults = int(key_entities.get("adults", 1))

        # 计算入住晚数
        stay_nights = 1
        if duration:
            try:
                stay_nights = int("".join(filter(str.isdigit, str(duration)))) or 1
            except Exception:
                stay_nights = 1

        # 从前序交通结果获取到达枢纽
        transport_info = self._extract_transport_info(previous_results)
        arrival_station = transport_info.get("arrival_station", "")

        # 兜底：从 context 读取 orchestrate_node 注入的交通信息
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
        if not destination:
            destination = transport_info.get("destination", "")
        if not date:
            date = transport_info.get("date", "")

        if not destination:
            return {"error": "缺少目的地信息，无法推荐住宿"}

        # ── 用户偏好 ──
        user_preferences = context.get("user_preferences", {})
        raw_brands = user_preferences.get("hotel_brands", [])
        # 兼容字符串格式（如"万豪酒店和希尔顿"）和列表格式
        if isinstance(raw_brands, str):
            hotel_brands: List[str] = [b.strip() for b in raw_brands.replace("和", ",").replace("、", ",").split(",") if b.strip()]
        else:
            hotel_brands: List[str] = raw_brands or []
        budget_level: str = user_preferences.get("budget_level", "")
        other_prefs: Dict = user_preferences.get("other_preferences", {})

        # ══════════════════════════════════════════════════════
        # Step A：调用 RollingGo MCP 搜索真实酒店
        # ══════════════════════════════════════════════════════
        check_in_date = date if date else None
        hotel_results = await self._search_hotels_via_mcp(
            destination=destination,
            check_in_date=check_in_date,
            stay_nights=stay_nights,
            adults=adults,
            hotel_brands=hotel_brands,
            budget_level=budget_level,
        )

        mcp_data_section = ""
        if hotel_results:
            try:
                mcp_data_section = f"""
【真实酒店数据（来自 RollingGo MCP，共 {len(hotel_results)} 条）】
{json.dumps(hotel_results, ensure_ascii=False, indent=2)}

请基于以上真实数据进行分析和推荐，优先使用这些真实酒店（含真实价格、位置、星级），
不要虚构酒店名称或价格。如数据不足，可补充合理的通用建议。
"""
            except Exception:
                mcp_data_section = ""

        if not mcp_data_section:
            mcp_data_section = "【注意】当前无真实酒店数据，请基于你的知识给出合理推荐，并注明价格为估算。"

        # ══════════════════════════════════════════════════════
        # Step B：LLM 对数据进行分析，生成结构化推荐
        # ══════════════════════════════════════════════════════
        location_hint = ""
        if arrival_station:
            location_hint = f"\n【到达交通枢纽】用户将抵达 {arrival_station}，请优先推荐该枢纽附近酒店。"

        brand_hint = ""
        if hotel_brands:
            brand_hint = f"\n用户偏好品牌: {'、'.join(hotel_brands)}"

        budget_hint = f"\n用户预算等级: {budget_level}" if budget_level else ""

        other_hint = ""
        if other_prefs:
            lines = [f"  - {k}: {v}" for k, v in other_prefs.items() if v]
            if lines:
                other_hint = "\n其他偏好:\n" + "\n".join(lines)

        prompt = f"""你是一个专业的住宿推荐专家（AccommodationAgent）。
请为用户在【{destination}】的住宿提供分析和推荐。

【入住信息】
- 目的地: {destination}
- 入住日期: {date or '未指定'}
- 行程时长: {duration or '未指定'}（约 {stay_nights} 晚）
- 成人人数: {adults}
{location_hint}{brand_hint}{budget_hint}{other_hint}

{mcp_data_section}

【输出格式要求】
请严格输出以下JSON格式，不要包含任何其他文本：
{{
    "destination": "{destination}",
    "arrival_station": "{arrival_station or '未知'}",
    "mcp_data_used": {"true" if hotel_results else "false"},
    "analysis": "住宿选址分析（结合到达枢纽和目的地情况）",
    "recommended_areas": [
        {{
            "area_name": "推荐区域名称",
            "reason": "推荐理由",
            "distance_to_station": "距到达枢纽的距离/交通时间"
        }}
    ],
    "options": [
        {{
            "tier": "档次（经济型/舒适型/高端型）",
            "hotel_name": "酒店名称",
            "hotel_id": "酒店ID（若有MCP数据则填写，否则填null）",
            "area": "所在区域",
            "price_range": "每晚价格区间（CNY）",
            "star": "星级",
            "highlights": "亮点（含早餐、地铁直达等）",
            "distance_info": "距枢纽/景点距离",
            "pros": "优点",
            "cons": "缺点"
        }}
    ],
    "recommendation": {{
        "best_choice": "综合最推荐的酒店/区域",
        "reason": "推荐理由",
        "booking_tips": "预订建议"
    }}
}}
"""

        try:
            messages = [
                {"role": "system", "content": "你是一个住宿推荐专家。只输出JSON，不含任何额外文本。"},
                {"role": "user", "content": prompt},
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
                "accommodation_plan": result,
                "mcp_hotels_count": len(hotel_results),   # 透传 MCP 原始数量
            }

        except Exception as e:
            logger.error(f"AccommodationAgent LLM failed: {e}")
            # 降级：直接返回 MCP 原始数据（如有）
            if hotel_results:
                return {
                    "accommodation_plan": {
                        "destination": destination,
                        "mcp_data_used": True,
                        "raw_hotels": hotel_results,
                        "analysis": "LLM 分析失败，以下为 MCP 原始酒店数据",
                    },
                    "mcp_hotels_count": len(hotel_results),
                }
            return {"error": str(e)}
