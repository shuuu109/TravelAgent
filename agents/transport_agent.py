"""
交通专家智能体 TransportAgent
职责：通过 12306 MCP 查询真实车次数据，结合 LLM 进行比价分析和推荐。
"""
import asyncio
import json
import logging

from mcp_clients.train_client import train_client
from mcp_clients.flight_client import flight_client
from utils.json_parser import robust_json_parse

logger = logging.getLogger(__name__)


class TransportAgent:
    def __init__(self, name: str = "TransportAgent", model=None, **kwargs):
        self.name = name
        self.model = model

    async def run(self, input_data: dict) -> dict:
        context = input_data.get("context", {})
        key_entities = context.get("key_entities", {})
        user_preferences = context.get("user_preferences", {})

        origin = key_entities.get("origin", "") or user_preferences.get("home_location", "")
        destination = key_entities.get("destination", "")
        date = key_entities.get("date", "")

        missing = []
        if not origin:
            missing.append("出发地")
        if not destination:
            missing.append("目的地")
        if missing:
            return {"error": f"缺少{'和'.join(missing)}，请补充后再查询。"}

        # =====================================================================
        # 日期格式标准化：将中文格式（2026年04月05日）转换为 YYYY-MM-DD
        # =====================================================================
        import re as _re
        def _normalize_date(d: str) -> str:
            if not d:
                return d
            # 已是标准格式
            if _re.match(r"^\d{4}-\d{2}-\d{2}", d):
                return d[:10]
            # 中文格式：2026年4月5日 / 2026年04月05日
            m = _re.search(r"(\d{4})[年/](\d{1,2})[月/](\d{1,2})[日号]?", d)
            if m:
                return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
            return d

        date = _normalize_date(date)

        # =====================================================================
        # 第一步：通过 12306 MCP 查询真实车次数据
        # =====================================================================
        transport_data_text = ""
        query_success = False
        try:
            if not date:
                from datetime import datetime, timedelta
                date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
                date_label = f"{date}（系统默认查询日期，用户未指定）"
            else:
                date_label = date

            # 并发查询火车、航班、目的地机场天气（三路并发）
            # flight_client.query_tickets() 内部已映射城市名→IATA三字码
            from mcp_clients.flight_client import CITY_TO_IATA
            arr_iata = CITY_TO_IATA.get(destination, destination)

            train_raw, train_price_raw, flight_raw, weather_raw = await asyncio.gather(
                train_client.query_tickets(date, origin, destination),
                train_client.query_ticket_price(date, origin, destination),
                flight_client.query_tickets(date, origin, destination),
                flight_client.get_airport_weather(arr_iata),
            )
            date = date_label

            # 格式化数据
            train_str       = train_raw       if isinstance(train_raw,       str) else json.dumps(train_raw,       ensure_ascii=False)
            train_price_str = train_price_raw if isinstance(train_price_raw, str) else json.dumps(train_price_raw, ensure_ascii=False)
            flight_str      = flight_raw      if isinstance(flight_raw,      str) else json.dumps(flight_raw,      ensure_ascii=False)
            weather_str     = weather_raw     if isinstance(weather_raw,     str) else json.dumps(weather_raw,     ensure_ascii=False)

            # 将火车、票价、航班、目的地天气数据一并喂给大模型
            transport_data_text = (
                f"【12306火车余票/车次数据】\n{train_str}\n\n"
                f"【12306火车票价数据（含各席别实时价格）】\n{train_price_str}\n\n"
                f"【航班数据】\n{flight_str}\n\n"
                f"【目的地（{destination}）天气预报】\n{weather_str}"
            )
            query_success = True
            logger.info(f"交通查询成功(包含火车+航班): {origin} → {destination}, date={date}")
        except Exception as e:
            logger.warning(f"交通查询失败: {e}")
            transport_data_text = f"交通查询失败: {str(e)}"

        # =====================================================================
        # 第二步：将真实数据交给 LLM 进行分析和推荐
        # =====================================================================
        if query_success:
            prompt = f"""你是一个专业的旅游大交通规划专家（TransportAgent）。
用户需要从【{origin}】到【{destination}】，日期【{date}】出行。

以下是查询到的真实车次和航班数据：
{transport_data_text}

【分析维度要求】
1. 从查询结果中综合筛选出最优的3-5个交通方案（包含高铁和飞机，综合考虑出发时间、耗时、票价、余位情况）。
2. 时间统筹：考虑"市区到高铁站/机场的接驳时间"+"候机/候车时间"，估算实际总耗时。飞机方案请务必考虑较高的提前到达机场和安检时间。
3. 性价比分析：结合时间成本和金钱成本，给出"最快方案"和"最具性价比方案"。
4. 如果上下文提及用户偏好（如带小孩/老人、偏好舒适），需调整推荐权重。
5. 天气提醒：根据目的地天气预报，在 weather_reminder 字段给出简洁提醒（如极端天气需特别强调对航班延误的影响）。

【输出格式要求】
请严格输出以下JSON格式，不要包含任何其他文本：
{{
    "query_info": {{
        "origin": "{origin}",
        "destination": "{destination}",
        "date": "{date}",
        "data_source": "实时火车与航班查询"
    }},
    "analysis": "基于真实火车和航班数据的综合分析，对比空铁优劣",
    "options": [
        {{
            "transport_type": "高铁/飞机",
            "transport_no": "车次号或航班号（如G1234 或 CA1234）",
            "departure_time": "出发时间（如08:00）",
            "arrival_time": "到达时间（如12:30）",
            "duration": "运行时长（如4小时30分）",
            "departure_hub": "出发车站/机场全称",
            "arrival_hub": "到达车站/机场全称",
            "price_range": "各席别/舱位价格和余票（如二等座¥553, 一等座¥887）",
            "is_recommended": false,
            "data_source": "realtime",
            "pros": "优点",
            "cons": "缺点"
        }}
    ],
    "recommendation": {{
        "fastest": "最快方案及理由",
        "best_value": "性价比最高方案及理由",
        "best_choice": "最终推荐方案",
        "arrival_hub": "到达枢纽名（如'上海虹桥站'或'浦东机场'），用于后续住宿推荐",
        "reason": "推荐理由"
    }},
    "weather_reminder": "目的地天气简报及出行提醒，如有极端天气需注明对航班的潜在影响"
}}
"""
        else:
            # 查询失败时，退化为 LLM 通用分析
            prompt = f"""你是一个专业的旅游大交通规划专家（TransportAgent）。
用户需要从【{origin}】到【{destination}】，日期【{date}】出行。

注意：实时交通查询暂时不可用（{transport_data_text}），请基于你的知识提供交通方案对比。

【分析维度要求】
1. 对比不同交通方式（高铁 vs 飞机 vs 自驾等）的优劣势。
2. 时间统筹：考虑"市区到机场/高铁站的接驳时间"+"安检/候车时间"，估算实际总耗时。
3. 性价比计算：综合时间成本和金钱成本，给出"最快方案"和"最具性价比方案"。

【输出格式要求】
请严格输出以下JSON格式，不要包含任何其他文本：
{{
    "query_info": {{
        "origin": "{origin}",
        "destination": "{destination}",
        "date": "{date}",
        "data_source": "LLM知识推断（实时查询不可用）"
    }},
    "analysis": "对不同交通方式的优劣势详细分析",
    "options": [
        {{
            "transport_type": "高铁/飞机/自驾",
            "transport_no": null,
            "departure_time": null,
            "arrival_time": null,
            "duration": "总耗时估算（如约5小时）",
            "departure_hub": "出发枢纽或城市",
            "arrival_hub": "到达枢纽或城市",
            "price_range": "预估价格区间（如¥300-600）",
            "is_recommended": false,
            "data_source": "llm",
            "pros": "优点",
            "cons": "缺点"
        }}
    ],
    "recommendation": {{
        "best_choice": "最终推荐的交通方式",
        "arrival_hub": "到达的交通枢纽（如'上海虹桥站'或'浦东机场'），用于后续住宿推荐",
        "reason": "推荐理由"
    }}
}}
"""

        try:
            messages = [
                {"role": "system", "content": "你是一个交通专家。只输出JSON。"},
                {"role": "user", "content": prompt}
            ]
            response = await self.model.ainvoke(messages)
            text = response.content

            # 使用更鲁棒的提取方式：截取第一个 { 到最后一个 } 之间的内容
            start_idx = text.find('{')
            end_idx = text.rfind('}')

            if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
                text = text[start_idx:end_idx+1]
            else:
                raise ValueError(f"无法从大模型回复中提取JSON结构。大模型原始回复截断: {text[:200]}")

            result = robust_json_parse(text)
            return {"transport_plan": result}

        except Exception as e:
            logger.error(f"TransportAgent failed: {e}")
            return {"error": str(e)}
