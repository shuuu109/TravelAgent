"""
意图识别节点 intent_node (P1)
职责：把用户自然语言查询转成结构化的意图分类、关键实体、住宿偏好与景点搜索提示词，
      为下游静态 fan-out（rag_node / transport_node / poi_fetch_node 等）提供
      路由依据与上下文。

设计要点（相比 LLM 动态调度版本）：
  - 不再产出 agent_schedule；分支由 intent_type 字段静态确定
  - intent_type 五值枚举：
        planning         — 行程规划主链路（fan-out → P3 → P4 → P5）
        preference_only  — 仅更新偏好（→ preference_node → respond）
        memory_only      — 仅查询用户历史记忆
        info_only        — 仅查询客观信息
        unknown          — 兜底（→ respond）
  - W mirror：当用户在同一句中表达偏好语气（"我喜欢/偏好/习惯..."）且 LLM 提取到
    accommodation_prefs.brand_keywords 时，立即同步到 long_term.preferences["hotel_brands"]，
    既让本轮 P4 消费 brand_keywords，又使下次会话能跨轮命中。
"""
import json
import logging
import re
from datetime import datetime, timedelta
from typing import Any

from langchain_core.messages import BaseMessage
from utils.skill_loader import SkillLoader
from utils.date_resolver import resolve_date_in_entities, resolve_relative_date
from graph.state import TravelGraphState

logger = logging.getLogger(__name__)


# 5 值意图分类，使路由判断有限确定
_VALID_INTENT_TYPES = {"planning", "preference_only", "memory_only", "info_only", "unknown"}

# 偏好语气词（W mirror 触发条件之一）：用户在自述长期偏好
_PREFERENCE_TONE_PATTERN = re.compile(
    r"我喜欢|我偏好|我习惯|我常住|我常坐|我喜爱|我钟爱|我比较喜欢|我经常住|我一般住"
)


def create_intent_node(llm, memory_manager=None):
    """
    工厂函数：将 LLM 与 memory_manager 通过闭包注入。

    Args:
        llm: LangChain ChatOpenAI 实例
        memory_manager: MemoryManager 实例（可选）。提供时启用 W mirror，
            将住宿品牌偏好同步到长期记忆。

    Returns:
        async 节点函数 intent_node(state) -> dict
    """
    skill_loader = SkillLoader()

    async def intent_node(state: TravelGraphState) -> dict:
        messages: list[BaseMessage] = state.get("messages", [])
        if not messages:
            return {
                "intent_data": {},
                "intent_type": "unknown",
                "user_query": "",
            }

        # ── 提取用户 query 与历史对话 ─────────────────────────────────
        user_query: str = messages[-1].content if messages else ""
        history_msgs = messages[:-1]

        conversation_history: list[str] = []
        for msg in history_msgs:
            if hasattr(msg, "content") and hasattr(msg, "type"):
                msg_type = msg.type  # 'human' / 'ai' / 'system'
                if msg_type == "system":
                    conversation_history.append(f"[系统记忆]\n{msg.content}")
                else:
                    role_name = "用户" if msg_type == "human" else "助手"
                    content = msg.content[:800] if len(msg.content) > 800 else msg.content
                    if len(msg.content) > 800:
                        content += "..."
                    conversation_history.append(f"{role_name}: {content}")

        system_memory: str | None = None
        dialogue_history: list[str] = []
        for item in conversation_history:
            if item.startswith("[系统记忆]"):
                system_memory = item
            else:
                dialogue_history.append(item)

        context_parts: list[str] = []
        if system_memory:
            context_parts.append(system_memory)
        if dialogue_history:
            context_parts.extend(dialogue_history)
        context_str = "\n".join(context_parts) if context_parts else "无历史对话"

        # ── 当前时间 + 相对日期示例 ───────────────────────────────────
        now = datetime.now()
        current_time = now.strftime("%Y年%m月%d日 %H:%M")
        weekday = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"][now.weekday()]
        _tomorrow = (now + timedelta(days=1)).strftime("%Y-%m-%d")
        _day_after = (now + timedelta(days=2)).strftime("%Y-%m-%d")
        _days_to_next_mon = (7 - now.weekday()) % 7 or 7
        _next_mon = (now + timedelta(days=_days_to_next_mon)).strftime("%Y-%m-%d")
        _next_sat = (now + timedelta(days=_days_to_next_mon + 5)).strftime("%Y-%m-%d")

        # ── Prompt 构建：intent_type 五值 + key_entities + 提示词 ──────
        prompt = f"""你是一个高级意图识别专家（IntentionRecognitionAgent）。请分析用户查询，识别意图并输出结构化决策。

【当前时间】
{current_time} {weekday}
（重要：当用户使用相对时间表达时，必须根据当前时间计算并输出具体日期，示例：
  - "明天"   → {_tomorrow}
  - "后天"   → {_day_after}
  - "下周六" → {_next_sat}（下一个周六）
  - "下周一" → {_next_mon}（下一个周一）
  - "3月15日" → 推断年份后输出完整 YYYY-MM-DD
  [WARNING] key_entities.date 字段必须输出具体的 YYYY-MM-DD 格式，
     禁止输出"下周六"、"后天"等模糊表达，这些会导致下游解析失败！）

【用户Query】
{user_query}

【对话历史上下文】
{context_str}

【意图分类（intent_type）必须从以下 5 个值中选择一个】
- "planning"        : 用户在规划未来行程（包含目的地+天数/日期/出发地等关键要素），如"我想去北京玩3天"
- "preference_only" : 用户**仅**在告知长期偏好且无任何规划要素，如"我喜欢全季和桔子酒店"、"我家在上海"
- "memory_only"     : 用户在询问自己的历史，如"我去过北京吗"、"我之前最常住哪种酒店"
- "info_only"       : 用户在询问与自己历史无关的客观信息，如"北京有什么好玩的"
- "unknown"         : 以上都不符合
重要规则：
  - 当用户同时表达"规划+偏好"时（如"我去北京玩3天，我喜欢全季"），intent_type="planning"，
    accommodation_prefs.brand_keywords 中提取品牌；规划链路本轮即可消费该偏好。
  - 当用户使用"我去过/我之前/我的"等带历史指向的表达时优先 memory_only。

【输出格式要求】
直接输出以下JSON，不要有其他文本：

{{
    "reasoning": "一句话说明意图识别依据",
    "intent_type": "5 值之一",
    "intents": [
        {{
            "type": "意图类型（如：itinerary_planning, preference_collection, information_query等，仅作辅助说明）",
            "confidence": 0.95,
            "description": "该意图的具体说明",
            "reason": "为什么识别出该意图的原因"
        }}
    ],

    "key_entities": {{
        "origin": "出发地（如果有）",
        "destination": "目的地（如果有）",
        "date": "日期（如果有）",
        "duration": "时长（如果有）",
        "pax": "出行人数（纯数字，如无提及则不填）",
        "budget": "预算金额（纯数字，人民币元，如有提及则填，例如用户说'预算5000'则填5000）",
        "budget_type": "预算类型，仅在填写了budget时填写：'人均' 表示每人预算，'总额' 表示所有人的总预算；若用户说'人均500'→'人均'，若说'总共5000'或'3个人共5000'→'总额'；若无法判断则不填",
        "other": "其他关键信息"
    }},

    "travel_style": "旅行风格，必填，从以下选项中选择一个：亲子（带孩子/家庭出游/亲子游）、老人（带老人/腿脚不便/轻松养生）、情侣（两人世界/蜜月/情侣/约会）、特种兵（特种兵/打卡/高效/多景点）、普通（默认值，未明确说明时使用）",

    "attraction_hints": [
        "根据用户原始需求生成2-4条高德地图景点搜索关键词，每条格式为'城市+景点类型或具体景点名'。",
        "覆盖用户提到的具体兴趣点（如大熊猫则写'成都 大熊猫基地'）。",
        "**严禁**包含：酒店/宾馆/连锁/民宿（属于住宿，写入 accommodation_prefs）；",
        "餐厅/美食/小吃/火锅（属于餐饮，由周边搜索覆盖）；地铁/机场/车站/打车（交通设施）。",
        "若用户提到上述住宿/餐饮词，**只**写入对应的 accommodation_prefs 字段，**不要**出现在 attraction_hints 中。",
        "仅在 destination 已知且 intent_type 为 planning 时填写，否则返回空列表 []。"
    ],

    "accommodation_prefs": {{
        "brand_keywords": "用户提到的住宿品牌/连锁词列表，如['连锁','汉庭','7天']；无则空列表 []",
        "type":           "住宿类型，从 ['连锁','经济','豪华','民宿',''] 中选一个，未提及填空字符串 ''",
        "price_range":    "若用户给出住宿单价区间填字符串如'300-500元/晚'，未提及则 null"
    }},

    "rewritten_query": "标准化、补全后的查询内容"
}}

请开始分析，直接输出JSON：
"""

        # ── 调用 LLM ─────────────────────────────────────────────────
        try:
            messages_list = [
                {"role": "system", "content": "你是一个高级意图识别专家。只输出JSON格式的结果，不要输出其他文本。"},
                {"role": "user", "content": prompt},
            ]
            response = await llm.ainvoke(messages_list)
            text = response.content.strip()

            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

            try:
                result = json.loads(text)
            except json.JSONDecodeError as e1:
                start_idx = text.find("{")
                end_idx = text.rfind("}")
                if start_idx != -1 and end_idx != -1:
                    json_str = text[start_idx:end_idx + 1]
                    try:
                        result = json.loads(json_str)
                    except json.JSONDecodeError as e2:
                        logger.error(f"JSON parse failed. Text sample: {json_str[:100]}")
                        raise ValueError(f"Failed to parse JSON. Error: {e2}")
                else:
                    raise ValueError(f"No JSON found in response. Parse error: {e1}")

        except Exception as e:
            logger.error(f"Intent recognition failed: {e}")
            result = _build_fallback_from_query(user_query)

        # ── 后处理 ────────────────────────────────────────────────────
        result = _ensure_travel_style(user_query, result)
        intent_type = _normalize_intent_type(result, user_query)
        result["intent_type"] = intent_type

        # 相对日期解析
        if "key_entities" in result and isinstance(result["key_entities"], dict):
            result["key_entities"] = resolve_date_in_entities(result["key_entities"])

        # attraction_hints：过滤住宿/餐饮/交通词；缺失时按 travel_style 模板兜底
        llm_hints = [
            h for h in result.get("attraction_hints", [])
            if isinstance(h, str) and h.strip() and not _looks_like_non_attraction(h)
        ]
        attraction_hints = llm_hints or _build_fallback_attraction_hints(result)
        result["attraction_hints"] = attraction_hints

        # accommodation_prefs：归一化
        accommodation_prefs = _normalize_accommodation_prefs(result.get("accommodation_prefs"))
        result["accommodation_prefs"] = accommodation_prefs

        # ── W mirror：偏好语气词 + 品牌词 → 同步长期记忆 ─────────────────
        # 适用场景：用户在 planning/preference_only 任意分支说"我喜欢XXX酒店"，
        # 把品牌写入 long_term.preferences['hotel_brands']，使下次会话能直接命中。
        if memory_manager and accommodation_prefs.get("brand_keywords"):
            if _PREFERENCE_TONE_PATTERN.search(user_query):
                _mirror_brands_to_long_term(
                    accommodation_prefs["brand_keywords"], memory_manager
                )

        # ── CityKnowledgeDB 补充结构化字段（季节/枢纽）─────────────────
        destination = (result.get("key_entities") or {}).get("destination", "")
        best_season: str = ""
        transport_hubs: list = []
        if destination:
            try:
                from utils.knowledge_parser import CityKnowledgeDB
                _kb = CityKnowledgeDB.get_instance()
                best_season = _kb.get_best_season(destination)
                transport_hubs = _kb.get_transport_hubs(destination)
                if best_season:
                    logger.info(f"[intent_node] {destination} 最佳季节: {best_season}")
                if transport_hubs:
                    logger.info(f"[intent_node] {destination} 交通枢纽: {transport_hubs}")
            except Exception as e:
                logger.warning(f"[intent_node] CityKnowledgeDB 查询失败，跳过: {e}")

        return {
            "intent_data": result,
            "intent_type": intent_type,
            "user_query": user_query,
            "travel_style": result.get("travel_style", "普通"),
            "travel_days": _parse_travel_days(result),
            "attraction_hints": attraction_hints,
            "accommodation_prefs": accommodation_prefs,
            "destination_best_season": best_season,
            "destination_transport_hubs": transport_hubs,
        }

    return intent_node


# =============================================================================
# 辅助函数
# =============================================================================

def _ensure_travel_style(user_query: str, result: dict) -> dict:
    """兜底：travel_style 缺失或非法时按关键词推断，最终默认 '普通'。"""
    valid_styles = {"亲子", "老人", "情侣", "特种兵", "普通"}
    current = result.get("travel_style", "")
    if current in valid_styles:
        return result

    if re.search(r"带孩子|家庭出游|亲子游|亲子", user_query):
        result["travel_style"] = "亲子"
    elif re.search(r"带老人|腿脚不便|轻松养生|老年人", user_query):
        result["travel_style"] = "老人"
    elif re.search(r"两人世界|蜜月|情侣|约会", user_query):
        result["travel_style"] = "情侣"
    elif re.search(r"特种兵|高效.*景点|多景点|打卡", user_query):
        result["travel_style"] = "特种兵"
    else:
        result["travel_style"] = "普通"
    return result


def _normalize_intent_type(result: dict, user_query: str) -> str:
    """
    intent_type 归一化：
      - LLM 输出已是 5 值之一直接返回
      - 否则结合 intents 列表 + 关键词做确定性兜底
    """
    raw = str(result.get("intent_type", "")).strip().lower()
    if raw in _VALID_INTENT_TYPES:
        return raw

    intents = result.get("intents", []) or []
    intent_types = {str(i.get("type", "")).lower() for i in intents}

    # planning 关键意图
    if intent_types & {"itinerary_planning", "plan_trip", "trip_planning"}:
        return "planning"
    # 偏好类
    if intent_types & {"preference", "preference_collection"}:
        return "preference_only"
    # 记忆类
    if intent_types & {"memory_query", "history_query"}:
        return "memory_only"
    # 信息查询类
    if "information_query" in intent_types:
        return "info_only"

    # 关键词兜底
    key_entities = result.get("key_entities") or {}
    if key_entities.get("destination") and (key_entities.get("date") or key_entities.get("duration")):
        return "planning"
    if re.search(r"我去过|我之前|我的", user_query):
        return "memory_only"
    if _PREFERENCE_TONE_PATTERN.search(user_query):
        return "preference_only"
    return "unknown"


def _mirror_brands_to_long_term(brand_keywords: list, memory_manager) -> None:
    """
    W mirror：把当前 query 提取的住宿品牌词同步到 long_term.preferences['hotel_brands']。
    采用追加 + 去重，避免覆盖历史品牌偏好。
    """
    try:
        current = memory_manager.long_term.get_preference()
        existing = current.get("hotel_brands")
        if isinstance(existing, list):
            merged = list(existing)
        elif existing:
            merged = [existing]
        else:
            merged = []
        added: list[str] = []
        for b in brand_keywords:
            if b and b not in merged:
                merged.append(b)
                added.append(b)
        if added:
            memory_manager.long_term.save_preference("hotel_brands", merged)
            logger.info(f"[intent_node] W mirror 追加 hotel_brands: {added} → {merged}")
    except Exception as e:
        logger.warning(f"[intent_node] W mirror 同步失败: {e}")


def _build_fallback_from_query(user_query: str) -> dict:
    """
    LLM 调用失败时，从 user_query 用正则提取关键实体，构建兜底意图结果。
    不再产出 agent_schedule —— 路由由 intent_type 决定。
    """
    # ── 出发地 ──
    origin = ""
    for pat in [
        r"从([^\s，,出去到]{1,6})(?:出发|启程|乘|坐)",
        r"([^\s，,]{1,6})出发",
    ]:
        m = re.search(pat, user_query)
        if m:
            origin = m.group(1).strip()
            break

    # ── 目的地 ──
    destination = ""
    for pat in [
        r"去([^\s，,。！？出]{1,6})(?:玩|旅游|旅行|游|参观|看)",
        r"到([^\s，,。！？]{1,6})(?:玩|旅游|旅行|游|参观|看)",
        r"去([^\s，,。！？出]{1,6})(?:\s|，|,)",
    ]:
        m = re.search(pat, user_query)
        if m:
            destination = m.group(1).strip()
            break

    # ── 行程天数 ──
    days_m = re.search(r"(\d+)\s*[天日]", user_query)
    duration = f"{days_m.group(1)}天" if days_m else ""

    # ── 出行日期 ──
    date_m = re.search(
        r"(下下周[一二三四五六日天]|下周[一二三四五六日天]|本周[一二三四五六日天]|这周[一二三四五六日天]"
        r"|今天|明天|后天|大后天"
        r"|周[一二三四五六日天]"
        r"|\d{4}年\d{1,2}月\d{1,2}日|\d{1,2}月\d{1,2}[日号])",
        user_query,
    )
    raw_date = date_m.group(1) if date_m else ""
    date = resolve_relative_date(raw_date) or raw_date

    has_destination = bool(destination)
    travel_kw = re.compile(r"行程|规划|旅游|旅行|游玩|出游|出发|去.{1,6}玩|游记|安排|攻略")
    is_travel = bool(travel_kw.search(user_query)) or (bool(origin) and has_destination)

    if is_travel:
        intent_type = "planning"
    elif _PREFERENCE_TONE_PATTERN.search(user_query):
        intent_type = "preference_only"
    elif re.search(r"我去过|我之前|我的", user_query):
        intent_type = "memory_only"
    else:
        intent_type = "info_only"

    intents = [{
        "type": {
            "planning": "itinerary_planning",
            "preference_only": "preference_collection",
            "memory_only": "memory_query",
            "info_only": "information_query",
        }.get(intent_type, "unknown"),
        "confidence": 0.5,
        "description": "LLM 降级，正则兜底",
        "reason": "LLM 调用失败",
    }]

    logger.info(
        f"_build_fallback_from_query: intent_type={intent_type!r}, "
        f"origin={origin!r}, destination={destination!r}, duration={duration!r}"
    )

    return {
        "reasoning": (
            f"LLM 调用失败，正则兜底提取："
            f"intent_type={intent_type!r}, origin={origin!r}, destination={destination!r}, "
            f"duration={duration!r}, date={date!r}"
        ),
        "intent_type": intent_type,
        "intents": intents,
        "key_entities": {
            "origin": origin or None,
            "destination": destination or None,
            "date": date or None,
            "duration": duration or None,
        },
        "travel_style": "普通",
        "rewritten_query": user_query,
        "attraction_hints": [],
        "accommodation_prefs": {},
    }


# 各旅行风格的兜底景点搜索关键词模板
_FALLBACK_HINTS_MAP: dict[str, list[str]] = {
    "亲子":  ["{city}亲子景点 儿童乐园", "{city}动物园 博物馆 科技馆", "{city}主题公园 游乐场"],
    "老人":  ["{city}休闲景点 公园", "{city}历史文化 寺庙", "{city}园林 古迹"],
    "情侣":  ["{city}浪漫景点 网红打卡", "{city}古镇 观景台", "{city}艺术馆 美术馆"],
    "特种兵":["{city}必去景点 热门景区", "{city}古迹 博物馆", "{city}网红景点 打卡地"],
    "普通":  ["{city}著名景点", "{city}历史文化 博物馆", "{city}风景名胜 公园"],
}

# 用于过滤 LLM 在 attraction_hints 里偶发混入的住宿/餐饮/交通词
_NON_ATTRACTION_KEYWORDS: tuple[str, ...] = (
    "酒店", "宾馆", "民宿", "连锁", "客栈", "旅馆",
    "餐厅", "美食", "小吃", "火锅",
    "机场", "地铁", "高铁", "火车站", "汽车站", "打车",
)


def _looks_like_non_attraction(hint: str) -> bool:
    return any(kw in hint for kw in _NON_ATTRACTION_KEYWORDS)


def _build_fallback_attraction_hints(result: dict) -> list[str]:
    city: str = (result.get("key_entities") or {}).get("destination", "") or ""
    if not city:
        return []
    style: str = result.get("travel_style", "普通")
    templates = _FALLBACK_HINTS_MAP.get(style, _FALLBACK_HINTS_MAP["普通"])
    return [t.format(city=city) for t in templates]


def _normalize_accommodation_prefs(raw: Any) -> dict:
    """归一化 accommodation_prefs 为 {brand_keywords, type, price_range}。"""
    if not isinstance(raw, dict):
        return {"brand_keywords": [], "type": "", "price_range": None}

    brands_raw = raw.get("brand_keywords", [])
    brand_keywords = (
        [b for b in brands_raw if isinstance(b, str) and b.strip()]
        if isinstance(brands_raw, list) else []
    )

    type_raw = raw.get("type", "")
    valid_types = {"连锁", "经济", "豪华", "民宿", ""}
    acc_type = type_raw if isinstance(type_raw, str) and type_raw in valid_types else ""

    price_range = raw.get("price_range")
    if not (isinstance(price_range, str) and price_range.strip()):
        price_range = None

    return {"brand_keywords": brand_keywords, "type": acc_type, "price_range": price_range}


def _parse_travel_days(result: dict) -> int:
    duration = (result.get("key_entities") or {}).get("duration") or ""
    m = re.search(r"(\d+)", str(duration))
    return int(m.group(1)) if m else 0
