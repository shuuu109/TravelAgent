"""
意图识别节点 intent_node
职责：将 IntentionAgent.reply() 逻辑转换为 LangGraph 节点函数

改动点（相比 agents/intention_agent.py）：
- 函数签名：async def intent_node(state: TravelGraphState) -> dict
- 输入：从 state["messages"] 获取，无需 Msg 包装
- LLM 调用：await llm.ainvoke(messages_list) 返回 AIMessage，取 .content
- 输出：{"intent_data": result, "intent_schedule": ..., "user_query": ...}
- 使用工厂函数将 LLM 实例通过闭包注入
"""
import json
import logging
from datetime import datetime

from langchain_core.messages import BaseMessage
from utils.skill_loader import SkillLoader
from graph.state import TravelGraphState

logger = logging.getLogger(__name__)


def create_intent_node(llm):
    """
    工厂函数：将 LLM 实例通过闭包注入到节点函数中。

    Args:
        llm: LangChain ChatOpenAI 实例（或任意实现 ainvoke 的 LLM）

    Returns:
        async 节点函数 intent_node(state) -> dict
    """
    skill_loader = SkillLoader()

    async def intent_node(state: TravelGraphState) -> dict:
        """
        意图识别节点主流程：
        1. 从 state["messages"] 提取用户 query 和历史对话
        2. 构建意图识别 Prompt（含动态 skill 描述、当前时间、上下文）
        3. 调用 LLM（ainvoke）获取 JSON 结果
        4. 解析并返回 intent_data、intent_schedule、user_query
        """
        messages: list[BaseMessage] = state.get("messages", [])
        if not messages:
            return {
                "intent_data": {},
                "intent_schedule": [],
                "user_query": ""
            }

        # =====================================================================
        # 提取用户 query 和历史对话（复用 IntentionAgent 的拆分逻辑）
        # =====================================================================
        user_query: str = messages[-1].content if messages else ""
        history_msgs = messages[:-1]

        conversation_history = []
        for msg in history_msgs:
            if hasattr(msg, 'content') and hasattr(msg, 'type'):
                msg_type = msg.type  # 'human' / 'ai' / 'system'
                if msg_type == "system":
                    conversation_history.append(f"[系统记忆]\n{msg.content}")
                else:
                    role_name = "用户" if msg_type == "human" else "助手"
                    content = msg.content[:800] if len(msg.content) > 800 else msg.content
                    if len(msg.content) > 800:
                        content += "..."
                    conversation_history.append(f"{role_name}: {content}")

        # 分离长期记忆 vs 对话历史
        system_memory = None
        dialogue_history = []
        for item in conversation_history:
            if item.startswith("[系统记忆]"):
                system_memory = item
            else:
                dialogue_history.append(item)

        context_parts = []
        if system_memory:
            context_parts.append(system_memory)
        if dialogue_history:
            context_parts.extend(dialogue_history)
        context_str = "\n".join(context_parts) if context_parts else "无历史对话"

        # =====================================================================
        # 当前时间
        # =====================================================================
        current_time = datetime.now().strftime("%Y年%m月%d日 %H:%M")
        weekday = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"][datetime.now().weekday()]

        # =====================================================================
        # 动态获取 Skills 描述（与 IntentionAgent 完全一致）
        # =====================================================================
        skill_mapping = {
            "memory-query": "memory_query",
            "plan-trip": "itinerary_planning",
            "preference": "preference",
            "query-info": "information_query",
            "ask-question": "rag_knowledge",
            "event-collection": "event_collection",
            "transport-query": "transport_query",
            "accommodation-query": "accommodation_query"
        }
        dynamic_skills_prompt = skill_loader.get_skill_prompt(skill_mapping)

        # =====================================================================
        # 构建意图识别 Prompt（与 IntentionAgent.reply() 完全一致）
        # =====================================================================
        prompt = f"""你是一个高级意图识别专家（IntentionRecognitionAgent）。请分析用户查询，识别意图并输出结构化的决策。

【当前时间】
{current_time} {weekday}
（重要：当用户说"2月28日"或"明天"等相对时间时，请根据当前时间进行推断完整日期）

【用户Query】
{user_query}

【对话历史上下文】
{context_str}

【可调度的子智能体 (Skills)】
{dynamic_skills_prompt}

【重要 - 意图区分原则】
请基于语义理解判断意图，不要机械匹配关键词。同一个词在不同语境下可能对应不同意图：
- "我去过北京吗？" → memory_query（询问自己的历史）
- "北京怎么样？" / "北京有什么好玩的？" → information_query（询问客观信息）
- "我想去北京" → itinerary_planning（规划未来行程）

优先级规则：
- memory_query 优先于 information_query（当问题涉及用户自己的历史时）
- 如果用户明确询问"我的"、"我过去的"，必须识别为 memory_query
- 当发现用户有明确的**跨城移动**需求时，必须包含 transport_query 获取真实车次
- **【出发地缺失规则】** 如果用户表达了行程规划意图，但未提供明确的出发地（origin），且对话历史和系统记忆中也无法确定出发地，则**不得调度 transport_query**，应优先调度 event_collection 收集完整信息（包括出发地）；待出发地明确后，再由后续轮次调度 transport_query

【任务要求】
请按以下步骤进行分析：

**第1步：推理过程生成**
- 分析用户query的核心诉求
- 识别query中的关键实体和意图信号
- 判断是否需要结合对话历史进行消歧
- 说明如何融合上下文信息进行推理

**第2步：多意图识别（原因）**
- 识别所有可能的用户意图（可以是多个）
- 为每个意图分配置信度（0-1之间）
- 说明为什么识别出该意图的原因

**第3步：智能Query改写**
- 识别口语化表达，进行标准化
- 补全省略的上下文信息
- 提取和重组关键信息

**第4步：构建结构化决策**
- 基于识别的意图，决定调用哪些子智能体
- 说明调用顺序和优先级
- 输出结构化的调用策略

【输出格式要求】
必须严格按照以下JSON格式输出（**只输出JSON，不要有其他文本**）：

{{
    "reasoning": "这里是详细的推理过程，包含第1步的分析，说明如何理解用户query，如何结合上下文，如何识别意图信号",

    "intents": [
        {{
            "type": "意图类型（如：itinerary_planning, preference_collection, information_query等）",
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
        "other": "其他关键信息"
    }},

    "travel_style": "旅行风格，必填，从以下选项中选择一个：亲子（带孩子/家庭出游/亲子游）、老人（带老人/腿脚不便/轻松养生）、情侣（两人世界/蜜月/情侣/约会）、特种兵（特种兵/打卡/高效/多景点）、普通（默认值，未明确说明时使用）",

    "rewritten_query": "标准化、补全后的查询内容",

    "agent_schedule": [
        {{
            "agent_name": "子智能体名称",
            "priority": 1,
            "reason": "调用该智能体的原因和依据",
            "expected_output": "期望该智能体提供什么输出"
        }}
    ]
}}

【重要提示 - 优先级设置规则】
优先级数字相同的智能体会**并行执行**，不同优先级按顺序批次执行。

**所有智能体优先级分组：**

**Priority 1（并行执行）- 信息收集类：**
- memory_query: 记忆查询智能体
- event_collection: 事项收集智能体
- preference: 偏好管理智能体
- information_query: 信息查询智能体（联网搜索）
- rag_knowledge: RAG知识库智能体（检索本地旅游攻略知识库，获取目的地景点推荐、游览时长、注意事项等旅行攻略内容）
  【触发条件】必须同时满足：① destination 已知；② 意图为 itinerary_planning 或与旅游相关的 information_query
  【不触发】纯交通查询、偏好更新、历史记忆查询、destination 未知时
- transport_query: 大交通查询智能体（查12306车票、航班，必须在行程规划前执行）
- accommodation_query: 住宿推荐智能体（根据到达枢纽和偏好推荐酒店，依赖transport_query结果时放Priority 2）

**Priority 2（依赖 Priority 1）- 住宿查询类：**
- accommodation_query: 住宿推荐智能体（依赖 transport_query 的到达枢纽；若无 transport_query 则可放 Priority 1）

**Priority 2 或 3（依赖信息收集+住宿）- 行程规划类：**
- itinerary_planning: 行程规划智能体（需要事项收集的结果；若有 accommodation_query，必须在其之后执行，确保行程中酒店信息与推荐一致）

**说明：**
- Priority 1 的智能体都是信息获取，互不依赖，可并行执行提升速度
- accommodation_query 依赖 transport_query，必须在 transport_query 之后
- itinerary_planning 依赖 accommodation_query（若存在），必须在 accommodation_query 之后，否则行程里的酒店与推荐不一致
- 示例A：用户说"我要从天津去北京，喜欢住汉庭"（出发地明确）
  → Priority 1: preference + event_collection + transport_query（并行）
  → Priority 2: itinerary_planning（使用 Priority 1 的结果）

- 示例B：用户说"我想去北京"（出发地不明确，记忆中也无 home_location）
  → Priority 1: event_collection（收集出发地等缺失信息）
  → **不调度 transport_query**，因为没有出发地无法查询车次
  → Priority 2: itinerary_planning（待 event_collection 补全信息后再规划）

- 示例C：用户说"我后天从上海去北京旅游，帮我查下交通和住宿"（出发地明确，且显式要求查交通和住宿）
  → 必须包含 transport_query（用户明确要求查交通）
  → 必须包含 accommodation_query（用户明确要求查住宿）
  → Priority 1: transport_query（先查交通获取到达枢纽）
  → Priority 2: accommodation_query（依赖 transport_query 的到达枢纽）
  → Priority 3: itinerary_planning（依赖 accommodation_query 的推荐酒店，确保行程中酒店信息一致）

**【关键规则】** 当用户在 query 中使用以下任何词语时，**无论意图判断结果如何，必须调度对应 agent**：
- "交通"、"车票"、"高铁"、"火车"、"航班"、"飞机"、"查下" + 跨城移动 → 必须调度 transport_query
- "住宿"、"酒店"、"宾馆"、"住哪" → 必须调度 accommodation_query

请开始分析，直接输出JSON：
"""

        # =====================================================================
        # 调用 LLM（LangChain ainvoke，返回 AIMessage）
        # =====================================================================
        try:
            messages_list = [
                {"role": "system", "content": "你是一个高级意图识别专家。只输出JSON格式的结果，不要输出其他文本。"},
                {"role": "user", "content": prompt}
            ]
            # 新调用方式：ainvoke 返回 AIMessage，直接取 .content
            response = await llm.ainvoke(messages_list)
            text = response.content

            # 清理 markdown 代码块
            text = text.strip()
            if text.startswith('```json'):
                text = text[7:]
            if text.startswith('```'):
                text = text[3:]
            if text.endswith('```'):
                text = text[:-3]
            text = text.strip()

            # 解析 JSON
            try:
                result = json.loads(text)
            except json.JSONDecodeError as e1:
                start_idx = text.find('{')
                end_idx = text.rfind('}')
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

        # =====================================================================
        # 后处理：travel_style 兜底 + 关键词驱动确保必要 agent 不遗漏
        # =====================================================================
        result = _ensure_travel_style(user_query, result)
        result = _ensure_required_agents(user_query, result)
        result = _inject_poi_fetch(result)
        result = _inject_rag_knowledge(result)

        return {
            "intent_data": result,
            "intent_schedule": result.get("agent_schedule", []),
            "user_query": user_query,
            # 在 intent_node 立即写入 state，不再依赖 poi_fetch 执行后才写入
            "travel_style": result.get("travel_style", "普通"),
            "travel_days": _parse_travel_days(result),
        }

    return intent_node


def _ensure_travel_style(user_query: str, result: dict) -> dict:
    """
    兜底：确保 travel_style 始终有值。
    优先使用 LLM 返回值，若缺失或非法则从关键词推断，最终默认 "普通"。
    """
    import re

    valid_styles = {"亲子", "老人", "情侣", "特种兵", "普通"}
    current = result.get("travel_style", "")

    if current in valid_styles:
        return result

    # 关键词推断
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


def _inject_poi_fetch(result: dict) -> dict:
    """
    当检测到旅行规划意图时，自动向 agent_schedule 注入 poi_fetch 任务（priority=1）。
    若 poi_fetch 已存在则跳过。
    """
    intents = result.get("intents", [])
    intent_types = {i.get("type", "") for i in intents}
    planning_intents = {"plan_trip", "itinerary_planning"}

    if not (intent_types & planning_intents):
        return result

    schedule: list = result.get("agent_schedule", [])
    if any(t.get("agent_name") == "poi_fetch" for t in schedule):
        return result

    destination = result.get("key_entities", {}).get("destination", "")
    travel_style = result.get("travel_style", "普通")

    schedule.append({
        "agent_name": "poi_fetch",
        "priority": 1,
        "reason": "旅行规划意图触发，获取目的地 POI 数据以辅助行程规划",
        "expected_output": "目的地景点/POI列表",
        "params": {
            "destination": destination,
            "travel_style": travel_style
        }
    })
    logger.info(f"Injected poi_fetch for destination={destination!r}, travel_style={travel_style!r}")
    result["agent_schedule"] = schedule
    return result


def _inject_rag_knowledge(result: dict) -> dict:
    """
    后处理：双向保险确保 rag_knowledge 调度策略正确。

    注入条件（同时满足）：
      1. destination 已知
      2. 意图包含 itinerary_planning 或旅游相关 information_query
    移除条件（任一满足）：
      - destination 为空
      - 意图仅为 transport_query / preference / memory_query 等非规划类

    设计参考 _inject_poi_fetch()，与其并列执行（均为 priority=1）。
    """
    intents = result.get("intents", [])
    intent_types = {i.get("type", "") for i in intents}
    destination = (result.get("key_entities") or {}).get("destination", "") or ""

    # 判断是否满足触发条件
    planning_or_travel_info = bool(
        intent_types & {"itinerary_planning", "plan_trip"}
        or (
            "information_query" in intent_types
            # 排除纯交通/偏好/记忆的情况
            and not (intent_types <= {"information_query", "transport_query",
                                      "preference", "memory_query"})
        )
    )
    should_inject = bool(destination) and planning_or_travel_info

    schedule: list = result.get("agent_schedule", [])
    already_in = any(t.get("agent_name") == "rag_knowledge" for t in schedule)

    if should_inject and not already_in:
        # 注入：priority=1，与 poi_fetch 并行
        schedule.append({
            "agent_name": "rag_knowledge",
            "priority": 1,
            "reason": f"行程规划触发，检索「{destination}」旅游攻略以辅助 POI 评分和行程描述",
            "expected_output": "目的地景点推荐、游览时长建议、同游搭配提示、实用 tips"
        })
        logger.info(f"Injected rag_knowledge for destination={destination!r}")
    elif not should_inject and already_in:
        # 移除：不满足触发条件时，过滤掉 LLM 误调度的 rag_knowledge
        schedule = [t for t in schedule if t.get("agent_name") != "rag_knowledge"]
        logger.info("Removed rag_knowledge from schedule (trigger conditions not met)")

    result["agent_schedule"] = schedule
    return result


def _ensure_required_agents(user_query: str, result: dict) -> dict:
    """
    后处理兜底：当用户 query 包含明确的交通/住宿关键词时，
    确保 transport_query / accommodation_query 出现在调度计划中。
    只补充缺失的 agent，不删除或修改已有 agent。
    """
    import re

    schedule: list = result.get("agent_schedule", [])
    scheduled_names = {t.get("agent_name") for t in schedule}
    key_entities: dict = result.get("key_entities", {})
    origin = key_entities.get("origin", "")
    destination = key_entities.get("destination", "")
    has_cross_city = bool(origin and destination)

    # travel_style 关键词兜底（LLM 未识别时补充）
    if result.get("travel_style", "普通") == "普通":
        if re.search(r"带孩子|家庭出游|亲子游|亲子", user_query):
            result["travel_style"] = "亲子"
        elif re.search(r"两人世界|蜜月|情侣|约会", user_query):
            result["travel_style"] = "情侣"
        elif re.search(r"特种兵|高效.*景点|多景点|打卡", user_query):
            result["travel_style"] = "特种兵"

    # 交通关键词（需要有明确出发地才能查）
    transport_keywords = re.compile(r"交通|车票|高铁|火车|动车|航班|飞机|班次|怎么去|怎么到|查下.*去|去.*查")
    need_transport = (
        transport_keywords.search(user_query)
        and has_cross_city
        and "transport_query" not in scheduled_names
    )

    # 住宿关键词
    accommodation_keywords = re.compile(r"住宿|酒店|宾馆|住哪|住在哪|订房|找房")
    need_accommodation = (
        accommodation_keywords.search(user_query)
        and "accommodation_query" not in scheduled_names
    )

    if not need_transport and not need_accommodation:
        return result

    # 找到当前最低 priority（准备插入到正确位置）
    existing_priorities = [t.get("priority", 1) for t in schedule]
    max_priority = max(existing_priorities) if existing_priorities else 1

    # itinerary_planning 若已存在，将其推后到 transport/accommodation 之后
    transport_priority = 1
    accommodation_priority = 2 if need_transport else 1
    planning_priority = accommodation_priority + 1

    for t in schedule:
        if t.get("agent_name") == "itinerary_planning":
            t["priority"] = planning_priority

    if need_transport:
        schedule.insert(0, {
            "agent_name": "transport_query",
            "priority": transport_priority,
            "reason": "用户明确要求查询交通信息（关键词触发兜底）",
            "expected_output": "上海到北京的真实车次/航班列表及推荐方案"
        })
        logger.info("Fallback: added missing transport_query to schedule")

    if need_accommodation:
        insert_priority = accommodation_priority
        schedule.append({
            "agent_name": "accommodation_query",
            "priority": insert_priority,
            "reason": "用户明确要求查询住宿信息（关键词触发兜底）",
            "expected_output": "目的地酒店推荐列表"
        })
        logger.info("Fallback: added missing accommodation_query to schedule")

    result["agent_schedule"] = schedule
    return result


def _build_fallback_from_query(user_query: str) -> dict:
    """
    LLM 调用失败时，从 user_query 用正则提取关键实体，构建兜底意图结果。
    相比原来的最简 fallback，此函数能正确识别跨城行程意图，
    注入 transport_query / poi_fetch / accommodation_query，避免级联失败。
    """
    import re

    # ── 出发地（从XX出发 / XX出发）──
    origin = ""
    for pat in [
        r'从([^\s，,出去到]{1,6})(?:出发|启程|乘|坐)',
        r'([^\s，,]{1,6})出发',
    ]:
        m = re.search(pat, user_query)
        if m:
            origin = m.group(1).strip()
            break

    # ── 目的地（去XX玩 / 到XX游 / 去XX旅游）──
    destination = ""
    for pat in [
        r'去([^\s，,。！？出]{1,6})(?:玩|旅游|旅行|游|参观|看)',
        r'到([^\s，,。！？]{1,6})(?:玩|旅游|旅行|游|参观|看)',
        r'去([^\s，,。！？出]{1,6})(?:\s|，|,)',
    ]:
        m = re.search(pat, user_query)
        if m:
            destination = m.group(1).strip()
            break

    # ── 行程天数 ──
    days_m = re.search(r'(\d+)\s*[天日]', user_query)
    duration = f"{days_m.group(1)}天" if days_m else ""

    # ── 出行日期 ──
    date_m = re.search(
        r'(下周[一二三四五六日天]|下下周[一二三四五六日天]|明天|后天|大后天'
        r'|\d{4}年\d{1,2}月\d{1,2}日|\d{1,2}月\d{1,2}日)',
        user_query
    )
    date = date_m.group(1) if date_m else ""

    has_cross_city = bool(origin and destination)

    # ── 意图判断 ──
    travel_kw = re.compile(r'行程|规划|旅游|旅行|游玩|出游|出发|去.{1,6}玩|游记|安排|攻略')
    is_travel = bool(travel_kw.search(user_query)) or has_cross_city

    if is_travel:
        intents = [{
            "type": "itinerary_planning",
            "confidence": 0.7,
            "description": "行程规划",
            "reason": "LLM 降级，关键词/OD 对检测"
        }]
    else:
        intents = [{
            "type": "information_query",
            "confidence": 0.5,
            "description": "信息查询",
            "reason": "LLM 降级，默认查询"
        }]

    # ── 构建 agent_schedule ──
    schedule = []
    if is_travel:
        if has_cross_city:
            schedule.append({
                "agent_name": "transport_query",
                "priority": 1,
                "reason": f"跨城出行：{origin} → {destination}",
                "expected_output": "车次/航班列表及推荐方案"
            })
        schedule.append({
            "agent_name": "poi_fetch",
            "priority": 1,
            "reason": "获取目的地 POI 数据以辅助行程规划",
            "expected_output": "景点/餐厅/体验 POI 列表",
            "params": {"destination": destination, "travel_style": "普通"}
        })
        if re.search(r'住宿|酒店|宾馆|住哪|订房', user_query):
            acc_priority = 2 if has_cross_city else 1
            schedule.append({
                "agent_name": "accommodation_query",
                "priority": acc_priority,
                "reason": "用户要求推荐住宿",
                "expected_output": "酒店推荐列表"
            })
    else:
        schedule.append({
            "agent_name": "information_query",
            "priority": 1,
            "reason": "默认信息查询",
            "expected_output": "查询结果"
        })

    logger.info(
        f"_build_fallback_from_query: origin={origin!r}, destination={destination!r}, "
        f"duration={duration!r}, schedule={[t['agent_name'] for t in schedule]}"
    )

    return {
        "reasoning": (
            f"LLM 调用失败，正则兜底提取："
            f"origin={origin!r}, destination={destination!r}, "
            f"duration={duration!r}, date={date!r}"
        ),
        "intents": intents,
        "key_entities": {
            "origin": origin or None,
            "destination": destination or None,
            "date": date or None,
            "duration": duration or None,
        },
        "travel_style": "普通",  # 会被 _ensure_travel_style 覆盖
        "rewritten_query": user_query,
        "agent_schedule": schedule,
    }


def _parse_travel_days(result: dict) -> int:
    """
    从 intent_data 的 key_entities.duration 提取旅行总天数。
    提取失败返回 0（下游节点自行降级）。
    """
    import re
    duration = (result.get("key_entities") or {}).get("duration") or ""
    m = re.search(r"(\d+)", str(duration))
    return int(m.group(1)) if m else 0
