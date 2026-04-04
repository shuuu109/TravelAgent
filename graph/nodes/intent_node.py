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
- rag_knowledge: RAG知识库智能体（查询企业知识库）
- transport_query: 大交通查询智能体（查12306车票、航班，必须在行程规划前执行）
- accommodation_query: 住宿推荐智能体（根据到达枢纽和偏好推荐酒店，依赖transport_query结果时放Priority 2）

**Priority 2（依赖 Priority 1）- 行程规划类：**
- itinerary_planning: 行程规划智能体（需要事项收集的结果）

**说明：**
- Priority 1 的智能体都是信息获取，互不依赖，可并行执行提升速度
- Priority 2 的智能体需要使用 Priority 1 收集的信息
- 示例A：用户说"我要从天津去北京，喜欢住汉庭"（出发地明确）
  → Priority 1: preference + event_collection + transport_query（并行）
  → Priority 2: itinerary_planning（使用 Priority 1 的结果）

- 示例B：用户说"我想去北京"（出发地不明确，记忆中也无 home_location）
  → Priority 1: event_collection（收集出发地等缺失信息）
  → **不调度 transport_query**，因为没有出发地无法查询车次
  → Priority 2: itinerary_planning（待 event_collection 补全信息后再规划）

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
            result = {
                "reasoning": f"意图识别出错，使用默认策略。错误: {str(e)}",
                "intents": [
                    {
                        "type": "information_query",
                        "confidence": 0.5,
                        "description": "默认查询意图",
                        "reason": "无法解析用户意图，使用默认策略"
                    }
                ],
                "key_entities": {},
                "rewritten_query": user_query,
                "agent_schedule": [
                    {
                        "agent_name": "information_query",
                        "priority": 1,
                        "reason": "默认查询",
                        "expected_output": "查询结果"
                    }
                ]
            }

        return {
            "intent_data": result,
            "intent_schedule": result.get("agent_schedule", []),
            "user_query": user_query
        }

    return intent_node
