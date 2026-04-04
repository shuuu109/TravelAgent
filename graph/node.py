"""
Travel Agent Graph Node Module
===============================

该模块定义了LangGraph工作流中的各个处理节点(Node)。
每个节点负责处理状态的特定方面，如约束提取、冲突检验等。

主要功能:
  - 硬约束提取：从用户输入中提取行程的必要信息
  - 约束融合：合并新旧约束，确保信息的累积和更新
  - 缺失信息检测：识别仍需用户提供的信息

依赖:
  - LangChain: LLM调用和提示词处理
  - 项目状态类型: TravelGraphState, HardConstraints
"""

import sys
import os
import json
import re
from typing import Dict, Any

# =============================================================================
# 路径配置和依赖导入
# =============================================================================

# 添加项目根目录到 Python 路径，便于导入配置和状态模块
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from config import LLM_CONFIG
from context.memory_manager import MemoryManager
import requests
from typing import Dict, Any, Tuple
from graph.state import TravelGraphState, RuleViolation, SoftConstraints, HardConstraints
from langchain_core.messages import HumanMessage
from mcp_clients.amap_client import amap_mcp_session

# =============================================================================
# MCP 工具加载
# =============================================================================
# 尝试导入langchain_mcp_adapters，如果不可用则使用基础方式
from langchain_mcp_adapters.tools import load_mcp_tools



# =============================================================================
# LLM 初始化和配置
# =============================================================================

# 初始化大语言模型实例
# 使用项目配置中的 LLM_CONFIG，支持 OpenAI 兼容的 API 接口
llm = ChatOpenAI(
    openai_api_key=LLM_CONFIG["api_key"],
    openai_api_base=LLM_CONFIG["base_url"],
    model_name=LLM_CONFIG["model_name"],
    temperature=0.1,  # 提取任务需要较低的温度以保证确定性和一致性
    max_tokens=LLM_CONFIG.get("max_tokens", 8192)
)

# 绑定结构化输出到 LLM，使其返回符合 HardConstraints 数据模型的结果
extractor_llm = llm.with_structured_output(HardConstraints)

# =============================================================================
# 节点函数定义
# =============================================================================

def extract_hard_constraints(state: TravelGraphState) -> Dict[str, Any]:
    """
    硬约束提取节点
    
    从用户的最新输入中提取行程的必要约束条件，包括出发地、目的地、
    时间和人数等。该节点负责约束的初始提取和逐步累积。
    
    处理流程:
      1. 获取对话历史中用户的最新输入
      2. 结合当前已知的约束信息构建LLM提示
      3. 通过LLM提取新的或更新的约束信息
      4. 融合新旧约束，保留已有信息
      5. 识别仍缺失的必要信息
    
    参数:
        state (TravelGraphState): 当前的图执行状态，包含对话消息和约束信息
    
    返回:
        Dict[str, Any]: 包含以下键的状态增量字典：
            - hard_constraints (HardConstraints): 融合后的约束对象
            - missing_info (List[str]): 仍需收集的缺失信息列表
    """
    # 从状态中提取对话历史和当前硬约束
    messages = state.get("messages", [])
    current_constraints = state.get("hard_constraints")
    
    # 如果状态中还未初始化硬约束，创建一个新的空约束对象
    if not current_constraints:
        current_constraints = HardConstraints()
    
    # 检查是否存在用户消息，如不存在返回初始缺失信息
    if not messages:
        return {
            "hard_constraints": current_constraints,
            "missing_info": ["origin", "destination", "start_date"]
        }
        
    # 获取对话历史中最后一条消息（用户的最新输入）
    last_message = messages[-1].content

    # =========================================================================
    # 构建LLM提示词
    # =========================================================================
    # 将当前已知的约束信息包含在系统提示中，帮助LLM理解上下文
    # 这样LLM可以识别哪些信息已经收集，只需要提取或更新必要的部分
    system_prompt = f"""
    你是一个专业的出行规划助手。你的任务是从用户的输入中提取行程的核心约束条件。
    
    当前已知的约束信息如下：
    - 出发地 (Origin): {current_constraints.origin or '未知'}
    - 目的地 (Destination): {current_constraints.destination or '未知'}
    - 出发时间 (Start Date): {current_constraints.start_date or '未知'}
    - 返程时间 (End Date): {current_constraints.end_date or '未知'}
    - 出行人数 (Pax): {current_constraints.pax}
    
    请分析用户的最新输入，提取缺失的信息，或者根据用户的要求更新已有的信息。
    注意：只提取明确提到的信息，不要进行无根据的猜测。
    """
    
    # 创建提示词模板，包括系统消息和用户输入
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{user_input}")
    ])
    
    # =========================================================================
    # 执行约束提取
    # =========================================================================
    # 组建处理链：提示词模板 | 结构化输出的LLM
    chain = prompt | extractor_llm
    
    # 调用链处理用户输入，获取结构化的约束对象
    new_constraints: HardConstraints = chain.invoke({"user_input": last_message})
    
    # =========================================================================
    # 约束融合：保留已有信息，更新新的信息
    # =========================================================================
    # 使用"新值优先，否则保留旧值"的策略合并约束
    # 确保已经确认的信息不会被空值覆盖，同时允许用户修改信息
    merged_constraints = HardConstraints(
        origin=new_constraints.origin or current_constraints.origin,
        destination=new_constraints.destination or current_constraints.destination,
        start_date=new_constraints.start_date or current_constraints.start_date,
        end_date=new_constraints.end_date or current_constraints.end_date,
        pax=new_constraints.pax if new_constraints.pax != 1 else current_constraints.pax 
    )
    
    # =========================================================================
    # 检测缺失信息
    # =========================================================================
    # 识别行程规划所必需但仍缺失的信息
    missing_info = []
    if not merged_constraints.origin:
        missing_info.append("出发地")
    if not merged_constraints.destination:
        missing_info.append("目的地")
    if not merged_constraints.start_date:
        missing_info.append("出发时间")
        
    # 返回状态增量：更新后的约束和缺失信息列表
    # 这些增量会被LangGraph自动合并到全局状态中
    return {
        "hard_constraints": merged_constraints,
        "missing_info": missing_info
    }


def enrich_soft_constraints(state: TravelGraphState) -> Dict[str, Any]:
    """
    软约束增强节点
    
    从长期记忆系统（PostgreSQL/JSON文件）中提取用户的历史偏好，
    并将其注入到当前的全局状态中。该节点确保用户的过往偏好设置
    能够被后续的行程规划步骤所使用。
    
    在完整的MCP架构就绪前，直接调用现有的 MemoryManager 进行偏好读取。
    
    处理流程:
      1. 初始化或获取当前的软约束状态
      2. 实例化 MemoryManager 连接长期记忆存储
      3. 从长期记忆中读取用户的所有历史偏好
      4. 将无结构的偏好数据映射到 SoftConstraints 强类型字段
      5. 处理扩展性偏好（如餐食、常驻地等）到 other_preferences 字典
      6. 返回更新后的软约束状态
    
    参数:
        state (TravelGraphState): 当前的图执行状态，包含软约束等信息
    
    返回:
        Dict[str, Any]: 包含以下键的状态增量字典：
            - soft_constraints (SoftConstraints): 融合了历史偏好的软约束对象
    
    注意:
        - 使用固定的 user_id="default_user"，生产环境应从 State 或配置中获取
        - 如果用户没有历史偏好记录，直接返回当前状态而不报错
        - 动态偏好会被兜底存储在 other_preferences 字典中以支持扩展
    """
    # 初始化或获取当前的软约束状态
    current_soft = state.get("soft_constraints")
    if not current_soft:
        current_soft = SoftConstraints()
    
    # =========================================================================
    # 实例化记忆管理器并获取长期偏好
    # =========================================================================
    # MemoryManager 负责与持久化存储（PostgreSQL/JSON）的交互
    # user_id 使用默认值，生产环境应将其参数化
    # llm_model 设为 None 因为仅需读取操作，无需 LLM 调用
    memory_manager = MemoryManager(
        user_id="default_user", 
        session_id="system_enrichment", 
        llm_model=None 
    )
    
    # 从长期记忆系统中检索用户的所有偏好数据（字典格式）
    raw_prefs = memory_manager.long_term.get_preference()
    
    # 如果没有历史偏好记录，直接返回原始状态
    if not raw_prefs:
        return {"soft_constraints": current_soft}
    
    # =========================================================================
    # 数据清洗与类型映射
    # =========================================================================
    # 将从存储中读取的无结构化偏好数据映射到 SoftConstraints 的强类型字段
    
    # 处理酒店品牌偏好：确保为列表类型
    hotel_brands = raw_prefs.get("hotel_brands", [])
    if isinstance(hotel_brands, str): 
        hotel_brands = [hotel_brands]
        
    # 处理航空公司偏好：确保为列表类型
    airlines = raw_prefs.get("airlines", [])
    if isinstance(airlines, str): 
        airlines = [airlines]
    
    # =========================================================================
    # 更新 SoftConstraints 的标准字段
    # =========================================================================
    # 将清洗后的数据赋值给对应的强类型字段
    current_soft.hotel_brands = hotel_brands
    current_soft.airlines = airlines
    current_soft.seat_preference = raw_prefs.get("seat_preference")
    current_soft.budget_level = raw_prefs.get("budget_level")
    
    # =========================================================================
    # 处理动态扩展偏好
    # =========================================================================
    # 系统可能在未来追加新的偏好类型（如餐食偏好、活动偏好等）
    # 这些未映射到强类型字段的偏好会被存储到 other_preferences 字典
    known_keys = {"hotel_brands", "airlines", "seat_preference", "budget_level"}
    for k, v in raw_prefs.items():
        if k not in known_keys and v:
            current_soft.other_preferences[k] = v
    
    # 返回状态增量，LangGraph 会自动将其融合到全局状态中
    return {"soft_constraints": current_soft}


async def validate_rule_constraints(state: TravelGraphState) -> Dict[str, Any]:
    """
    规则约束校验节点 (Agentic 版本)
    
    校验硬约束中的出发地、目的地和时间信息是否符合真实物理世界的约束规则。
    该节点通过拉起一个内置高德官方 MCP 工具的 ReAct 子智能体，
    让大模型自主查询真实的路网距离和预估时间，并进行复杂的时空逻辑推理。
    
    处理流程:
      1. 检查硬约束的完整性（出发地、目的地是否存在）。
      2. 建立与高德官方 MCP Server 的异步会话。
      3. 将官方工具转化为 LangChain Tools，并注入到子智能体中。
      4. 引导智能体综合验证“距离规则”和“时间安排”（原规则A与规则B）。
      5. 解析智能体的判定结果，生成标准的违规记录。
    """
    hard_constraints = state.get("hard_constraints")
    new_violations = []
    
    # =========================================================================
    # 前置检查：验证必要信息是否完整
    # =========================================================================
    if not hard_constraints or not hard_constraints.origin or not hard_constraints.destination:
        return {"rule_violations": new_violations}
    
    origin = hard_constraints.origin
    destination = hard_constraints.destination
    start_date = hard_constraints.start_date or "未指定"
    end_date = hard_constraints.end_date or "未指定"
    
    # =========================================================================
    # 启动 MCP 会话并进行 Agentic 校验
    # =========================================================================
    try:
        async with amap_mcp_session() as session:
            # 获取高德官方提供的所有 MCP 工具（如路线规划、地理编码等）
            amap_tools = await load_mcp_tools(session)
            
            # 创建专门用于规则校验的 ReAct 子智能体
            validator_agent = create_react_agent(llm, amap_tools)
            
            # 构建高强度的 Prompt
            system_prompt = f"""
            你是一个严谨的交通规划专家。请审查以下用户的出行意图：
            - 出发地: {origin}
            - 目的地: {destination}
            - 出发日期: {start_date}
            - 返程日期: {end_date}
            
            请使用你手中的高德地图工具执行以下验证：
            1. 距离规则：查出两地的实际驾车或公共交通路线距离（公里）和耗时。如果距离极长（如超过800公里），驾车成本极高。
            2. 时间规则：结合用户的出发和返程日期，判断用户的时间安排是否能容纳所需的往返交通时间。例如，如果距离遥远却要求“当日往返”，在物理上是不现实的。
            
            请严格按照以下 JSON 格式输出你的最终结论（不要输出 markdown 代码块，纯 JSON）：
            {{
                "is_valid": true/false,
                "violation_type": "long_distance_warning/time_conflict/none",
                "description": "详细说明发现的物理冲突或距离时间数据（仅当is_valid为false时填写）",
                "suggestion": "给用户的具体交通方式或行程修改建议（仅当is_valid为false时填写）"
            }}
            """
            
            # 1. 极简初始化：去掉 state_modifier 参数，完美绕过版本差异
            validator_agent = create_react_agent(llm, tools=amap_tools)
            
            # 2. 触发调用：在传入 messages 列表时，直接把 system_prompt 放在最前面
            response = await validator_agent.ainvoke({
                "messages": [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content="请开始调用高德工具执行物理规则与时间的校验，并严格只返回最终的 JSON 结果。")
                ]
            })
            
            # =================================================================
            # 提取最终结果并解析为 JSON
            # 重要：JSON 解析必须在 MCP session 内部做好容错，
            # 否则异常会穿透 session cleanup 链，被 anyio TaskGroup
            # 包装成 ExceptionGroup，导致误报为"MCP 连接失败"。
            # =================================================================
            final_answer = response["messages"][-1].content.strip()

            try:
                # 清理可能的 markdown 代码块标记
                if "```" in final_answer:
                    # 提取 ``` 和 ``` 之间的内容
                    match = re.search(r"```(?:json)?\s*(.*?)```", final_answer, re.DOTALL)
                    if match:
                        final_answer = match.group(1).strip()

                # 如果 LLM 在 JSON 前后加了废话，尝试提取第一个 {...}
                if not final_answer.startswith("{"):
                    match = re.search(r"\{.*\}", final_answer, re.DOTALL)
                    if match:
                        final_answer = match.group(0)

                result = json.loads(final_answer)
            except (json.JSONDecodeError, AttributeError) as parse_err:
                print(f"\n⚠️ [JSON 解析失败] LLM 原始回复:\n{response['messages'][-1].content[:300]}")
                print(f"   解析错误: {parse_err}")
                # 解析失败时视为校验通过（不阻塞用户流程）
                result = {"is_valid": True}

            # =================================================================
            # 解析校验结果并生成 RuleViolation
            # =================================================================
            if not result.get("is_valid", True):
                violation = RuleViolation(
                    violation_type=result.get("violation_type", "spatial_temporal_conflict"),
                    description=result.get("description", f"在规划从 {origin} 到 {destination} 的行程时发现物理限制。"),
                    suggestion=result.get("suggestion", "请检查您的行程距离和时间安排是否合理。")
                )
                new_violations.append(violation)
                
    except Exception as e:
        import traceback
        print("\n❌ [Agent 内部崩溃] 正在拆解 TaskGroup 真实报错：")
        
        # 拆解 TaskGroup 子异常
        if hasattr(e, 'exceptions'):
            for i, sub_e in enumerate(e.exceptions):
                print(f"  🔴 子异常 {i+1} [{type(sub_e).__name__}]: {sub_e}")
        else:
            print(f"  🔴 常规异常 [{type(e).__name__}]: {e}")
            
        print("\n📜 完整错误堆栈:")
        traceback.print_exc()
        
        # 兜底容错机制
        violation = RuleViolation(
            violation_type="system_error",
            description="连接高德空间推理服务时出现异常，无法验证路网规则。",
            suggestion="您可以继续行程，但请自行确认路程时间是否合理。"
        )
        new_violations.append(violation)

    return {"rule_violations": new_violations}


def negotiate_constraints(state: TravelGraphState) -> Dict[str, Any]:
    """
    协商节点：当信息缺失或存在物理规则冲突时，负责向用户解释并提出修正建议。
    """
    missing_info = state.get("missing_info", [])
    violations = state.get("rule_violations", [])

    prompt_parts = ["你是一个专业的智能出行规划师。当前用户的请求无法直接生成最终行程，请向用户委婉地解释原因并提问：\n"]

    # 1. 处理信息缺失
    if missing_info:
        prompt_parts.append(f"- 缺失核心信息：我们需要知道用户的 {', '.join(missing_info)}。请询问用户这些信息。")

    # 2. 处理物理规则冲突
    if violations:
        prompt_parts.append("- 存在以下物理时空约束冲突：")
        for v in violations:
            prompt_parts.append(f"  * {v.description} 建议：{v.suggestion}")

    prompt_parts.append("\n请基于以上情况，生成一段自然、专业、贴心的回复，引导用户补充信息或调整行程计划。不要输出其他多余的思考过程。")

    # 调用 LLM 生成回复
    # 注意：state["messages"] 是 LangChain BaseMessage 对象列表，
    # 需要统一使用 BaseMessage 而非 dict 来构建输入
    system_msg = SystemMessage(content="\n".join(prompt_parts))
    history = state.get("messages", [])

    response = llm.invoke([system_msg] + history)

    # 将助手的回复追加到 messages 中，返回给前端
    return {"messages": [response]}

def plan_itinerary(state: TravelGraphState) -> Dict[str, Any]:
    """
    行程规划节点：当所有硬约束集齐，且没有物理冲突时，整合软约束生成最终的行程表。
    这是"状态回退重规划"的目标节点——用户局部修改后只需从此节点重跑。
    """
    hard = state["hard_constraints"]
    soft = state.get("soft_constraints") or SoftConstraints()
    messages = state.get("messages", [])

    system_prompt = f"""
    你是一个金牌旅行规划师。所有的时空约束和物理校验均已通过，请为用户生成一份详尽的行程计划。

    【行程框架】
    出发地: {hard.origin}
    目的地: {hard.destination}
    出发时间: {hard.start_date}
    返程时间: {hard.end_date or '未指定（请规划合理时长）'}
    出行人数: {hard.pax}

    【个性化偏好】
    酒店偏好: {'、'.join(soft.hotel_brands) if soft.hotel_brands else '无特殊要求'}
    航司偏好: {'、'.join(soft.airlines) if soft.airlines else '无特殊要求'}
    座位偏好: {soft.seat_preference or '无特殊要求'}
    预算等级: {soft.budget_level or '中等'}
    其他偏好: {soft.other_preferences if soft.other_preferences else '无'}

    请输出一份结构清晰的行程单，包含：
    1. 行程总览（目的地简介与推荐理由）
    2. 每日详细安排（景点 + 交通衔接 + 餐饮推荐 + 时间分配）
    3. 住宿推荐（参照酒店偏好，给出具体区域和品牌建议）
    4. 往返交通方案（交通工具选择、票务建议）
    5. 预算估算（分项：交通/住宿/餐饮/门票/其他）
    6. 出行注意事项（天气/当地习俗等）
    """

    system_msg = SystemMessage(content=system_prompt)
    response = llm.invoke([system_msg] + messages)

    # 记录最终计划并追加消息
    return {
        "current_plan": {
            "status": "completed",
            "details": response.content,
            "origin": hard.origin,
            "destination": hard.destination,
            "start_date": hard.start_date,
            "end_date": hard.end_date,
            "pax": hard.pax,
        },
        "messages": [response],
    }