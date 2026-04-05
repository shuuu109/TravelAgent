"""
Travel Agent State Module
========================

该模块定义了智能旅行助手系统的核心数据结构和全局状态管理。
包括：
  - 硬约束（OD信息、时间）
  - 软约束（用户偏好）
  - 规则校验（地理和物理限制）
  - LangGraph 全局状态类型定义

使用 Pydantic 模型确保数据验证和结构化提取。
"""

import operator
from typing import TypedDict, Annotated, List, Optional, Dict, Any
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


# =============================================================================
# 第一部分：约束数据模型定义
# =============================================================================

class HardConstraints(BaseModel):
    """
    硬约束数据模型
    
    表示行程的必选项，包括出发地、目的地和时间信息。
    这些约束条件对于生成有效的行程计划至关重要。
    
    属性:
        origin (Optional[str]): 出发地城市或地点
        destination (Optional[str]): 目的地城市或地点
        start_date (Optional[str]): 出发日期
        end_date (Optional[str]): 返程日期
        pax (Optional[int]): 出行人数，默认为1
    """
    origin: Optional[str] = Field(default=None, description="出发地 (Origin)")
    destination: Optional[str] = Field(default=None, description="目的地 (Destination)")
    start_date: Optional[str] = Field(default=None, description="出发时间")
    end_date: Optional[str] = Field(default=None, description="返程时间")
    pax: Optional[int] = Field(default=1, description="出行人数")
    
    def is_complete(self) -> bool:
        """
        检查核心硬约束是否已收集完毕
        
        仅检查出发地、目的地和出发日期是否都已提供，
        这是生成初步行程的最少必要信息。
        
        返回:
            bool: 当出发地、目的地和出发日期都不为空时返回 True，否则返回 False
        """
        return all([self.origin, self.destination, self.start_date])


class SoftConstraints(BaseModel):
    """
    软约束数据模型
    
    用户的偏好设置，包括酒店品牌、航空公司、座位偏好等。
    这些约束不是强制性的，但可以用来优化行程方案的选择。
    
    属性:
        hotel_brands (List[str]): 用户偏好的酒店品牌列表
        airlines (List[str]): 用户偏好的航空公司列表
        seat_preference (Optional[str]): 座位偏好，如"靠窗"、"过道"等
        budget_level (Optional[str]): 预算等级，如"经济"、"豪华"等
        other_preferences (Dict[str, Any]): 其他动态追加的偏好，用于扩展性
    """
    hotel_brands: List[str] = Field(default_factory=list, description="偏好的酒店品牌")
    airlines: List[str] = Field(default_factory=list, description="偏好的航空公司")
    seat_preference: Optional[str] = Field(default=None, description="座位偏好 (如：靠窗, 过道)")
    budget_level: Optional[str] = Field(default=None, description="预算等级 (如：经济, 豪华)")
    other_preferences: Dict[str, Any] = Field(default_factory=dict, description="其他动态追加的偏好")


class RuleViolation(BaseModel):
    """
    规则冲突数据模型
    
    记录违反物理常识或地理空间限制的情况。
    当系统检测到不可能的行程安排时，会创建此类的实例以供后续处理。
    
    属性:
        violation_type (str): 冲突类型，如"distance_error"（距离错误）、"time_conflict"（时间冲突）等
        description (str): 冲突的具体描述，例如"一天内无法步行从南京到北京，距离1000公里"
        suggestion (Optional[str]): 给用户的修正建议，如"建议更换交通方式为高铁"
    """
    violation_type: str = Field(description="冲突类型 (如: distance_error, time_conflict)")
    description: str = Field(description="冲突的具体描述 (如: 一天内无法步行从南京到北京，距离1000公里)")
    suggestion: Optional[str] = Field(default=None, description="给用户的修正建议 (如: 建议更换交通方式为高铁)")


# =============================================================================
# 第二部分：LangGraph 全局状态定义
# =============================================================================
class TravelOption(BaseModel):
    """统一的交通选项，抽象高铁和航班差异"""
    transport_type: str                  # "高铁" | "飞机"
    transport_no: Optional[str] = None   # G1234 / CA1234
    departure_time: Optional[str] = None
    arrival_time: Optional[str] = None
    duration: Optional[str] = None       # 运行时长
    departure_hub: Optional[str] = None  # 出发枢纽（站/机场）
    arrival_hub: Optional[str] = None    # 到达枢纽（站/机场）→ 住宿推荐关键字段
    price_range: Optional[str] = None    # 价格区间
    is_recommended: bool = False
    data_source: str = "llm"             # "realtime" | "llm"

class TravelGraphState(TypedDict):
    """
    智能旅行助手的全局状态类型定义

    该类定义了LangGraph中各个节点（Node）间的数据流动和状态管理。
    所有状态值在图的执行过程中会逐步被更新和传递，支持并行节点的安全写入。

    状态结构分为四层：
    1. 对话层：消息历史记录
    2. 约束层：硬约束、软约束、规则冲突
    3. 规划层：行程计划、缺失信息、交通选项
    4. 编排层：用户意图、技能执行结果、最终回复

    属性:
        messages (Annotated[list[BaseMessage], add_messages]):
            对话消息记录。使用 add_messages reducer 实现消息追加而不是覆盖，
            确保完整的对话历史得以保留。支持并行节点同时添加新消息。

        hard_constraints (HardConstraints):
            硬约束对象，包含必须的行程信息（出发地、目的地、日期等）。
            由 information_collection_node 逐步填充，是行程规划的基础。

        soft_constraints (SoftConstraints):
            软约束对象，包含用户的偏好设置（酒店品牌、航空公司、座位等）。
            可从用户消息或已保存的偏好中获取，用于优化行程方案选择。

        rule_violations (List[RuleViolation]):
            规则冲突列表，记录在行程规划中发现的所有不合理之处
            （如距离过远、时间不可行等）。供验证节点和用户反馈使用。

        missing_info (List[str]):
            缺失信息列表，记录还需从用户处收集的硬约束信息。
            由 check_completeness_node 维护，直接指导会话流向和用户提示内容。

        current_plan (Dict[str, Any]):
            当前行程计划，存储生成的行程草案或最终结果。
            结构灵活，可包含多日行程、地点、活动列表、预订信息、预估成本等。

        transport_options (List[Dict[str, Any]]):
            可选的交通工具选项列表。由查询节点获取（如航班、高铁、汽车等），
            供 planning_node 在生成行程时参考和选择。

        user_query (str):
            原始用户输入文本。从最新消息 messages[-1].content 提取，
            作为意图识别的输入，便于追溯和调试。

        intent_data (Dict[str, Any]):
            IntentionAgent 的完整输出，包含 reasoning（推理过程）、
            intents（识别的意图列表）、key_entities（关键实体）、
            agent_schedule（后续执行的技能调度清单）。

        intent_schedule (List[Dict[str, Any]]):
            agent_schedule 的提取版本，包含要执行的所有技能及其参数。
            方便 orchestrate_node 直接迭代和调度技能执行。

        skill_results (Annotated[List[Dict[str, Any]], operator.add]):
            技能执行结果列表。使用 operator.add reducer 支持并行节点安全地
            追加结果，无需显式同步。每个结果包含技能名称、输出、执行时间等。

        final_response (str):
            respond_node 生成的最终文字回复。基于行程计划、技能结果等信息
            生成的用户友好的回复文本，是对话的最终输出。

        travel_style (str):
            旅行风格标签，由意图/偏好节点写入。取值为 "亲子" | "情侣" | "特种兵" | "普通"。

        travel_days (int):
            旅行总天数，由 hard_constraints 中的 start_date/end_date 计算后写入，
            避免各下游节点重复计算。

        poi_candidates (List[Dict]):
            P2 poi_fetch 节点的原始 POI 结果列表。替换语义，每次写入覆盖旧值。

        daily_itinerary (List[Dict]):
            P3 clustering 节点输出的每日 POI 分组。替换语义，每次写入覆盖旧值。

        daily_routes (List[Dict]):
            P3 TSP 优化后的每日路线列表，包含景点顺序和建议交通方式。替换语义。

        rag_snippets (List[Dict]):
            P2 rag_knowledge 的原始检索文档列表，结构为
            [{"content": str, "metadata": dict}, ...]。替换语义。
            供 P3 itinerary_planning_node 做 POI 亲和度权重偏移，
            供 P5 respond_node 为每个景点填充攻略描述和实用 tips。
    """
    # ==================== 对话层 ====================
    # 消息记录：使用 add_messages 实现消息追加而不是覆盖，支持并行写入
    messages: Annotated[list[BaseMessage], add_messages]

    # ==================== 约束层 ====================
    # 硬约束：必须的行程信息（出发地、目的地、日期、人数）
    hard_constraints: HardConstraints

    # 软约束：用户偏好（酒店品牌、航空公司、座位等），非强制但影响方案优化
    soft_constraints: SoftConstraints

    # 规则冲突：在规划过程中发现的不可行之处和修正建议
    rule_violations: List[RuleViolation]

    # ==================== 规划层 ====================
    # 缺失信息：还需收集的硬约束字段名，引导用户交互流
    missing_info: List[str]

    # 行程计划：生成的行程结构（多日行程、地点、活动、预订等）
    current_plan: Dict[str, Any]

    # 交通选项：经 TravelOption 验证后的交通方式列表（model_dump() 序列化为 dict），供规划节点选择
    transport_options: List[Dict[str, Any]]

    # 旅行风格：亲子 | 情侣 | 特种兵 | 普通
    travel_style: str

    # 旅行天数：从 start_date/end_date 计算后显式存储，避免各节点重复计算
    travel_days: int

    # POI 候选列表：P2 poi_fetch 的原始结果（替换语义，非追加）
    poi_candidates: List[Dict]

    # 每日 POI 分组：P3 clustering 后每天的景点安排（替换语义）
    daily_itinerary: List[Dict]

    # 每日路线：P3 TSP 优化后每天的路线，含交通方式（替换语义）
    daily_routes: List[Dict]

    # RAG 检索片段：P2 rag_knowledge 的原始检索文档列表（替换语义）
    # 结构：[{"content": str, "metadata": dict}, ...]
    # 供 P3 itinerary_planning_node 做 POI 权重偏移，供 P5 respond_node 填充景点描述
    rag_snippets: List[Dict]

    # POI 搜索提示词：由 intent_node LLM 根据用户完整原始输入生成（替换语义）
    # 结构：["成都 大熊猫基地", "成都 宽窄巷子", ...]，2-4条
    # 供 poi_fetch agent 替代静态 keywords_map，语义上更贴近用户真实兴趣
    poi_search_hints: List[str]

    # ==================== 编排层 ====================
    # 用户原始输入：从最新消息提取，用于意图识别和追溯
    user_query: str

    # 意图数据：IntentionAgent 的完整输出（推理、意图、实体、技能调度）
    intent_data: Dict[str, Any]

    # 技能调度：待执行的技能列表及参数，方便编排节点调度
    intent_schedule: List[Dict[str, Any]]

    # 技能结果：各技能执行的输出结果，使用 add reducer 支持并行安全追加
    skill_results: Annotated[List[Dict[str, Any]], operator.add]

    # 最终回复：生成给用户的文字回复，是对话的最终输出
    final_response: str
