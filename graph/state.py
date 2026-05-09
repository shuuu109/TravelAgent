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
from typing import TypedDict, Annotated, List, Optional, Dict, Any, Union
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


# =============================================================================
# skill_results reducer
# =============================================================================
# 跨轮累加问题：orchestrate_node 和 accommodation_node 都会在同一轮内向 skill_results
# 追加内容，因此需要 reducer 语义。但跨轮（LangGraph checkpointer 续跑时）会把上一轮
# 的 skill_results 继续累积，导致列表无限增长且包含陈旧数据。
#
# 解决方案：通过 sentinel 字符串 SKILL_RESULTS_RESET 触发重置。
#   - 新轮开始时，extract_constraints_node 返回 {"skill_results": SKILL_RESULTS_RESET}
#     → reducer 检测到 sentinel，清空为 []
#   - 同轮内各节点返回 {"skill_results": [...]} → reducer 正常追加
#   - 同轮内 orchestrate_node 一次性写入完整列表，reducer 等价于 replace
SKILL_RESULTS_RESET: str = "__SKILL_RESULTS_RESET__"


def skill_results_reducer(
    left: Optional[List[Dict[str, Any]]],
    right: Union[str, List[Dict[str, Any]], None],
) -> List[Dict[str, Any]]:
    """
    支持 sentinel 重置的 skill_results reducer。

    - right == SKILL_RESULTS_RESET -> 清空，返回 []
    - right is None                -> 不变，返回 left 或 []
    - right is list                -> 追加到 left
    """
    if right == SKILL_RESULTS_RESET:
        return []
    if right is None:
        return list(left) if left else []
    if not isinstance(right, list):
        # 防御性兜底：非预期类型视为 no-op
        return list(left) if left else []
    return (list(left) if left else []) + right


# =============================================================================
# RAG 结构化抽取输出模型
# =============================================================================

class ExperienceOutput(BaseModel):
    """
    rag_experience_node 的结构化抽取结果。

    不做自由摘要，而是显式提取两个字段，防止 LLM 丢弃具体细节。

    属性:
        tips (List[str]):
            可操作的旅行建议，每条保留原文的具体细节
            （例如："灵隐寺需先买飞来峰票再买香花券，勿走路边绕道"）。
        best_for (List[str]):
            该目的地特别适合当前旅行风格的理由，供 respond_node 在行程介绍里使用。
    """
    tips: List[str] = Field(default_factory=list, description="可操作的旅行建议，保留原文具体细节")
    best_for: List[str] = Field(default_factory=list, description="适合当前旅行风格的理由")


class RiskOutput(BaseModel):
    """
    rag_risk_node 的结构化抽取结果。

    每条风险项要求保留"具体场景 + 潜在后果 + 规避建议"三要素，
    不允许 LLM 压缩成泛泛警告。

    属性:
        risks (List[str]):
            避坑条目列表，每条包含场景、后果和建议三要素
            （例如："西湖周边打车高峰期易堵，若赶班次建议提前1小时出发或改乘地铁"）。
    """
    risks: List[str] = Field(default_factory=list, description="避坑条目，每条含场景+后果+建议")


class RAGContext(BaseModel):
    """
    P2 编排阶段全部 RAG 输出的统一容器。

    orchestrate_node 在并行执行 rag_experience_node + rag_risk_node 后，
    将两个 agent 的结果聚合写入此容器，一次性写入 state["rag_context"]，
    避免三个散落字段（rag_snippets / rag_experience / rag_risks）生命周期割裂。

    属性:
        rag_snippets (List[Dict]):
            原始检索文档列表，结构 [{"content": str, "metadata": dict}, ...]。
            由 rag_experience_node 返回，供 P3 itinerary_planning_node 做 POI 关键词权重偏移。
        rag_experience (Optional[ExperienceOutput]):
            结构化旅行建议，含 tips（可操作建议）和 best_for（适合当前风格的理由）。
            供 P5 respond_node 渲染"旅行小贴士"区块。
        rag_risks (Optional[RiskOutput]):
            结构化避坑条目，每条含场景+后果+建议三要素。
            供 P5 respond_node 渲染"避坑提示"区块。
    """
    rag_snippets: List[Dict] = Field(default_factory=list)
    rag_experience: Optional[ExperienceOutput] = None
    rag_risks: Optional[RiskOutput] = None


# =============================================================================
# POI 时间信息模型（供 itinerary_planning_node._fetch_poi_time_info 使用）
# =============================================================================

class PoiTimeInfo(BaseModel):
    """
    单个景点的游览时间信息，由 LLM 内置知识批量生成。

    属性:
        poi_name (str): 景点名称，需与 POI dict 中的 name 字段精确一致。
        estimated_hours (float): 建议游览时长（小时），默认 1.5。
        best_period (str): 最佳游览时段，取值范围：
            morning   — 适合上午（如寺庙、古迹、需排队的热门景区）
            afternoon — 适合下午
            evening   — 适合傍晚或夜间（如夜市、灯会、酒吧街）
            flexible  — 全天均可
    """
    poi_name: str = Field(description="景点名称")
    estimated_hours: float = Field(default=1.5, description="建议游览时长（小时）")
    best_period: str = Field(
        default="flexible",
        description="最佳游览时段: morning/afternoon/evening/flexible",
    )


class PoiTimeInfoList(BaseModel):
    """
    批量 POI 时间信息的顶层容器，供 with_structured_output 使用。

    属性:
        items (List[PoiTimeInfo]): 各景点的时间信息列表，顺序与请求一致。
    """
    items: List[PoiTimeInfo] = Field(default_factory=list)


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
    total_budget: Optional[float] = Field(default=None, description="人均总预算（人民币元），由用户输入后归一化为人均值")
    
    def is_complete(self) -> bool:
        """
        检查核心硬约束是否已收集完毕

        仅检查出发地、目的地和出发日期是否都已提供，
        这是生成初步行程的最少必要信息。

        返回:
            bool: 当出发地、目的地和出发日期都不为空时返回 True，否则返回 False
        """
        return all([self.origin, self.destination, self.start_date])


def ensure_hard_constraints(obj: Any) -> "HardConstraints":
    """
    将 state 中可能以 None / dict / HardConstraints 三种形态存在的硬约束
    统一归一为 HardConstraints 模型，供下游节点用纯 attribute 访问字段，
    避免每处都写 hasattr / isinstance 双重判断。

    三种形态来源：
      - None:              初始首轮尚未写入
      - dict:              LangGraph checkpointer 反序列化可能生成 dict
      - HardConstraints:   extract_constraints_node 正常写入
    """
    if isinstance(obj, HardConstraints):
        return obj
    if isinstance(obj, dict):
        return HardConstraints(**obj)
    return HardConstraints()


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
    # critical = 可通过重排消除，触发 P3 回环；warning = 结构性限制（孤岛POI），直接透传给 respond_node
    severity: str = Field(default="critical", description="严重程度: critical | warning")


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

        daily_restaurants (List[Dict]):
            P3 基于每天景点地理重心，由高德周边搜索获取的餐厅推荐，每天推荐 5 家。替换语义。
            结构：[{"day": 1, "restaurants": [{"name": ..., "distance_m": ..., "amap_rating": ...}]}]

        rag_context (Optional[RAGContext]):
            P2 编排阶段 RAG 输出的统一容器，替换原有散落的 rag_snippets /
            rag_experience / rag_risks 三字段，由 orchestrate_node 一次性写入。
            setter: orchestrate_node；consumer: itinerary_planning_node (rag_snippets),
            respond_node (rag_experience, rag_risks)。
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

    # 每日周边餐厅：P3 基于每天景点重心搜索的餐厅推荐（替换语义）
    # 结构：[{"day": 1, "restaurants": [{"name": ..., "distance_m": ..., "amap_rating": ...}, ...]}, ...]
    daily_restaurants: List[Dict]

    # RAG 上下文容器：P2 orchestrate_node 将 rag_experience + rag_risk 两个并行 agent 的结果
    # 聚合后一次性写入（替换语义）。内含：
    #   rag_snippets    — 原始检索文档，供 P3 itinerary_planning_node 做 POI 关键词权重偏移
    #   rag_experience  — 结构化旅行建议，供 P5 respond_node 渲染"旅行小贴士"区块
    #   rag_risks       — 结构化避坑条目，供 P5 respond_node 渲染"避坑提示"区块
    rag_context: Optional[RAGContext]

    # POI 体验描述索引：P3.5 poi_enrich_node 的输出（替换语义）
    # 结构：{poi_name: description}，key 为景点名，value 为 1-2 句提炼后的体验描述
    # 查询词即景点名，语义对齐精准，供 P5 respond_node 按景点名注入行程介绍
    poi_descriptions: Dict[str, str]

    # 人均每日落地预算：P3 itinerary_planning_node 在扣除往返交通费后写入（替换语义）
    # 计算逻辑：(total_budget_per_person - min_transport_cost) / travel_days
    # 供 accommodation_node(P4) 计算住宿价格上限
    daily_budget_per_person: Optional[float]

    # 住宿降级等级：P4.6 budget_check_node 触发降级时递增（替换语义）
    # 0=初始（高端），1=第一次降级（舒适型），2=第二次降级（经济型），超过 2 次静默放行
    accommodation_downgrade_level: int

    # 每天3档住宿备选：accommodation_node 初始轮从 AccommodationAgent 输出中提取并写入（替换语义）
    # 格式：[{day, high: {hotel_name, price_per_night, area}, mid: {...}, low: {...}}]
    # 供 budget_check_node 读取价格、accommodation_node 降级轮次直接复用跳过 MCP
    daily_options_by_tier: List[Dict]

    # 预算检查结论：由 budget_check_node 写入，respond_node 渲染（替换语义）
    # "预算符合预期" = 交通+住宿 <= 总预算70%；None = 未检查或静默放行
    budget_fit_message: Optional[str]

    # 景点搜索提示词：由 intent_node LLM 根据用户完整原始输入生成（替换语义）
    # 结构：["成都 大熊猫基地", "成都 宽窄巷子", ...]，2-4条
    # 仅用于景点搜索，禁止包含住宿/餐厅/交通词，由 prompt 负面约束保证。
    # 供 poi_fetch agent 替代静态 keywords_map，语义上更贴近用户真实兴趣
    attraction_hints: List[str]

    # 住宿偏好（来自当前 query 的 P1 提取，非历史偏好）：替换语义
    # 结构（容忍缺字段）：
    #   {
    #       "brand_keywords": List[str],   # 用户提到的品牌/连锁词，如 ["连锁","汉庭"]
    #       "type":           str,          # "连锁" | "经济" | "豪华" | "民宿" | ""
    #       "price_range":    Optional[str] # 用户预算下的住宿单价区间，未提则 None
    #   }
    # 设计意图：把"我想住连锁酒店"这类住宿意图与景点搜索词分流，
    # 由 P4 accommodation_node 消费，避免污染 attraction_hints。
    accommodation_prefs: Dict[str, Any]

    # 目的地最佳旅游季节：intent_node 从 CityKnowledgeDB 查表写入（替换语义）
    # 如 "3-4月（春）；9-11月（秋）"；知识库无数据时为空字符串
    # 供 poi_select_node 按季节调整景点优先级
    destination_best_season: str

    # 目的地交通枢纽列表：intent_node 从 CityKnowledgeDB 查表写入（替换语义）
    # 如 ["杭州东站", "萧山国际机场"]；知识库无数据时为空列表
    # 供 accommodation_node 在 MCP 未返回到达枢纽时做兜底参考
    destination_transport_hubs: List[str]

    # ==================== 编排层 ====================
    # 用户原始输入：从最新消息提取，用于意图识别和追溯
    user_query: str

    # 意图数据：IntentionAgent 的完整输出（推理、意图、实体、技能调度）
    intent_data: Dict[str, Any]

    # 技能调度：待执行的技能列表及参数，方便编排节点调度
    intent_schedule: List[Dict[str, Any]]

    # 技能结果：各技能执行的输出结果。
    # 使用 skill_results_reducer（带 sentinel 重置语义）替代 operator.add：
    #   - 同轮内 orchestrate_node 和 accommodation_node 均会追加 → reducer 支持并发安全追加
    #   - 跨轮开始时由 extract_constraints_node 发送 SKILL_RESULTS_RESET sentinel 触发清空，
    #     避免 LangGraph checkpointer 续跑时历史结果无限累积且覆盖陈旧数据
    skill_results: Annotated[List[Dict[str, Any]], skill_results_reducer]

    # 最终回复：生成给用户的文字回复，是对话的最终输出
    final_response: str

    # 规划自检重试计数器：P4.5 itinerary_review_node 使用，防止回环死循环
    # 默认为 0，最多允许 1 次回环到 P3 重规划
    review_retry_count: int

    # 孤岛 POI 列表：P3 聚类时写入，记录与所有其他 POI 的最小通勤时间均超阈值的景点名称
    # 供 P4.5 review 节点将涉及孤岛的 long_transit_leg 降级为 warning，避免无效回环
    isolated_pois: List[str]

    # 跨轮累积的拆分约束：每次回环时追加新的 split_hints 而不是覆盖
    # 确保历史约束在后续重试中持续生效，防止"左手倒右手"循环
    accumulated_split_hints: List[List[str]]

    # 跨轮累积的删除约束：每次回环时追加新的 remove_hints
    # 与 accumulated_split_hints 对称，防止同一个问题 POI 在下一轮重新被锚定/填入
    accumulated_remove_hints: List[str]
