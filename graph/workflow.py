from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from config import LLM_CONFIG
from graph.state import TravelGraphState
from graph.nodes.intent_node import create_intent_node
from graph.nodes.extract_constraints_node import create_extract_constraints_node
from graph.nodes.validate_node import create_validate_constraints_node
from graph.nodes.negotiate_node import create_negotiate_node
from graph.nodes.orchestrate_node import create_orchestrate_node
from graph.nodes.respond_node import create_respond_node
from graph.nodes.itinerary_planning_node import create_itinerary_planning_node
from graph.nodes.poi_enrich_node import create_poi_enrich_node
from graph.nodes.accommodation_node import create_accommodation_node
from graph.nodes.itinerary_review_node import create_itinerary_review_node
from typing import Literal

# P4.5 自检最大回环次数：第 0 / 1 次规划失败后允许回环，第 2 次仍有违规则放行到 respond
# 并由 respond_node 渲染 "已知限制" 区块提醒用户
REVIEW_MAX_RETRIES: int = 2


def route_after_review(state: TravelGraphState) -> Literal["itinerary_planning", "respond"]:
    """
    P4.5 自检后的路由判断。

    - 存在违规且 retry_count < REVIEW_MAX_RETRIES → 回环到 P3 重规划
    - 无违规，或已重试满 REVIEW_MAX_RETRIES 次 → 进入 P5 响应
    violations 会随 state 传给 respond_node，可作为 warning 渲染。
    """
    violations = state.get("rule_violations") or []
    retry_count = state.get("review_retry_count", 0)
    if violations and retry_count < REVIEW_MAX_RETRIES:
        return "itinerary_planning"
    return "respond"


def route_after_validation(state: TravelGraphState) -> Literal["orchestrate", "negotiate"]:
    """
    P1.5 验证后的路由判断。

    两种情况走 negotiate 分支，本轮规划终止：
      - rule_violations 包含阻塞类违规（time_conflict / spatial_temporal_conflict / system_error）
      - missing_info 非空：硬约束信息不完整（由 extract_constraints_node 计算）

    非阻塞类违规（long_distance_warning 等提示性信息）不中断流程，
    violations 会随 state 传给 respond_node 作为 warning 渲染。
    """
    # 只有这些类型才真正阻断 P2 流程
    BLOCKING_TYPES = {"time_conflict", "spatial_temporal_conflict", "system_error"}

    violations = state.get("rule_violations") or []
    missing    = state.get("missing_info") or []

    blocking = [
        v for v in violations
        if (v.violation_type if hasattr(v, "violation_type") else v.get("violation_type", ""))
        in BLOCKING_TYPES
    ]

    if blocking or missing:
        return "negotiate"
    return "orchestrate"


def build_graph(memory_manager, checkpointer=None):
    llm = ChatOpenAI(
        openai_api_key=LLM_CONFIG["api_key"],
        openai_api_base=LLM_CONFIG["base_url"],
        model_name=LLM_CONFIG["model_name"],
        temperature=LLM_CONFIG.get("temperature", 0.7),
        max_tokens=LLM_CONFIG.get("max_tokens", 8192),
    )

    from agents.lazy_agent_registry import LazyAgentRegistry
    registry = LazyAgentRegistry(model=llm, cache={}, memory_manager=memory_manager)

    # 注册独立 agent（transport、poi_fetch）
    from agents.transport_agent import TransportAgent
    from agents.poi_agent import POIFetchAgent
    registry["transport_query"] = TransportAgent(name="TransportAgent", model=llm)
    registry["poi_fetch"] = POIFetchAgent(name="POIFetchAgent")

    # ── 节点实例化（工厂模式，LLM/依赖在此注入）────────────────────────────────
    intent_node = create_intent_node(llm)
    extract_constraints_node = create_extract_constraints_node()            # P1.4：轻量级映射，无 LLM
    validate_constraints_node = create_validate_constraints_node(llm)       # P1.5：MCP ReAct 子智能体
    negotiate_node = create_negotiate_node(llm)                              # P1.5b：协商终止
    orchestrate_node = create_orchestrate_node(registry, memory_manager)
    itinerary_planning_node = create_itinerary_planning_node(llm=llm)
    poi_enrich_node = create_poi_enrich_node(llm)                           # P3.5：POI 体验补充
    accommodation_node = create_accommodation_node(llm, memory_manager)
    itinerary_review_node = create_itinerary_review_node()                   # P4.5：行程自检
    respond_node = create_respond_node(llm)

    workflow = StateGraph(TravelGraphState)
    workflow.add_node("intent", intent_node)
    workflow.add_node("extract_constraints", extract_constraints_node)       # P1.4：hard_constraints 单一真源
    workflow.add_node("validate_constraints", validate_constraints_node)     # P1.5：时空可行性卫兵
    workflow.add_node("negotiate", negotiate_node)                           # P1.5b：冲突协商终止节点
    workflow.add_node("orchestrate", orchestrate_node)
    workflow.add_node("itinerary_planning", itinerary_planning_node)
    workflow.add_node("poi_enrich", poi_enrich_node)                         # P3.5
    workflow.add_node("accommodation", accommodation_node)
    workflow.add_node("itinerary_review", itinerary_review_node)             # P4.5
    workflow.add_node("respond", respond_node)

    # ── 边连接 ────────────────────────────────────────────────────────────────
    workflow.add_edge(START, "intent")
    # P1 → P1.4：意图识别完成后立即结构化 hard_constraints + 跨轮清理
    workflow.add_edge("intent", "extract_constraints")
    # P1.4 → P1.5：结构化完毕后做时空物理校验
    workflow.add_edge("extract_constraints", "validate_constraints")
    workflow.add_conditional_edges(
        "validate_constraints",
        route_after_validation,
        {"orchestrate": "orchestrate", "negotiate": "negotiate"},
    )
    workflow.add_edge("negotiate", END)                                       # 协商完毕，本轮结束
    workflow.add_edge("orchestrate", "itinerary_planning")
    workflow.add_edge("itinerary_planning", "poi_enrich")                     # P3 → P3.5
    workflow.add_edge("poi_enrich", "accommodation")                          # P3.5 → P4
    workflow.add_edge("accommodation", "itinerary_review")                    # P4 → P4.5
    workflow.add_conditional_edges(
        "itinerary_review",
        route_after_review,
        {"itinerary_planning": "itinerary_planning", "respond": "respond"},
    )
    workflow.add_edge("respond", END)

    return workflow.compile(checkpointer=checkpointer)
