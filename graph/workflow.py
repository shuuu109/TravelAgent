from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from config import LLM_CONFIG
from graph.state import TravelGraphState
from graph.nodes.intent_node import create_intent_node
from graph.nodes.extract_constraints_node import create_extract_constraints_node
from graph.nodes.validate_node import create_validate_constraints_node
from graph.nodes.negotiate_node import create_negotiate_node
from graph.nodes.respond_node import create_respond_node
from graph.nodes.itinerary_planning_node_newcluster import create_itinerary_planning_node
from graph.nodes.poi_enrich_node import create_poi_enrich_node
from graph.nodes.restaurant_node import create_restaurant_node
from graph.nodes.accommodation_node import create_accommodation_node
from graph.nodes.itinerary_review_node import create_itinerary_review_node
from graph.nodes.budget_check_node import create_budget_check_node, route_after_budget_check
from graph.nodes.rag_node import create_rag_node
from graph.nodes.transport_node import create_transport_node
from graph.nodes.poi_fetch_node import create_poi_fetch_node
from graph.nodes.preference_node import create_preference_node
from graph.nodes.branch_nodes import create_memory_query_node, create_info_query_node
from typing import Literal

# P4.5 自检最大回环次数：第 0 / 1 次规划失败后允许回环，第 2 次仍有违规则放行到 respond
# 并由 respond_node 渲染 "已知限制" 区块提醒用户
REVIEW_MAX_RETRIES: int = 2


def route_after_review(state: TravelGraphState) -> Literal["itinerary_planning", "restaurant"]:
    """
    P4.5 自检后的路由判断。

    - 存在 critical 违规且 retry_count < REVIEW_MAX_RETRIES → 回环到 P3 重规划
    - 仅剩 warning 违规（结构性孤岛等）/ 无违规 / 已用完重试次数 → 进入 P3.6 餐厅推荐
      之后再依次走 accommodation → budget_check → respond
    所有 violations（含 warning）会随 state 传给 respond_node 渲染为"已知限制"。

    注：餐厅推荐放在 review 通过出口而不是 itinerary_planning 内部，
    避免 P3 回环重规划时重复消耗高德 API 配额。
    """
    violations = state.get("rule_violations") or []
    retry_count = state.get("review_retry_count", 0)
    critical = [
        v for v in violations
        if (v.severity if hasattr(v, "severity") else v.get("severity", "critical")) == "critical"
    ]
    if critical and retry_count < REVIEW_MAX_RETRIES:
        return "itinerary_planning"
    return "restaurant"


def route_after_validation(state: TravelGraphState):
    """
    P1.5 验证 + 意图分支路由（取代旧 LLM 动态调度）。

    返回值语义：
      - 单个节点名 → 路由到该节点
      - 节点名列表 → fan-out 到列表中所有节点（LangGraph 原生支持）

    1) 阻塞类违规 / 必填信息缺失 → "negotiate"（本轮终止，向用户澄清）
    2) 否则按 intent_type 静态分支：
        planning        → ["rag", "transport", "poi_fetch"]   (并行 fan-out)
        preference_only → "preference"
        memory_only     → "memory_query"
        info_only       → "info_query"
        unknown         → "respond"
    """
    BLOCKING_TYPES = {"time_conflict", "spatial_temporal_conflict", "system_error"}

    violations = state.get("rule_violations") or []
    missing = state.get("missing_info") or []

    blocking = [
        v for v in violations
        if (v.violation_type if hasattr(v, "violation_type") else v.get("violation_type", ""))
        in BLOCKING_TYPES
    ]
    if blocking or missing:
        return "negotiate"

    intent_type = state.get("intent_type", "unknown")
    if intent_type == "planning":
        return ["rag", "transport", "poi_fetch"]
    if intent_type == "preference_only":
        return "preference"
    if intent_type == "memory_only":
        return "memory_query"
    if intent_type == "info_only":
        return "info_query"
    return "respond"


def build_graph(memory_manager, checkpointer=None):
    llm = ChatOpenAI(
        openai_api_key=LLM_CONFIG["api_key"],
        openai_api_base=LLM_CONFIG["base_url"],
        model_name=LLM_CONFIG["model_name"],
        temperature=LLM_CONFIG.get("temperature", 0.7),
        max_tokens=LLM_CONFIG.get("max_tokens", 8192),
    )

    # intent_node 专用 LLM：复杂查询含 5+ agent 调度时，2500 会截断 JSON → 提升到 4096
    intent_llm = ChatOpenAI(
        openai_api_key=LLM_CONFIG["api_key"],
        openai_api_base=LLM_CONFIG["base_url"],
        model_name=LLM_CONFIG["model_name"],
        temperature=0.3,
        max_tokens=4096,
    )

    from agents.lazy_agent_registry import LazyAgentRegistry
    registry = LazyAgentRegistry(model=llm, cache={}, memory_manager=memory_manager)

    # 注册独立 agent（transport、poi_fetch）
    from agents.transport_agent import TransportAgent
    from agents.poi_agent import POIFetchAgent
    registry["transport_query"] = TransportAgent(name="TransportAgent", model=llm)
    registry["poi_fetch"] = POIFetchAgent(name="POIFetchAgent")

    # ── 节点实例化（工厂模式，LLM/依赖在此注入）────────────────────────────────
    intent_node = create_intent_node(intent_llm, memory_manager=memory_manager)  # P1：含 W mirror
    extract_constraints_node = create_extract_constraints_node(memory_manager=memory_manager)  # P1.4
    validate_constraints_node = create_validate_constraints_node(llm)             # P1.5
    negotiate_node = create_negotiate_node(llm)                                   # P1.5b
    # P2 fan-out 节点（取代旧 orchestrate_node）
    rag_node = create_rag_node(registry)
    transport_node = create_transport_node(registry)
    poi_fetch_node = create_poi_fetch_node(registry)
    # P2 单 skill 分支节点
    preference_node = create_preference_node(registry, memory_manager=memory_manager)
    memory_query_node = create_memory_query_node(registry)
    info_query_node = create_info_query_node(registry)
    # P3+
    itinerary_planning_node = create_itinerary_planning_node(llm=llm)
    poi_enrich_node = create_poi_enrich_node(llm)                                  # P3.5
    restaurant_node = create_restaurant_node()                                     # P3.6
    accommodation_node = create_accommodation_node(llm, memory_manager)
    itinerary_review_node = create_itinerary_review_node()                         # P4.5
    budget_check_node = create_budget_check_node()                                  # P4.6
    respond_node = create_respond_node(llm, memory_manager=memory_manager)         # P5（含 save_trip_history）

    workflow = StateGraph(TravelGraphState)
    workflow.add_node("intent", intent_node)
    workflow.add_node("extract_constraints", extract_constraints_node)
    workflow.add_node("validate_constraints", validate_constraints_node)
    workflow.add_node("negotiate", negotiate_node)
    # P2 fan-out 三节点 + 单 skill 分支三节点
    workflow.add_node("rag", rag_node)
    workflow.add_node("transport", transport_node)
    workflow.add_node("poi_fetch", poi_fetch_node)
    workflow.add_node("preference", preference_node)
    workflow.add_node("memory_query", memory_query_node)
    workflow.add_node("info_query", info_query_node)
    # P3+
    workflow.add_node("itinerary_planning", itinerary_planning_node)
    workflow.add_node("poi_enrich", poi_enrich_node)
    workflow.add_node("restaurant", restaurant_node)
    workflow.add_node("accommodation", accommodation_node)
    workflow.add_node("itinerary_review", itinerary_review_node)
    workflow.add_node("budget_check", budget_check_node)
    workflow.add_node("respond", respond_node)

    # ── 边连接 ────────────────────────────────────────────────────────────────
    workflow.add_edge(START, "intent")
    workflow.add_edge("intent", "extract_constraints")
    workflow.add_edge("extract_constraints", "validate_constraints")

    # P1.5 验证 + intent_type 静态路由（路径函数返回 list 时自动 fan-out）
    workflow.add_conditional_edges(
        "validate_constraints",
        route_after_validation,
        ["negotiate", "rag", "transport", "poi_fetch", "preference", "memory_query", "info_query", "respond"],
    )

    workflow.add_edge("negotiate", END)

    # planning fan-in：rag + transport + poi_fetch 全部完成后进入 P3 行程规划
    workflow.add_edge(["rag", "transport", "poi_fetch"], "itinerary_planning")

    workflow.add_edge("itinerary_planning", "poi_enrich")                   # P3 → P3.5
    workflow.add_edge("poi_enrich", "itinerary_review")                     # P3.5 → P4.5
    workflow.add_conditional_edges(
        "itinerary_review",
        route_after_review,
        {"itinerary_planning": "itinerary_planning", "restaurant": "restaurant"},
    )
    workflow.add_edge("restaurant", "accommodation")                        # P3.6 → P4
    workflow.add_edge("accommodation", "budget_check")                      # P4 → P4.6
    workflow.add_conditional_edges(
        "budget_check",
        route_after_budget_check,
        {"accommodation": "accommodation", "respond": "respond"},
    )

    # 单 skill 分支：完成后直接进 respond
    workflow.add_edge("preference", "respond")
    workflow.add_edge("memory_query", "respond")
    workflow.add_edge("info_query", "respond")

    workflow.add_edge("respond", END)

    return workflow.compile(checkpointer=checkpointer)
