from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from config import LLM_CONFIG
from graph.state import TravelGraphState
from graph.nodes.intent_node import create_intent_node
from graph.nodes.orchestrate_node import create_orchestrate_node
from graph.nodes.respond_node import create_respond_node
from graph.nodes.itinerary_planning_node import create_itinerary_planning_node
from graph.nodes.poi_enrich_node import create_poi_enrich_node
from graph.nodes.accommodation_node import create_accommodation_node
from graph.nodes.itinerary_review_node import create_itinerary_review_node
from graph.node import (extract_hard_constraints, enrich_soft_constraints,
                        validate_rule_constraints, negotiate_constraints, plan_itinerary)
from typing import Literal


def route_after_review(state: TravelGraphState) -> Literal["itinerary_planning", "respond"]:
    """
    P4.5 自检后的路由判断。

    存在违规且尚未重试过（retry_count == 0）→ 回环到 P3 重规划。
    无违规，或已重试一次（retry_count >= 1）→ 进入 P5 响应。
    violations 会随 state 传入 respond_node，可作为 warning 渲染。
    """
    violations = state.get("rule_violations", [])
    retry_count = state.get("review_retry_count", 0)
    if violations and retry_count == 0:
        return "itinerary_planning"
    return "respond"


def route_after_validation(state: TravelGraphState) -> Literal["orchestrate", "negotiate"]:
    """
    P1.5 验证后的路由判断。

    两种情况都走 negotiate 分支，本轮规划终止：
      - rule_violations 非空：检测到物理时空冲突
      - missing_info 非空：硬约束信息不完整

    其余情况正常进入 P2 编排节点。
    """
    violations = state.get("rule_violations", [])
    missing = state.get("missing_info", [])
    if violations or missing:
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

    intent_node = create_intent_node(llm)
    orchestrate_node = create_orchestrate_node(registry, memory_manager)
    itinerary_planning_node = create_itinerary_planning_node(llm=llm)
    poi_enrich_node = create_poi_enrich_node(llm)          # P3.5: Post-Retrieval Augmentation
    accommodation_node = create_accommodation_node(llm, memory_manager)
    itinerary_review_node = create_itinerary_review_node() # P4.5: 行程自检
    respond_node = create_respond_node(llm)

    workflow = StateGraph(TravelGraphState)
    workflow.add_node("intent", intent_node)
    workflow.add_node("validate_constraints", validate_rule_constraints)  # P1.5: 时空可行性卫兵
    workflow.add_node("negotiate", negotiate_constraints)                  # P1.5b: 冲突协商终止节点
    workflow.add_node("orchestrate", orchestrate_node)
    workflow.add_node("itinerary_planning", itinerary_planning_node)
    workflow.add_node("poi_enrich", poi_enrich_node)         # P3.5: 景点体验描述补充
    workflow.add_node("accommodation", accommodation_node)
    workflow.add_node("itinerary_review", itinerary_review_node)  # P4.5: 行程自检
    workflow.add_node("respond", respond_node)

    workflow.add_edge(START, "intent")
    # P1 → P1.5：先过卫兵节点，再根据结果分流
    workflow.add_edge("intent", "validate_constraints")
    workflow.add_conditional_edges(
        "validate_constraints",
        route_after_validation,
        {"orchestrate": "orchestrate", "negotiate": "negotiate"},
    )
    workflow.add_edge("negotiate", END)                          # 协商完毕，本轮对话结束
    workflow.add_edge("orchestrate", "itinerary_planning")
    workflow.add_edge("itinerary_planning", "poi_enrich")        # P3 → P3.5
    workflow.add_edge("poi_enrich", "accommodation")             # P3.5 → P4
    workflow.add_edge("accommodation", "itinerary_review")       # P4 → P4.5
    workflow.add_conditional_edges(
        "itinerary_review",
        route_after_review,
        {"itinerary_planning": "itinerary_planning", "respond": "respond"},
    )
    workflow.add_edge("respond", END)

    return workflow.compile(checkpointer=checkpointer)
