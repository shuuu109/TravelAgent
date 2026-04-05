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
from graph.node import (extract_hard_constraints, enrich_soft_constraints,
                        validate_rule_constraints, negotiate_constraints, plan_itinerary)


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
    itinerary_planning_node = create_itinerary_planning_node()
    poi_enrich_node = create_poi_enrich_node(llm)   # P3.5: Post-Retrieval Augmentation
    accommodation_node = create_accommodation_node(llm, memory_manager)
    respond_node = create_respond_node(llm)

    workflow = StateGraph(TravelGraphState)
    workflow.add_node("intent", intent_node)
    workflow.add_node("orchestrate", orchestrate_node)
    workflow.add_node("itinerary_planning", itinerary_planning_node)
    workflow.add_node("poi_enrich", poi_enrich_node)   # P3.5: 景点体验描述补充
    workflow.add_node("accommodation", accommodation_node)
    workflow.add_node("respond", respond_node)

    workflow.add_edge(START, "intent")
    workflow.add_edge("intent", "orchestrate")
    workflow.add_edge("orchestrate", "itinerary_planning")
    workflow.add_edge("itinerary_planning", "poi_enrich")   # P3 → P3.5
    workflow.add_edge("poi_enrich", "accommodation")        # P3.5 → P4
    workflow.add_edge("accommodation", "respond")
    workflow.add_edge("respond", END)

    return workflow.compile(checkpointer=checkpointer)
