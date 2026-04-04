from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from config import LLM_CONFIG
from graph.state import TravelGraphState
from graph.nodes.intent_node import create_intent_node
from graph.nodes.orchestrate_node import create_orchestrate_node
from graph.nodes.respond_node import create_respond_node
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
    
    # 注册独立 agent（transport、accommodation）
    from agents.transport_agent import TransportAgent
    from agents.accommodation_agent import AccommodationAgent
    registry["transport_query"] = TransportAgent(name="TransportAgent", model=llm)
    registry["accommodation_query"] = AccommodationAgent(name="AccommodationAgent", model=llm)
    
    intent_node = create_intent_node(llm)
    orchestrate_node = create_orchestrate_node(registry, memory_manager)
    respond_node = create_respond_node(llm)

    workflow = StateGraph(TravelGraphState)
    workflow.add_node("intent", intent_node)
    workflow.add_node("orchestrate", orchestrate_node)
    workflow.add_node("respond", respond_node)
    
    workflow.add_edge(START, "intent")
    workflow.add_edge("intent", "orchestrate")
    workflow.add_edge("orchestrate", "respond")
    workflow.add_edge("respond", END)
    
    return workflow.compile(checkpointer=checkpointer)
