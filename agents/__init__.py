"""
Aligo Multi-Agent System - Agents Package (LangGraph 架构)
"""
# IntentionAgent / OrchestrationAgent 已迁移为 LangGraph 节点，不再作为独立类存在
# 实际加载通过 lazy_agent_registry 动态进行
from .transport_agent import TransportAgent
from .accommodation_agent import AccommodationAgent
from .lazy_agent_registry import LazyAgentRegistry

__all__ = [
    'TransportAgent',
    'AccommodationAgent',
    'LazyAgentRegistry',
]
