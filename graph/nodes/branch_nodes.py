"""
轻分支节点 (P2 静态分支)
用于 intent_type = memory_only / info_only 时的单 skill 直通调用。

设计：每个节点只调用一个 agent、把结果 flatten 进 skill_results。
preference 分支因为有"按 action 落库"的副作用复杂度，拆到独立 preference_node.py。
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from graph.state import TravelGraphState

logger = logging.getLogger(__name__)


def create_memory_query_node(registry):
    """memory_only 分支节点：调用 memory_query agent 回答用户对自身历史的询问。"""

    async def memory_query_node(state: TravelGraphState) -> dict:
        if state.get("intent_type") != "memory_only":
            return {}
        return await _run_single_agent(state, registry, "memory_query")

    return memory_query_node


def create_info_query_node(registry):
    """info_only 分支节点：调用 information_query agent 检索客观信息。"""

    async def info_query_node(state: TravelGraphState) -> dict:
        if state.get("intent_type") != "info_only":
            return {}
        return await _run_single_agent(state, registry, "information_query")

    return info_query_node


async def _run_single_agent(
    state: TravelGraphState,
    registry,
    agent_name: str,
) -> Dict[str, Any]:
    try:
        agent = registry[agent_name]
    except KeyError:
        logger.warning(f"[branch_nodes] {agent_name} agent 未注册")
        return {}

    intent_data: Dict[str, Any] = state.get("intent_data") or {}
    context: Dict[str, Any] = {
        "key_entities": intent_data.get("key_entities", {}),
        "rewritten_query": intent_data.get("rewritten_query", ""),
        "intents": intent_data.get("intents", []),
        "reasoning": intent_data.get("reasoning", ""),
    }
    input_data = {"context": context}

    try:
        result = await agent.run(input_data)
    except Exception as e:
        logger.error(f"[branch_nodes] {agent_name} 执行失败: {e}")
        return {"skill_results": [{
            "agent_name": agent_name,
            "status": "error",
            "data": {"error": str(e)},
            "message": str(e),
        }]}

    if isinstance(result, dict) and result.get("error"):
        return {"skill_results": [{
            "agent_name": agent_name,
            "status": "error",
            "data": result,
            "message": result.get("error", ""),
        }]}

    return {"skill_results": [{
        "agent_name": agent_name,
        "status": "success",
        "data": result,
    }]}
