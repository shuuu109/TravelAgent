"""
交通查询共享辅助 _transport_helper

被 transport_outbound_node 与 transport_return_node 共用，
封装"单次调用 TransportAgent + 校验 options"的逻辑，避免两个节点重复代码。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from graph.state import TravelOption

logger = logging.getLogger(__name__)


def make_skill_result(
    agent_name: str,
    status: str,
    data: Dict[str, Any],
    message: str = "",
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"agent_name": agent_name, "status": status, "data": data}
    if message:
        out["message"] = message
    return out


async def query_one_date(
    agent,
    origin: str,
    destination: str,
    date: str,
    intent_data: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    单次调用 TransportAgent 查询指定日期的交通方案。

    Returns:
        (transport_plan, error_message)
            - 成功：(plan_dict, None)
            - 失败：(None, error_str)
    """
    key_entities: Dict[str, Any] = dict(intent_data.get("key_entities") or {})
    key_entities["origin"] = origin
    key_entities["destination"] = destination
    key_entities["date"] = date

    context: Dict[str, Any] = {
        "key_entities": key_entities,
        "rewritten_query": intent_data.get("rewritten_query", ""),
        "travel_style": intent_data.get("travel_style", "普通"),
    }
    input_data = {"context": context}

    try:
        result = await agent.run(input_data)
    except Exception as e:  # noqa: BLE001
        logger.error(f"[transport_helper] TransportAgent 执行失败 ({origin}→{destination}, {date}): {e}")
        return None, str(e)

    if isinstance(result, dict) and "error" in result:
        return None, str(result.get("error", ""))

    plan = (result or {}).get("transport_plan")
    if not plan:
        return None, "transport_plan empty"
    return plan, None


def validate_options(raw_options: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """对 transport_plan.options 做 Pydantic 校验，过滤非法项。"""
    validated: List[Dict[str, Any]] = []
    for i, opt in enumerate(raw_options or []):
        try:
            validated.append(TravelOption(**opt).model_dump())
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[transport_helper] TravelOption[{i}] 校验失败，跳过: {e}")
    return validated
