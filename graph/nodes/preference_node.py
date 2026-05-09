"""
偏好节点 preference_node
职责：仅在 intent_type == "preference_only" 分支下执行；调用 PreferenceAgent
      抽取结构化偏好，并按 action（append/replace）持久化到长期记忆。

不在 fan-out 链路中：planning 链路里的住宿品牌偏好已由 intent_node 的"W mirror"
直接同步到 long_term.preferences['hotel_brands']，避免重复落库。

action 语义（沿用旧 orchestrate_node._update_memory 的处理规则）：
  - append  → 已存在列表则去重追加；非列表则与新值组装成 list 后保存
  - replace → 直接覆盖（默认）
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from graph.state import TravelGraphState

logger = logging.getLogger(__name__)


def create_preference_node(registry, memory_manager=None):
    """
    工厂函数：将 registry + memory_manager 通过闭包注入。
    PreferenceAgent 自身需要 memory_manager 读取当前偏好（已由 LazyAgentRegistry 自动注入）。
    """

    async def preference_node(state: TravelGraphState) -> dict:
        if state.get("intent_type") != "preference_only":
            return {}

        try:
            agent = registry["preference"]
        except KeyError:
            logger.warning("[preference_node] preference agent 未注册")
            return {}

        intent_data: Dict[str, Any] = state.get("intent_data") or {}
        context: Dict[str, Any] = {
            "key_entities": intent_data.get("key_entities", {}),
            "rewritten_query": intent_data.get("rewritten_query", ""),
        }
        input_data = {"context": context}

        try:
            result = await agent.run(input_data)
        except Exception as e:
            logger.error(f"[preference_node] PreferenceAgent 执行失败: {e}")
            return {"skill_results": [_skill_result(
                "preference", "error",
                {"error": str(e)}, message=str(e),
            )]}

        if isinstance(result, dict) and result.get("error"):
            return {"skill_results": [_skill_result(
                "preference", "error", result, message=result.get("error", ""),
            )]}

        # 落库：从 result.preferences 中按 action 写入长期记忆
        if memory_manager:
            _persist_preferences(result, memory_manager)

        return {"skill_results": [_skill_result("preference", "success", result)]}

    return preference_node


def _persist_preferences(data: Dict[str, Any], memory_manager) -> None:
    """
    将 PreferenceAgent 输出的偏好按 action 落到长期记忆。
    支持两种结构：
      - List[{type, value, action}]：新结构，逐条按 action 写入
      - Dict[str, value]：旧结构，整体 replace
    """
    preferences_data = data.get("preferences", {})

    if isinstance(preferences_data, list):
        for item in preferences_data:
            if not isinstance(item, dict):
                continue
            pref_type = item.get("type")
            pref_value = item.get("value")
            pref_action = item.get("action", "replace")
            if not pref_type or not pref_value:
                continue

            if pref_action == "append":
                current = memory_manager.long_term.get_preference()
                existing = current.get(pref_type)
                if isinstance(existing, list):
                    if pref_value not in existing:
                        existing.append(pref_value)
                    memory_manager.long_term.save_preference(pref_type, existing)
                    logger.info(f"[preference_node] append {pref_type}: {pref_value} → {existing}")
                else:
                    new_list: List[Any] = [existing, pref_value] if existing else [pref_value]
                    memory_manager.long_term.save_preference(pref_type, new_list)
                    logger.info(f"[preference_node] append (new list) {pref_type}: {new_list}")
            else:
                memory_manager.long_term.save_preference(pref_type, pref_value)
                logger.info(f"[preference_node] replace {pref_type}: {pref_value}")
    elif isinstance(preferences_data, dict):
        for pref_type, value in preferences_data.items():
            if value and pref_type not in ("has_preferences", "error"):
                memory_manager.long_term.save_preference(pref_type, value)
                logger.info(f"[preference_node] legacy replace {pref_type}: {value}")


def _skill_result(
    agent_name: str,
    status: str,
    data: Dict[str, Any],
    message: str = "",
) -> Dict[str, Any]:
    out = {"agent_name": agent_name, "status": status, "data": data}
    if message:
        out["message"] = message
    return out
