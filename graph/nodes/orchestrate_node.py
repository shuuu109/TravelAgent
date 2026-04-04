"""
编排节点 orchestrate_node
职责：将 OrchestrationAgent.reply() + _execute_parallel_agents() + _execute_agent() 逻辑
      转换为 LangGraph 节点函数

改动点（相比 agents/orchestration_agent.py）：
- 函数签名：async def orchestrate_node(state: TravelGraphState) -> dict
- 输入：从 state["intent_schedule"]、state["intent_data"] 读取，无需 Msg 解析
- Agent 调用：agent.run(input_data_dict) 代替 agent.reply(Msg(...))
  result 已是 dict，不需要 json.loads
- 输出：{"skill_results": all_results}（通过 operator.add reducer 追加到全局状态）
- 使用工厂函数将 registry 和 memory_manager 通过闭包注入
"""
import json
import logging
import asyncio
from typing import Dict, Any, List

from graph.state import TravelGraphState, TravelOption

logger = logging.getLogger(__name__)


def create_orchestrate_node(registry, memory_manager=None):
    """
    工厂函数：将 agent registry 和 memory_manager 通过闭包注入。

    Args:
        registry: dict[str, agent]，agent 需实现 async run(input_data: dict) -> dict
        memory_manager: MemoryManager 实例（可选），用于读取短期/长期记忆上下文

    Returns:
        async 节点函数 orchestrate_node(state) -> dict
    """

    async def orchestrate_node(state: TravelGraphState) -> dict:
        """
        编排节点主流程：
        1. 从 state 读取 intent_schedule 和 intent_data
        2. 准备上下文（记忆 + 意图数据）
        3. 按优先级分组，同优先级并行执行
        4. 聚合所有结果，更新记忆
        5. 返回 {"skill_results": all_results}
        """
        agent_schedule: List[Dict] = state.get("intent_schedule", [])
        intent_data: Dict[str, Any] = state.get("intent_data", {})

        if not agent_schedule:
            return {"skill_results": []}

        # 按优先级排序
        sorted_schedule = sorted(agent_schedule, key=lambda x: x.get("priority", 999))
        logger.info(f"Orchestrating {len(sorted_schedule)} agents")

        # 后处理：accommodation_query 依赖 transport_query 的到达枢纽，
        # 若两者同处 priority=1（并行），自动将 accommodation_query 升为 priority=2
        has_transport = any(t.get("agent_name") == "transport_query" for t in sorted_schedule)
        if has_transport:
            for t in sorted_schedule:
                if t.get("agent_name") == "accommodation_query" and t.get("priority") == 1:
                    t["priority"] = 2
                    logger.info("Auto-elevated accommodation_query to priority=2 (depends on transport_query)")

        # 后处理：itinerary_planning 需要使用 accommodation_query 的结果，
        # 若两者同处同一 priority（并行），自动将 itinerary_planning 升至 accommodation_query priority + 1
        has_accommodation = any(t.get("agent_name") == "accommodation_query" for t in sorted_schedule)
        has_itinerary = any(t.get("agent_name") == "itinerary_planning" for t in sorted_schedule)
        if has_accommodation and has_itinerary:
            accommodation_priority = next(
                t.get("priority", 1) for t in sorted_schedule if t.get("agent_name") == "accommodation_query"
            )
            for t in sorted_schedule:
                if t.get("agent_name") == "itinerary_planning" and t.get("priority", 0) <= accommodation_priority:
                    t["priority"] = accommodation_priority + 1
                    logger.info(
                        f"Auto-elevated itinerary_planning to priority={accommodation_priority + 1} "
                        "(depends on accommodation_query)"
                    )

        # 准备上下文
        context = _prepare_context(intent_data, memory_manager)

        # 按优先级分组并行执行
        all_results: List[Dict] = []
        current_priority = None
        parallel_tasks: List[Dict] = []

        for task in sorted_schedule:
            priority = task.get("priority", 0)
            if current_priority is not None and priority != current_priority:
                if parallel_tasks:
                    batch = await _execute_parallel_agents(parallel_tasks, context, all_results, registry)
                    all_results.extend(batch)
                    # 每批执行完后，将 transport_query / accommodation_query 结果注入 context，供后续批次使用
                    _enrich_context_with_transport(batch, context)
                    _enrich_context_with_accommodation(batch, context)
                    parallel_tasks = []
            current_priority = priority
            parallel_tasks.append(task)

        if parallel_tasks:
            batch = await _execute_parallel_agents(parallel_tasks, context, all_results, registry)
            all_results.extend(batch)
            _enrich_context_with_transport(batch, context)
            _enrich_context_with_accommodation(batch, context)

        # 更新记忆（使用原始嵌套结构，_update_memory 依赖 result["result"] 层）
        if memory_manager:
            _update_memory(intent_data, all_results, memory_manager)

        # 展平 all_results：将嵌套的 result 层提升，使 respond_node 可直接读取
        # 原结构: {"agent_name": ..., "priority": ..., "result": {"status": ..., "data": ...}}
        # 展平后: {"agent_name": ..., "status": ..., "data": ...}
        flat_results = []
        for r in all_results:
            inner = r.get("result", {})
            flat = {
                "agent_name": r.get("agent_name", ""),
                "status": inner.get("status", ""),
                "data": inner.get("data", {}),
            }
            if "message" in inner:
                flat["message"] = inner["message"]
            flat_results.append(flat)

        # 从 transport_query 结果中提取 transport_options 写回 state
        state_updates: dict = {"skill_results": flat_results}
        for r in flat_results:
            if r.get("agent_name") == "transport_query":
                data = r.get("data", {})
                transport_plan = data.get("transport_plan", {})
                raw_options = transport_plan.get("options", [])
                if raw_options:
                    validated_options = []
                    for i, opt in enumerate(raw_options):
                        try:
                            validated_options.append(TravelOption(**opt).model_dump())
                        except Exception as e:
                            logger.warning(f"TravelOption validation failed for option[{i}], skipping: {e}")
                    if validated_options:
                        state_updates["transport_options"] = validated_options
                break

        return state_updates

    return orchestrate_node


# =============================================================================
# 内部辅助函数（模块私有，非节点接口）
# =============================================================================

def _prepare_context(intent_data: Dict[str, Any], memory_manager) -> Dict[str, Any]:
    """构建传递给各 skill agent 的上下文（与 OrchestrationAgent._prepare_context 一致）"""
    context = {
        "reasoning": intent_data.get("reasoning", ""),
        "intents": intent_data.get("intents", []),
        "key_entities": intent_data.get("key_entities", {}),
        "rewritten_query": intent_data.get("rewritten_query", "")
    }
    if memory_manager:
        recent_context = memory_manager.short_term.get_recent_context(3)
        context["recent_dialogue"] = recent_context
        preferences = memory_manager.long_term.get_preference()
        context["user_preferences"] = preferences
    return context


async def _execute_parallel_agents(
    tasks: List[Dict],
    context: Dict[str, Any],
    previous_results: List[Dict],
    registry: dict
) -> List[Dict]:
    """同优先级的任务并行执行（与 OrchestrationAgent._execute_parallel_agents 一致）"""
    if not tasks:
        return []

    if len(tasks) == 1:
        task = tasks[0]
        result = await _execute_agent(
            agent_name=task.get("agent_name"),
            context=context,
            reason=task.get("reason", ""),
            expected_output=task.get("expected_output", ""),
            previous_results=previous_results,
            registry=registry
        )
        return [{
            "agent_name": task.get("agent_name"),
            "priority": task.get("priority", 0),
            "result": result
        }]

    logger.info(f"Executing {len(tasks)} agents in parallel")

    parallel_coroutines = []
    for task in tasks:
        agent_name = task.get("agent_name")
        priority = task.get("priority", 0)
        reason = task.get("reason", "")
        expected_output = task.get("expected_output", "")
        logger.info(f"Parallel executing agent: {agent_name} (priority={priority})")
        coro = _execute_agent(
            agent_name=agent_name,
            context=context,
            reason=reason,
            expected_output=expected_output,
            previous_results=previous_results,
            registry=registry
        )
        parallel_coroutines.append((agent_name, priority, coro))

    execution_results = await asyncio.gather(
        *[coro for _, _, coro in parallel_coroutines],
        return_exceptions=True
    )

    results = []
    for (agent_name, priority, _), exec_result in zip(parallel_coroutines, execution_results):
        if isinstance(exec_result, Exception):
            logger.error(f"Parallel agent execution failed: {agent_name}, error: {exec_result}")
            result = {
                "status": "error",
                "agent_name": agent_name,
                "data": {"error": str(exec_result)},
                "message": f"并行执行失败: {str(exec_result)}"
            }
        else:
            result = exec_result
        results.append({
            "agent_name": agent_name,
            "priority": priority,
            "result": result
        })

    return results


async def _execute_agent(
    agent_name: str,
    context: Dict[str, Any],
    reason: str,
    expected_output: str,
    previous_results: List[Dict],
    registry: dict
) -> Dict[str, Any]:
    """
    执行单个 skill agent。

    改动点：
    - 旧：input_msg = Msg(...); response = await agent.reply(input_msg); result = json.loads(response.content)
    - 新：input_data = dict; result = await agent.run(input_data)  # result 已是 dict
    """
    if agent_name not in registry:
        logger.warning(f"Agent not registered: {agent_name}")
        return {
            "status": "error",
            "message": f"智能体未注册: {agent_name}"
        }

    agent = registry[agent_name]

    # 新：直接传 dict，不再包装为 Msg
    input_data = {
        "context": context,
        "reason": reason,
        "expected_output": expected_output,
        "previous_results": previous_results
    }

    try:
        # 新：agent.run(input_data) 返回 dict，不需要 json.loads
        result = await agent.run(input_data)

        if isinstance(result, dict) and "error" in result:
            error_msg = result.get("error", "未知错误")
            return {
                "status": "error",
                "agent_name": agent_name,
                "data": result,
                "message": error_msg
            }

        return {
            "status": "success",
            "agent_name": agent_name,
            "data": result
        }

    except Exception as e:
        logger.error(f"Agent execution failed: {agent_name}, error: {e}")
        return {
            "status": "error",
            "agent_name": agent_name,
            "data": {"error": str(e)},
            "message": f"智能体执行失败: {str(e)}"
        }


def _enrich_context_with_transport(batch: List[Dict], context: Dict[str, Any]) -> None:
    """批次执行完后，将 transport_query 的结果注入 context，供下一批次（如 accommodation_query）使用。"""
    for r in batch:
        if r.get("agent_name") == "transport_query":
            data = r.get("result", {}).get("data", {})
            transport_plan = data.get("transport_plan", {})
            options = transport_plan.get("options", [])
            recommendation = transport_plan.get("recommendation", {})
            if options:
                context["transport_options"] = options
            if recommendation:
                context["transport_recommendation"] = recommendation
            break


def _enrich_context_with_accommodation(batch: List[Dict], context: Dict[str, Any]) -> None:
    """批次执行完后，将 accommodation_query 的结果注入 context，供下一批次（itinerary_planning）使用。"""
    for r in batch:
        if r.get("agent_name") == "accommodation_query":
            data = r.get("result", {}).get("data", {})
            recommendations = data.get("recommendations", [])
            recommendation = data.get("recommendation", {})
            if recommendations:
                context["accommodation_recommendations"] = recommendations
            if recommendation:
                context["accommodation_recommendation"] = recommendation
            logger.info("Injected accommodation_query results into context for itinerary_planning")
            break


def _update_memory(intent_data: Dict[str, Any], results: List[Dict], memory_manager):
    """
    更新长期/短期记忆（逻辑与 OrchestrationAgent._update_memory 完全一致）
    """
    for result in results:
        agent_name = result["agent_name"]
        data = result["result"].get("data", {})

        # 交通查询日志
        if agent_name == "transport_query" and isinstance(data, dict):
            if "transport_plan" in data:
                options = data["transport_plan"].get("options", [])
                logger.info(f"[Transport] 交通查询成功，共查询到 {len(options)} 个方案。")
            else:
                logger.warning(f"[Transport] 交通查询失败: {data.get('error', '未知原因')}")

        # 偏好管理
        if agent_name == "preference" and isinstance(data, dict):
            preferences_data = data.get("preferences", {})

            if isinstance(preferences_data, list):
                for pref_item in preferences_data:
                    if not isinstance(pref_item, dict):
                        continue
                    pref_type = pref_item.get("type")
                    pref_value = pref_item.get("value")
                    pref_action = pref_item.get("action", "replace")

                    if not pref_type or not pref_value:
                        continue

                    if pref_action == "append":
                        current_prefs = memory_manager.long_term.get_preference()
                        existing_value = current_prefs.get(pref_type)
                        if isinstance(existing_value, list):
                            if pref_value not in existing_value:
                                existing_value.append(pref_value)
                            memory_manager.long_term.save_preference(pref_type, existing_value)
                            logger.info(f"Appended to {pref_type}: {pref_value}, total: {existing_value}")
                        else:
                            new_list = [existing_value, pref_value] if existing_value else [pref_value]
                            memory_manager.long_term.save_preference(pref_type, new_list)
                            logger.info(f"Created list for {pref_type}: {new_list}")
                    else:
                        memory_manager.long_term.save_preference(pref_type, pref_value)
                        logger.info(f"Replaced {pref_type}: {pref_value}")

            elif isinstance(preferences_data, dict):
                for pref_type, value in preferences_data.items():
                    if value and pref_type != "has_preferences" and pref_type != "error":
                        memory_manager.long_term.save_preference(pref_type, value)
                        logger.info(f"Updated {pref_type}: {value} (legacy format)")

        # 行程规划 → 写入行程历史
        if agent_name == "itinerary_planning" and isinstance(data, dict):
            itinerary = data.get("itinerary", {})
            if itinerary:
                event_data = {}
                for r in results:
                    if r["agent_name"] == "event_collection":
                        event_data = r["result"].get("data", {})
                        break

                destination = event_data.get("destination")
                if destination:
                    memory_manager.long_term.save_trip_history({
                        "origin": event_data.get("origin"),
                        "destination": destination,
                        "start_date": event_data.get("start_date"),
                        "end_date": event_data.get("end_date"),
                        "purpose": event_data.get("trip_purpose", "旅游")
                    })
                    logger.info(f"Saved trip to long-term memory: {event_data.get('origin')} -> {destination}")

    logger.info("Memory updated after orchestration")
