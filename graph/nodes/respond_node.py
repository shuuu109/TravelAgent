"""
回复生成节点 respond_node
职责：根据 skill_results 和 intent_data 生成用户可读的自然语言回复

来源：提取自 cli.py 的 _display_results / _generate_human_response 逻辑，
      并新增 LLM 兜底汇总（当规则生成无输出时）。

输入（从 state 读取）：
  - skill_results: List[Dict]   各 skill 的执行结果
  - intent_data: Dict           IntentionAgent 的完整输出

输出：
  {"final_response": response_text, "messages": [AIMessage(content=response_text)]}
"""
import logging
from typing import List, Dict, Any

from langchain_core.messages import AIMessage
from graph.state import TravelGraphState, ensure_hard_constraints

from graph.nodes._respond.agent_format import (
    _format_agent_result,
    _get_agent_display_name,
)
from graph.nodes._respond.llm_refine import (
    _llm_summarize,
    _build_user_context,
    _llm_refine_tips_risks,
    _parse_refined_tips_risks_lists,
)
from graph.nodes._respond.tips_risks import _collect_raw_tips_risks
from graph.nodes._respond.chat_summary import _build_chat_summary
from graph.nodes._respond.chat_summary.headline import _format_headline_text

logger = logging.getLogger(__name__)


def create_respond_node(llm, memory_manager=None):
    """
    工厂函数：将 LLM 与 memory_manager 通过闭包注入。

    Args:
        llm: LangChain ChatOpenAI 实例（用于兜底汇总 / 个性化润色）
        memory_manager: MemoryManager 实例（可选）。提供时启用 save_trip_history：
            当 intent_type == "planning" 且 hard_constraints 完整时，把本轮行程
            写入长期记忆，使后续会话可命中"我去过 X"等记忆查询。

    Returns:
        async 节点函数 respond_node(state) -> dict
    """

    async def respond_node(state: TravelGraphState) -> dict:
        """
        回复生成节点：
        - planning + daily_routes 路径：直出 chat_summary（结构化）+ headline 文本
        - 其他意图路径：保留规则渲染 markdown final_response
        - planning 路径完成后写 save_trip_history（长期记忆归档）
        """
        intent_type: str = state.get("intent_type", "")
        daily_routes: List[Dict] = state.get("daily_routes") or []

        # ── planning 分支：跳过 markdown 拼接，直出结构化 chat_summary ──
        if intent_type == "planning" and daily_routes:
            result = await _build_planning_response(state, llm)

            # 长期记忆：行程归档（仅 hard_constraints 完整时）
            if memory_manager:
                try:
                    hc = ensure_hard_constraints(state.get("hard_constraints"))
                    if hc.is_complete():
                        memory_manager.long_term.save_trip_history({
                            "origin": hc.origin,
                            "destination": hc.destination,
                            "start_date": hc.start_date,
                            "end_date": hc.end_date,
                            "purpose": "旅游",
                        })
                        logger.info(
                            f"[respond_node] save_trip_history: "
                            f"{hc.origin} -> {hc.destination} ({hc.start_date}~{hc.end_date})"
                        )
                except Exception as e:
                    logger.warning(f"[respond_node] save_trip_history 失败（非阻塞）: {e}")

            return result

        # ── 非 planning 分支：维持原规则渲染 markdown 逻辑 ──
        skill_results: List[Dict] = state.get("skill_results", [])
        intent_data: Dict[str, Any] = state.get("intent_data", {})
        text_parts: List[str] = []

        # 用规则逻辑生成各 agent 的文字片段
        if not skill_results:
            text_parts.append("好的，我已记录下来。您可以继续补充信息，或尝试规划行程、查询信息。")
        else:
            # 保留首次出现的 agent 顺序，但取最后一条同名结果
            # 目的：accommodation 降级后 skill_results 中存在两条同名结果，渲染最新的（降级后）那条
            _last_by_agent: dict = {}
            for r in skill_results:
                _last_by_agent[r.get("agent_name", "")] = r

            seen_agents: set = set()
            results_to_render = []
            for r in skill_results:
                name = r.get("agent_name", "")
                if name not in seen_agents:
                    seen_agents.add(name)
                    results_to_render.append(_last_by_agent[name])

            for result in results_to_render:
                agent_name = result.get("agent_name", "")
                status = result.get("status", "")
                data = result.get("data", {})

                if status == "error":
                    error_msg = data.get("error", "未知错误")
                    display_name = _get_agent_display_name(agent_name)
                    text_parts.append(f"{display_name}执行失败: {error_msg}")
                    continue

                if status != "success":
                    continue

                # rag_experience / rag_risk 的内容已从结构化 state 字段渲染，跳过 skill_results
                if agent_name in ("rag_experience", "rag_risk"):
                    continue

                part = _format_agent_result(agent_name, data, skill_results)
                if part:
                    text_parts.append(part)

        # 若规则无输出，用 LLM 做兜底汇总
        if not text_parts and skill_results:
            llm_summary = await _llm_summarize(skill_results, intent_data, llm)
            text_parts.append(llm_summary)

        response_text = "\n\n".join(text_parts) if text_parts else "已处理您的请求。"

        return {
            "final_response": response_text,
            "messages": [AIMessage(content=response_text)]
        }

    return respond_node


# =============================================================================
# planning 分支专用：跳过 markdown 拼接，直出 chat_summary + headline
# =============================================================================
async def _build_planning_response(state: TravelGraphState, llm) -> dict:
    """
    planning + daily_routes 路径专用：
      1. 从 rag_context 收集 raw_tips / raw_risks
      2. 用 LLM 结合用户特征 + 本次行程 POI 润色，得到 raw content
      3. 解析为 list[str] 喂给 _build_chat_summary
      4. final_response 降级为一行 headline 文本（详细内容由前端读 chat_summary 渲染）

    返回字典包含 final_response / chat_summary / messages，
    save_trip_history 由 respond_node 在调用本函数后统一处理。
    """
    daily_routes: List[Dict] = state.get("daily_routes") or []
    raw_tips, raw_risks = _collect_raw_tips_risks(state)

    refined_tips: List[str] = []
    refined_risks: List[str] = []
    if raw_tips or raw_risks:
        user_ctx = _build_user_context(state)
        # 抽取本次行程实际 POI 名称（去重保序），让 LLM 产出与具体景点相关的建议
        itinerary_pois: List[str] = []
        seen_pois: set = set()
        for day in daily_routes:
            for poi in day.get("ordered_pois", []) or []:
                name = (poi.get("name") or "").strip()
                if name and name not in seen_pois:
                    seen_pois.add(name)
                    itinerary_pois.append(name)

        raw_content = await _llm_refine_tips_risks(
            raw_tips, raw_risks, user_ctx, llm, itinerary_pois
        )
        if raw_content:
            refined_tips, refined_risks = _parse_refined_tips_risks_lists(raw_content)

    chat_summary = _build_chat_summary(state, refined_tips, refined_risks)
    final_response = _format_headline_text(chat_summary.get("headline", {}))

    return {
        "final_response": final_response,
        "chat_summary": chat_summary,
        "messages": [AIMessage(content=final_response)],
    }
