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
import json
import logging
from typing import List, Dict, Any

from langchain_core.messages import AIMessage
from graph.state import TravelGraphState

logger = logging.getLogger(__name__)


def create_respond_node(llm):
    """
    工厂函数：将 LLM 实例通过闭包注入，用于兜底汇总。

    Args:
        llm: LangChain ChatOpenAI 实例

    Returns:
        async 节点函数 respond_node(state) -> dict
    """

    async def respond_node(state: TravelGraphState) -> dict:
        """
        回复生成节点：
        1. 若 daily_routes 有值，用结构化路线格式渲染每日行程
        2. 优先用规则逻辑（从 cli.py 提取）生成各 skill 文字片段
        3. 若规则未产生输出，调用 LLM 做自然语言汇总
        4. 返回 final_response 和追加到 messages 的 AIMessage
        """
        skill_results: List[Dict] = state.get("skill_results", [])
        intent_data: Dict[str, Any] = state.get("intent_data", {})
        daily_routes: List[Dict] = state.get("daily_routes", [])

        # =====================================================================
        # 第一步：daily_routes 优先路径 — 结构化行程渲染
        # =====================================================================
        text_parts: List[str] = []
        has_daily_routes = bool(daily_routes)

        if has_daily_routes:
            text_parts.append(_format_daily_routes(daily_routes))

        # =====================================================================
        # 第二步：用规则逻辑生成各 agent 的文字片段
        # =====================================================================
        if not skill_results and not has_daily_routes:
            text_parts.append("好的，我已记录下来。您可以继续补充信息，或尝试规划行程、查询信息。")
        else:
            for result in skill_results:
                agent_name = result.get("agent_name", "")
                status = result.get("status", "")
                data = result.get("data", {})

                # daily_routes 已渲染行程，跳过 skill_results 中的 itinerary_planning 避免重复
                if agent_name == "itinerary_planning" and has_daily_routes:
                    continue

                if status == "error":
                    error_msg = data.get("error", "未知错误")
                    display_name = _get_agent_display_name(agent_name)
                    text_parts.append(f"{display_name}执行失败: {error_msg}")
                    continue

                if status != "success" and not (agent_name == "rag_knowledge" and status == "no_knowledge"):
                    continue

                part = _format_agent_result(agent_name, data, skill_results)
                if part:
                    text_parts.append(part)

        # =====================================================================
        # 第二步：若规则无输出，用 LLM 做兜底汇总
        # =====================================================================
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
# 内部辅助：各 agent 结果格式化（提取自 cli.py _generate_human_response）
# =============================================================================

def _format_agent_result(agent_name: str, data: dict, all_results: List[Dict]) -> str:
    """将单个 agent 的 data 格式化为纯文本段落，返回空字符串表示无内容。"""
    lines: List[str] = []

    # --- 行程规划 ---
    if agent_name == "itinerary_planning":
        itinerary = data.get("itinerary") or data.get("data", {}).get("itinerary")
        if itinerary:
            title = itinerary.get("title", "行程规划")
            lines.append(f"【{title}】")
            lines.append(f"时长: {itinerary.get('duration', '未知')}")
            for day_plan in itinerary.get("daily_plans", []):
                day_num = day_plan.get("day", 1)
                lines.append(f"\n第 {day_num} 天")
                activities = day_plan.get("activities") or day_plan.get("time_slots") or []
                for slot in activities:
                    time = slot.get("time", "")
                    activity = slot.get("activity") or slot.get("location") or ""
                    description = slot.get("description", "")
                    transport = slot.get("transport", "")
                    lines.append(f"  {time} - {activity}")
                    if description:
                        lines.append(f"    {description}")
                    if transport:
                        lines.append(f"    交通: {transport}")
                meals = day_plan.get("meals", {})
                if meals:
                    if meals.get("lunch"):
                        lines.append(f"  午餐: {meals['lunch']}")
                    if meals.get("dinner"):
                        lines.append(f"  晚餐: {meals['dinner']}")
            notes = itinerary.get("notes", [])
            if notes:
                lines.append("\n注意事项:")
                for note in notes:
                    lines.append(f"  • {note}")

    # --- 偏好管理 ---
    elif agent_name == "preference":
        raw_prefs = data.get("preferences") or data.get("data", {}).get("preferences")
        if isinstance(raw_prefs, dict):
            prefs_list = raw_prefs.get("preferences", [])
        else:
            prefs_list = raw_prefs if isinstance(raw_prefs, list) else []

        if prefs_list:
            lines.append("已更新您的偏好设置:")
            type_names = {
                "home_location": "常驻地",
                "transportation_preference": "交通偏好",
                "hotel_brands": "酒店偏好",
                "airlines": "航空公司偏好",
                "seat_preference": "座位偏好",
                "meal_preference": "餐食偏好",
                "budget_level": "预算等级"
            }
            for pref in prefs_list:
                pref_type = pref.get("type", "")
                pref_value = pref.get("value", "")
                action = pref.get("action", "replace")
                display_type = type_names.get(pref_type, pref_type)
                action_text = "追加" if action == "append" else "设置为"
                lines.append(f"  • {display_type} {action_text} {pref_value}")
            has_itinerary = any(r.get("agent_name") == "itinerary_planning" for r in all_results)
            if not has_itinerary:
                lines.append("下次规划行程时会参考这些偏好。")
        elif data.get("error"):
            lines.append(f"偏好未保存: {data['error']}")

    # --- 事项收集 ---
    elif agent_name == "event_collection":
        origin = data.get("origin") or data.get("data", {}).get("origin")
        destination = data.get("destination") or data.get("data", {}).get("destination")
        start_date = data.get("start_date") or data.get("data", {}).get("start_date")
        end_date = data.get("end_date") or data.get("data", {}).get("end_date")
        missing_info = data.get("missing_info") or data.get("data", {}).get("missing_info") or []

        has_itinerary = any(r.get("agent_name") == "itinerary_planning" for r in all_results)
        if not has_itinerary and (origin or destination):
            lines.append("已收集行程信息:")
            if origin:
                lines.append(f"  • 出发地: {origin}")
            if destination:
                lines.append(f"  • 目的地: {destination}")
            if start_date:
                lines.append(f"  • 出发日期: {start_date}")
            if end_date:
                lines.append(f"  • 返程日期: {end_date}")
        if missing_info:
            _MISSING_FIELD_NAMES = {
                "end_date": "返回日期",
                "duration_days": "行程天数",
                "return_location": "返程地点",
                "origin": "出发地",
                "destination": "目的地",
                "start_date": "出发日期",
                "trip_purpose": "出行目的",
            }
            missing_cn = [_MISSING_FIELD_NAMES.get(f, f) for f in missing_info]
            lines.append(f"还需要补充: {', '.join(missing_cn)}")

    # --- 信息查询 ---
    elif agent_name == "information_query":
        query_results = data.get("results") or data.get("data", {}).get("results") or data
        if not isinstance(query_results, dict):
            query_results = {}
        summary = query_results.get("summary", "")
        sources = query_results.get("sources", []) or []
        message = query_results.get("message", "")
        error = query_results.get("error", "")

        if summary:
            lines.append(summary)
        elif message:
            lines.append(message)
        elif error:
            lines.append(error)

        if sources:
            lines.append("\n参考来源:")
            for i, source in enumerate(sources[:3], 1):
                url = source.get("url", "") if isinstance(source, dict) else str(source)
                lines.append(f"  {i}. {url}")

    # --- RAG 知识库 ---
    elif agent_name == "rag_knowledge":
        answer = data.get("answer") or data.get("data", {}).get("answer") \
                 or data.get("content") or data.get("data", {}).get("content")
        if isinstance(answer, dict):
            answer = answer.get("answer", str(answer))
        if isinstance(answer, str) and answer.strip().startswith("{") and answer.strip().endswith("}"):
            try:
                json_obj = json.loads(answer)
                if isinstance(json_obj, dict) and "answer" in json_obj:
                    answer = json_obj["answer"]
            except Exception:
                pass
        if answer:
            lines.append(str(answer))

    # --- 记忆查询 ---
    elif agent_name == "memory_query":
        query_result = (
            data.get("answer") or data.get("result") or data.get("content")
            or data.get("data", {}).get("answer")
            or data.get("data", {}).get("result")
            or data.get("data", {}).get("content")
        )
        if query_result:
            lines.append(str(query_result))

    # --- 交通查询 ---
    elif agent_name == "transport_query":
        transport_plan = data.get("transport_plan") or data.get("data", {}).get("transport_plan")
        if transport_plan:
            qi = transport_plan.get("query_info", {})
            date_str = qi.get("date", "")
            lines.append(f"## 交通方案 ({date_str})")
            analysis = transport_plan.get("analysis", "")
            if analysis:
                lines.append(analysis)
            options = transport_plan.get("options", [])
            if options:
                lines.append("")
                lines.append("| 类型 | 班次 | 时间 | 耗时 | 参考价格 |")
                lines.append("|------|------|------|------|----------|")
                for opt in options:
                    t_type = opt.get("transport_type", "")
                    t_no = opt.get("transport_no") or "-"
                    dep = opt.get("departure_time") or "-"
                    arr = opt.get("arrival_time") or "-"
                    dur = opt.get("duration", "-")
                    price = opt.get("price_range", "暂无")
                    lines.append(f"| {t_type} | {t_no} | {dep}→{arr} | {dur} | {price} |")
            rec = transport_plan.get("recommendation", {})
            if rec:
                best = rec.get("best_choice", "")
                reason = rec.get("reason", "")
                arrival_hub = rec.get("arrival_hub") or rec.get("arrival_station", "")
                if best:
                    lines.append(f"\n**推荐**: {best}")
                if reason:
                    lines.append(f"**理由**: {reason}")
                if arrival_hub:
                    lines.append(f"**到达枢纽**: {arrival_hub}")
        elif data.get("error"):
            lines.append(f"交通查询失败: {data['error']}")

    # --- 住宿推荐 ---
    elif agent_name == "accommodation_query":
        acc_plan = data.get("accommodation_plan") or data.get("data", {}).get("accommodation_plan")
        if acc_plan:
            dest = acc_plan.get("destination", "")
            arrival_station = acc_plan.get("arrival_station", "")
            mcp_used = acc_plan.get("mcp_data_used", False)
            data_note = "（真实MCP数据）" if mcp_used else "（LLM推断）"
            lines.append(f"## 住宿推荐 - {dest} {data_note}")
            analysis = acc_plan.get("analysis", "")
            if analysis:
                lines.append(analysis)
            if arrival_station and arrival_station != "未知":
                lines.append(f"\n**到达枢纽**: {arrival_station}")
            areas = acc_plan.get("recommended_areas", [])
            if areas:
                lines.append("\n**推荐区域**:")
                for area in areas[:3]:
                    lines.append(f"  - {area.get('area_name', '')}：{area.get('reason', '')}")
            options = acc_plan.get("options", [])
            if options:
                lines.append("\n**酒店方案**:")
                for opt in options[:4]:
                    tier = opt.get("tier", "")
                    name = opt.get("hotel_name", "")
                    price = opt.get("price_range", "")
                    highlights = opt.get("highlights", "")
                    lines.append(f"  - [{tier}] **{name}** {price}  {highlights}")
            rec = acc_plan.get("recommendation", {})
            if rec:
                best = rec.get("best_choice", "")
                reason = rec.get("reason", "")
                tips = rec.get("booking_tips", "")
                if best:
                    lines.append(f"\n**综合推荐**: {best}")
                if reason:
                    lines.append(f"**推荐理由**: {reason}")
                if tips:
                    lines.append(f"**预订建议**: {tips}")
        elif data.get("error"):
            lines.append(f"住宿查询失败: {data['error']}")

    # --- 通用兜底 ---
    if not lines:
        common_keys = ["answer", "content", "result", "message", "summary", "text", "description"]
        for k in common_keys:
            if k in data and isinstance(data[k], str) and data[k].strip():
                lines.append(data[k])
                break
        if not lines and "data" in data and isinstance(data["data"], dict):
            for k in common_keys:
                if k in data["data"] and isinstance(data["data"][k], str) and data["data"][k].strip():
                    lines.append(data["data"][k])
                    break

    return "\n".join(lines)


async def _llm_summarize(skill_results: List[Dict], intent_data: Dict, llm) -> str:
    """当规则生成无任何文本时，用 LLM 将 skill_results 汇总为自然语言。"""
    results_text = json.dumps(skill_results, ensure_ascii=False, indent=2)
    user_query = intent_data.get("rewritten_query", "")
    prompt = f"""你是一个旅行助手。请根据以下各智能体的执行结果，生成一段简洁、自然的中文回复给用户。

用户问题：{user_query}

智能体执行结果：
{results_text}

要求：直接输出给用户看的文字，不要有额外的解释或JSON。"""

    try:
        response = await llm.ainvoke([{"role": "user", "content": prompt}])
        return response.content.strip()
    except Exception as e:
        logger.error(f"LLM summarize failed: {e}")
        return "已处理您的请求。"


def _format_daily_routes(daily_routes: List[Dict]) -> str:
    """
    将 daily_routes 渲染为结构化行程文本。

    格式（每天）：
      **第 N 天**：区域名
      景点A → (步行15分钟) → 景点B → (地铁20分钟) → 景点C
      总交通时长: X小时Y分钟
    """
    lines = ["## 每日行程路线"]

    for day_data in daily_routes:
        day_num = day_data.get("day", 1)
        ordered_pois = day_data.get("ordered_pois", [])
        legs = day_data.get("legs", [])
        total_duration = day_data.get("total_duration", 0)

        if not ordered_pois:
            continue

        region = _infer_region(ordered_pois)
        header = f"**第 {day_num} 天**"
        if region:
            header += f"：{region}"
        lines.append("")
        lines.append(header)

        # 构建 POI → (交通) → POI 链
        route_parts = [ordered_pois[0].get("name", "")]
        for i, leg in enumerate(legs):
            transport_str = _format_leg(leg)
            route_parts.append(f"({transport_str})")
            next_idx = i + 1
            if next_idx < len(ordered_pois):
                route_parts.append(ordered_pois[next_idx].get("name", ""))

        lines.append(" → ".join(route_parts))

        if total_duration > 0:
            lines.append(_format_duration(total_duration, prefix="总交通时长: "))

    return "\n".join(lines)


def _infer_region(pois: List[Dict]) -> str:
    """
    从 POI 地址列表中推断区域名（取各 POI 地址的区级前缀最长公共部分）。
    地址通常为 "XX市XX区XX路..."，尝试提取区名。
    """
    addresses = [p.get("address", "") for p in pois if p.get("address")]
    if not addresses:
        return ""

    # 尝试提取 "XX区" / "XX县" / "XX镇"
    import re
    district_pattern = re.compile(r"[\u4e00-\u9fa5]{1,5}[区县镇]")
    candidates: List[str] = []
    for addr in addresses:
        m = district_pattern.search(addr)
        if m:
            candidates.append(m.group())

    if not candidates:
        return ""

    # 返回出现次数最多的区名
    from collections import Counter
    most_common = Counter(candidates).most_common(1)[0][0]
    return most_common


def _format_leg(leg: Dict) -> str:
    """
    将单段交通 leg 格式化为简短说明，如 "步行15分钟"、"地铁20分钟"。
    若 steps 包含线路信息（如"地铁2号线"），则附加线路名。
    """
    mode = leg.get("mode", "") or "交通"
    duration = leg.get("duration", 0)
    steps = leg.get("steps", []) or []

    # 从 steps 中提取首条地铁/公交线路名
    line_name = ""
    if steps and isinstance(steps, list):
        for step in steps:
            if isinstance(step, dict):
                name = step.get("line_name") or step.get("lineName") or step.get("name", "")
            else:
                name = str(step)
            if name and any(kw in name for kw in ["号线", "路", "线路", "巴士"]):
                line_name = name
                break

    parts = [mode]
    if line_name:
        parts.append(line_name)
    if duration > 0:
        parts.append(_format_duration(duration))

    return "".join(parts)


def _format_duration(minutes: int, prefix: str = "") -> str:
    """将分钟数格式化为 'X小时Y分钟' 或 'Y分钟'。"""
    if minutes <= 0:
        return ""
    hours, mins = divmod(int(minutes), 60)
    if hours > 0 and mins > 0:
        result = f"{hours}小时{mins}分钟"
    elif hours > 0:
        result = f"{hours}小时"
    else:
        result = f"{mins}分钟"
    return f"{prefix}{result}"


def _get_agent_display_name(agent_name: str) -> str:
    """获取 agent 的中文显示名称"""
    agent_display_names = {
        "event_collection": "事项收集",
        "preference": "偏好管理",
        "itinerary_planning": "行程规划",
        "information_query": "信息查询",
        "rag_knowledge": "知识库查询",
        "memory_query": "记忆查询",
        "transport_query": "交通查询",
        "accommodation_query": "住宿查询",
    }
    return agent_display_names.get(agent_name, agent_name)
