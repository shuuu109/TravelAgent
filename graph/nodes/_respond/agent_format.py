"""Agent 结果格式化辅助：单 agent data → 纯文本段落 + 中文显示名映射。"""
from typing import Dict, List


def _format_agent_result(
    agent_name: str,
    data: dict,
    all_results: List[Dict],
) -> str:
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
                    lines.append(f"  - {note}")

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
                lines.append(f"  - {display_type} {action_text} {pref_value}")
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
                lines.append(f"  - 出发地: {origin}")
            if destination:
                lines.append(f"  - 目的地: {destination}")
            if start_date:
                lines.append(f"  - 出发日期: {start_date}")
            if end_date:
                lines.append(f"  - 返程日期: {end_date}")
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

def _get_agent_display_name(agent_name: str) -> str:
    """获取 agent 的中文显示名称"""
    agent_display_names = {
        "event_collection": "事项收集",
        "preference": "偏好管理",
        "itinerary_planning": "行程规划",
        "information_query": "信息查询",
        "rag_experience": "经验建议查询",
        "rag_risk": "避坑风险查询",
        "memory_query": "记忆查询",
        "transport_query": "交通查询",
        "accommodation_query": "住宿查询",
    }
    return agent_display_names.get(agent_name, agent_name)