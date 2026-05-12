# Timeline 构造（chat_summary.timeline）
from graph.state import TravelGraphState, ensure_hard_constraints
from typing import Any, Dict, List, Optional, Tuple


# 各类事件 icon（emoji 用 unicode 转义写入源码，避免 Windows 终端编码问题；
# 前端 Web 端直接渲染为图标）
ICON_TRANSPORT: str = "✈"        # 飞机
ICON_POI:        str = "🗺"   # 地图
ICON_HOTEL:      str = "🏨"   # 旅馆

# ISO 周一=0 .. 周日=6
WEEKDAY_CN: List[str] = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]


def _format_day_label(start_date: str, day_index: int, total_days: int) -> Tuple[str, str]:
    """
    根据 start_date("YYYY-MM-DD") 与 0-based day_index 计算该日的
    (iso_date, display_label)。

    - 有效 start_date  ->  ("2026-05-13", "05-13 周三")
    - 无效或缺失       ->  ("", f"第 {day_index+1} 天")
    """
    if not start_date:
        return "", f"第 {day_index + 1} 天"
    try:
        from datetime import datetime, timedelta
        base = datetime.strptime(start_date, "%Y-%m-%d")
        d = base + timedelta(days=day_index)
        iso = d.strftime("%Y-%m-%d")
        label = f"{d.strftime('%m-%d')} {WEEKDAY_CN[d.weekday()]}"
        return iso, label
    except (ValueError, IndexError):
        return "", f"第 {day_index + 1} 天"



def _make_transport_event(
    options: List[Dict[str, Any]],
    *,
    is_return: bool,
    origin: str,
    destination: str,
) -> Optional[Dict[str, Any]]:
    """
    从 transport options 中选首项（is_recommended 优先，否则 [0]），
    构造 timeline 交通事件。

    title 用「出发城市 → 到达城市」（返程时反向），保持示例口径；
    detail 拼装 transport_type + transport_no + price_range。

    Returns:
        event dict 或 None（options 为空）
    """
    if not options:
        return None
    chosen = next((o for o in options if o.get("is_recommended")), options[0])

    transport_type = chosen.get("transport_type", "") or ""
    transport_no   = chosen.get("transport_no") or ""
    dep_time       = chosen.get("departure_time") or ""
    price_range    = chosen.get("price_range") or ""

    if is_return:
        title = f"{destination} → {origin}" if origin and destination else "返程"
    else:
        title = f"{origin} → {destination}" if origin and destination else "去程"

    detail_parts: List[str] = []
    if transport_type:
        detail_parts.append(transport_type)
    if transport_no:
        detail_parts.append(transport_no)
    if price_range:
        detail_parts.append(price_range)

    return {
        "type":   "transport_return" if is_return else "transport_outbound",
        "icon":   ICON_TRANSPORT,
        "time":   dep_time,
        "title":  title,
        "detail": " ".join(detail_parts),
    }



def _make_poi_event(poi: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    构造单个 POI timeline 事件。

    title  : 景点 name
    detail : "约 Xh"（从 poi.estimated_hours 取，缺失则空串）

    poi 没有 name 时返回 None。
    """
    name = (poi.get("name") or "").strip()
    if not name:
        return None

    detail = ""
    hours = poi.get("estimated_hours")
    if isinstance(hours, (int, float)) and hours > 0:
        # 整数小时不带小数点，半小时保留 .5
        if abs(hours - round(hours)) < 0.05:
            detail = f"约 {int(round(hours))}h"
        else:
            detail = f"约 {hours:.1f}h"

    return {
        "type":   "poi",
        "icon":   ICON_POI,
        "title":  name,
        "detail": detail,
    }



def _make_hotel_event(
    current: Optional[Dict[str, Any]],
    prev: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    决定是否输出酒店 timeline 事件，并构造它。

    判定规则：
      - current 为空（当晚无住宿，如最后一天返程） -> None
      - prev 为空（首晚） -> action="入住"
      - prev 存在但 hotel_name 不同 -> action="换住"
      - 同酒店连住 -> None（不刷屏）

    title  : suggested_hotel
    detail : "¥X/晚"（price_per_night 存在时）
    """
    if not current:
        return None
    hotel = (current.get("suggested_hotel") or "").strip()
    if not hotel:
        return None

    if prev is None:
        action = "入住"
    else:
        prev_hotel = (prev.get("suggested_hotel") or "").strip()
        if prev_hotel == hotel:
            return None
        action = "换住"

    price = current.get("price_per_night")
    detail = f"¥{int(round(float(price)))}/晚" if isinstance(price, (int, float)) and price > 0 else ""

    return {
        "type":   "hotel",
        "icon":   ICON_HOTEL,
        "action": action,
        "title":  hotel,
        "detail": detail,
    }



def _build_timeline(state: TravelGraphState) -> List[Dict[str, Any]]:
    """
    装配 chat_summary.timeline：按天展开的事件流。

    数据源:
      - hard_constraints  -> origin / destination / start_date
      - state.travel_days
      - state.daily_routes        -> 每天 POI 事件
      - state.transport_options   -> Day 0 去程
      - state.transport_return_options -> 末日返程
      - skill_results[accommodation_query].accommodation_plan.daily_suggestions
        -> 入住 / 换住事件

    Returns:
        [{date, label, events: [...]}, ...]
        events 顺序：去程交通 -> POI -> 酒店 -> 返程交通
        最后一天若有返程交通，则当天不出酒店事件（避免显示无意义的换住）。
    """
    hard_constraints = ensure_hard_constraints(state.get("hard_constraints"))
    origin      = hard_constraints.origin or ""
    destination = hard_constraints.destination or ""
    start_date  = hard_constraints.start_date or ""

    # 旅行天数兜底（依次：state.travel_days -> 日期算 -> daily_routes 长度 -> 1）
    travel_days: int = state.get("travel_days") or 0
    if travel_days <= 0 and start_date and hard_constraints.end_date:
        try:
            from datetime import datetime
            sd = datetime.strptime(start_date, "%Y-%m-%d")
            ed = datetime.strptime(hard_constraints.end_date, "%Y-%m-%d")
            travel_days = (ed - sd).days + 1
        except Exception:
            travel_days = 0
    if travel_days <= 0:
        travel_days = len(state.get("daily_routes") or []) or 1

    # 按 day 索引 daily_routes / daily_suggestions（数据源用 1-based day 字段）
    routes_by_day: Dict[int, Dict[str, Any]] = {
        r.get("day"): r for r in (state.get("daily_routes") or []) if isinstance(r, dict)
    }

    acc_data: Dict[str, Any] = {}
    for r in state.get("skill_results", []) or []:
        if r.get("agent_name") == "accommodation_query" and r.get("status") == "success":
            acc_data = r.get("data") or {}
    suggestions_by_day: Dict[int, Dict[str, Any]] = {
        s.get("day"): s
        for s in (acc_data.get("accommodation_plan") or {}).get("daily_suggestions") or []
        if isinstance(s, dict)
    }

    outbound_opts = state.get("transport_options") or []
    return_opts   = state.get("transport_return_options") or []
    has_return    = bool(return_opts)

    timeline: List[Dict[str, Any]] = []

    for day_idx in range(travel_days):
        iso, label = _format_day_label(start_date, day_idx, travel_days)
        day_num = day_idx + 1
        events: List[Dict[str, Any]] = []

        # 1) Day 0 去程
        if day_idx == 0:
            ev = _make_transport_event(
                outbound_opts, is_return=False, origin=origin, destination=destination,
            )
            if ev:
                events.append(ev)

        # 2) 当天 POI
        day_route = routes_by_day.get(day_num) or {}
        for poi in day_route.get("ordered_pois") or []:
            ev = _make_poi_event(poi)
            if ev:
                events.append(ev)

        # 3) 酒店事件
        #    最后一天若有返程，跳过（current 置 None）
        is_last_day = (day_idx == travel_days - 1)
        if is_last_day and has_return:
            current_sugg = None
        else:
            current_sugg = suggestions_by_day.get(day_num)
        prev_sugg = suggestions_by_day.get(day_num - 1) if day_idx > 0 else None
        ev = _make_hotel_event(current_sugg, prev_sugg)
        if ev:
            events.append(ev)

        # 4) 末日返程
        if is_last_day and has_return:
            ev = _make_transport_event(
                return_opts, is_return=True, origin=origin, destination=destination,
            )
            if ev:
                events.append(ev)

        timeline.append({"date": iso, "label": label, "events": events})

    return timeline