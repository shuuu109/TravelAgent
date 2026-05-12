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
    options_by_name: Optional[Dict[str, Dict[str, Any]]] = None,
    address_by_hotel: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """
    决定是否输出酒店 timeline 事件，并构造它。

    判定规则：
      - current 为空（当晚无住宿，如最后一天返程） -> None
      - prev 为空（首晚） -> action="入住"
      - prev 存在但 hotel_name 不同 -> action="换住"
      - 同酒店连住 -> None（不刷屏）

    title  : suggested_hotel
    detail : "¥X/晚"（优先 daily_suggestions.price_per_night，
             兜底从 options[hotel_name].price_range 解析数字）
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

    detail = _format_hotel_price(current, hotel, options_by_name)
    address = ""
    if address_by_hotel:
        entry = address_by_hotel.get(hotel) or address_by_hotel.get(hotel.strip())
        if entry:
            address = (entry.get("address") or "").strip()
    # 兜底：options[hotel].area（可能是区域名而非完整地址，但聊胜于无）
    if not address and options_by_name:
        opt = options_by_name.get(hotel) or options_by_name.get(hotel.strip())
        if opt:
            address = (opt.get("area") or "").strip()

    event: Dict[str, Any] = {
        "type":   "hotel",
        "icon":   ICON_HOTEL,
        "action": action,
        "title":  hotel,
        "detail": detail,
    }
    if address:
        event["address"] = address
    return event


def _format_hotel_price(
    current: Dict[str, Any],
    hotel: str,
    options_by_name: Optional[Dict[str, Dict[str, Any]]],
) -> str:
    """
    生成酒店事件的 detail 价格字符串。

    优先级：
      1. current.price_per_night（数值）
      2. options[hotel].price_per_night
      3. options[hotel].price_range 内首段数字
    全部失败返回 ""。
    """
    direct = _coerce_price(current.get("price_per_night"))
    if direct:
        return direct

    if options_by_name:
        opt = options_by_name.get(hotel) or options_by_name.get(hotel.strip())
        if opt:
            from_opt = _coerce_price(opt.get("price_per_night"))
            if from_opt:
                return from_opt
            parsed = _parse_price_range(opt.get("price_range"))
            if parsed:
                return parsed
    return ""


def _coerce_price(value: Any) -> str:
    """把 price_per_night 转成 "¥X/晚"，兼容数值与可解析的字符串。"""
    if isinstance(value, bool):
        return ""
    if isinstance(value, (int, float)) and value > 0:
        return f"¥{int(round(float(value)))}/晚"
    if isinstance(value, str):
        return _parse_price_range(value)
    return ""


def _parse_price_range(price_range: Any) -> str:
    """
    从形如 "680元/晚" / "500-800 元/晚" / "¥680/晚" 中抽出首段整数，
    返回 "¥680/晚"。无法解析则返回 ""。
    """
    if not isinstance(price_range, str) or not price_range.strip():
        return ""
    import re
    m = re.search(r"\d+(?:\.\d+)?", price_range)
    if not m:
        return ""
    try:
        val = float(m.group(0))
    except ValueError:
        return ""
    if val <= 0:
        return ""
    return f"¥{int(round(val))}/晚"



def _build_daily_hotels(state: TravelGraphState) -> List[Dict[str, Any]]:
    """
    构建按天展开的酒店元数据（供前端在地图上叠加酒店标记）。

    与 _build_timeline 的差异：
      - 不做"连住去重"，每天若有酒店都会输出一条；
      - 携带坐标 lng/lat（解析自 address_by_hotel[name].location "lng,lat"）。

    Returns:
        [{"day": 1, "name": "...", "address": "..."?, "lng": 120.1?, "lat": 30.2?}, ...]
        没有 daily_suggestions 时返回 []。
    """
    acc_data: Dict[str, Any] = {}
    for r in state.get("skill_results", []) or []:
        if r.get("agent_name") == "accommodation_query" and r.get("status") == "success":
            acc_data = r.get("data") or {}
    accommodation_plan = acc_data.get("accommodation_plan") or {}
    suggestions = accommodation_plan.get("daily_suggestions") or []
    address_by_hotel: Dict[str, Dict[str, Any]] = (
        accommodation_plan.get("address_by_hotel") or {}
    )
    options_by_name: Dict[str, Dict[str, Any]] = {
        (o.get("hotel_name") or "").strip(): o
        for o in accommodation_plan.get("options") or []
        if isinstance(o, dict) and (o.get("hotel_name") or "").strip()
    }

    out: List[Dict[str, Any]] = []
    for s in suggestions:
        if not isinstance(s, dict):
            continue
        day = s.get("day")
        name = (s.get("suggested_hotel") or "").strip()
        if not isinstance(day, int) or not name:
            continue
        entry: Dict[str, Any] = {"day": day, "name": name}

        addr_meta = address_by_hotel.get(name) or {}
        addr = (addr_meta.get("address") or "").strip()
        if not addr:
            opt = options_by_name.get(name) or {}
            addr = (opt.get("area") or "").strip()
        if addr:
            entry["address"] = addr

        loc = (addr_meta.get("location") or "").strip()
        if loc and "," in loc:
            try:
                lng_str, lat_str = loc.split(",", 1)
                lng = float(lng_str)
                lat = float(lat_str)
                entry["lng"] = lng
                entry["lat"] = lat
            except (ValueError, TypeError):
                pass
        out.append(entry)
    return out


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
    accommodation_plan = acc_data.get("accommodation_plan") or {}
    suggestions_by_day: Dict[int, Dict[str, Any]] = {
        s.get("day"): s
        for s in accommodation_plan.get("daily_suggestions") or []
        if isinstance(s, dict)
    }
    # 按 hotel_name 索引 options，作为 daily_suggestions 缺失 price_per_night 时的兜底价格源
    options_by_name: Dict[str, Dict[str, Any]] = {
        (o.get("hotel_name") or "").strip(): o
        for o in accommodation_plan.get("options") or []
        if isinstance(o, dict) and (o.get("hotel_name") or "").strip()
    }
    # 由 accommodation_agent 填充：{hotel_name: {address, location("lng,lat")}}
    address_by_hotel: Dict[str, Dict[str, Any]] = (
        accommodation_plan.get("address_by_hotel") or {}
    )

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
        ev = _make_hotel_event(current_sugg, prev_sugg, options_by_name, address_by_hotel)
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