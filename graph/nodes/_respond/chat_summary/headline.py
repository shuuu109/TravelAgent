from typing import Any, Dict, List

def _format_headline_text(headline: Dict[str, Any]) -> str:
    """
    将 chat_summary.headline 渲染为一行用户友好的中文摘要文本。

    示例输出：
      "已为您生成 杭州 → 成都 共 5 天的行程方案（2026-05-12 ~ 2026-05-16，2 人）。详细内容请查看右侧。"

    字段缺失时优雅降级，绝不抛错。
    """
    origin = headline.get("origin") or ""
    destination = headline.get("destination") or ""
    start_date = headline.get("start_date") or ""
    end_date = headline.get("end_date") or ""
    travel_days = headline.get("travel_days") or 0
    pax = headline.get("pax") or 1

    od = f"{origin} → {destination}" if origin and destination else (destination or "本次")
    days_part = f"共 {travel_days} 天的行程方案" if travel_days else "的行程方案"

    date_pax_parts: List[str] = []
    if start_date and end_date:
        date_pax_parts.append(f"{start_date} ~ {end_date}")
    elif start_date:
        date_pax_parts.append(start_date)
    if pax and pax > 0:
        date_pax_parts.append(f"{pax} 人")
    suffix = f"（{', '.join(date_pax_parts)}）" if date_pax_parts else ""

    return f"已为您生成 {od} {days_part}{suffix}。详细内容请查看右侧面板。"