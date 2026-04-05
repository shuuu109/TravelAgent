"""
date_resolver.py
职责：将中文相对日期表达式（下周六、后天、大后天等）确定性地转换为 YYYY-MM-DD 格式日期。

支持的表达式：
  - 今天 / 明天 / 后天 / 大后天
  - 本周X / 这周X（含本周末）
  - 下周X（含下周末）
  - 下下周X
  - X月X日 / X月X号（无年份时根据当前时间推断）
  - YYYY年X月X日 / YYYY-MM-DD（已是具体日期时原样返回规范格式）

返回值：
  - 解析成功：str，格式 "YYYY-MM-DD"
  - 无法解析：None（调用方保留原始字符串）
"""

import re
from datetime import date, timedelta
from typing import Optional


# ── 星期中文映射 ──────────────────────────────────────────────
_WEEKDAY_MAP: dict[str, int] = {
    "一": 0, "周一": 0, "礼拜一": 0,
    "二": 1, "周二": 1, "礼拜二": 1,
    "三": 2, "周三": 2, "礼拜三": 2,
    "四": 3, "周四": 3, "礼拜四": 3,
    "五": 4, "周五": 4, "礼拜五": 4,
    "六": 5, "周六": 5, "礼拜六": 5, "末": 5,
    "日": 6, "天": 6, "周日": 6, "周天": 6, "礼拜天": 6, "礼拜日": 6,
}


def _weekday_key(text: str) -> Optional[int]:
    """从文本片段提取 weekday 编号（0=周一 … 6=周日），失败返回 None。"""
    # 优先匹配两字符，再匹配单字符，避免 "六" 误匹配 "礼拜六" 的局部
    for key in sorted(_WEEKDAY_MAP, key=len, reverse=True):
        if text.endswith(key) or text == key:
            return _WEEKDAY_MAP[key]
    return None


def resolve_relative_date(expr: str, today: Optional[date] = None) -> Optional[str]:
    """
    将中文相对日期表达式解析为 YYYY-MM-DD 字符串。

    Args:
        expr:  原始日期表达式，如 "下周六"、"后天"、"3月15日" 等
        today: 基准日期，默认使用 date.today()

    Returns:
        "YYYY-MM-DD" 字符串，或 None（无法解析时）
    """
    if not expr:
        return None

    base = today or date.today()
    expr = expr.strip()

    # ── 1. 已是完整日期 YYYY-MM-DD ────────────────────────────
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", expr)
    if m:
        return expr  # 已经是标准格式

    # ── 2. YYYY年M月D日 ────────────────────────────────────────
    m = re.search(r"(\d{4})[年/](\d{1,2})[月/-](\d{1,2})[日号]?", expr)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return date(y, mo, d).strftime("%Y-%m-%d")
        except ValueError:
            pass

    # ── 3. 今天 / 明天 / 后天 / 大后天 ────────────────────────
    # 注意：必须按字符串长度从长到短排列，防止"后天"子串提前匹配"大后天"
    _offset_map = [
        ("大后天", 3),
        ("后天",   2), ("后日", 2),
        ("明天",   1), ("明日", 1),
        ("今天",   0), ("今日", 0),
    ]
    for keyword, offset in _offset_map:
        if keyword in expr:
            return (base + timedelta(days=offset)).strftime("%Y-%m-%d")

    # ── 4. 下下周X ─────────────────────────────────────────────
    m = re.search(r"下下周(.+)", expr)
    if m:
        target_wd = _weekday_key(m.group(1))
        if target_wd is not None:
            # 下下周 = 本周起 +14 天后所在周
            days_to_next_monday = (7 - base.weekday()) % 7 or 7
            next_week_monday = base + timedelta(days=days_to_next_monday)
            target = next_week_monday + timedelta(days=7 + target_wd)
            return target.strftime("%Y-%m-%d")

    # ── 5. 下周X ───────────────────────────────────────────────
    m = re.search(r"下周(.+)", expr)
    if m:
        target_wd = _weekday_key(m.group(1))
        if target_wd is not None:
            # 下周一 = 距本周一 +7 天
            days_to_next_monday = (7 - base.weekday()) % 7 or 7
            next_week_monday = base + timedelta(days=days_to_next_monday)
            target = next_week_monday + timedelta(days=target_wd)
            return target.strftime("%Y-%m-%d")

    # ── 6. 本周X / 这周X ──────────────────────────────────────
    m = re.search(r"(?:本周|这周)(.+)", expr)
    if m:
        target_wd = _weekday_key(m.group(1))
        if target_wd is not None:
            this_week_monday = base - timedelta(days=base.weekday())
            target = this_week_monday + timedelta(days=target_wd)
            return target.strftime("%Y-%m-%d")

    # ── 7. X月X日 / X月X号（无年份）──────────────────────────
    m = re.search(r"(\d{1,2})月(\d{1,2})[日号]?", expr)
    if m:
        mo, d = int(m.group(1)), int(m.group(2))
        # 优先推断为当年；如果已过则推断为明年
        try:
            candidate = date(base.year, mo, d)
            if candidate < base:
                candidate = date(base.year + 1, mo, d)
            return candidate.strftime("%Y-%m-%d")
        except ValueError:
            pass

    return None  # 无法解析


def resolve_date_in_entities(key_entities: dict, today: Optional[date] = None) -> dict:
    """
    对 intent_data['key_entities'] 中的 date / start_date / end_date 字段
    批量执行相对日期解析。原地修改并返回 key_entities（方便链式调用）。

    Args:
        key_entities: intent_data["key_entities"] 字典
        today:        基准日期，默认 date.today()

    Returns:
        更新后的 key_entities 字典
    """
    if not isinstance(key_entities, dict):
        return key_entities

    for field in ("date", "start_date", "end_date"):
        raw = key_entities.get(field)
        if raw and isinstance(raw, str):
            resolved = resolve_relative_date(raw, today)
            if resolved:
                key_entities[field] = resolved

    return key_entities
