"""
约束提取节点 extract_constraints_node (P1.4)

职责：作为 hard_constraints 的单一真源（Single Source of Truth）。
      P1 intent_node 已经把自然语言日期/时长解析成 key_entities 中的结构化字段，
      本节点只做轻量级映射与合并（Route A：不再调用 LLM）。


  1. 结构化 hard_constraints：
     - 从 intent_data.key_entities 读 origin / destination / date / duration
     - 映射 date → start_date；由 start_date + duration 推算 end_date
     - 与已存在的 hard_constraints（可能来自 checkpointer 续跑）做"新值优先"合并

  2. 衍生字段刷新：
     - missing_info：origin / destination / start_date 三项缺失即列入
     - travel_days：优先用 start_date / end_date 计算，退化为 duration 字符串解析

  3. 跨轮状态清理（配合 LangGraph checkpointer）：
     - rule_violations → []
     - review_retry_count → 0
     - skill_results → SKILL_RESULTS_RESET sentinel
       （触发 state.py 中 skill_results_reducer 清空，避免上一轮残留污染）

设计要点：
  - 无 LLM 调用：完全依赖 intent_node 输出，确定性高、不引入额外延迟
  - 支持 dict 形态的 hard_constraints：防御 checkpointer 反序列化差异
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from graph.state import (
    HardConstraints,
    SKILL_RESULTS_RESET,
    TravelGraphState,
)

logger = logging.getLogger(__name__)

# 硬约束中必填的三个字段（用于生成 missing_info）
_REQUIRED_FIELDS: List[Tuple[str, str]] = [
    ("origin", "出发地"),
    ("destination", "目的地"),
    ("start_date", "出发时间"),
]


def _parse_duration_days(duration: Any) -> int:
    """
    从 key_entities.duration 提取天数。

    支持："3天"、"3日"、"5天4晚" → 3 / 3 / 5；无法解析返回 0。
    中文汉字（如"三天"）未覆盖，intent_node 端应输出阿拉伯数字。
    """
    if duration is None:
        return 0
    m = re.search(r"(\d+)", str(duration))
    return int(m.group(1)) if m else 0


def _compute_end_date(start_date: Optional[str], days: int) -> Optional[str]:
    """
    由 start_date（YYYY-MM-DD）和天数推算 end_date，含首含尾。

    例：start_date=2026-04-25, days=3 → 2026-04-27（4/25、4/26、4/27 共 3 天）。
    解析失败返回 None，不抛异常。
    """
    if not start_date or days <= 0:
        return None
    try:
        dt = datetime.strptime(start_date, "%Y-%m-%d")
    except ValueError:
        return None
    return (dt + timedelta(days=days - 1)).strftime("%Y-%m-%d")


def _compute_travel_days(
    start_date: Optional[str],
    end_date: Optional[str],
    fallback_days: int,
    state_days: int,
) -> int:
    """
    travel_days 计算优先级：
      1. start_date + end_date 都存在 → 日期差 +1（含首含尾）
      2. 本轮解析到 duration_days → 用之
      3. state 中已有 travel_days → 保留（checkpoint 续跑友好）
      4. 兜底 0，由下游节点自行降级
    """
    if start_date and end_date:
        try:
            s = datetime.strptime(start_date, "%Y-%m-%d")
            e = datetime.strptime(end_date, "%Y-%m-%d")
            days = (e - s).days + 1
            if days > 0:
                return days
        except ValueError:
            pass
    if fallback_days > 0:
        return fallback_days
    return state_days if state_days > 0 else 0


def create_extract_constraints_node(memory_manager=None):
    """
    工厂函数：返回 extract_constraints_node 异步节点。

    Args:
        memory_manager: MemoryManager 实例（可选），用于从 home_location 偏好回填缺失的 origin。

    与其他节点保持 create_xxx_node() 风格；无需 LLM 依赖。
    """

    async def extract_constraints_node(state: TravelGraphState) -> Dict[str, Any]:
        intent_data: Dict[str, Any] = state.get("intent_data") or {}
        key_entities: Dict[str, Any] = intent_data.get("key_entities") or {}

        # ── 1. 读取本轮 key_entities 中的新值（空字符串也当作缺失处理）────────
        def _clean(v: Any) -> Optional[str]:
            if v is None:
                return None
            s = str(v).strip()
            return s if s else None

        new_origin: Optional[str] = _clean(key_entities.get("origin"))
        new_destination: Optional[str] = _clean(key_entities.get("destination"))
        new_start_date: Optional[str] = _clean(key_entities.get("date"))
        duration_days: int = _parse_duration_days(key_entities.get("duration"))

        # ── 2. 先加载已有 hard_constraints，供后续合并和换算使用 ─────────────────
        existing = state.get("hard_constraints")
        if existing is None:
            existing = HardConstraints()
        elif isinstance(existing, dict):
            existing = HardConstraints(**existing)

        # ── pax：从本轮 key_entities 解析，失败则保留已有值 ────────────────────
        new_pax: Optional[int] = None
        raw_pax = key_entities.get("pax")
        if raw_pax is not None:
            try:
                new_pax = int(str(raw_pax).strip())
            except (ValueError, TypeError):
                pass

        # ── budget：解析金额 + 类型，归一化为人均值 ──────────────────────────
        new_budget_per_person: Optional[float] = None
        raw_budget = key_entities.get("budget")
        raw_budget_type = str(key_entities.get("budget_type") or "").strip()
        if raw_budget is not None:
            try:
                budget_amount = float(str(raw_budget).strip())
                # pax 用于"总额"→"人均"换算：优先用本轮解析值，再用已有值，最后默认1
                budget_pax = new_pax or existing.pax or 1
                if raw_budget_type == "总额" and budget_pax > 1:
                    new_budget_per_person = budget_amount / budget_pax
                else:
                    new_budget_per_person = budget_amount
            except (ValueError, TypeError):
                pass

        merged_origin: Optional[str] = new_origin or existing.origin
        # 二级回退：query 和上轮 state 均无 origin 时，从 home_location 偏好补填
        if not merged_origin and memory_manager:
            home_loc = memory_manager.long_term.get_preference("home_location")
            if home_loc and isinstance(home_loc, str):
                merged_origin = home_loc
                logger.info(f"[extract_constraints] origin 缺失，使用 home_location 偏好回填: {home_loc}")

        merged_destination: Optional[str] = new_destination or existing.destination
        merged_start_date: Optional[str] = new_start_date or existing.start_date

        # end_date：若本轮能解析到 duration 则基于 merged_start_date 重算，
        # 否则保留已有值。这样 "仅改 duration" 或 "仅改 start_date + duration" 都能正确刷新。
        if duration_days > 0 and merged_start_date:
            merged_end_date: Optional[str] = _compute_end_date(
                merged_start_date, duration_days
            )
        else:
            merged_end_date = existing.end_date

        merged_pax: int = new_pax or existing.pax or 1
        merged_budget: Optional[float] = (
            new_budget_per_person if new_budget_per_person is not None
            else existing.total_budget
        )

        merged = HardConstraints(
            origin=merged_origin,
            destination=merged_destination,
            start_date=merged_start_date,
            end_date=merged_end_date,
            pax=merged_pax,
            total_budget=merged_budget,
        )

        # ── 3. missing_info：遍历必填字段（中文字段名，供 negotiate 节点直接引用）──
        missing_info: List[str] = [
            field_label
            for attr, field_label in _REQUIRED_FIELDS
            if not getattr(merged, attr)
        ]

        # ── 4. travel_days：日期差优先 → duration → state 已有值 → 0 ─────────
        travel_days: int = _compute_travel_days(
            merged.start_date,
            merged.end_date,
            duration_days,
            state.get("travel_days") or 0,
        )

        logger.info(
            f"[extract_constraints] origin={merged.origin!r}, "
            f"destination={merged.destination!r}, "
            f"start_date={merged.start_date!r}, end_date={merged.end_date!r}, "
            f"travel_days={travel_days}, pax={merged_pax}, "
            f"total_budget={merged_budget}, missing={missing_info}"
        )

        # ── 5. 返回状态增量（含跨轮清理 sentinel）─────────────────────────────
        return {
            "hard_constraints": merged,
            "missing_info": missing_info,
            "travel_days": travel_days,
            # 跨轮清理：避免 checkpointer 续跑时携带上一轮污染状态
            "rule_violations": [],
            "review_retry_count": 0,
            "skill_results": SKILL_RESULTS_RESET,
        }

    return extract_constraints_node
