"""
行程自检节点 itinerary_review_node (P4.5)
职责：对 P3 生成的 daily_routes 进行四项物理合理性检查，
      发现违规时写入 rule_violations，由路由函数决定是否回环重规划。

检查项（全部 critical，命中即触发 P3 回环重规划）：
  Check 1 - daily_time_overload    单日总时长超出旅行风格上限 25%
  Check 2 - long_transit_leg       相邻景点单段交通时间超阈值（驾车>60 / 公共交通>90 分钟）
  Check 3 - time_slot_mismatch     best_period 与位置时段不符（如夜市排上午、寺庙排傍晚）
  Check 4 - isolated_poi_in_day    某天行程内含 P3 识别的孤岛 POI（必须移除）

已取消的检查：
  - 餐厅距离（旧 Check 5）
  - 同类景点集中（旧 Check 4 category_concentration）：同类本身不构成错误，
    且 _select_pois 在选 POI 阶段已有 category_quota 多样性约束，review 端不再重复判定
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from graph.state import TravelGraphState, RuleViolation
from utils.llm_resilience import retry_with_backoff

logger = logging.getLogger(__name__)

# ── 单日时间上限（小时），按旅行风格 ──────────────────────────────────────────
_MAX_DAILY_HOURS: Dict[str, float] = {
    "亲子": 7.0,
    "情侣": 8.0,
    "普通": 9.0,
    "特种兵": 11.0,
}

# 超出多少百分比视为 critical（触发回环）
# 调高至 0.25：原先 0.15 过紧，正常波动就触发回环；
# 用户反馈"5 项检查限制太死"，对软指标提高容忍度，硬物理违规仍由 long_transit_leg 等覆盖。
_OVERLOAD_RATIO_THRESHOLD = 0.25

# 单段交通时间上限（分钟），按交通模式区分
# mode 字段来自高德 MCP get_transit_route 返回的 recommended_mode
_LONG_LEG_MINUTES: Dict[str, int] = {
    "驾车":   60,
    "公共交通": 90,
}
# 兜底阈值：mode 未知或未匹配时使用
_LONG_LEG_DEFAULT_MINUTES = 60

# 按位置推算时段：前两个景点视为上午，最后一个视为傍晚/夜间，其余为下午
_MORNING_MAX_INDEX = 1    # index 0、1 属于上午档
_EVENING_MIN_OFFSET = 1   # 距末尾 offset >= 1 即不是最后一个，则不属于傍晚档


def _pick_intra_day_outlier(
    ordered_pois: List[Dict],
    legs: List[Dict],
) -> Dict:
    """
    在单日行程中挑选"日内孤岛"——相邻交通腿之和最大的 POI。
    用于 daily_time_overload 的删除建议：超时大多由长腿交通造成，
    移除游览时间最长的景点通常救不了，移除日内空间外离群点更有效。

    Args:
        ordered_pois: 当日按访问顺序排列的 POI 列表（≥1 个）。
        legs: 相邻景点间交通腿，长度 = len(ordered_pois) - 1，
              每个元素含 duration（分钟）。

    Returns:
        最适合移除的 POI dict。若 ordered_pois 长度为 1，原样返回。
    """
    n = len(ordered_pois)
    if n <= 1:
        return ordered_pois[0]

    best_idx = 0
    best_adjacent = -1
    for i in range(n):
        in_t = legs[i - 1].get("duration", 0) if i > 0 and (i - 1) < len(legs) else 0
        out_t = legs[i].get("duration", 0) if i < len(legs) else 0
        adjacent = in_t + out_t
        if adjacent > best_adjacent:
            best_adjacent = adjacent
            best_idx = i
    return ordered_pois[best_idx]

def create_itinerary_review_node():
    """
    工厂函数，返回 itinerary_review_node 异步节点。
    保持与其他节点一致的 create_xxx_node() 风格。
    """

    async def itinerary_review_node(state: TravelGraphState) -> Dict[str, Any]:
        """
        P4.5 行程自检节点。

        读取 state["daily_routes"]，对每一天依次执行 Check 1-3，
        随后基于 state["isolated_pois"] 做 Check 4 全行程兜底扫描。
        所有违规聚合后写入 state["rule_violations"]（替换语义）。
        不负责重试计数，由 route_after_review 路由函数决策。
        """
        daily_routes: List[Dict] = state.get("daily_routes") or []
        travel_style: str = state.get("travel_style") or "普通"
        max_hours: float = _MAX_DAILY_HOURS.get(travel_style, 9.0)

        if not daily_routes:
            logger.warning("[itinerary_review] daily_routes 为空，跳过自检")
            return {"rule_violations": []}

        violations: List[RuleViolation] = []

        for day_data in daily_routes:
            day: int = day_data.get("day", 0)
            ordered_pois: List[Dict] = day_data.get("ordered_pois", [])
            legs: List[Dict] = day_data.get("legs", [])

            # ── Check 1：单日总时长预算 ──────────────────────────────────────
            poi_hours: float = sum(
                p.get("estimated_hours", 1.5) for p in ordered_pois
            )
            transit_hours: float = sum(
                leg.get("duration", 0) for leg in legs
            ) / 60.0
            total_hours: float = poi_hours + transit_hours

            if total_hours > max_hours * (1 + _OVERLOAD_RATIO_THRESHOLD):
                # 挑选"日内孤岛"作为建议移除对象：相邻交通腿之和最大的 POI。
                # 经验上单日时长超标主要由长腿交通造成，移除空间外离群点比
                # 移除游览时长最长的景点更能直接降低 total_hours。
                outlier_poi = (
                    _pick_intra_day_outlier(ordered_pois, legs)
                    if ordered_pois
                    else None
                )
                suggestion = (
                    f"建议将「{outlier_poi['name']}」移至其他天，或缩减当天景点数量"
                    if outlier_poi
                    else "建议减少当天景点数量"
                )
                violations.append(RuleViolation(
                    violation_type="daily_time_overload",
                    description=(
                        f"第{day}天总时长约{total_hours:.1f}小时"
                        f"（景点{poi_hours:.1f}h + 交通{transit_hours:.1f}h），"
                        f"超出{travel_style}风格上限{max_hours}h"
                    ),
                    suggestion=suggestion,
                ))
                logger.info(
                    f"[itinerary_review] Check1 violation: Day{day} "
                    f"total={total_hours:.1f}h > max={max_hours}h"
                )

            # ── Check 2：单段交通时间过长（阈值按交通模式区分）────────────────
            for leg in legs:
                duration: int = leg.get("duration", 0)
                mode: str = leg.get("mode", "未知")
                threshold: int = _LONG_LEG_MINUTES.get(mode, _LONG_LEG_DEFAULT_MINUTES)
                if duration > threshold:
                    violations.append(RuleViolation(
                        violation_type="long_transit_leg",
                        description=(
                            f"第{day}天 {leg['from']}→{leg['to']} "
                            f"交通（{mode}）{duration}分钟，"
                            f"单段超过{mode}阈值{threshold}分钟"
                        ),
                        suggestion=(
                            f"建议将「{leg['from']}」和「{leg['to']}」"
                            f"拆分到不同天，或替换为距离更近的景点"
                        ),
                    ))
                    logger.info(
                        f"[itinerary_review] Check2 violation: Day{day} "
                        f"{leg['from']}→{leg['to']} {duration}min ({mode}) > threshold {threshold}min"
                    )

            # ── Check 3：best_period 时段冲突 ────────────────────────────────
            last_index = len(ordered_pois) - 1
            for idx, poi in enumerate(ordered_pois):
                best_period: str = poi.get("best_period", "flexible")
                name: str = poi.get("name", "未知景点")

                if best_period == "morning" and idx > _MORNING_MAX_INDEX:
                    violations.append(RuleViolation(
                        violation_type="time_slot_mismatch",
                        description=(
                            f"第{day}天「{name}」适合上午游览，"
                            f"但排在第{idx + 1}个景点（预计下午到达）"
                        ),
                        suggestion=f"建议将「{name}」调整为当天第1或第2个景点",
                    ))
                    logger.info(
                        f"[itinerary_review] Check3 morning violation: "
                        f"Day{day} {name} at index {idx}"
                    )

                elif best_period == "evening" and idx < last_index:
                    violations.append(RuleViolation(
                        violation_type="time_slot_mismatch",
                        description=(
                            f"第{day}天「{name}」适合傍晚/夜间游览，"
                            f"但排在第{idx + 1}个景点，而非最后一个"
                        ),
                        suggestion=f"建议将「{name}」调整为当天最后一个景点",
                    ))
                    logger.info(
                        f"[itinerary_review] Check3 evening violation: "
                        f"Day{day} {name} at index {idx}"
                    )

        # ── Check 4：孤岛 POI 漏入某天（兜底）─────────────────────────────────
        # P3 itinerary_planning_node 已在聚类前识别并过滤孤岛 POI；
        # 但当过滤后剩余 POI < travel_days 时，P3 会放弃过滤（warn-only），
        # 此时孤岛仍可能落入某天。此 check 作为兜底，强制 remove 该 POI。
        # state["isolated_pois"] 由 P3 写入（POI 名称列表）。
        isolated_names: set = set(state.get("isolated_pois") or [])
        if isolated_names:
            for day_data in daily_routes:
                day_num: int = day_data.get("day", 0)
                for poi in day_data.get("ordered_pois", []):
                    pname: str = poi.get("name", "")
                    if pname and pname in isolated_names:
                        violations.append(RuleViolation(
                            violation_type="isolated_poi_in_day",
                            description=(
                                f"第{day_num}天行程包含孤岛景点「{pname}」，"
                                f"该景点与所有其他候选景点最小通勤时间均超 60 分钟，"
                                f"无法与同天其他景点合理串联"
                            ),
                            suggestion=f"建议从行程中移除「{pname}」",
                        ))
                        logger.info(
                            f"[itinerary_review] Check4 violation: Day{day_num} "
                            f"isolated POI {pname}"
                        )

        if violations:
            logger.warning(
                f"[itinerary_review] 发现 {len(violations)} 处违规，"
                f"写入 rule_violations，等待路由判断是否回环"
            )
        else:
            logger.info("[itinerary_review] 三项检查全部通过，无违规")

        return {"rule_violations": violations}

    return itinerary_review_node
