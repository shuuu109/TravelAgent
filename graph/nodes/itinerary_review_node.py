"""
行程自检节点 itinerary_review_node (P4.5)
职责：对 P3 生成的 daily_routes 进行四项物理合理性检查，
      发现违规时写入 rule_violations，由路由函数决定是否回环重规划。

检查项：
  Check 1 - daily_time_overload    : 单日总时长（景点游览 + 交通）超出旅行风格上限
  Check 2 - long_transit_leg       : 相邻景点单段交通时间超过阈值（驾车>60min / 公共交通>90min）
  Check 3 - time_slot_mismatch     : 景点的 best_period 与实际所在位置推算的时段不符
  Check 4 - category_concentration : 同一天出现 2 个以上同大类景点（自然公园/古镇/博物馆等）

不检查：
  - 餐厅距离（Check 5 已取消）
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List

from graph.state import TravelGraphState, RuleViolation
from utils.poi_category import get_category_for_poi

logger = logging.getLogger(__name__)

# ── 单日时间上限（小时），按旅行风格 ──────────────────────────────────────────
_MAX_DAILY_HOURS: Dict[str, float] = {
    "亲子": 7.0,
    "情侣": 8.0,
    "普通": 9.0,
    "特种兵": 11.0,
}

# 超出多少百分比视为 critical（触发回环）
_OVERLOAD_RATIO_THRESHOLD = 0.15

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

# Check 4 大类判断已迁移至 utils/poi_category.py：
#   - 优先用 poi["amap_type"]（高德 typecode 前缀）匹配大类
#   - typecode 未命中时降级为名称关键词匹配
# 此文件不再维护内联关键词字典，统一通过 get_category_for_poi() 获取大类标签。


def create_itinerary_review_node():
    """
    工厂函数，返回 itinerary_review_node 异步节点。
    保持与其他节点一致的 create_xxx_node() 风格。
    """

    async def itinerary_review_node(state: TravelGraphState) -> Dict[str, Any]:
        """
        P4.5 行程自检节点。

        读取 state["daily_routes"]，对每一天依次执行三项检查，
        将所有违规聚合后写入 state["rule_violations"]（替换语义）。
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
                # 找出游览时长最长的景点作为建议移除对象
                longest_poi = max(
                    ordered_pois,
                    key=lambda p: p.get("estimated_hours", 1.5),
                    default=None,
                )
                suggestion = (
                    f"建议将「{longest_poi['name']}」移至其他天，或缩减当天景点数量"
                    if longest_poi
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

            # ── Check 4：同类景点集中 ────────────────────────────────────────
            # 用 get_category_for_poi() 对当天所有 POI 分组：
            #   优先取 poi["amap_type"] typecode 前缀 → 大类标签；
            #   typecode 未命中时降级为名称关键词匹配。
            # 每个 POI 只归入一个大类，避免旧方案的重复计数问题。
            cat_groups: Dict[str, List[Dict]] = defaultdict(list)
            for p in ordered_pois:
                cat_label = get_category_for_poi(p)
                if cat_label:
                    cat_groups[cat_label].append(p)

            for category_label, same_cat_pois in cat_groups.items():
                if len(same_cat_pois) < 2:
                    continue

                # 找出游览时长最长的作为"建议删除"对象
                longest_poi = max(
                    same_cat_pois, key=lambda p: p.get("estimated_hours", 1.5)
                )
                # 取前两个（通常就是问题对）作为 split_hints 提取源
                poi_a = same_cat_pois[0]["name"]
                poi_b = same_cat_pois[1]["name"]

                violations.append(RuleViolation(
                    violation_type="category_concentration",
                    description=(
                        f"第{day}天同时安排了 {len(same_cat_pois)} 个{category_label}类景点："
                        f"{'、'.join(p['name'] for p in same_cat_pois)}，"
                        f"体验高度同质且游览时长叠加过重"
                    ),
                    suggestion=(
                        f"建议将「{poi_a}」和「{poi_b}」拆分到不同天，"
                        f"或删除其中耗时较长的「{longest_poi['name']}」"
                    ),
                ))
                logger.info(
                    f"[itinerary_review] Check4 violation: Day{day} "
                    f"category={category_label} pois={[p['name'] for p in same_cat_pois]}"
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
