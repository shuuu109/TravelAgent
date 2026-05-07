"""
预算检查节点 budget_check_node (P4.6)
职责：在行程与住宿生成完毕后，检查往返交通+住宿总成本是否在预算70%以内。

通过   → budget_fit_message = "预算符合预期"，路由到 respond
超支且 accommodation_downgrade_level < 2 → 递增降级等级，路由回 accommodation 重新规划
超支且已降级 2 次 → 静默放行（不写 budget_fit_message），路由到 respond
"""
import logging
from typing import Optional, Literal

from graph.state import TravelGraphState, ensure_hard_constraints

logger = logging.getLogger(__name__)

_BUDGET_THRESHOLD_RATIO = 0.70
_MAX_DOWNGRADE_LEVEL    = 2


def create_budget_check_node():
    async def budget_check_node(state: TravelGraphState) -> dict:
        hard_constraints = ensure_hard_constraints(state.get("hard_constraints"))
        total_budget     = hard_constraints.total_budget

        if not total_budget:
            return {"budget_fit_message": None}

        # 往返交通成本由 daily_budget_per_person 反推
        # 公式：transport = total_budget - daily_budget_per_person × travel_days
        daily_budget = state.get("daily_budget_per_person")
        travel_days  = state.get("travel_days") or 0
        if not daily_budget or not travel_days:
            logger.info("budget_check: 缺少 daily_budget_per_person 或 travel_days，跳过检查")
            return {"budget_fit_message": None}

        transport_cost       = total_budget - daily_budget * travel_days
        accommodation_total  = _extract_accommodation_total(state)

        if accommodation_total is None:
            logger.info("budget_check: 无住宿价格数据，跳过检查")
            return {"budget_fit_message": None}

        fixed_cost = transport_cost + accommodation_total
        threshold  = total_budget * _BUDGET_THRESHOLD_RATIO
        ratio      = fixed_cost / total_budget

        logger.info(
            f"budget_check: transport={transport_cost:.0f} + acc={accommodation_total:.0f} "
            f"= {fixed_cost:.0f} / {total_budget:.0f} ({ratio:.1%}), "
            f"threshold={threshold:.0f}"
        )

        if fixed_cost <= threshold:
            return {"budget_fit_message": "预算符合预期"}

        downgrade_level = state.get("accommodation_downgrade_level") or 0
        if downgrade_level < _MAX_DOWNGRADE_LEVEL:
            new_level = downgrade_level + 1
            logger.info(
                f"budget_check: 超出阈值，触发住宿降级 {downgrade_level} -> {new_level}"
            )
            return {"accommodation_downgrade_level": new_level}

        logger.info("budget_check: 已达最大降级次数，静默放行")
        return {"budget_fit_message": None}

    return budget_check_node


def route_after_budget_check(
    state: TravelGraphState,
) -> Literal["accommodation", "respond"]:
    """
    budget_check_node 后的路由判断。

    若 accommodation_query 最新结果的 downgrade_level 小于当前
    accommodation_downgrade_level，说明刚刚触发了降级，需要回环到 P4 重新规划。
    否则放行到 P5 respond。
    """
    current_level = state.get("accommodation_downgrade_level") or 0
    if current_level == 0:
        return "respond"

    for result in reversed(state.get("skill_results") or []):
        if result.get("agent_name") == "accommodation_query":
            result_level = (result.get("data") or {}).get("downgrade_level", 0)
            if result_level < current_level:
                return "accommodation"
            break

    return "respond"


def _extract_accommodation_total(state: TravelGraphState) -> Optional[float]:
    """从 skill_results 中取最新一条 accommodation_query 的住宿总成本估算。"""
    for result in reversed(state.get("skill_results") or []):
        if result.get("agent_name") == "accommodation_query":
            total = (result.get("data") or {}).get("estimated_accommodation_total")
            if isinstance(total, (int, float)) and total > 0:
                return float(total)
    return None
