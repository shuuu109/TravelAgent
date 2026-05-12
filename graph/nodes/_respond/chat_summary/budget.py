from graph.state import TravelGraphState, ensure_hard_constraints
from typing import Any, Dict, List, Optional


FOOD_TIER_PRICES: Dict[str, int] = {
    "高端": 200,
    "普通": 100,
    "经济": 60,
    "穷游": 30,
}


def _select_food_tier(state: TravelGraphState) -> tuple:
    """
    决定餐饮档位与单餐人均价。

    优先级：
      1. soft_constraints.budget_level（用户显式预算等级，做关键词匹配）
      2. travel_style 兜底（特种兵 -> 经济；其余 -> 普通）

    Returns:
        (tier_name, price_per_meal_per_person)
        tier_name: 高端 / 普通 / 经济 / 穷游
        price:     200 / 100 / 60 / 30
    """
    soft = state.get("soft_constraints")
    budget_level: str = ""
    if isinstance(soft, dict):
        budget_level = soft.get("budget_level") or ""
    elif soft is not None:
        budget_level = getattr(soft, "budget_level", "") or ""

    if budget_level:
        bl = budget_level.strip()
        if any(kw in bl for kw in ("豪华", "高端", "奢华")):
            return "高端", FOOD_TIER_PRICES["高端"]
        if any(kw in bl for kw in ("穷游", "超低", "省钱")):
            return "穷游", FOOD_TIER_PRICES["穷游"]
        if "经济" in bl:
            return "经济", FOOD_TIER_PRICES["经济"]
        return "普通", FOOD_TIER_PRICES["普通"]

    style: str = state.get("travel_style") or ""
    if style == "特种兵":
        return "经济", FOOD_TIER_PRICES["经济"]
    return "普通", FOOD_TIER_PRICES["普通"]



def _build_budget(state: TravelGraphState, food_tier: tuple) -> Dict[str, Any]:
    """
    构建预算明细 dict，供 chat_summary.budget 使用。

    取数规则：
      - 交通: transport_outbound / transport_return options 中
              is_recommended 优先，否则首项；price_range 中取最低数字 * pax
      - 住宿: accommodation_plan.daily_suggestions[].price_per_night 直接累加
              （视作每天总房费，不乘房数）
      - 餐饮: travel_days * pax * 2餐 * price_per_meal
      - 总预算: hard_constraints.total_budget (人均) * pax；缺失 -> limit=None, fit=unknown

    任一档计算失败置 amount=0 并在 note 标注，不抛异常。
    """
    import re

    tier_name, price_per_meal = food_tier

    hard_constraints = ensure_hard_constraints(state.get("hard_constraints"))
    pax: int = hard_constraints.pax or 1
    travel_days: int = state.get("travel_days") or 0
    if travel_days <= 0 and hard_constraints.start_date and hard_constraints.end_date:
        try:
            from datetime import datetime
            sd = datetime.strptime(hard_constraints.start_date, "%Y-%m-%d")
            ed = datetime.strptime(hard_constraints.end_date, "%Y-%m-%d")
            travel_days = (ed - sd).days + 1
        except Exception:
            travel_days = 1
    travel_days = max(travel_days, 1)

    items: List[Dict[str, Any]] = []
    total: float = 0.0

    def _pick_recommended(options: List[Dict]) -> Optional[Dict]:
        if not options:
            return None
        for o in options:
            if o.get("is_recommended"):
                return o
        return options[0]

    def _min_price(price_range: Any) -> Optional[float]:
        nums = re.findall(r"(\d+(?:\.\d+)?)", str(price_range or ""))
        if not nums:
            return None
        return float(min(nums, key=float))

    def _transport_item(label: str, options: List[Dict]) -> Dict[str, Any]:
        nonlocal total
        opt = _pick_recommended(options or [])
        if not opt:
            return {"label": label, "amount": 0, "note": "暂无数据"}
        price = _min_price(opt.get("price_range"))
        if price is None:
            return {"label": label, "amount": 0, "note": "暂无价格"}
        amount = int(round(price * pax))
        total += amount
        t_type = opt.get("transport_type", "") or "交通"
        t_no = opt.get("transport_no") or ""
        note_parts = [t_type]
        if t_no:
            note_parts.append(t_no)
        note_parts.append(f"¥{int(price)} × {pax}人")
        return {"label": label, "amount": amount, "note": " ".join(note_parts)}

    items.append(_transport_item("去程交通", state.get("transport_options") or []))
    items.append(_transport_item("返程交通", state.get("transport_return_options") or []))

    # ---- 住宿 ----
    acc_data: Dict[str, Any] = {}
    for r in state.get("skill_results", []) or []:
        if r.get("agent_name") == "accommodation_query" and r.get("status") == "success":
            acc_data = r.get("data") or {}
    acc_plan = acc_data.get("accommodation_plan") or {}
    daily_suggestions = acc_plan.get("daily_suggestions") or []

    acc_sum: float = 0.0
    acc_nights: int = 0
    for ds in daily_suggestions:
        ppn = ds.get("price_per_night")
        if isinstance(ppn, (int, float)) and ppn > 0:
            acc_sum += float(ppn)
            acc_nights += 1

    if acc_sum > 0:
        amount = int(round(acc_sum))
        total += amount
        avg = int(round(acc_sum / acc_nights)) if acc_nights else 0
        items.append({"label": "住宿", "amount": amount, "note": f"{acc_nights}晚 × ¥{avg}/晚"})
    else:
        items.append({"label": "住宿", "amount": 0, "note": "暂无数据"})

    # ---- 餐饮 ----
    food_amount = travel_days * pax * 2 * price_per_meal
    total += food_amount
    items.append({
        "label": "餐饮",
        "amount": int(food_amount),
        "note": f"{tier_name}档 ¥{price_per_meal} × 2餐 × {pax}人 × {travel_days}天",
    })

    # ---- 预算上限 / fit ----
    limit: Optional[int] = None
    fit: str = "unknown"
    if hard_constraints.total_budget:
        limit = int(round(float(hard_constraints.total_budget) * pax))
        fit = "under" if int(total) <= limit else "over"

    return {
        "currency": "¥",
        "total": int(total),
        "limit": limit,
        "fit": fit,
        "items": items,
    }
