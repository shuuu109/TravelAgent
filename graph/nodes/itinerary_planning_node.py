"""
行程规划节点 itinerary_planning_node
职责：基于 poi_candidates 完成三步规划：
  6a. _select_pois        — 按旅行风格 & 天数筛选合适数量的 POI
  6b. _cluster_by_geography — 贪心地理聚类，将 POI 分配到各天
  6c. _optimize_daily_route — 每天内部 TSP 优化 + 高德路线查询

改动点（相比旧 itinerary_planning skill）：
- 函数签名：async def itinerary_planning_node(state: TravelGraphState) -> dict
- 输入：从 state["poi_candidates"]、state["travel_style"]、state["travel_days"] 读取
- MCP session：整个节点只建立一次 amap_mcp_session()，各子函数共享同一 session
- 输出：{"daily_itinerary": [...], "daily_routes": [...]}（替换语义，非 operator.add）
- Fallback：MCP 失败时降级为原始评分顺序，不让整个规划中断
"""
from __future__ import annotations

import logging
from itertools import permutations
from math import sqrt
from typing import Any, Dict, List, Optional, Tuple

from graph.state import TravelGraphState
from mcp_clients.amap_client import amap_mcp_session, get_distance_matrix, get_transit_route

logger = logging.getLogger(__name__)

# POI 每天数量上限（按旅行风格）
_POIS_PER_DAY: Dict[str, int] = {
    "老人": 2,
    "亲子": 2,
    "情侣": 2,
    "普通": 3,
    "特种兵": 4,
}


# =============================================================================
# 工厂函数（供 workflow.py 调用，闭包注入未来可能的依赖）
# =============================================================================

def create_itinerary_planning_node():
    """
    返回 itinerary_planning_node async 函数。
    当前不需要外部注入依赖，保持与其他节点工厂一致的接口。
    """

    async def itinerary_planning_node(state: TravelGraphState) -> dict:
        """
        行程规划节点主流程：
        1. 从 state 读取 poi_candidates、travel_style、travel_days、目的地城市
        2. _select_pois：按风格 & 天数筛选 POI
        3. _cluster_by_geography：贪心地理聚类，按天分组
        4. _optimize_daily_route：TSP 优化 + 高德路线（共享单个 MCP session）
        5. 写入 state: daily_itinerary, daily_routes
        """
        poi_candidates: List[Dict] = state.get("poi_candidates", [])
        travel_style: str = state.get("travel_style", "普通")
        travel_days: int = state.get("travel_days") or 1
        if travel_days != state.get("travel_days"):
            logger.warning("itinerary_planning_node: travel_days 无效，降级为 1 天")

        # 从 hard_constraints 提取目的地城市（同时兼容 Pydantic model 和 dict）
        hard_constraints = state.get("hard_constraints")
        if hard_constraints is None:
            city = ""
        elif hasattr(hard_constraints, "destination"):
            city = hard_constraints.destination or ""
        elif isinstance(hard_constraints, dict):
            city = hard_constraints.get("destination", "") or ""
        else:
            city = ""

        if not poi_candidates:
            logger.warning("itinerary_planning_node: poi_candidates 为空，跳过规划")
            return {}

        logger.info(
            f"itinerary_planning_node: city={city}, style={travel_style}, "
            f"days={travel_days}, poi_count={len(poi_candidates)}"
        )

        # 6a — 筛选 POI
        selected_pois = _select_pois(poi_candidates, travel_style, travel_days)
        logger.info(f"_select_pois: 筛选后 {len(selected_pois)} 个 POI")

        # 6b — 地理聚类
        daily_itinerary = _cluster_by_geography(selected_pois, travel_days)
        logger.info(f"_cluster_by_geography: {len(daily_itinerary)} 天行程分组完成")

        # 6c — TSP 路线优化（整个节点共享一个 MCP session）
        daily_routes: List[Dict] = []
        try:
            async with amap_mcp_session() as session:
                for day_group in daily_itinerary:
                    route = await _optimize_daily_route(
                        day_pois=day_group["pois"],
                        city=city,
                        session=session,
                    )
                    daily_routes.append({"day": day_group["day"], **route})
            logger.info("itinerary_planning_node: TSP 路线优化完成")
        except Exception as e:
            logger.error(
                f"itinerary_planning_node: MCP session 失败: {e}，降级为评分顺序"
            )
            # Fallback：不调用 MCP，按原始顺序保留
            for day_group in daily_itinerary:
                daily_routes.append({
                    "day": day_group["day"],
                    "ordered_pois": day_group["pois"],
                    "legs": [],
                    "total_duration": 0,
                })

        return {
            "daily_itinerary": daily_itinerary,
            "daily_routes": daily_routes,
        }

    return itinerary_planning_node


# =============================================================================
# 6a — POI 筛选
# =============================================================================

def _select_pois(
    pois: List[Dict],
    travel_style: str,
    travel_days: int,
) -> List[Dict]:
    """
    按旅行风格决定每日 POI 数量，从候选列表中选出 total_needed 个 POI。

    策略：
    - 景点占 ~60%，餐厅占 ~20%，体验占 ~20%
    - 每类按 rating 降序，保证高分优先
    - 若各类总量不足 total_needed，从剩余候选中按 rating 补足
    """
    pois_per_day = _POIS_PER_DAY.get(travel_style, 3)
    total_needed = pois_per_day * travel_days

    # 按类别分桶
    attractions = sorted(
        [p for p in pois if p.get("category") == "景点"],
        key=lambda p: p.get("rating", 0.0), reverse=True,
    )
    restaurants = sorted(
        [p for p in pois if p.get("category") == "餐厅"],
        key=lambda p: p.get("rating", 0.0), reverse=True,
    )
    experiences = sorted(
        [p for p in pois if p.get("category") == "体验"],
        key=lambda p: p.get("rating", 0.0), reverse=True,
    )

    # 配额：景点 60%、餐厅 20%、体验 20%，四舍五入后剩余补给体验
    n_attr = min(len(attractions), round(total_needed * 0.6))
    n_rest = min(len(restaurants), round(total_needed * 0.2))
    n_exp  = min(len(experiences), total_needed - n_attr - n_rest)

    selected = attractions[:n_attr] + restaurants[:n_rest] + experiences[:n_exp]

    # 若仍不足，从整个候选集的剩余条目中按评分补齐
    if len(selected) < total_needed:
        selected_set = {id(p) for p in selected}
        remaining = sorted(
            [p for p in pois if id(p) not in selected_set],
            key=lambda p: p.get("rating", 0.0), reverse=True,
        )
        selected.extend(remaining[: total_needed - len(selected)])

    return selected[:total_needed]


# =============================================================================
# 6b — 地理聚类（贪心经度分组）
# =============================================================================

def _cluster_by_geography(
    pois: List[Dict],
    travel_days: int,
) -> List[Dict[str, Any]]:
    """
    贪心地理聚类：找整体重心后，用最近邻贪心把 POI 分配到各天，使同天 POI 地理集中。

    算法：
    1. 计算所有 POI 的地理重心（平均经纬度）
    2. 第一天锚点 = 距整体重心最远的 POI（从地理边界处开始）
    3. 后续天锚点 = 距上一天 POI 集合重心最远的未分配 POI（使各天地理分散）
    4. 每天从锚点出发，最近邻贪心地追加未分配 POI，直到当天配额满

    为什么不用 k-means：景点数量少（6-12 个），k-means 需要指定 k 且结果不稳定；
    贪心最近邻在小样本下简单可靠。

    返回：[{"day": 1, "pois": [...]}, {"day": 2, "pois": [...]}, ...]
    """
    if not pois or travel_days <= 0:
        return []

    n = len(pois)
    base_count = n // travel_days
    remainder = n % travel_days
    # 各天配额（前 remainder 天各多分一个）
    quotas = [base_count + (1 if d < remainder else 0) for d in range(travel_days)]

    # 1. 整体地理重心
    centroid: Tuple[float, float] = (
        sum(p["lng"] for p in pois) / n,
        sum(p["lat"] for p in pois) / n,
    )

    unassigned: List[int] = list(range(n))   # 待分配的 POI 索引
    groups: List[Dict[str, Any]] = []

    for day_idx, quota in enumerate(quotas):
        if not unassigned:
            break

        day_indices: List[int] = []

        # 2 & 3. 选锚点
        if day_idx == 0:
            # 第一天：选距整体重心最远的 POI
            anchor = max(
                unassigned,
                key=lambda i: _euclidean((pois[i]["lng"], pois[i]["lat"]), centroid),
            )
        else:
            # 后续天：选距上一天 POI 集合重心最远的未分配 POI
            prev_pois = groups[-1]["pois"]
            prev_centroid: Tuple[float, float] = (
                sum(p["lng"] for p in prev_pois) / len(prev_pois),
                sum(p["lat"] for p in prev_pois) / len(prev_pois),
            )
            anchor = max(
                unassigned,
                key=lambda i: _euclidean((pois[i]["lng"], pois[i]["lat"]), prev_centroid),
            )

        day_indices.append(anchor)
        unassigned.remove(anchor)

        # 4. 最近邻贪心：从锚点出发，依次追加距当前末尾最近的未分配 POI
        while len(day_indices) < quota and unassigned:
            current = day_indices[-1]
            nearest = min(
                unassigned,
                key=lambda i: _euclidean(
                    (pois[current]["lng"], pois[current]["lat"]),
                    (pois[i]["lng"], pois[i]["lat"]),
                ),
            )
            day_indices.append(nearest)
            unassigned.remove(nearest)

        groups.append({"day": day_idx + 1, "pois": [pois[i] for i in day_indices]})

    return groups


# =============================================================================
# 6c — 每日路线 TSP 优化（调用高德 MCP）
# =============================================================================

async def _optimize_daily_route(
    day_pois: List[Dict],
    city: str,
    session,
) -> Dict[str, Any]:
    """
    对单日的 POI 列表进行 TSP 优化，并获取相邻景点间的公交路线。

    步骤：
    1. 调用 get_distance_matrix 获取时间矩阵（失败则降级为欧氏距离）
    2. n <= 4：暴力枚举全排列（最多 4! = 24 种），找最短路线
       n > 4：最近邻贪心 TSP
    3. 按最优顺序调用 get_transit_route 获取相邻段路线

    Fallback：MCP 调用失败时降级为按 rating 降序的原始顺序，legs 为空。
    """
    n = len(day_pois)

    if n <= 1:
        return {"ordered_pois": day_pois, "legs": [], "total_duration": 0}

    coords = [f"{p['lng']},{p['lat']}" for p in day_pois]

    # --- 1. 获取时间矩阵 ---
    matrix: Optional[List[List[float]]] = None
    try:
        matrix = await get_distance_matrix(session, coords, coords)
    except Exception as e:
        logger.warning(f"_optimize_daily_route: get_distance_matrix 失败: {e}，用欧氏距离")

    # --- 2. TSP 求最优访问顺序 ---
    if matrix is not None:
        best_order = (
            _tsp_brute_force_matrix(matrix, n)
            if n <= 4
            else _tsp_nearest_neighbor_matrix(matrix, n)
        )
    else:
        points = [(p["lng"], p["lat"]) for p in day_pois]
        best_order = (
            _tsp_brute_force_euclidean(points)
            if n <= 4
            else _tsp_nearest_neighbor_euclidean(points)
        )

    ordered_pois = [day_pois[i] for i in best_order]

    # --- 3. 获取相邻景点间的公交路线 ---
    legs: List[Dict] = []
    total_duration = 0
    for i in range(len(ordered_pois) - 1):
        src = ordered_pois[i]
        dst = ordered_pois[i + 1]
        try:
            route = await get_transit_route(
                session,
                origin=f"{src['lng']},{src['lat']}",
                destination=f"{dst['lng']},{dst['lat']}",
                city=city,
            )
            legs.append({
                "from": src["name"],
                "to": dst["name"],
                "duration": route["duration"],
                "mode": route["recommended_mode"],
                "steps": route["steps"],
            })
            total_duration += route["duration"]
        except Exception as e:
            logger.warning(
                f"_optimize_daily_route: transit {src['name']}→{dst['name']} 失败: {e}"
            )
            legs.append({
                "from": src["name"],
                "to": dst["name"],
                "duration": 0,
                "mode": "unknown",
                "steps": [],
            })

    return {
        "ordered_pois": ordered_pois,
        "legs": legs,
        "total_duration": total_duration,
    }


# =============================================================================
# TSP 辅助函数（纯计算，不涉及 MCP）
# =============================================================================

def _tsp_brute_force_matrix(
    matrix: List[List[float]],
    n: int,
) -> List[int]:
    """
    暴力枚举：遍历所有 n! 排列，返回总时间最小的顺序。
    仅用于 n <= 4（最多 24 种排列）。
    """
    best_cost = float("inf")
    best_perm: List[int] = list(range(n))

    for perm in permutations(range(n)):
        cost = sum(
            matrix[perm[i]][perm[i + 1]]
            for i in range(n - 1)
        )
        if cost < best_cost:
            best_cost = cost
            best_perm = list(perm)

    return best_perm


def _tsp_nearest_neighbor_matrix(
    matrix: List[List[float]],
    n: int,
) -> List[int]:
    """
    最近邻贪心 TSP：从节点 0 出发，每次选择最近未访问节点。
    用于 n > 4。
    """
    visited = [False] * n
    order = [0]
    visited[0] = True

    for _ in range(n - 1):
        current = order[-1]
        nearest = min(
            (i for i in range(n) if not visited[i]),
            key=lambda i: matrix[current][i],
        )
        order.append(nearest)
        visited[nearest] = True

    return order


def _tsp_brute_force_euclidean(
    points: List[Tuple[float, float]],
) -> List[int]:
    """
    欧氏距离暴力枚举 TSP（MCP 不可用时的 fallback）。
    仅用于 n <= 4。
    """
    n = len(points)
    best_cost = float("inf")
    best_perm: List[int] = list(range(n))

    for perm in permutations(range(n)):
        cost = sum(
            _euclidean(points[perm[i]], points[perm[i + 1]])
            for i in range(n - 1)
        )
        if cost < best_cost:
            best_cost = cost
            best_perm = list(perm)

    return best_perm


def _tsp_nearest_neighbor_euclidean(
    points: List[Tuple[float, float]],
) -> List[int]:
    """
    欧氏距离最近邻贪心 TSP（MCP 不可用时的 fallback）。
    用于 n > 4。
    """
    n = len(points)
    visited = [False] * n
    order = [0]
    visited[0] = True

    for _ in range(n - 1):
        current = order[-1]
        nearest = min(
            (i for i in range(n) if not visited[i]),
            key=lambda i: _euclidean(points[current], points[i]),
        )
        order.append(nearest)
        visited[nearest] = True

    return order


def _euclidean(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """欧氏距离（经纬度近似，仅用于相对排序）"""
    return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
