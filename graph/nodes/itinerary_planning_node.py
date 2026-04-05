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
        2. 解析 rag_snippets，提取 POI 关键词权重集合和同游景点约束对
        3. _select_pois：按风格 & 天数筛选 POI，RAG 提及的景点评分加权
        4. _cluster_by_geography：贪心地理聚类，RAG "可同天游"的景点强制同组
        5. _optimize_daily_route：TSP 优化 + 高德路线（共享单个 MCP session）
        6. 写入 state: daily_itinerary, daily_routes
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

        # ── 解析 RAG 攻略片段，提取权重信息 ──────────────────────────────
        rag_snippets: List[Dict] = state.get("rag_snippets") or []
        rag_boosted_names, rag_joint_hints = _parse_rag_hints(rag_snippets)
        logger.info(
            f"RAG hints: boosted_names={rag_boosted_names}, joint_hints={rag_joint_hints}"
        )

        logger.info(
            f"itinerary_planning_node: city={city}, style={travel_style}, "
            f"days={travel_days}, poi_count={len(poi_candidates)}"
        )

        # 6a — 筛选 POI（RAG 命中景点评分上调）
        selected_pois = _select_pois(
            poi_candidates, travel_style, travel_days, rag_boosted_names
        )
        logger.info(f"_select_pois: 筛选后 {len(selected_pois)} 个 POI")

        # 6b — 地理聚类（RAG 同游对强制同组）
        daily_itinerary = _cluster_by_geography(
            selected_pois, travel_days, rag_joint_hints
        )
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
    rag_boosted_names: Optional[set] = None,
) -> List[Dict]:
    """
    按旅行风格决定每日 POI 数量，从候选列表中选出 total_needed 个 POI。

    策略：
    - 景点占 ~60%，餐厅占 ~20%，体验占 ~20%
    - 每类按 rating 降序，保证高分优先
    - RAG 攻略中明确推荐的景点，评分额外加 +1.5 以提升排序权重
    - 若各类总量不足 total_needed，从剩余候选中按 rating 补足

    Args:
        rag_boosted_names: RAG 攻略中提及的景点名称集合（模糊匹配）
    """
    rag_boosted_names = rag_boosted_names or set()

    def _effective_rating(poi: Dict) -> float:
        """计算 POI 的有效评分：RAG 命中的景点额外加 1.5 分。"""
        base = poi.get("rating", 0.0) or 0.0
        name = poi.get("name", "")
        # 模糊匹配：POI 名称包含任一 RAG 关键词即命中
        if any(keyword in name for keyword in rag_boosted_names if keyword):
            return base + 1.5
        return base

    pois_per_day = _POIS_PER_DAY.get(travel_style, 3)
    total_needed = pois_per_day * travel_days

    # 按类别分桶（使用 _effective_rating 排序）
    attractions = sorted(
        [p for p in pois if p.get("category") == "景点"],
        key=_effective_rating, reverse=True,
    )
    restaurants = sorted(
        [p for p in pois if p.get("category") == "餐厅"],
        key=_effective_rating, reverse=True,
    )
    experiences = sorted(
        [p for p in pois if p.get("category") == "体验"],
        key=_effective_rating, reverse=True,
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
    rag_joint_hints: Optional[List[Tuple[str, str]]] = None,
) -> List[Dict[str, Any]]:
    """
    贪心地理聚类：找整体重心后，用最近邻贪心把 POI 分配到各天，使同天 POI 地理集中。

    算法：
    1. 预处理 RAG 同游约束：将"可同天游"的 POI 对提前绑定为同组种子
    2. 计算所有 POI 的地理重心（平均经纬度）
    3. 第一天锚点 = 距整体重心最远的 POI
    4. 后续天锚点 = 距上一天集合重心最远的未分配 POI
    5. 每天从锚点出发，最近邻贪心地追加未分配 POI，直到配额满
       — 若 RAG 约束的配对 POI 仍未分配，优先追加它

    Args:
        rag_joint_hints: RAG 提取的"可同天游"POI 名称对列表，
                         例如 [("都江堰", "青城山"), ("熊猫基地", "武侯祠")]

    返回：[{"day": 1, "pois": [...]}, {"day": 2, "pois": [...]}, ...]
    """
    if not pois or travel_days <= 0:
        return []

    rag_joint_hints = rag_joint_hints or []

    n = len(pois)
    base_count = n // travel_days
    remainder = n % travel_days
    # 各天配额（前 remainder 天各多分一个）
    quotas = [base_count + (1 if d < remainder else 0) for d in range(travel_days)]

    # ── 预处理 RAG 同游约束 ─────────────────────────────────────────────
    # 构建 POI 索引到其"约束伙伴"索引的映射
    # 用景点名称模糊匹配（POI name 包含 hint 关键词即命中）
    joint_pairs: List[Tuple[int, int]] = []
    for hint_a, hint_b in rag_joint_hints:
        idx_a = next(
            (i for i, p in enumerate(pois) if hint_a in p.get("name", "")), None
        )
        idx_b = next(
            (i for i, p in enumerate(pois) if hint_b in p.get("name", "")), None
        )
        if idx_a is not None and idx_b is not None and idx_a != idx_b:
            joint_pairs.append((idx_a, idx_b))
            logger.info(
                f"RAG joint hint matched: {pois[idx_a]['name']} + {pois[idx_b]['name']}"
            )

    # 伙伴映射（双向）
    partner_of: Dict[int, int] = {}
    for a, b in joint_pairs:
        partner_of[a] = b
        partner_of[b] = a

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
            anchor = max(
                unassigned,
                key=lambda i: _euclidean((pois[i]["lng"], pois[i]["lat"]), centroid),
            )
        else:
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

        # 若锚点有 RAG 同游伙伴且未分配，立即追加（配额允许时）
        if anchor in partner_of:
            partner = partner_of[anchor]
            if partner in unassigned and len(day_indices) < quota:
                day_indices.append(partner)
                unassigned.remove(partner)

        # 4. 最近邻贪心：优先追加当天已有 POI 的 RAG 伙伴，其次选地理最近的
        while len(day_indices) < quota and unassigned:
            # 检查当天已有 POI 是否有尚未分配的 RAG 伙伴
            rag_priority = next(
                (partner_of[i] for i in day_indices
                 if i in partner_of and partner_of[i] in unassigned),
                None,
            )
            if rag_priority is not None:
                day_indices.append(rag_priority)
                unassigned.remove(rag_priority)
            else:
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


# =============================================================================
# RAG 攻略解析辅助函数
# =============================================================================

def _parse_rag_hints(
    rag_snippets: List[Dict],
) -> Tuple[set, List[Tuple[str, str]]]:
    """
    解析 RAG 检索到的旅游攻略片段，提取两类信息：

    1. rag_boosted_names (set[str])：
       攻略中明确提及的景点名称关键词，用于在 _select_pois 中给对应 POI 加权。
       提取策略：扫描 content 中的 2-8 字中文词组，过滤停用词后收录。

    2. rag_joint_hints (List[Tuple[str, str]])：
       攻略中出现"A + B 可同天游"/"A 和 B 建议同天"等表达时，提取 (A, B) 对，
       用于在 _cluster_by_geography 中强制同组。

    Args:
        rag_snippets: orchestrate_node 写入 state 的 retrieved_documents 列表，
                      每项结构为 {"content": str, "metadata": dict}

    Returns:
        (boosted_names, joint_hints)
    """
    import re

    boosted_names: set = set()
    joint_hints: List[Tuple[str, str]] = []

    # 常见旅游场景中文景点名称模式（2-8 字中文）
    poi_name_pattern = re.compile(r'[\u4e00-\u9fa5]{2,8}')

    # 同游表达的正则：匹配"A和B可同天游"/"A+B建议同天"等
    joint_day_patterns = [
        re.compile(
            r'([\u4e00-\u9fa5]{2,8})[和与、及]?([\u4e00-\u9fa5]{2,8})'
            r'(?:可以?|建议|推荐)?同[一]?天(?:游览?|参观?|游玩?)'
        ),
        re.compile(
            r'([\u4e00-\u9fa5]{2,8})(?:与|和)([\u4e00-\u9fa5]{2,8})'
            r'(?:距离较近|相邻|毗邻|顺路).*?(?:可|建议)同[一]?天'
        ),
    ]

    # 停用词（避免把"景点"、"推荐"等误识别为 POI 名称）
    stopwords = {
        "景点", "推荐", "建议", "注意", "适合", "游览", "参观", "时间", "门票",
        "交通", "路线", "行程", "旅游", "旅行", "攻略", "必去", "必玩", "必看",
        "亲子", "情侣", "老人", "特种兵", "普通", "人群", "游客", "日期", "季节",
    }

    for snippet in rag_snippets:
        content = snippet.get("content", "") if isinstance(snippet, dict) else ""
        if not content:
            continue

        # 1. 提取景点关键词
        for match in poi_name_pattern.findall(content):
            if match not in stopwords and len(match) >= 2:
                boosted_names.add(match)

        # 2. 提取同游约束对
        for pattern in joint_day_patterns:
            for m in pattern.finditer(content):
                a, b = m.group(1), m.group(2)
                if a not in stopwords and b not in stopwords and a != b:
                    joint_hints.append((a, b))

    logger.info(
        f"_parse_rag_hints: {len(boosted_names)} boosted keywords, "
        f"{len(joint_hints)} joint hints"
    )
    return boosted_names, joint_hints
