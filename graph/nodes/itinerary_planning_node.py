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
from utils.knowledge_parser import CityKnowledgeDB
from mcp_clients.amap_client import (
    amap_mcp_session,
    get_distance_matrix,
    get_transit_route,
    search_restaurants_nearby,
)

logger = logging.getLogger(__name__)

# POI 每天数量上限（按旅行风格）
_POIS_PER_DAY: Dict[str, int] = {
    "老人": 2,
    "亲子": 2,
    "情侣": 2,
    "普通": 3,
    "特种兵": 4,
}

# 旅游景点常见尾字/尾词白名单，用于从 RAG 文本中精准识别景点名称
# 提取到的词组若以这些词结尾，才视为 POI 名称候选
_ATTRACTION_SUFFIXES: Tuple = (
    # 单字后缀
    '寺', '塔', '楼', '桥', '街', '湖', '园', '宫', '庙', '观', '府', '坊',
    '阁', '殿', '院', '堂', '岛', '峰', '山', '城', '洞', '谷',
    # 双字/多字后缀
    '景区', '公园', '遗址', '故居', '博物馆', '纪念馆', '广场', '老街',
    '古镇', '古街', '名胜', '风景区', '风光带',
)

# 明确不是景点名的高频词（用于 boosted_names 过滤，补充 stopwords）
_RAG_NOISE_WORDS: frozenset = frozenset([
    '地铁', '公交', '步行', '打车', '机场', '车站', '高铁', '动车',
    '分钟', '小时', '公里', '米', '元', '号线', '路线', '攻略',
    '推荐', '建议', '注意', '适合', '游览', '参观', '早场', '晚场',
    '交通', '住宿', '美食', '餐厅', '酒店', '民宿', '综合', '指南',
    '核心', '本地', '环湖', '顺路', '枢纽', '商圈', '繁华', '入口',
])


def _is_likely_poi(name: str) -> bool:
    """
    判断词组是否可能是景点名称。

    策略（严格模式，用于 boosted_names 提取）：
    - 必须以景点类后缀结尾（如 寺/塔/湖/街/园/峰 等）
    - 不包含 _RAG_NOISE_WORDS 中的噪声子串（过滤"综合攻略"、"早场建议"等）
    - 不使用 4字+ catch-all，避免把形容词、句子片段误认为景点名
    """
    if not name or len(name) < 2:
        return False
    # 精确噪声词命中
    if name in _RAG_NOISE_WORDS:
        return False
    # 子串噪声命中（如 "综合攻略" 包含 "综合"，"早场建议" 包含 "建议"）
    if any(nw in name for nw in _RAG_NOISE_WORDS if len(nw) >= 2):
        return False
    # 必须以景点后缀结尾
    for suffix in _ATTRACTION_SUFFIXES:
        if name.endswith(suffix):
            return True
    return False


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
        3. 从 skill_results 中找到 RAG answer，用 jieba 提取推荐景点序列
        4. _select_pois：Phase-1 RAG锚定 + Phase-2 评分填充（仅景点）
        5. _cluster_by_geography：贪心地理聚类
        6. _optimize_daily_route：TSP 优化 + 高德路线（共享单个 MCP session）
        6d. 基于每天景点地理重心，调用高德搜索周边餐厅（每天推荐 5 家）
        7. 写入 state: daily_itinerary, daily_routes, daily_restaurants
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

        # Fallback：hard_constraints 缺失时从 intent_data 或 skill_results 中补取目的地
        if not city:
            intent_data: dict = state.get("intent_data") or {}
            city = (
                intent_data.get("key_entities", {}).get("destination", "")
                or intent_data.get("destination", "")
                or ""
            )
        if not city:
            # 最后兜底：从 skill_results 中找 event_collection 的 destination
            for sr in state.get("skill_results", []):
                if sr.get("agent_name") == "event_collection":
                    city = sr.get("data", {}).get("destination", "") or ""
                    if city:
                        break
        if city:
            logger.info(f"itinerary_planning_node: 解析到目的地城市='{city}'")

        if not poi_candidates:
            logger.warning("itinerary_planning_node: poi_candidates 为空，跳过规划")
            return {}

        # ── 解析 RAG 攻略原始片段，提取加权关键词 ────────────────────────────
        rag_snippets: List[Dict] = state.get("rag_snippets") or []
        rag_boosted_names, rag_joint_hints = _parse_rag_hints(rag_snippets)
        logger.info(
            f"RAG hints: boosted_names={rag_boosted_names}, joint_hints={rag_joint_hints}"
        )

        # ── 优先从结构化知识库获取城市必去景点（直接查表，零 NLP 损耗）────────
        # 知识库按城市整理好"必去景点"有序列表，直接用于 Phase-1 锚定，
        # 彻底绕过 RAG answer → jieba 提取的高损耗链路（jieba 会切断复合地名、
        # 引入"路边""浙大"等噪声词）。
        # Fallback：城市不在知识库时，降级为原有 RAG answer + jieba 提取逻辑。
        knowledge_db = CityKnowledgeDB.get_instance()
        route_combos: List[List[str]] = []

        if city and knowledge_db.has_city(city):
            rag_preferred_pois = knowledge_db.get_must_visit_names(city)
            route_combos = knowledge_db.get_route_combos(city)
            logger.info(
                f"itinerary_planning_node: 知识库命中城市='{city}', "
                f"必去景点={rag_preferred_pois}, 顺路组合={len(route_combos)} 条"
            )
        else:
            # Fallback：城市不在知识库，降级为 RAG answer + jieba 提取
            rag_preferred_pois = []
            for sr in state.get("skill_results", []):
                if sr.get("agent_name") == "rag_knowledge" and sr.get("status") == "success":
                    raw_data = sr.get("data", {})
                    if isinstance(raw_data, str):
                        try:
                            import json as _json
                            raw_data = _json.loads(raw_data)
                        except Exception:
                            raw_data = {}
                    rag_answer_text = (
                        raw_data.get("answer")
                        or raw_data.get("data", {}).get("answer", "")
                        or ""
                    )
                    if isinstance(rag_answer_text, str) and rag_answer_text.strip().startswith("{"):
                        try:
                            import json as _json
                            inner = _json.loads(rag_answer_text)
                            rag_answer_text = inner.get("answer", rag_answer_text)
                        except Exception:
                            pass
                    if rag_answer_text:
                        rag_preferred_pois = _extract_rag_preferred_pois(rag_answer_text)
                    break
            logger.info(
                f"itinerary_planning_node: 城市='{city}' 不在知识库, "
                f"降级为 RAG 提取, rag_preferred={rag_preferred_pois}"
            )

        logger.info(
            f"itinerary_planning_node: city={city}, style={travel_style}, "
            f"days={travel_days}, poi_count={len(poi_candidates)}, "
            f"rag_preferred={rag_preferred_pois}"
        )

        # 6a — 筛选 POI（Phase-1 RAG锚定 + Phase-2 评分填充）
        selected_pois = _select_pois(
            poi_candidates, travel_style, travel_days,
            rag_boosted_names, rag_preferred_pois,
        )
        logger.info(f"_select_pois: 筛选后 {len(selected_pois)} 个 POI")

        # 6b — 地理聚类（RAG 同游对 + 知识库顺路组合强制同组）
        # 将顺路组合展开为相邻 POI 对，追加到 rag_joint_hints，
        # 使同一线路的景点（如 断桥→白堤→苏堤→雷峰塔）倾向分配到同一天。
        combo_joint_hints: List[Tuple[str, str]] = [
            (combo[i], combo[i + 1])
            for combo in route_combos
            for i in range(len(combo) - 1)
        ]
        combined_hints = rag_joint_hints + combo_joint_hints
        daily_itinerary = _cluster_by_geography(
            selected_pois, travel_days, combined_hints
        )
        logger.info(f"_cluster_by_geography: {len(daily_itinerary)} 天行程分组完成")

        # 6c — TSP 路线优化（整个节点共享一个 MCP session）
        daily_routes: List[Dict] = []
        daily_restaurants: List[Dict] = []  # 6d 输出：每天周边餐厅推荐
        try:
            async with amap_mcp_session() as session:
                for day_group in daily_itinerary:
                    # 6c: TSP 优化
                    route = await _optimize_daily_route(
                        day_pois=day_group["pois"],
                        city=city,
                        session=session,
                    )
                    daily_routes.append({"day": day_group["day"], **route})

                    # 6d: 基于当天景点地理重心，搜索周边餐厅
                    day_restaurants = await _fetch_day_restaurants(
                        day_pois=route.get("ordered_pois", day_group["pois"]),
                        session=session,
                        city=city,
                    )
                    daily_restaurants.append({
                        "day": day_group["day"],
                        "restaurants": day_restaurants,
                    })

            logger.info("itinerary_planning_node: TSP 路线优化 + 周边餐厅搜索完成")
        except Exception as e:
            logger.error(
                f"itinerary_planning_node: MCP session 失败: {e}，降级为评分顺序"
            )
            # Fallback：不调用 MCP，按原始顺序保留，餐厅列表为空
            for day_group in daily_itinerary:
                daily_routes.append({
                    "day": day_group["day"],
                    "ordered_pois": day_group["pois"],
                    "legs": [],
                    "total_duration": 0,
                })
                daily_restaurants.append({
                    "day": day_group["day"],
                    "restaurants": [],
                })

        return {
            "daily_itinerary": daily_itinerary,
            "daily_routes": daily_routes,
            "daily_restaurants": daily_restaurants,
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
    rag_preferred_pois: Optional[List[str]] = None,
) -> List[Dict]:
    """
    按旅行风格决定每日 POI 数量，从候选列表中选出 total_needed 个 POI。

    两阶段策略：
    ① RAG 优先锚定（若 rag_preferred_pois 非空）：
       将 RAG 行程安排中提到的景点与 Amap 候选 POI 进行模糊匹配，
       匹配到的 POI 优先锚定进入选集，保证游客真正想去的名胜不被过滤。
    ② 剩余配额按评分/搜索排名填充：
       - 景点占 ~60%、餐厅占 ~20%、体验占 ~20%
       - rating=0 的景点用 search_rank 换算基准分（排名越靠前分越高）
       - RAG boosted_names 命中的 POI 额外加 +1.5 分

    Args:
        rag_boosted_names:  RAG 原始片段中提取的景点关键词集合（用于加权）
        rag_preferred_pois: RAG 综合回答的行程安排中提取的有序景点名列表（用于锚定）
    """
    rag_boosted_names = rag_boosted_names or set()
    rag_preferred_pois = rag_preferred_pois or []

    pois_per_day = _POIS_PER_DAY.get(travel_style, 3)
    total_needed = pois_per_day * travel_days

    def _effective_rating(poi: Dict) -> float:
        """
        计算 POI 的有效评分，兼顾以下两种情况：
        1. 景点类 POI（尤其是高德地图景区）biz_ext.rating 通常为空（=0），
           此时用搜索排名（search_rank）换算出基准分：排名越靠前分越高，
           最高折算约 2.0 分（rank=1），随排名线性衰减。
        2. 餐厅/体验类 POI 有真实评分（4.x），直接使用。
        3. RAG 攻略中明确推荐的任意类 POI，额外加 1.5 分提升权重。
        """
        base = poi.get("rating", 0.0) or 0.0
        # 当 rating 缺失（=0）时，用搜索排名作为代理指标
        if base == 0.0:
            rank = poi.get("search_rank", 20)
            # 排名1 → 2.0分，排名20+ → 接近0分，线性衰减
            base = max(0.0, (21 - rank) / 21 * 2.0)
        name = poi.get("name", "")
        rag_boost = 1.5 if any(kw in name for kw in rag_boosted_names if kw) else 0.0
        return base + rag_boost

    # ─── Phase 1: RAG 优先锚定 ───────────────────────────────────────────────
    # 将 RAG 行程安排中的景点名与 Amap 候选 POI 进行名称模糊匹配（优先精确，其次子串）
    anchored: List[Dict] = []
    anchored_ids: set = set()

    if rag_preferred_pois:
        for rag_name in rag_preferred_pois:
            if len(anchored) >= total_needed:
                break
            best_match: Optional[Dict] = None
            # 精确匹配优先
            for poi in pois:
                if id(poi) not in anchored_ids and poi.get("name", "") == rag_name:
                    best_match = poi
                    break
            # 子串匹配次之：rag_name 是 POI name 的子串，或 POI name 是 rag_name 的子串
            if best_match is None:
                for poi in pois:
                    if id(poi) in anchored_ids:
                        continue
                    poi_name = poi.get("name", "")
                    if rag_name in poi_name or (len(rag_name) >= 2 and poi_name in rag_name):
                        best_match = poi
                        break
            if best_match is not None:
                anchored.append(best_match)
                anchored_ids.add(id(best_match))

        logger.info(
            f"_select_pois: RAG锚定 {len(anchored)}/{len(rag_preferred_pois)} 个景点: "
            f"{[p['name'] for p in anchored]}"
        )

    # ─── Phase 2: 剩余配额按评分/排名填充（仅景点）────────────────────────────
    # 注意：poi_candidates 已确保全为景点类别（餐厅由 6d 步骤单独搜索），
    # 此处直接按有效评分排序填充，不再做类别配额分割。
    remaining_needed = total_needed - len(anchored)
    remaining_pois = [p for p in pois if id(p) not in anchored_ids]

    fill: List[Dict] = []
    if remaining_needed > 0:
        fill = sorted(remaining_pois, key=_effective_rating, reverse=True)[:remaining_needed]

    selected = anchored + fill
    logger.info(f"_select_pois: 最终选出 {len(selected)} 个POI: {[p['name'] for p in selected]}")
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
# 6d — 周边餐厅搜索（基于当天景点地理重心）
# =============================================================================

async def _fetch_day_restaurants(
    day_pois: List[Dict],
    session: Any,
    city: str,
    radius: int = 3000,
    count: int = 5,
) -> List[Dict]:
    """
    计算当天所有景点的地理重心，以此为中心调用高德周边搜索获取附近餐厅。

    选择重心而非第一个景点的原因：重心更接近全天活动区域的几何中心，
    推荐的餐厅对全天行程的覆盖更均匀，减少"只推荐第一个景点附近"的偏差。

    Args:
        day_pois:  当天经 TSP 排序后的景点列表（含 lng, lat 字段）。
        session:   与调用方共享的 amap MCP ClientSession。
        city:      城市名，传递给高德 API 辅助过滤。
        radius:    搜索半径（米），默认 3000m。
        count:     返回餐厅数量上限，默认 5 家。

    Returns:
        餐厅列表，结构同 search_restaurants_nearby() 的返回值。
        若 day_pois 为空或搜索失败，返回空列表。
    """
    if not day_pois:
        return []

    # 计算所有景点的地理重心（平均经纬度）
    valid_pois = [p for p in day_pois if p.get("lng") and p.get("lat")]
    if not valid_pois:
        return []

    centroid_lng = sum(p["lng"] for p in valid_pois) / len(valid_pois)
    centroid_lat = sum(p["lat"] for p in valid_pois) / len(valid_pois)
    centroid_location = f"{centroid_lng:.6f},{centroid_lat:.6f}"

    try:
        restaurants = await search_restaurants_nearby(
            session=session,
            location=centroid_location,
            radius=radius,
            city=city,
            count=count,
        )
        logger.info(
            f"_fetch_day_restaurants: 重心={centroid_location}, "
            f"搜索到 {len(restaurants)} 家餐厅"
        )
        return restaurants
    except Exception as e:
        logger.warning(f"_fetch_day_restaurants: 餐厅搜索失败: {e}")
        return []


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
    解析 RAG 检索到的旅游攻略原始片段，提取两类信息：

    1. rag_boosted_names (set[str])：
       攻略中明确提及的景点名称关键词，用于在 _select_pois 中给对应 POI 加权。
       提取策略：使用 _is_likely_poi 白名单过滤，只保留以景点后缀结尾的词组，
       同时规范化空格（修复 OCR 断字，如 "灵 隐寺"→"灵隐寺"）。

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

    # 提取 2-6 字连续中文词组（不含空格，避免匹配整句）
    # 在逐条 content 中先合并字间孤立空格（修复 OCR 断字，如 "灵 隐寺"→"灵隐寺"），再提取
    poi_name_pattern = re.compile(r'[\u4e00-\u9fa5]{2,6}')

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

    for snippet in rag_snippets:
        content = snippet.get("content", "") if isinstance(snippet, dict) else ""
        if not content:
            continue

        # 规范化：合并字间孤立空格（修复 OCR 断字，如 "灵 隐寺"→"灵隐寺"）
        normalized_content = re.sub(r'(?<=[\u4e00-\u9fa5]) (?=[\u4e00-\u9fa5])', '', content)

        # 1. 提取景点关键词（严格后缀过滤：只保留真正的景点名）
        for match in poi_name_pattern.findall(normalized_content):
            if _is_likely_poi(match):
                boosted_names.add(match)

        # 2. 提取同游约束对（规范化后再匹配）
        for pattern in joint_day_patterns:
            for m in pattern.finditer(normalized_content):
                a, b = m.group(1), m.group(2)
                if _is_likely_poi(a) and _is_likely_poi(b) and a != b:
                    joint_hints.append((a, b))

    logger.info(
        f"_parse_rag_hints: {len(boosted_names)} boosted keywords, "
        f"{len(joint_hints)} joint hints"
    )
    return boosted_names, joint_hints


def _extract_rag_preferred_pois(rag_answer: str) -> List[str]:
    """
    从 RAG 综合回答的【行程安排】段落中提取有序景点名称列表。

    主路径（jieba posseg）：
        使用 jieba 词性标注，筛选 ns（地名）、nz（其他专名）、s（处所词）标签的词。
        这些词天然就是名词，不受时间副词（傍晚）、动词（逛）的干扰。
        jieba 自带大量旅游地名词典，可正确识别西湖、灵隐寺、雷峰塔、断桥、苏堤等。

    Fallback（正则 + 景点后缀白名单）：
        若 jieba 未安装，退化为仅匹配以 _ATTRACTION_SUFFIXES 中词结尾的词组。
        准确率较低（断桥、苏堤等无标准后缀的地名会被漏掉），但不引入噪声。

    Args:
        rag_answer: RAG skill 输出的综合回答文本（已解析 JSON 后的 answer 字段）

    Returns:
        有序景点名称列表（按行程描述中出现的顺序排列），可能为空
    """
    import re

    if not rag_answer or not isinstance(rag_answer, str):
        return []

    # 规范化：去除行内多余空格（修复 OCR 断字），保留换行
    normalized = re.sub(r'[ \t]+', '', rag_answer)

    # 优先截取【行程安排】段落；若无此标记则用全文
    section_match = re.search(r'【行程安排】(.*?)(?:【|$)', normalized, re.DOTALL)
    target = section_match.group(1) if section_match else normalized

    # 过滤词：即使被标注为地名，也应排除的非景点词汇
    _NON_SPOT_WORDS: frozenset = frozenset([
        '地铁', '公交', '步行', '机场', '车站', '高铁', '动车',
        '分钟', '小时', '公里', '早餐', '午餐', '晚餐',
        '美食', '民宿', '酒店', '商场',
    ])

    # ─── 主路径：jieba posseg 词性标注 ──────────────────────────────────────
    try:
        import jieba.posseg as pseg  # type: ignore

        # jieba 初始化时加载词典会有 INFO 日志，仅首次调用，后续自动缓存
        pois: List[str] = []
        seen: set = set()

        for word, flag in pseg.cut(target):
            # ns=地名、nz=其他专名、s=处所词
            # 排除单字词（通常是量词或介词的误标）
            if (
                flag in ('ns', 'nz', 's')
                and len(word) >= 2
                and word not in seen
                and word not in _NON_SPOT_WORDS
                and not any(kw in word for kw in _NON_SPOT_WORDS)
            ):
                pois.append(word)
                seen.add(word)

        logger.info(
            f"_extract_rag_preferred_pois [jieba]: 提取到 {len(pois)} 个景点: {pois}"
        )
        return pois

    except ImportError:
        logger.warning(
            "_extract_rag_preferred_pois: jieba 未安装，降级为正则+后缀方案。"
            " 建议运行 `pip install jieba` 提升提取准确率。"
        )

    # ─── Fallback：正则 + 景点后缀白名单 ────────────────────────────────────
    # 只匹配以 _ATTRACTION_SUFFIXES 中的词结尾的词组，准确率有限但无误报
    pattern = re.compile(
        r'[\u4e00-\u9fa5]{1,5}(?:'
        + '|'.join(re.escape(s) for s in _ATTRACTION_SUFFIXES)
        + r')'
    )
    seen_fb: set = set()
    fallback_pois: List[str] = []
    for m in pattern.finditer(target):
        name = m.group()
        if (
            len(name) >= 2
            and name not in seen_fb
            and not any(kw in name for kw in _NON_SPOT_WORDS)
        ):
            fallback_pois.append(name)
            seen_fb.add(name)

    logger.info(
        f"_extract_rag_preferred_pois [regex fallback]: 提取到 {len(fallback_pois)} 个景点: {fallback_pois}"
    )
    return fallback_pois
