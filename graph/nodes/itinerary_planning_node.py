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

from graph.state import TravelGraphState, PoiTimeInfo, PoiTimeInfoList
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

# 各旅行风格每日最大游览时长（小时），含通勤，用于 _cluster_by_geography 时间预算检查
_MAX_DAILY_HOURS: Dict[str, float] = {
    "老人": 6.0,
    "亲子": 6.0,
    "情侣": 7.0,
    "普通": 8.0,
    "特种兵": 10.0,
}

# best_period 到排序权重的映射：morning 优先，evening 靠后
_PERIOD_ORDER: Dict[str, int] = {
    "morning":   0,
    "flexible":  1,
    "afternoon": 2,
    "evening":   3,
}

# 同天相邻景点单段通勤时间上限（分钟）。
# 来自高德 MCP 距离矩阵的真实行程时间；超过此值的候选景点跳过，留给其他天。
# 90min = 行程约 1.5h，对应市内跨区通勤的合理上限（如西溪→良渚 116min 会被拦截）。
_MAX_SAME_DAY_TRANSIT_MIN: float = 90.0


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
# 时间信息辅助：解析知识库 duration 字符串 + 批量 LLM 查询
# =============================================================================

def _parse_duration_str(duration_str: str) -> float:
    """
    将知识库中的游览时长字符串解析为小时数（float）。

    支持以下格式：
      "2-3小时"  ->  2.5  （取均值）
      "1.5小时"  ->  1.5
      "2小时"    ->  2.0
      "90分钟"   ->  1.5
      其他/空值  ->  1.5  （默认）

    Args:
        duration_str: 原始时长字符串，可为空。

    Returns:
        对应的小时数，解析失败时返回 1.5。
    """
    import re
    if not duration_str:
        return 1.5

    # 范围格式：2-3小时 / 1~2小时
    range_match = re.search(r'([\d.]+)\s*[-~到]\s*([\d.]+)\s*小时', duration_str)
    if range_match:
        lo = float(range_match.group(1))
        hi = float(range_match.group(2))
        return (lo + hi) / 2.0

    # 单值小时
    hour_match = re.search(r'([\d.]+)\s*小时', duration_str)
    if hour_match:
        return float(hour_match.group(1))

    # 分钟
    min_match = re.search(r'([\d.]+)\s*分钟', duration_str)
    if min_match:
        return float(min_match.group(1)) / 60.0

    return 1.5


async def _fetch_poi_time_info(
    pois: List[Dict],
    city: str,
    knowledge_db: CityKnowledgeDB,
    llm,
) -> None:
    """
    批量获取 POI 游览时长和适宜时段，就地写入每个 POI dict 的
    "estimated_hours" 和 "best_period" 字段。

    优先级（三级降级）：
      1. 知识库 duration 字段 -> estimated_hours（精确来源，最高可信度）
      2. LLM with_structured_output   -> 填补知识库未覆盖的 estimated_hours；
                                         所有 POI 的 best_period 均由 LLM 提供
      3. 默认兜底 -> estimated_hours=1.5, best_period="flexible"

    注意：此函数不向 state 写入新字段，修改直接发生在传入的 pois list 的各元素上，
    后续 _cluster_by_geography 可直接读取 poi["estimated_hours"] 和
    poi["best_period"]。

    Args:
        pois:         _select_pois 筛选后的 POI 列表（含 name 字段）。
        city:         目的地城市名，用于知识库查表。
        knowledge_db: CityKnowledgeDB 单例。
        llm:          LangChain ChatOpenAI 实例，为 None 时跳过 LLM 调用。
    """
    if not pois:
        return

    # ── 第一步：从知识库填充已知 duration ────────────────────────────────────
    kb_duration_map: Dict[str, float] = {}
    if city and knowledge_db.has_city(city):
        for poi_info in knowledge_db.get_must_visit(city):
            raw_dur = getattr(poi_info, "duration", None)
            if raw_dur:
                kb_duration_map[poi_info.name] = _parse_duration_str(raw_dur)

    for poi in pois:
        name = poi.get("name", "")
        if name in kb_duration_map:
            poi["estimated_hours"] = kb_duration_map[name]

    # ── 第二步：一次 LLM 调用批量查所有 POI 的 best_period（和缺失的 estimated_hours）──
    all_names = [poi.get("name", "") for poi in pois if poi.get("name")]
    if not all_names or llm is None:
        for poi in pois:
            poi.setdefault("estimated_hours", 1.5)
            poi.setdefault("best_period", "flexible")
        return

    try:
        structured_llm = llm.with_structured_output(PoiTimeInfoList)
        names_str = "、".join(all_names)
        prompt = (
            f"城市：{city}\n"
            f"请为以下景点提供建议游览时长（estimated_hours，单位小时）"
            f"和最佳游览时段（best_period）。\n"
            f"景点列表：{names_str}\n\n"
            f"best_period 取值说明：\n"
            f"  morning   = 适合上午（如寺庙、古迹、需排队的热门景区，光线好、人少）\n"
            f"  afternoon = 适合下午\n"
            f"  evening   = 适合傍晚或夜间（如夜市、灯会、酒吧街）\n"
            f"  flexible  = 全天均可\n\n"
            f"要求：严格按景点列表顺序，每个景点输出一条记录，poi_name 与列表完全一致。"
        )

        result: PoiTimeInfoList = await structured_llm.ainvoke(prompt)

        # 将 LLM 结果写回 POI dict
        name_to_info: Dict[str, PoiTimeInfo] = {
            item.poi_name: item for item in (result.items or [])
        }
        for poi in pois:
            name = poi.get("name", "")
            info = name_to_info.get(name)
            if info:
                # 知识库时长优先（已写入），LLM 时长仅在尚未设置时补充
                poi.setdefault("estimated_hours", info.estimated_hours)
                poi["best_period"] = info.best_period
            else:
                poi.setdefault("estimated_hours", 1.5)
                poi.setdefault("best_period", "flexible")

        logger.info(
            f"_fetch_poi_time_info: LLM 成功返回 {len(name_to_info)} 条时间信息"
        )

    except Exception as exc:
        logger.warning(f"_fetch_poi_time_info: LLM 调用失败: {exc}，使用默认值")
        for poi in pois:
            poi.setdefault("estimated_hours", 1.5)
            poi.setdefault("best_period", "flexible")


# =============================================================================
# 工厂函数（供 workflow.py 调用，闭包注入未来可能的依赖）
# =============================================================================

def create_itinerary_planning_node(llm=None):
    """
    返回 itinerary_planning_node async 函数。

    Args:
        llm: LangChain ChatOpenAI 实例，供 _fetch_poi_time_info 调用
             with_structured_output 批量获取 POI 时间信息。
             为 None 时跳过 LLM 调用，统一使用默认值（1.5h, flexible）。
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

        # ── P4.5 回环处理：读取上轮自检的违规记录 ────────────────────────────
        # 若存在 rule_violations，说明本次是由 itinerary_review_node 触发的重规划。
        # 1. 递增 review_retry_count，防止路由再次回环（最多重试 1 次）
        # 2. 从 long_transit_leg 违规中提取强制拆分对，传给 _cluster_by_geography
        review_violations = state.get("rule_violations") or []
        review_retry_count: int = state.get("review_retry_count", 0)
        retry_state_update: dict = {}

        # 提取 long_transit_leg 违规 → split_hints（格式：[("景点A", "景点B"), ...]）
        split_hints: List[Tuple[str, str]] = []
        if review_violations:
            retry_state_update["review_retry_count"] = review_retry_count + 1
            for v in review_violations:
                if v.get("violation_type") == "long_transit_leg" if isinstance(v, dict) else v.violation_type == "long_transit_leg":
                    desc = v.get("description") if isinstance(v, dict) else v.description
                    # description 格式：第N天 景点A→景点B 交通（模式）XX分钟...
                    # 直接从 suggestion 里提取更可靠：建议将「A」和「B」拆分到不同天
                    suggestion = v.get("suggestion") if isinstance(v, dict) else v.suggestion
                    if suggestion:
                        import re as _re
                        matches = _re.findall(r'「([^」]+)」', suggestion)
                        if len(matches) >= 2:
                            split_hints.append((matches[0], matches[1]))
            if split_hints:
                logger.info(
                    f"[itinerary_planning] P4.5 回环重规划，"
                    f"split_hints={split_hints}，retry_count={review_retry_count + 1}"
                )

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

        # 从 route_combos 展开所有子景点名称集合，供 _select_pois Phase-2 combo_boost 使用
        # 包含所有 combo 子景点（含 must_visit 已覆盖的），保证断桥/雷峰塔等都能获得加分
        combo_spot_names: set = {
            spot
            for combo in route_combos
            for spot in combo
        }

        # 6a — 筛选 POI（Phase-1 KB锚定 + Phase-2 评分填充）
        selected_pois = _select_pois(
            poi_candidates, travel_style, travel_days,
            rag_boosted_names, rag_preferred_pois,
            combo_spot_names=combo_spot_names,
        )
        logger.info(f"_select_pois: 筛选后 {len(selected_pois)} 个 POI")

        # 6a.5 — 批量获取 POI 游览时长和适宜时段（就地写入 estimated_hours / best_period）
        # 结果直接写入 selected_pois 各元素，后续 _cluster_by_geography 直接读取，
        # 无需跨节点传递，也无需新增 state 字段。
        await _fetch_poi_time_info(
            pois=selected_pois,
            city=city,
            knowledge_db=knowledge_db,
            llm=llm,
        )
        logger.info(
            f"_fetch_poi_time_info: 完成。"
            f"示例: {[(p['name'], p.get('estimated_hours'), p.get('best_period')) for p in selected_pois[:3]]}"
        )

        # 6b/6c/6d — 单一 MCP session 完成聚类前置矩阵 + TSP + 餐厅搜索
        # 将顺路组合展开为相邻 POI 对，让同路线景点（断桥→白堤→苏堤→雷峰塔）优先同天
        combo_joint_hints: List[Tuple[str, str]] = [
            (combo[i], combo[i + 1])
            for combo in route_combos
            for i in range(len(combo) - 1)
        ]
        combined_hints = rag_joint_hints + combo_joint_hints

        daily_itinerary: List[Dict] = []
        daily_routes: List[Dict] = []
        daily_restaurants: List[Dict] = []

        try:
            async with amap_mcp_session() as session:

                # 6a.7 — 预取 POI 间真实通勤时间矩阵，用于聚类（比欧氏距离更准确）
                transit_matrix: Optional[List[List[float]]] = None
                try:
                    coords = [f"{p['lng']},{p['lat']}" for p in selected_pois]
                    transit_matrix = await get_distance_matrix(session, coords, coords)
                    logger.info(
                        f"transit_matrix 获取成功: {len(selected_pois)}x{len(selected_pois)}，"
                        f"示例[0][1]={transit_matrix[0][1]:.1f}min" if len(selected_pois) > 1 else
                        f"transit_matrix 获取成功: {len(selected_pois)} 个POI"
                    )
                except Exception as e:
                    logger.warning(f"transit_matrix 获取失败: {e}，降级为欧氏距离聚类")

                # 6b — 基于真实通勤时间的地理聚类（transit_matrix 为 None 时自动降级欧氏）
                # split_hints 来自 P4.5 自检回传，强制将违规 POI 对拆分到不同天
                daily_itinerary = _cluster_by_geography(
                    selected_pois, travel_days, combined_hints,
                    travel_style=travel_style,
                    transit_matrix=transit_matrix,
                    split_hints=split_hints,
                )
                logger.info(
                    f"_cluster_by_geography: {len(daily_itinerary)} 天行程分组完成  "
                    + "  ".join(
                        f"Day{g['day']}=[{','.join(p['name'] for p in g['pois'])}]"
                        for g in daily_itinerary
                    )
                )

                # 6c + 6d — TSP 路线优化 + 周边餐厅搜索
                for day_group in daily_itinerary:
                    route = await _optimize_daily_route(
                        day_pois=day_group["pois"],
                        city=city,
                        session=session,
                    )
                    daily_routes.append({"day": day_group["day"], **route})

                    day_restaurants = await _fetch_day_restaurants(
                        day_pois=route.get("ordered_pois", day_group["pois"]),
                        session=session,
                        city=city,
                    )
                    daily_restaurants.append({
                        "day": day_group["day"],
                        "restaurants": day_restaurants,
                    })

            logger.info("itinerary_planning_node: transit聚类 + TSP优化 + 餐厅搜索全部完成")

        except Exception as e:
            logger.error(
                f"itinerary_planning_node: MCP session 完全失败: {e}，"
                f"降级为欧氏距离聚类 + 空路线"
            )
            # Fallback：MCP session 整体不可用时，用欧氏距离做聚类，跳过 TSP 和餐厅
            if not daily_itinerary:
                daily_itinerary = _cluster_by_geography(
                    selected_pois, travel_days, combined_hints,
                    travel_style=travel_style,
                    transit_matrix=None,   # 无矩阵，纯欧氏兜底
                )
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
            **retry_state_update,   # 回环时写入 review_retry_count += 1
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
    combo_spot_names: Optional[set] = None,
) -> List[Dict]:
    """
    按旅行风格决定每日 POI 数量，从候选列表中选出 total_needed 个 POI。

    两阶段策略：
    ① KB 优先锚定（Phase-1）：
       将知识库 must_visit 景点名与 poi_candidates 进行模糊匹配，
       匹配成功的直接进入 anchored 列表，不参与评分竞争。
       由于 POIFetchAgent 已按 must_visit 名称精准查询高德，锚定基本必然成功。
    ② 剩余配额按有效评分填充（Phase-2）：
       - rating=0 时用 search_rank 换算基准分（排名越靠前分越高）
       - RAG 攻略关键词命中的 POI：+1.5 分（rag_boost）
       - 知识库顺路组合子景点命中的 POI：+0.8 分（combo_boost）
         确保断桥/法喜寺/龙井村等 combo 细粒度子景点在剩余配额中优先入选

    Args:
        rag_boosted_names:  RAG 原始片段中提取的景点关键词集合（用于 rag_boost）
        rag_preferred_pois: 知识库 must_visit 有序景点名列表（用于 Phase-1 锚定）
        combo_spot_names:   知识库顺路组合的所有子景点名集合（用于 combo_boost）
    """
    rag_boosted_names = rag_boosted_names or set()
    rag_preferred_pois = rag_preferred_pois or []
    combo_spot_names = combo_spot_names or set()

    pois_per_day = _POIS_PER_DAY.get(travel_style, 3)
    total_needed = pois_per_day * travel_days

    def _effective_rating(poi: Dict) -> float:
        """
        计算 POI 的有效评分（仅用于 Phase-2 填充排序，已锚定的 must_visit 不经此函数）：
        1. 景点类 POI rating 通常为 0，用 search_rank 换算基准分（rank1→2.0，线性衰减）
        2. rag_boost +1.5：RAG 攻略原文中明确提到的景点
        3. combo_boost +0.8：知识库顺路组合子景点（如断桥、法喜寺、龙井村）
           比普通高德结果有更强的"值得去"背书，在剩余配额中获得优先权
        """
        base = poi.get("rating", 0.0) or 0.0
        if base == 0.0:
            rank = poi.get("search_rank", 20)
            base = max(0.0, (21 - rank) / 21 * 2.0)
        name = poi.get("name", "")
        rag_boost = 1.5 if any(kw in name for kw in rag_boosted_names if kw) else 0.0
        combo_boost = 0.8 if any(
            kw in name or name in kw
            for kw in combo_spot_names if kw
        ) else 0.0
        return base + rag_boost + combo_boost

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
    travel_style: str = "普通",
    transit_matrix: Optional[List[List[float]]] = None,
    split_hints: Optional[List[Tuple[str, str]]] = None,
) -> List[Dict[str, Any]]:
    """
    基于实际通勤时间（或欧氏距离兜底）的贪心聚类，将 POI 分配到各天。

    算法：
    1. 预处理 RAG 同游约束：将"可同天游"的 POI 对提前绑定为同组种子
    2. 预处理强制拆分约束（split_hints）：来自 P4.5 自检的违规 POI 对，
       聚类时禁止将两者放入同一天
    3. 计算所有 POI 的地理重心（平均经纬度），用于 Day1 锚点选择
    4. Day1 锚点 = 距整体重心欧氏距离最远的 POI
       后续天锚点 = 对所有前一天 POI 的最小通勤时间（或欧氏距离）最大的未分配 POI
    5. 每天从锚点出发，按通勤时间（或欧氏距离）升序排列候选，依次检查：
       a. 强制拆分：候选 POI 的 split_partner 已在本天则跳过
       b. 通勤阈值：单段 > _MAX_SAME_DAY_TRANSIT_MIN (75min) 则跳过，留给其他天
       c. 时间预算：当天已用 + 候选游览时长 + 真实通勤时长 > 上限则跳过
       若所有候选均不满足，提前结束当天（不做 fallback 强塞）
    6. 聚类完成后按 best_period 排序每天内部 POI（morning → flexible → afternoon → evening）

    Args:
        rag_joint_hints:  RAG 提取的"可同天游"POI 名称对
        travel_style:     旅行风格，用于从 _MAX_DAILY_HOURS 查每日时间预算上限
        transit_matrix:   高德 MCP get_distance_matrix 返回的 N×N 时间矩阵（分钟）。
                          为 None 时降级为欧氏距离 + 0.5h/段通勤粗估。
        split_hints:      P4.5 自检回传的强制拆分 POI 名称对，这些对不能同天出现。

    返回：[{"day": 1, "pois": [...]}, {"day": 2, "pois": [...]}, ...]
    """
    if not pois or travel_days <= 0:
        return []

    rag_joint_hints = rag_joint_hints or []
    split_hints = split_hints or []
    n = len(pois)
    base_count = n // travel_days
    remainder = n % travel_days
    quotas = [base_count + (1 if d < remainder else 0) for d in range(travel_days)]

    # ── POI id → index 映射，用于后续从 group["pois"] 反查索引 ──────────────
    poi_id_to_idx: Dict[int, int] = {id(p): i for i, p in enumerate(pois)}

    # ── 预处理 RAG 同游约束 ─────────────────────────────────────────────────
    joint_pairs: List[Tuple[int, int]] = []
    for hint_a, hint_b in rag_joint_hints:
        idx_a = next((i for i, p in enumerate(pois) if hint_a in p.get("name", "")), None)
        idx_b = next((i for i, p in enumerate(pois) if hint_b in p.get("name", "")), None)
        if idx_a is not None and idx_b is not None and idx_a != idx_b:
            joint_pairs.append((idx_a, idx_b))
            logger.info(f"RAG joint hint matched: {pois[idx_a]['name']} + {pois[idx_b]['name']}")

    partner_of: Dict[int, int] = {}
    for a, b in joint_pairs:
        partner_of[a] = b
        partner_of[b] = a

    # ── 预处理强制拆分约束（来自 P4.5 自检） ─────────────────────────────────
    # split_partners[i] = {j, k, ...}：POI i 不能与这些 POI 同天
    split_partners: Dict[int, set] = {}
    for hint_a, hint_b in split_hints:
        idx_a = next((i for i, p in enumerate(pois) if hint_a in p.get("name", "")), None)
        idx_b = next((i for i, p in enumerate(pois) if hint_b in p.get("name", "")), None)
        if idx_a is not None and idx_b is not None and idx_a != idx_b:
            split_partners.setdefault(idx_a, set()).add(idx_b)
            split_partners.setdefault(idx_b, set()).add(idx_a)
            logger.info(
                f"[cluster] split hint applied: "
                f"{pois[idx_a]['name']} and {pois[idx_b]['name']} forced to different days"
            )

    # ── 距离/通勤时间辅助函数 ────────────────────────────────────────────────

    def _transit(i: int, j: int) -> float:
        """返回 i → j 的通勤时间（分钟）；无矩阵时返回欧氏距离×1000（量纲无关，仅供排序）。"""
        if transit_matrix is not None:
            return transit_matrix[i][j]
        return _euclidean((pois[i]["lng"], pois[i]["lat"]), (pois[j]["lng"], pois[j]["lat"])) * 1000

    def _transit_to_prev_day(candidate: int, prev_day_poi_ids: List[int]) -> float:
        """候选 POI 到前一天所有 POI 的最小通勤时间（最近距离，用于锚点选择）。"""
        return min(_transit(poi_id_to_idx[pid], candidate) for pid in prev_day_poi_ids)

    # ── 时间预算辅助函数（使用真实通勤时间）──────────────────────────────────

    max_daily_hours: float = _MAX_DAILY_HOURS.get(travel_style, 8.0)

    def _day_visit_hours(day_idx_list: List[int]) -> float:
        return sum(pois[i].get("estimated_hours", 1.5) for i in day_idx_list)

    def _day_transit_hours_real(day_idx_list: List[int]) -> float:
        """当天已有景点之间的实际通勤时长合计（小时）。"""
        if transit_matrix is None or len(day_idx_list) < 2:
            return max(0, len(day_idx_list) - 1) * 0.5  # 兜底：0.5h/段
        total = 0.0
        for k in range(len(day_idx_list) - 1):
            total += transit_matrix[day_idx_list[k]][day_idx_list[k + 1]] / 60.0
        return total

    def _new_transit_hours(day_idx_list: List[int], candidate: int) -> float:
        """加入 candidate 后新增一段通勤的时长（小时）。"""
        if transit_matrix is None or not day_idx_list:
            return 0.5
        return transit_matrix[day_idx_list[-1]][candidate] / 60.0

    def _fits_in_budget(day_idx_list: List[int], candidate: int) -> bool:
        visit  = _day_visit_hours(day_idx_list) + pois[candidate].get("estimated_hours", 1.5)
        transit = _day_transit_hours_real(day_idx_list) + _new_transit_hours(day_idx_list, candidate)
        return visit + transit <= max_daily_hours

    def _exceeds_transit_cap(current: int, candidate: int) -> bool:
        """单段通勤超过上限：不适合同天出行。无矩阵时不做限制。"""
        if transit_matrix is None:
            return False
        return transit_matrix[current][candidate] > _MAX_SAME_DAY_TRANSIT_MIN

    # ── 整体地理重心（Day1 锚点用，虚拟点，不受矩阵约束）─────────────────────
    centroid: Tuple[float, float] = (
        sum(p["lng"] for p in pois) / n,
        sum(p["lat"] for p in pois) / n,
    )

    unassigned: List[int] = list(range(n))
    groups: List[Dict[str, Any]] = []

    for day_idx, quota in enumerate(quotas):
        if not unassigned:
            break

        day_indices: List[int] = []

        # ── 选锚点 ───────────────────────────────────────────────────────────
        if day_idx == 0:
            # Day1：距整体重心欧氏距离最远
            anchor = max(
                unassigned,
                key=lambda i: _euclidean((pois[i]["lng"], pois[i]["lat"]), centroid),
            )
        else:
            # Day2+：对前一天所有 POI 的最小通勤时间最大（即最"孤立"的未分配 POI）
            prev_poi_ids = [id(p) for p in groups[-1]["pois"]]
            anchor = max(
                unassigned,
                key=lambda i: _transit_to_prev_day(i, prev_poi_ids),
            )

        day_indices.append(anchor)
        unassigned.remove(anchor)

        # ── 立即追加 RAG 伙伴（若满足通勤阈值和时间预算）────────────────────
        if anchor in partner_of:
            partner = partner_of[anchor]
            if (
                partner in unassigned
                and len(day_indices) < quota
                and not _exceeds_transit_cap(anchor, partner)
                and _fits_in_budget(day_indices, partner)
            ):
                day_indices.append(partner)
                unassigned.remove(partner)

        # ── 最近邻贪心填充 ───────────────────────────────────────────────────
        while len(day_indices) < quota and unassigned:
            current = day_indices[-1]

            # 优先：RAG 伙伴（满足通勤阈值和时间预算）
            rag_priority = next(
                (partner_of[i] for i in day_indices
                 if i in partner_of and partner_of[i] in unassigned),
                None,
            )
            if rag_priority is not None:
                if (
                    not _exceeds_transit_cap(current, rag_priority)
                    and _fits_in_budget(day_indices, rag_priority)
                ):
                    day_indices.append(rag_priority)
                    unassigned.remove(rag_priority)
                    continue
                # RAG 伙伴不满足约束，落入常规最近邻逻辑

            # 按通勤时间升序排列候选（无矩阵时用欧氏距离代理）
            candidates_sorted = sorted(unassigned, key=lambda i: _transit(current, i))

            added = False
            for candidate in candidates_sorted:
                # 强制拆分检查：候选的 split_partner 已在本天，跳过（不 break，继续找其他候选）
                if candidate in split_partners:
                    if split_partners[candidate] & set(day_indices):
                        logger.info(
                            f"_cluster_by_geography: 第{day_idx+1}天 "
                            f"跳过「{pois[candidate]['name']}」（P4.5 强制拆分约束）"
                        )
                        continue

                if _exceeds_transit_cap(current, candidate):
                    # 通勤超时，且后续候选只会更远（已排序），直接停止本天填充
                    logger.debug(
                        f"_cluster_by_geography: 第{day_idx+1}天 "
                        f"{pois[current]['name']} -> {pois[candidate]['name']} "
                        f"通勤={transit_matrix[current][candidate]:.0f}min > {_MAX_SAME_DAY_TRANSIT_MIN}min，停止"
                    )
                    break
                if _fits_in_budget(day_indices, candidate):
                    day_indices.append(candidate)
                    unassigned.remove(candidate)
                    added = True
                    break
                # 候选通勤在阈值内但时间预算不足：继续尝试下一个更近的候选
                logger.debug(
                    f"_cluster_by_geography: 第{day_idx+1}天 "
                    f"{pois[candidate]['name']} 游览时长超预算，跳过"
                )

            if not added:
                logger.info(
                    f"_cluster_by_geography: 第{day_idx+1}天配额={quota}，"
                    f"实分={len(day_indices)}，因通勤/预算约束提前结束"
                )
                break

        groups.append({"day": day_idx + 1, "pois": [pois[i] for i in day_indices]})

    # 未被分配的 POI 直接丢弃（时间/通勤约束下无法合理安排，不做强塞）
    if unassigned:
        logger.info(
            f"_cluster_by_geography: {len(unassigned)} 个POI因通勤/时间约束未能安排，"
            f"已舍弃: {[pois[i]['name'] for i in unassigned]}"
        )

    # 按 best_period 排序每天内部 POI（morning 优先，evening 靠后）
    for group in groups:
        group["pois"].sort(
            key=lambda p: _PERIOD_ORDER.get(p.get("best_period", "flexible"), 1)
        )

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
