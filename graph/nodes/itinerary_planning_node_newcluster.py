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

import json
import logging
from itertools import permutations
from math import sqrt
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.runnables import RunnableConfig

from graph.state import TravelGraphState, PoiTimeInfo, PoiTimeInfoList, ensure_hard_constraints, RAGContext
from utils.knowledge_parser import CityKnowledgeDB
from utils.llm_resilience import retry_with_backoff
from utils.poi_category import get_category_for_poi
from mcp_clients.amap_client import (
    amap_mcp_session,
    get_distance_matrix,
    get_transit_route,
)

logger = logging.getLogger(__name__)

# POI 每天数量上限（按旅行风格）
_POIS_PER_DAY: Dict[str, int] = {
    "老人": 3,
    "亲子": 3,
    "情侣": 3,
    "普通": 3,
    "特种兵": 4,
}

# 候选池放大系数（按旅行风格动态调整）：
# 候选池 size = ceil(pois_per_day * travel_days * _STYLE_POOL_FACTOR[style])
# 候选池放大后参与 K-means，分簇结束按 pois_per_day 砍尾，让地理上不顺路的低分 POI 自然出局。
_STYLE_POOL_FACTOR: Dict[str, float] = {
    "老人":   1.0,
    "亲子":   1.0,
    "情侣":   1.0,
    "普通":   1.2,
    "特种兵": 1.5,
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
    '售票处', '入口', '停车场', '服务区', '休息区', '信息台', '咨询台', '游客中心', '检票口',
    '分钟', '小时', '公里', '米', '元', '号线', '路线', '攻略',
    '推荐', '建议', '注意', '适合', '游览', '参观', '早场', '晚场',
    '交通', '住宿', '美食', '餐厅', '酒店', '民宿', '综合', '指南',
    '核心', '本地', '环湖', '顺路', '枢纽', '商圈', '繁华', '入口',
])

# 各旅行风格每日最大游览时长（小时），含通勤，用于 _cluster_by_geography 时间预算检查
_MAX_DAILY_HOURS: Dict[str, float] = {
    "老人": 7.0,
    "亲子": 7.0,
    "情侣": 8.0,
    "普通": 8.0,
    "特种兵": 10.0,
}

# best_period 到排序权重的映射：morning 优先，evening 靠后
_PERIOD_ORDER: Dict[str, int] = {
    "morning":   0,
    "flexible":  1,
    "evening":   2,
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
      "2-3小时"  ->  3.0   （区间取上界）
      "1.5小时"  ->  1.5
      "2小时"    ->  2.0
      "90分钟"   ->  1.5
      "1天" / "一天"      -> 8.0  （按一个游览日 8 小时计）
      "半天"              -> 4.0
      "1-2天"             -> 16.0
      其他/空值          ->  2.0  （默认）
    Args:
        duration_str: 原始时长字符串，可为空。
    Returns:
        对应的小时数，解析失败时返回 2。
    """
    import re
    if not duration_str:
        return 2.0

    HOURS_PER_DAY = 8.0  # 一个游览日按 8 小时计算

    # 中文数字到阿拉伯数字（仅覆盖常见的 1-10，足够 "一天/两天" 等表达）
    cn_num = {"一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5,
              "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
    s = duration_str
    for cn, ar in cn_num.items():
        s = s.replace(cn, str(ar))

    # 半天
    if "半天" in s:
        return HOURS_PER_DAY / 2.0

    # 天数：范围 "1-2天" -> 取上界 * 8
    if (m := re.search(r'([\d.]+)\s*[-~到]\s*([\d.]+)\s*天', s)):
        return float(m.group(2)) * HOURS_PER_DAY
    # 天数：单值 "1天" / "1.5天"
    if (m := re.search(r'([\d.]+)\s*天', s)):
        return float(m.group(1)) * HOURS_PER_DAY

    # 小时：范围 "2-3小时" -> 取上界
    if (m := re.search(r'([\d.]+)\s*[-~到]\s*([\d.]+)\s*小时', s)):
        return float(m.group(2))
    # 小时：单值
    if (m := re.search(r'([\d.]+)\s*小时', s)):
        return float(m.group(1))

    # 分钟
    if (m := re.search(r'([\d.]+)\s*分钟', s)):
        return float(m.group(1)) / 60.0

    return 2.0


# 大类游览时长下限（小时）。用于兜底：当 KB/LLM 给出的 estimated_hours 明显偏低时，
# 至少抬到该类景点公认的合理最小值。不会下调更高的值，只在低于下限时抬升。
# 取值依据：大型公园/博物馆通常需要 2.5h+ 才能基本逛完；古镇/遗址 1.5h；寺庙最小 1h。
_CATEGORY_HOUR_FLOOR: Dict[str, float] = {
    "自然公园": 2.5,
    "博物馆":   2.5,
    "古镇古街": 1.5,
    "遗址遗迹": 1.5,
    "宗教寺庙": 1.0,
}


def _apply_category_floor(pois: List[Dict]) -> None:
    """
    对所有 POI 的 estimated_hours 应用 category-based 下限兜底，就地修改。
    仅在当前值低于下限时抬升，避免误伤短时长的小景点。
    """
    from utils.poi_category import get_category_for_poi
    for poi in pois:
        category = get_category_for_poi(poi)
        floor = _CATEGORY_HOUR_FLOOR.get(category or "")
        if floor is None:
            continue
        current = poi.get("estimated_hours")
        if current is None or current < floor:
            poi["estimated_hours"] = floor


async def _fetch_poi_time_info(
    pois: List[Dict],
    city: str,
    knowledge_db: CityKnowledgeDB,
    llm,
) -> None:
    """
    批量获取 POI 游览时长和适宜时段，就地写入每个 POI dict 的
    "estimated_hours" 和 "best_period" 字段。
    策略：
      1. 知识库 duration 字段 -> estimated_hours（最高可信度，不调用LLM）
      2. LLM 批量查询 -> 为知识库缺失的获取 estimated_hours，为所有POI获取 best_period
      3. 默认兜底 -> estimated_hours=2.0, best_period="flexible"
    Args:
        pois:         _select_pois 筛选后的 POI 列表（含 name 字段）。
        city:         目的地城市名，用于知识库查表。
        knowledge_db: CityKnowledgeDB 单例。
        llm:          LangChain ChatOpenAI 实例，为 None 时跳过 LLM 调用。
    """
    from utils.poi_category import get_category_for_poi

    if not pois:
        return

    # ── 第一步：从知识库填充已知 duration ────────────────────────────────────
    kb_duration_map: Dict[str, float] = {}
    if city and knowledge_db.has_city(city):
        for poi_info in knowledge_db.get_must_visit(city):
            raw_dur = getattr(poi_info, "duration", None)
            if raw_dur:
                kb_duration_map[poi_info.name] = _parse_duration_str(raw_dur)
    # 先给所有POI设置知识库的时长
    for poi in pois:
        name = poi.get("name", "")
        if name in kb_duration_map:
            poi["estimated_hours"] = kb_duration_map[name]
    # ── 第二步：LLM 批量查询（所有POI的best_period + 知识库缺失的estimated_hours）──
    all_names = [poi.get("name", "") for poi in pois if poi.get("name")]
    if not all_names or llm is None:
        # 没有LLM时，用默认值
        for poi in pois:
            poi.setdefault("estimated_hours", 2.0)
            poi.setdefault("best_period", "flexible")
        _apply_category_floor(pois)
        return
    try:
        # 为每个POI收集类别信息
        names_with_category = []
        for name in all_names:
            poi = next((p for p in pois if p.get("name") == name), None)
            category = get_category_for_poi(poi) if poi else "普通景点"
            names_with_category.append(f"{name}({category})")   
        # 标识哪些是知识库已有，哪些需要LLM补充
        kb_covered_names = set(kb_duration_map.keys())
        need_duration_names = [n for n in all_names if n not in kb_covered_names]
        names_str = "、".join(all_names)
        names_with_category_str = "、".join(names_with_category)
        system_prompt = (
            "你是一个专业的旅游向导，熟悉各地旅游景点的推荐游览时长和最佳游览时段。\n"
            "在推荐游览时长时，请优先按下面的『景点体量分级』判断时长，再做细微调整：\n"
            "  - 大型（large）：主题乐园、综合景区、5A 大型自然/文化景区、大型博物馆等，"
            "通常 4-8 小时（如：上海迪士尼 8h、故宫博物院 4-5h、长隆 6-8h、世界之窗 4-5h、"
            "颐和园 3-4h、秦始皇陵 4-5h、东湖风景区 3-4h）。\n"
            "  - 中型（medium）：普通博物馆、中型公园、古镇古街、知名寺庙、城市地标公园等，"
            "通常 1.5-3 小时（如：大雁塔 1.5h、宋城 3-4h、鼓浪屿日光岩 1.5h）。\n"
            "  - 小型（small）：单点观景台、广场、网红打卡点、小型寺庙/塔/桥等，"
            "通常 0.5-1.5 小时（如：长江索道 0.5h、李子坝 0.5h、雷峰塔 1h、富里桥 1h）。\n"
            "其它需要考虑的因素：景点是否包含多个子景点/展览、是否需要排队、是否有"
            "体验项目或演出，以及是否需要预留拍照与休息时间。\n"
            "重要：不要因为输入清单短就压缩时长；请按景点本身的体量给出合理、充裕的时长，"
            "让游客能充分体验景点魅力。"
        )
        user_prompt = (
            f"城市：{city}\n"
            f"请为以下景点提供建议游览时长（estimated_hours，单位小时）"
            f"和最佳游览时段（best_period）。\n"
            f"景点列表（括号内为大类标签）：{names_with_category_str}\n\n"
            f"时长参考（按体量分级，先判规模再选范围）：\n"
            f"  - 大型主题乐园/综合度假区：6-8 小时（如迪士尼、长隆、方特）\n"
            f"  - 大型 5A 景区/皇家园林/著名博物馆：4-5 小时（如故宫、颐和园、秦始皇陵、世界之窗）\n"
            f"  - 中型自然公园/动物园/大型园区：2.5-4 小时\n"
            f"  - 中型博物馆/展览馆：2-3 小时\n"
            f"  - 古镇古街/历史街区：1.5-2.5 小时\n"
            f"  - 宗教寺庙：1-2 小时\n"
            f"  - 单点地标/观景台/网红打卡点：0.5-1.5 小时\n\n"
            f"best_period 取值说明：\n"
            f"  morning   = 适合上午（如寺庙、古迹、需排队的热门景区，光线好、人少）\n"
            f"  evening   = 适合傍晚或夜间（如夜市、灯会、酒吧街、看夜景）\n"
            f"  flexible  = 全天均可\n\n"
            f'输出格式（JSON）：{{"items": [{{"poi_name": "景点名", "estimated_hours": 2.0, "best_period": "morning"}}]}}\n'
            f"要求：\n"
            f"  1. 严格按景点列表顺序，每个景点输出一条记录\n"
            f"  2. poi_name 必须与输入的景点名完全一致（不要带括号里的类别）\n"
            f"  3. estimated_hours 取值范围：0.5-8.0，保留1位小数；大型景点不要给低于 4.0 的值\n"
            f"  4. best_period 只能是 morning/evening/flexible 3个值之一\n"
        )

        response = await retry_with_backoff(
            lambda: llm.ainvoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]),
            max_retries=2,
        )
        raw = json.loads(response.content)
        result = PoiTimeInfoList(items=[PoiTimeInfo(**item) for item in raw.get("items", [])])

        # 将 LLM 结果写回 POI dict
        name_to_info: Dict[str, PoiTimeInfo] = {
            item.poi_name: item for item in (result.items or [])
        }
        
        for poi in pois:
            name = poi.get("name", "")
            info = name_to_info.get(name)
            if info:
                # 知识库已有：只设置 best_period，不覆盖时长
                if name in kb_covered_names:
                    poi["best_period"] = info.best_period
                else:
                    # 知识库缺失：设置时长和 best_period
                    poi.setdefault("estimated_hours", info.estimated_hours)
                    poi["best_period"] = info.best_period
            else:
                # LLM 没有返回该POI信息：设置默认值
                poi.setdefault("estimated_hours", 2.0)
                poi.setdefault("best_period", "flexible")

        logger.info(
            f"_fetch_poi_time_info: LLM 成功返回 {len(name_to_info)} 条信息，"
            f"知识库已覆盖 {len(kb_covered_names)} 个，LLM补充 {len(need_duration_names)} 个"
        )

    except Exception as exc:
        logger.warning(f"_fetch_poi_time_info: LLM 调用失败: {exc}，使用默认值")
        for poi in pois:
            poi.setdefault("estimated_hours", 2.0)
            poi.setdefault("best_period", "flexible")

    # 最终统一兜底：按大类抬升明显偏低的 estimated_hours
    _apply_category_floor(pois)


# =============================================================================
# 预算辅助：从 transport_options 提取最低交通费用
# =============================================================================

def _parse_min_transport_cost(transport_options: List[Dict]) -> Optional[float]:
    """
    从 transport_options 列表中提取最低单人交通费（单程），单位元。
    price_range 字段为字符串（如 "¥250-350"、"二等座：250元"），用正则提取数字取最小值。
    解析失败返回 None。
    """
    import re
    min_cost: Optional[float] = None
    for opt in transport_options:
        raw = opt.get("price_range") or ""
        nums = re.findall(r"(\d+(?:\.\d+)?)", str(raw))
        if nums:
            candidate = float(min(nums, key=float))
            if min_cost is None or candidate < min_cost:
                min_cost = candidate
    return min_cost


def _compute_budget_tier(daily_spend_budget: float) -> str:
    """
    将每日可支配花销（餐饮+景点+市内交通，= daily_land_budget * 0.6）映射为预算档位。
    低于 100 → 经济；100-300 → 普通；300-500 → 舒适；500 以上 → 豪华。
    """
    if daily_spend_budget < 100:
        return "经济"
    if daily_spend_budget < 300:
        return "普通"
    if daily_spend_budget < 500:
        return "舒适"
    return "豪华"


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

    async def itinerary_planning_node(
        state: TravelGraphState,
        config: Optional[RunnableConfig] = None,
    ) -> dict:
        """
        行程规划节点主流程：
        1. 从 state 读取 poi_candidates、travel_style、travel_days、目的地城市
        2. 解析 rag_snippets，提取 POI 关键词权重集合和同游景点约束对
        3. 从 skill_results 中找到 RAG answer，用 jieba 提取推荐景点序列
        4. _select_pois：Phase-1 RAG锚定 + Phase-2 评分填充（仅景点）
        5. _cluster_by_geography：贪心地理聚类
        6. _optimize_daily_route：TSP 优化 + 高德路线（共享单个 MCP session）
        7. 写入 state: daily_itinerary, daily_routes
        注：周边餐厅搜索已迁出，由 restaurant_node 在 itinerary_review 通过后统一处理，
        避免 P3 回环重规划时重复消耗高德 API 配额。

        Web SSE 进度推送：
            FastAPI 层通过 RunnableConfig.configurable.progress_cb 注入异步回调，
            分别在 poi_select / daily_cluster / route_optimize 三个子步骤完成后调用。
            CLI 模式不传 config 或不带 progress_cb 时静默跳过，零侵入。
        """
        # ── 子步骤进度推送（SSE 模式才启用，CLI 模式 cb 为 None 自动跳过）─────
        progress_cb = None
        if config:
            progress_cb = (config.get("configurable") or {}).get("progress_cb")

        async def _emit(sub_node: str, label: str, data: Optional[Dict] = None) -> None:
            """progress_cb 的包装：cb 不存在或抛错都不阻断主流程"""
            if progress_cb is None:
                return
            try:
                await progress_cb(sub_node, label, data)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"progress_cb raised, ignored: {exc!r}")

        # 子步骤 1/3 起点：尽早通知前端 P3 已经开始干实事，
        # 避免 LLM 抽取 + 候选过滤期间出现长达 5s+ 的状态空白
        await _emit("poi_select", "正在根据旅行风格筛选合适景点")

        poi_candidates: List[Dict] = state.get("poi_candidates", [])
        travel_style: str = state.get("travel_style", "普通")
        travel_days: int = state.get("travel_days") or 0
        if travel_days <= 0:
            # extract_constraints_node 应已从 start_date/end_date 计算 travel_days；
            # 若仍为 0，说明两端日期缺失，尝试从 hard_constraints 兜底重算。
            _hc = ensure_hard_constraints(state.get("hard_constraints"))
            if _hc.start_date and _hc.end_date:
                from datetime import datetime as _dt
                try:
                    _s = _dt.strptime(_hc.start_date, "%Y-%m-%d")
                    _e = _dt.strptime(_hc.end_date, "%Y-%m-%d")
                    travel_days = max((_e - _s).days + 1, 1)
                except ValueError:
                    travel_days = 1
            else:
                travel_days = 1
            logger.warning(
                f"itinerary_planning_node: travel_days 未由 extract_constraints_node 写入 "
                f"(start={_hc.start_date!r}, end={_hc.end_date!r})，兜底计算得 travel_days={travel_days}"
            )

        # ── P4.5 回环处理：读取上轮自检的违规记录 ────────────────────────────
        # 若存在 rule_violations，说明本次是由 itinerary_review_node 触发的重规划。
        # 1. 递增 review_retry_count，防止路由无限回环（最多重试 REVIEW_MAX_RETRIES 次）
        # 2. 把 4 类违规解析为 P3 可消费的三类 hints：
        #      remove_hints  → 本轮跳过这些 POI（daily_time_overload、同类集中最长项）
        #      split_hints   → 强制拆分到不同天（long_transit_leg、同类集中 POI 对）
        #      reorder_hints → TSP 后按 best_period 二次稳定排序（time_slot_mismatch）
        review_violations = state.get("rule_violations") or []
        review_retry_count: int = state.get("review_retry_count", 0)
        retry_state_update: dict = {}

        remove_hints: set = set()
        split_hints: List[Tuple[str, str]] = []
        reorder_hints: set = set()

        # 跨轮累积的 remove_hints（与 accumulated_split_hints 对称）：
        # 一旦某 POI 在历史轮次被识别为问题，后续永久不再选；
        # 防止同一孤岛/超时景点在每轮重新被 KB 锚定 → 看似 retry 实则原地踏步
        accumulated_remove: set = set(state.get("accumulated_remove_hints") or [])

        if review_violations:
            retry_state_update["review_retry_count"] = review_retry_count + 1
            new_remove, new_splits, new_reorder = _parse_violation_hints(review_violations)
            # 累积历史 split_hints，防止每轮只看当前违规导致约束丢失
            prev_splits = state.get("accumulated_split_hints") or []
            split_hints = list({tuple(x) for x in [*prev_splits, *new_splits]})
            retry_state_update["accumulated_split_hints"] = [list(p) for p in split_hints]

            # 累积 remove_hints：本轮 new_remove 并入历史集合，下一轮继续生效
            accumulated_remove |= new_remove
            remove_hints = accumulated_remove
            retry_state_update["accumulated_remove_hints"] = sorted(accumulated_remove)

            reorder_hints = new_reorder
            logger.info(
                f"[itinerary_planning] P4.5 回环重规划 (retry={review_retry_count + 1}): "
                f"remove_hints(累积)={remove_hints}, split_hints={split_hints}, "
                f"reorder_hints={reorder_hints}"
            )
        elif accumulated_remove:
            # 无新违规但有历史累积（罕见路径，例如 checkpointer 续跑）：仍应用历史约束
            remove_hints = accumulated_remove

        # 应用 remove_hints：从候选中过滤（防止 _select_pois 再次锚定/填充同名 POI）
        if remove_hints:
            before_count = len(poi_candidates)
            poi_candidates = [
                p for p in poi_candidates if p.get("name") not in remove_hints
            ]
            logger.info(
                f"[itinerary_planning] remove_hints 过滤：{before_count} → {len(poi_candidates)} 个候选"
            )

        # 从 hard_constraints 提取目的地城市（由 extract_constraints_node 结构化保证形态）
        hard_constraints = ensure_hard_constraints(state.get("hard_constraints"))
        city = hard_constraints.destination or ""

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

        # ── 预算计算：扣除往返交通后得到落地预算，派生 budget_tier 用于 POI 筛选 ──
        daily_land_budget: Optional[float] = None
        budget_tier: Optional[str] = None

        if hard_constraints.total_budget is not None and travel_days > 0:
            transport_cost = _parse_min_transport_cost(state.get("transport_options") or [])
            land = hard_constraints.total_budget - (transport_cost or 0.0)
            # 交通费超出预算时 land < 0，仍继续规划（respond_node 会输出警告）
            land = max(land, 0.0)
            daily_land_budget = land / travel_days
            budget_tier = _compute_budget_tier(daily_land_budget * 0.6)
            logger.info(
                f"[itinerary_planning] 预算: total={hard_constraints.total_budget}, "
                f"transport_cost={transport_cost}, daily_land={daily_land_budget:.1f}, "
                f"budget_tier={budget_tier}"
            )

        if not poi_candidates:
            logger.warning("itinerary_planning_node: poi_candidates 为空，跳过规划")
            return {}

        # ── 解析 RAG 攻略原始片段，提取加权关键词 ────────────────────────────
        _rag_ctx: RAGContext = state.get("rag_context") or RAGContext()
        rag_snippets: List[Dict] = _rag_ctx.rag_snippets
        rag_boosted_names, rag_joint_hints = _parse_rag_hints(rag_snippets)
        logger.info(
            f"RAG hints: boosted_names={rag_boosted_names}, joint_hints={rag_joint_hints}"
        )

        # ── 种子景点：直接读取 P2 中段 llm_seed_extract_node 写入的 state.llm_seed_pois ─
        # 该字段已是「KB 必去（保序在前）⊕ LLM 抽取补充」的合并结果，无需在此重复抽取。
        # route_combos 仍需在此查表（itinerary_planning 专用，未透传到 state）。
        knowledge_db = CityKnowledgeDB.get_instance()
        route_combos: List[List[str]] = (
            knowledge_db.get_route_combos(city) if city and knowledge_db.has_city(city) else []
        )

        rag_preferred_pois: List[str] = list(state.get("llm_seed_pois") or [])

        # 极端兜底（按优先级递降）：
        #   1) seed 节点正常产出 → 直接使用（绝大多数路径）
        #   2) seed 为空但 KB 命中 → 直接拿 KB must_visit（seed 节点 LLM 调用失败时）
        #   3) seed + KB 都没有 → jieba 提取的 rag_boosted_names 作为种子
        if not rag_preferred_pois and city and knowledge_db.has_city(city):
            rag_preferred_pois = list(knowledge_db.get_must_visit_names(city))
            logger.warning(
                f"itinerary_planning_node: state.llm_seed_pois 为空, "
                f"降级为 KB must_visit: {rag_preferred_pois}"
            )
        if not rag_preferred_pois and rag_boosted_names:
            rag_preferred_pois = list(rag_boosted_names)
            logger.warning(
                f"itinerary_planning_node: seed+KB 均无产出, "
                f"最终降级为 rag_boosted_names: {rag_preferred_pois}"
            )

        logger.info(
            f"itinerary_planning_node: city={city}, style={travel_style}, "
            f"days={travel_days}, poi_count={len(poi_candidates)}, "
            f"seed_names(来自 state.llm_seed_pois)={rag_preferred_pois}"
        )

        # 从 route_combos 展开所有子景点名称集合，供 _select_pois Phase-2 combo_boost 使用
        # 包含所有 combo 子景点（含 must_visit 已覆盖的），保证断桥/雷峰塔等都能获得加分
        combo_spot_names: set = {
            spot
            for combo in route_combos
            for spot in combo
        }

        # 历史违规涉案 POI 集合（split_hints 两端展开）：未被直接删除但反复参与违规
        # 在 retry 中作为评分惩罚目标，迫使 Phase-2 填充探索其他候选
        penalty_pois: set = {name for pair in split_hints for name in pair}

        # 6a — 筛选 POI（Phase-1 KB锚定 + Phase-2 评分填充）
        selected_pois = _select_pois(
            poi_candidates, travel_style, travel_days,
            rag_boosted_names, rag_preferred_pois,
            combo_spot_names=combo_spot_names,
            budget_tier=budget_tier,
            penalty_pois=penalty_pois,
            retry_count=review_retry_count,
        )
        logger.info(f"_select_pois: 筛选后 {len(selected_pois)} 个 POI")

        # 子步骤 2/3：筛选完成 → 进入聚类
        await _emit(
            "daily_cluster",
            "正在按地理位置分配每日行程",
            {
                "selected_count": len(selected_pois),
                "total_candidates": len(poi_candidates),
            },
        )

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
        isolated_pois: List[str] = []

        try:
            async with amap_mcp_session() as session:

                # 6a.7 — 预取 POI 间真实通勤时间矩阵，用于聚类（比欧氏距离更准确）
                transit_matrix: Optional[List[List[float]]] = None
                try:
                    coords = [f"{p['lng']},{p['lat']}" for p in selected_pois]
                    # 高德 maps_distance 返回的 duration 单位是"秒"，
                    # 但 _cluster_by_geography / _cross_day_swap 内部的阈值
                    # (_MAX_SAME_DAY_TRANSIT_MIN=90min) 和预算换算 (/60 -> 小时)
                    # 全部假设输入单位是"分钟"。此处在边界处统一做 秒->分钟 转换，
                    # 避免单位错配导致每段通勤被误判超阈（实际只是几十秒就 > 90）。
                    transit_matrix_sec = await get_distance_matrix(session, coords, coords)
                    transit_matrix = [
                        [v / 60.0 for v in row] for row in transit_matrix_sec
                    ]
                    logger.info(
                        f"transit_matrix 获取成功: {len(selected_pois)}x{len(selected_pois)}，"
                        f"示例[0][1]={transit_matrix[0][1]:.1f}min" if len(selected_pois) > 1 else
                        f"transit_matrix 获取成功: {len(selected_pois)} 个POI"
                    )
                except Exception as e:
                    logger.warning(f"transit_matrix 获取失败: {e}，降级为欧氏距离聚类")

                # 6a.8 — 识别并过滤孤岛 POI（与所有其他 POI 的最小通勤时间均超阈值）
                # 孤岛 POI 若进入聚类，会因 anchor 选择策略（max min_transit_to_prev_day）
                # 被强制选作某天锚点，又因 _MAX_SAME_DAY_TRANSIT_MIN=90min 阻止同伴加入，
                # 最终单独成天且 4 项 review 全过 → 这里直接过滤掉以根除该现象。
                # 安全阈值：过滤后剩余 POI 必须 >= travel_days，否则放弃过滤以保证可填满每天。
                isolated_indices = _find_isolated_pois(selected_pois, transit_matrix)
                isolated_pois = [
                    selected_pois[i].get("name", "") for i in isolated_indices
                ]
                isolated_pois = [n for n in isolated_pois if n]

                if isolated_indices:
                    remaining_count = len(selected_pois) - len(isolated_indices)
                    if remaining_count >= travel_days:
                        keep_order = [
                            i for i in range(len(selected_pois))
                            if i not in set(isolated_indices)
                        ]
                        kept_pois = [selected_pois[i] for i in keep_order]
                        kept_matrix = (
                            [[transit_matrix[i][j] for j in keep_order] for i in keep_order]
                            if transit_matrix is not None else None
                        )
                        logger.info(
                            f"[itinerary_planning] 过滤孤岛 POI: "
                            f"{len(selected_pois)} -> {len(kept_pois)}，丢弃: {isolated_pois}"
                        )
                        selected_pois = kept_pois
                        transit_matrix = kept_matrix
                    else:
                        logger.warning(
                            f"[itinerary_planning] 检出孤岛 POI {isolated_pois}，"
                            f"但过滤后剩余 {remaining_count} < 行程 {travel_days} 天，"
                            f"放弃过滤以保证可填满每天"
                        )

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

                # 子步骤 3/3：聚类完成 → 进入 TSP 路线优化
                await _emit(
                    "route_optimize",
                    "正在优化每日游览路线",
                    {
                        "days": [
                            {"day": g["day"], "poi_count": len(g["pois"])}
                            for g in daily_itinerary
                        ],
                    },
                )

                # 6c — TSP 路线优化（周边餐厅搜索已迁出至 restaurant_node）
                # 按 best_period 分桶后再 TSP，时段顺序天然成立，
                # P4.5 time_slot_mismatch 违规不会再回环到这里重排。
                for day_group in daily_itinerary:
                    route = await _optimize_daily_route(
                        day_pois=day_group["pois"],
                        city=city,
                        session=session,
                    )
                    daily_routes.append({"day": day_group["day"], **route})

            logger.info("itinerary_planning_node: transit聚类 + TSP优化 全部完成")

        except Exception as e:
            logger.error(
                f"itinerary_planning_node: MCP session 完全失败: {e}，"
                f"降级为欧氏距离聚类 + 空路线"
            )
            # Fallback：MCP session 整体不可用时，用欧氏距离做聚类，跳过 TSP
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

        return {
            "daily_itinerary": daily_itinerary,
            "daily_routes": daily_routes,
            "daily_budget_per_person": daily_land_budget,
            "isolated_pois": isolated_pois,
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
    budget_tier: Optional[str] = None,
    penalty_pois: Optional[set] = None,
    retry_count: int = 0,
) -> List[Dict]:
    """
    按旅行风格决定候选池大小，从候选列表中选出 pool_size 个 POI（送入 K-means 聚类）。

    候选池放大设计：
       final_count = pois_per_day * travel_days       # 最终入选目标（每天 N 个景点）
       pool_size   = ceil(final_count * _STYLE_POOL_FACTOR[style])
                     # 候选池大小，特种兵 1.5x、普通 1.2x、其余 1.0x
       本函数返回 pool_size 个 POI，K-means 分簇后再按 pois_per_day 砍尾，
       让"地理上不顺路的低分 POI"在聚类阶段被自然剔除。

    两阶段策略：
    ① 锚定（Phase-1）：
       将 seed_names（KB must_visit + LLM RAG 推荐合并去重）与 poi_candidates
       进行模糊匹配，匹配成功的直接进入 anchored 列表，不参与评分竞争。
    ② 剩余配额按有效评分填充至 pool_size（Phase-2）：
       - rating=0 时用 search_rank 换算基准分（排名越靠前分越高）
       - RAG 攻略关键词命中的 POI：+1.5 分（rag_boost）
       - 知识库顺路组合子景点命中的 POI：+0.8 分（combo_boost）
       - 历史涉案 POI（penalty_pois）：-0.6 * retry_count，retry 时令次优候选上位

    Args:
        rag_boosted_names:  RAG 原始片段中提取的景点关键词集合（用于 rag_boost）
        rag_preferred_pois: KB must_visit + LLM RAG 推荐合并的有序景点名列表
                            （用于 Phase-1 锚定）
        combo_spot_names:   知识库顺路组合的所有子景点名集合（用于 combo_boost）
        penalty_pois:       历史轮次违规涉案 POI 名集合（split_hints 两端展开），
                            用于在 retry 中降低优先级，迫使探索其他候选
        retry_count:        当前是第几轮重试（0 表示首轮，无惩罚）
    """
    import math

    rag_boosted_names = rag_boosted_names or set()
    rag_preferred_pois = rag_preferred_pois or []
    combo_spot_names = combo_spot_names or set()
    penalty_pois = penalty_pois or set()

    pois_per_day = _POIS_PER_DAY.get(travel_style, 3)
    final_count = pois_per_day * travel_days
    style_factor = _STYLE_POOL_FACTOR.get(travel_style, 1.0)
    # 候选池大小：放大后的目标数；最小不低于 final_count，最大不超过候选总数
    pool_size = max(final_count, math.ceil(final_count * style_factor))
    pool_size = min(pool_size, len(pois)) if pois else pool_size
    total_needed = pool_size  # 后续 anchor/fill 逻辑统一用 total_needed 命名，便于阅读
    logger.info(
        f"_select_pois: style={travel_style}, days={travel_days}, "
        f"pois_per_day={pois_per_day}, final_count={final_count}, "
        f"style_factor={style_factor}, pool_size={pool_size}"
    )

    # retry 评分扰动力度：每多一轮重试，对涉案 POI 多扣 0.6 分
    # 0.6 量级足以让排名相邻的次优候选反超（base 满分约 2.0，rag_boost=1.5）
    retry_penalty_per_round: float = 0.6

    def _effective_rating(poi: Dict) -> float:
        """
        计算 POI 的有效评分（仅用于 Phase-2 填充排序，已锚定的 must_visit 不经此函数）：
        1. 景点类 POI rating 通常为 0，用 search_rank 换算基准分（rank1→2.0，线性衰减）
        2. rag_boost +1.5：RAG 攻略原文中明确提到的景点
        3. combo_boost +0.8：知识库顺路组合子景点（如断桥、法喜寺、龙井村）
           比普通高德结果有更强的"值得去"背书，在剩余配额中获得优先权
        4. retry_penalty -0.6 * retry_count：历史涉案 POI 在 retry 中降权
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
        # 根据预算档位对高消费景点类型降权
        budget_penalty = 0.0
        if budget_tier == "经济":
            _expensive_keywords = {"主题公园", "游乐", "滑雪", "温泉", "演艺", "水上乐园", "高尔夫"}
            if any(kw in name for kw in _expensive_keywords):
                budget_penalty = -1.5
        elif budget_tier == "普通":
            _expensive_keywords = {"主题公园", "滑雪", "高尔夫"}
            if any(kw in name for kw in _expensive_keywords):
                budget_penalty = -0.8
        elif budget_tier == "舒适":
            # 舒适档位对极端高消费项目轻微降权
            _very_expensive_keywords = {"高尔夫", "私人游艇"}
            if any(kw in name for kw in _very_expensive_keywords):
                budget_penalty = -0.3
        # 豪华档位无降权
        # retry 历史涉案惩罚：仅当 retry_count > 0 且 POI 名命中 penalty_pois 时生效
        retry_penalty = 0.0
        if retry_count > 0 and name in penalty_pois:
            retry_penalty = -retry_penalty_per_round * retry_count
        return base + rag_boost + combo_boost + budget_penalty + retry_penalty

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

    # ─── Phase 2: 剩余配额按评分/排名填充至 pool_size ─────────────────────────
    remaining_needed = total_needed - len(anchored)
    remaining_pois = [p for p in pois if id(p) not in anchored_ids]

    fill: List[Dict] = []
    if remaining_needed > 0:
        # 直接按有效评分降序排序选取
        for candidate in sorted(remaining_pois, key=_effective_rating, reverse=True):
            if len(fill) >= remaining_needed:
                break
            fill.append(candidate)

    selected = anchored + fill
    logger.info(f"_select_pois: 最终选出 {len(selected)} 个POI: {[p['name'] for p in selected]}")
    return selected[:total_needed]


# =============================================================================
# 6b-aux — 缺额天补丁（聚类后填充少于 quota-1 的天）
# =============================================================================

def _fill_undersized_days(
    groups: List[Dict[str, Any]],
    pois_per_day: int,
    max_daily_hours: float,
    transit_matrix: Optional[List[List[float]]],
    poi_id_to_idx: Dict[int, int],
    dropped_pool: List[Dict],
    max_transit_relaxed: float = 120.0,
    budget_buffer_h: float = 1.0,
) -> None:
    """
    扫描所有天，对景点数 < `pois_per_day - 1` 的"缺额天"执行两步补充（in-place 修改 groups）：

      Step A — 从 dropped_pool（trim/未分配）按"到当天最近 POI 通勤时间最短"补 1 个
      Step B — 若 dropped_pool 无可用候选，从富余天（> pois_per_day）借通勤最近的 POI

    放宽阈值（仅本补丁阶段生效，主聚类逻辑仍走严格 90min/_MAX_DAILY_HOURS）：
      - 通勤上限：max_transit_relaxed=120min（原 90min，+33%）
      - 时间预算：max_daily_hours + budget_buffer_h=1.0h

    任何一步成功补入即继续 while 循环，直到达到 quota-1 或两步都补不上。
    """
    min_quota = max(1, pois_per_day - 1)

    def _candidate_min_transit(candidate_idx: int, group_indices: List[int]) -> float:
        """候选 POI 到当天已有 POI 的最小通勤时间（分钟）。无矩阵时返回 0（放行）。"""
        if transit_matrix is None or not group_indices:
            return 0.0
        return min(transit_matrix[candidate_idx][i] for i in group_indices)

    def _day_total_hours(group: Dict[str, Any]) -> float:
        plist = group["pois"]
        visit = sum(p.get("estimated_hours", 1.5) for p in plist)
        if transit_matrix is None or len(plist) < 2:
            transit = max(0, len(plist) - 1) * 0.5
        else:
            idx_list = [poi_id_to_idx[id(p)] for p in plist if id(p) in poi_id_to_idx]
            transit = sum(
                transit_matrix[idx_list[k]][idx_list[k + 1]] / 60.0
                for k in range(len(idx_list) - 1)
            )
        return visit + transit

    def _can_fit(candidate: Dict, transit_min: float, current_hours: float) -> bool:
        cand_visit = candidate.get("estimated_hours", 1.5)
        return current_hours + cand_visit + transit_min / 60.0 <= max_daily_hours + budget_buffer_h

    for group in groups:
        guard = 5  # 防御性：单天最多补 5 次循环
        while len(group["pois"]) < min_quota and guard > 0:
            guard -= 1
            day_label = group.get("day", "?")
            current_hours = _day_total_hours(group)
            group_indices = [
                poi_id_to_idx[id(p)] for p in group["pois"] if id(p) in poi_id_to_idx
            ]

            # ── Step A：从 dropped_pool 找通勤最短的可纳入候选 ──────────────
            best_a: Optional[Dict] = None
            best_a_transit: float = float("inf")
            for cand in dropped_pool:
                if id(cand) not in poi_id_to_idx:
                    continue
                cidx = poi_id_to_idx[id(cand)]
                t_min = _candidate_min_transit(cidx, group_indices)
                if t_min > max_transit_relaxed:
                    continue
                if not _can_fit(cand, t_min, current_hours):
                    continue
                if t_min < best_a_transit:
                    best_a_transit = t_min
                    best_a = cand

            if best_a is not None:
                group["pois"].append(best_a)
                dropped_pool.remove(best_a)
                logger.info(
                    f"[fill_undersized] Day{day_label} 补充「{best_a.get('name')}」"
                    f"（来源=dropped_pool, transit={best_a_transit:.0f}min）"
                )
                continue

            # ── Step B：从富余天（>pois_per_day）借通勤最短的 ───────────────
            best_b: Optional[Dict] = None
            best_b_transit: float = float("inf")
            best_b_src: Optional[Dict[str, Any]] = None
            for src_group in groups:
                if src_group is group or len(src_group["pois"]) <= pois_per_day:
                    continue
                for cand in src_group["pois"]:
                    if id(cand) not in poi_id_to_idx:
                        continue
                    cidx = poi_id_to_idx[id(cand)]
                    t_min = _candidate_min_transit(cidx, group_indices)
                    if t_min > max_transit_relaxed:
                        continue
                    if not _can_fit(cand, t_min, current_hours):
                        continue
                    if t_min < best_b_transit:
                        best_b_transit = t_min
                        best_b = cand
                        best_b_src = src_group

            if best_b is not None and best_b_src is not None:
                best_b_src["pois"].remove(best_b)
                group["pois"].append(best_b)
                logger.info(
                    f"[fill_undersized] Day{day_label} 从 Day{best_b_src.get('day')} "
                    f"借「{best_b.get('name')}」（transit={best_b_transit:.0f}min）"
                )
                continue

            # 两步都没补到：缺额天保持现状
            logger.info(
                f"[fill_undersized] Day{day_label} 仍少于 {min_quota} "
                f"（当前 {len(group['pois'])}），无可用候选"
            )
            break


# =============================================================================
# 6b-aux — 跨天 2-opt 交换（聚类后全局通勤优化）
# =============================================================================

def _cross_day_swap(
    groups: List[Dict[str, Any]],
    transit_matrix: List[List[float]],
    poi_id_to_idx: Dict[int, int],
    pois: List[Dict],
    partner_of: Dict[int, int],
    split_partners: Dict[int, set],
    max_daily_hours: float,
    max_iter: int = 20,
) -> None:
    """
    对聚类结果做跨天 POI 两两交换，降低所有天的路线总通勤时间之和。

    算法（类 2-opt）：
      逐轮枚举所有跨天 POI 对 (poi_a from day_i, poi_b from day_j)：
        1. 计算交换后两天的路线代价变化（贪心最近邻路径估计）
        2. 若代价降低，检查约束后接受，并重启枚举
      循环直至无改善或达到 max_iter 次

    约束检查（三项）：
      a. split_partners：交换后两天均不得出现被强制拆分的 POI 对
      b. partner_of（RAG joint hints）：不得将已同天的 joint pair 拆散
      c. 时间预算：两天交换后均满足 max_daily_hours

    Args:
        groups:          _cluster_by_geography 的分组结果，原地修改。
        transit_matrix:  N×N 通勤时间矩阵（分钟），为 None 时跳过。
        poi_id_to_idx:   id(poi_dict) -> 全局 pois 列表中的索引。
        pois:            全局 POI 列表（仅用于日志输出名称）。
        partner_of:      RAG joint hint 的一对一映射（双向）。
        split_partners:  split_hints 的不可同天映射（双向）。
        max_daily_hours: 每日时间预算上限（小时）。
        max_iter:        最大交换轮数。
    """
    if transit_matrix is None or len(groups) < 2:
        return

    def _get_indices(group: Dict) -> List[int]:
        return [poi_id_to_idx[id(p)] for p in group["pois"]]

    def _path_cost(idx_list: List[int]) -> float:
        """贪心最近邻路径代价（分钟），用于估算当天路线总通勤。"""
        if len(idx_list) <= 1:
            return 0.0
        n = len(idx_list)
        visited = [False] * n
        order = [0]
        visited[0] = True
        for _ in range(n - 1):
            cur = order[-1]
            nearest = min(
                (k for k in range(n) if not visited[k]),
                key=lambda k: transit_matrix[idx_list[cur]][idx_list[k]],
            )
            order.append(nearest)
            visited[nearest] = True
        return sum(
            transit_matrix[idx_list[order[k]]][idx_list[order[k + 1]]]
            for k in range(n - 1)
        )

    def _fits_budget(pois_list: List[Dict], idx_list: List[int]) -> bool:
        visit_h = sum(p.get("estimated_hours", 1.5) for p in pois_list)
        transit_h = _path_cost(idx_list) / 60.0
        return visit_h + transit_h <= max_daily_hours

    def _is_valid_swap(
        pois_i_new: List[Dict], idx_i_new: List[int],
        pois_j_new: List[Dict], idx_j_new: List[int],
    ) -> bool:
        """检查交换后两天的 split_partners、joint pair、时间预算约束。"""
        set_i = set(idx_i_new)
        set_j = set(idx_j_new)

        # split_partners：两天内均不能出现被强制拆分的 POI 对
        for idx in idx_i_new:
            if split_partners.get(idx, set()) & set_i - {idx}:
                return False
        for idx in idx_j_new:
            if split_partners.get(idx, set()) & set_j - {idx}:
                return False

        # partner_of（RAG joint hints）：不得把已同天的 joint pair 拆到不同天
        for idx in idx_i_new:
            partner = partner_of.get(idx)
            if partner is not None and partner not in set_i and partner not in set_j:
                pass  # partner 本来就在其他天，允许
            if partner is not None and partner in set_j:
                return False  # 原来同天的 joint pair 被拆到两边，禁止

        # 时间预算
        return _fits_budget(pois_i_new, idx_i_new) and _fits_budget(pois_j_new, idx_j_new)

    for iteration in range(max_iter):
        improved = False

        for gi in range(len(groups)):
            for gj in range(gi + 1, len(groups)):
                g_i = groups[gi]
                g_j = groups[gj]
                idx_i = _get_indices(g_i)
                idx_j = _get_indices(g_j)
                cost_before = _path_cost(idx_i) + _path_cost(idx_j)

                for poi_a in g_i["pois"]:
                    for poi_b in g_j["pois"]:
                        ia = poi_id_to_idx[id(poi_a)]
                        ib = poi_id_to_idx[id(poi_b)]

                        new_idx_i = [x for x in idx_i if x != ia] + [ib]
                        new_idx_j = [x for x in idx_j if x != ib] + [ia]
                        cost_after = _path_cost(new_idx_i) + _path_cost(new_idx_j)

                        if cost_after >= cost_before - 1e-6:
                            continue

                        new_pois_i = [p for p in g_i["pois"] if p is not poi_a] + [poi_b]
                        new_pois_j = [p for p in g_j["pois"] if p is not poi_b] + [poi_a]

                        if not _is_valid_swap(new_pois_i, new_idx_i, new_pois_j, new_idx_j):
                            continue

                        logger.info(
                            f"[cross_day_swap] iter={iteration + 1}: "
                            f"Day{g_i['day']}「{poi_a.get('name')}」"
                            f" <-> Day{g_j['day']}「{poi_b.get('name')}」, "
                            f"transit {cost_before:.0f} -> {cost_after:.0f} min"
                        )
                        g_i["pois"] = new_pois_i
                        g_j["pois"] = new_pois_j
                        improved = True
                        break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

        if not improved:
            logger.debug(f"[cross_day_swap] 第 {iteration + 1} 轮无改善，提前终止")
            break


# =============================================================================
# 6b — 地理聚类（K-means 主路径 + 贪心兜底）
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
    地理聚类调度器：优先使用 balanced K-means（基于经纬度初分 + transit_matrix 后修正），
    失败时降级到原有贪心算法（_greedy_cluster_by_geography）。

    K-means 主路径 (_kmeans_cluster_by_geography) 步骤：
      1. balanced K-means（k=travel_days，每簇 size 限制 [avg-1, avg+1]）
      2. split_pair 后修正：必须拆分的 POI 对若同簇 → swap 到其他簇
      3. joint_pair 后修正：必须同天的 POI 对若跨簇 → 合并到同簇
      4. 候选池砍尾：每簇按"距质心远 + rating 低"裁到 pois_per_day
      5. 时间预算修正：超 _MAX_DAILY_HOURS 的簇丢掉最离群 POI
      6. 跨天 2-opt（_cross_day_swap）保留
      7. best_period 内部排序保留

    Fallback：k-means-constrained 包未安装、K-means 求解异常时，自动落到贪心。
    """
    if not pois or travel_days <= 0:
        return []

    try:
        return _kmeans_cluster_by_geography(
            pois, travel_days, rag_joint_hints, travel_style,
            transit_matrix, split_hints,
        )
    except Exception as exc:
        logger.warning(
            f"_cluster_by_geography: K-means 路径异常 ({type(exc).__name__}: {exc})，"
            f"降级到贪心聚类"
        )
        return _greedy_cluster_by_geography(
            pois, travel_days, rag_joint_hints, travel_style,
            transit_matrix, split_hints,
        )


def _greedy_cluster_by_geography(
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
    6. 聚类完成后按 best_period 排序每天内部 POI（morning → flexible → evening）

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

    def _dist_to_group_centroid(candidate: int, day_idx_list: List[int]) -> float:
        """
        候选 POI 到当天已有景点组重心的距离。
        有矩阵时：取 candidate 到组内各 POI 通勤时间的平均值（用均值代理重心通勤）。
        无矩阵时：计算候选点到组经纬度均值的欧氏距离。
        排序时用此代替"距上一个加入的 POI 的距离"，使分组更稳定。
        """
        if not day_idx_list:
            return 0.0
        if transit_matrix is not None:
            return sum(transit_matrix[i][candidate] for i in day_idx_list) / len(day_idx_list)
        centroid_lng = sum(pois[i]["lng"] for i in day_idx_list) / len(day_idx_list)
        centroid_lat = sum(pois[i]["lat"] for i in day_idx_list) / len(day_idx_list)
        return _euclidean(
            (pois[candidate]["lng"], pois[candidate]["lat"]),
            (centroid_lng, centroid_lat),
        )

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

            # 按候选点到当天已有景点组重心的距离升序排列
            # 比"距上一个加入的 POI"更稳定：避免因局部蛇形延伸导致重心飘移
            candidates_sorted = sorted(
                unassigned,
                key=lambda i: _dist_to_group_centroid(i, day_indices),
            )

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

                # 通勤上限检查：候选到组内最近 POI 的通勤超阈值，则整组都太远，停止填充。
                # 使用 min（最近 POI）而非 max（最远 POI），避免组内离群点导致过早 break。
                # 候选已按组重心距离升序排列，若最近邻仍超阈值，后续只会更远 → break 合理。
                if transit_matrix is not None:
                    min_transit_in_group = min(
                        transit_matrix[i][candidate] for i in day_indices
                    )
                    if min_transit_in_group > _MAX_SAME_DAY_TRANSIT_MIN:
                        logger.debug(
                            f"_cluster_by_geography: 第{day_idx+1}天 "
                            f"「{pois[candidate]['name']}」到组内最近距离"
                            f"={min_transit_in_group:.0f}min > {_MAX_SAME_DAY_TRANSIT_MIN}min，停止"
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

    # 未被分配的 POI 进入 dropped_pool，作为缺额天补丁的候选源（不再直接丢弃）
    dropped_pool: List[Dict] = [pois[i] for i in unassigned]
    if unassigned:
        logger.info(
            f"_greedy_cluster: {len(unassigned)} 个POI主聚类阶段未安排，"
            f"进入 dropped_pool 待补: {[p['name'] for p in dropped_pool]}"
        )

    # 缺额天补丁：< pois_per_day - 1 的天从 dropped_pool / 富余天补足
    pois_per_day = _POIS_PER_DAY.get(travel_style, 3)
    _fill_undersized_days(
        groups=groups,
        pois_per_day=pois_per_day,
        max_daily_hours=max_daily_hours,
        transit_matrix=transit_matrix,
        poi_id_to_idx=poi_id_to_idx,
        dropped_pool=dropped_pool,
    )

    # 跨天 2-opt 交换：在 best_period 排序前优化，降低全局通勤总时间
    _cross_day_swap(
        groups=groups,
        transit_matrix=transit_matrix,
        poi_id_to_idx=poi_id_to_idx,
        pois=pois,
        partner_of=partner_of,
        split_partners=split_partners,
        max_daily_hours=max_daily_hours,
    )

    # 按 best_period 排序每天内部 POI（morning 优先，evening 靠后）
    for group in groups:
        group["pois"].sort(
            key=lambda p: _PERIOD_ORDER.get(p.get("best_period", "flexible"), 1)
        )

    return groups


# =============================================================================
# 6b-kmeans — Balanced K-means 聚类 + 后修正
# =============================================================================

def _kmeans_cluster_by_geography(
    pois: List[Dict],
    travel_days: int,
    rag_joint_hints: Optional[List[Tuple[str, str]]] = None,
    travel_style: str = "普通",
    transit_matrix: Optional[List[List[float]]] = None,
    split_hints: Optional[List[Tuple[str, str]]] = None,
) -> List[Dict[str, Any]]:
    """
    Balanced K-means 主路径：经纬度初分 → 三步后修正 → cross_day_swap → period 排序。

    后修正顺序敏感（必须按下序执行）：
      1. split_pair：必须拆分的 POI 对若同簇 → 移到其它簇
      2. joint_pair：必须同天的 POI 对若跨簇 → 合并到同簇
      3. trim：候选池大于 final_count 时按"距质心远 + rating 低"砍尾每簇到 pois_per_day
      4. time_budget：超 _MAX_DAILY_HOURS 的簇剔除最离群 POI

    任何步骤失败都会向上抛异常，由 _cluster_by_geography dispatcher 捕获并降级到贪心。
    """
    from k_means_constrained import KMeansConstrained  # type: ignore
    import numpy as np

    rag_joint_hints = rag_joint_hints or []
    split_hints = split_hints or []
    n = len(pois)

    pois_per_day = _POIS_PER_DAY.get(travel_style, 3)
    max_daily_hours = _MAX_DAILY_HOURS.get(travel_style, 8.0)

    # 平均每簇大小（基于 pool_size，可能 > pois_per_day）
    avg_per_cluster = max(1, n // travel_days)
    remainder = n % travel_days
    # K-means 阶段每簇允许 ±1 浮动；avg-1 不能小于 1
    size_min = max(1, avg_per_cluster)
    size_max = avg_per_cluster + (1 if remainder > 0 else 0)
    # 极端情况下放宽：当 n 很小或 travel_days=1 时 size_min/max 退化
    if size_max < size_min:
        size_max = size_min
    # 单天行程 / n < travel_days 直接全部归一簇兜底
    if travel_days <= 1 or n <= travel_days:
        # 单天：所有 POI 归 Day1；POI 数 ≤ 天数：每天 1 个，剩余空着
        if travel_days <= 1:
            groups = [{"day": 1, "pois": list(pois)}]
        else:
            groups = [
                {"day": d + 1, "pois": [pois[d]] if d < n else []}
                for d in range(travel_days)
            ]
        return _kmeans_finalize(
            groups, pois, travel_style, transit_matrix,
            rag_joint_hints, split_hints, max_daily_hours,
            dropped_pool=[],
        )

    # ── Step 1: balanced K-means on (lng, lat) ─────────────────────────────
    coords = np.array([[p["lng"], p["lat"]] for p in pois], dtype=float)
    clf = KMeansConstrained(
        n_clusters=travel_days,
        size_min=size_min,
        size_max=max(size_max, size_min),
        random_state=42,
        n_init=10,
    )
    labels = clf.fit_predict(coords)
    logger.info(
        f"_kmeans_cluster: n={n}, k={travel_days}, "
        f"size=[{size_min},{size_max}], labels distribution="
        f"{[int((labels == i).sum()) for i in range(travel_days)]}"
    )

    # 构造初始 groups
    groups: List[Dict[str, Any]] = [
        {"day": d + 1, "pois": [pois[i] for i in range(n) if labels[i] == d]}
        for d in range(travel_days)
    ]

    # ── 索引辅助 ────────────────────────────────────────────────────────────
    poi_id_to_idx: Dict[int, int] = {id(p): i for i, p in enumerate(pois)}

    def _name_to_idx(name_a: str, name_b: str) -> Tuple[Optional[int], Optional[int]]:
        ia = next((i for i, p in enumerate(pois) if name_a in p.get("name", "")), None)
        ib = next((i for i, p in enumerate(pois) if name_b in p.get("name", "")), None)
        return ia, ib

    def _which_group(idx: int) -> Optional[int]:
        for gi, g in enumerate(groups):
            if any(poi_id_to_idx[id(p)] == idx for p in g["pois"]):
                return gi
        return None

    def _move_poi(idx: int, src_gi: int, dst_gi: int) -> bool:
        """把 pois[idx] 从 src 簇移到 dst 簇。返回是否成功。"""
        src = groups[src_gi]["pois"]
        for k, p in enumerate(src):
            if poi_id_to_idx[id(p)] == idx:
                moved = src.pop(k)
                groups[dst_gi]["pois"].append(moved)
                return True
        return False

    def _cluster_centroid(gi: int) -> Tuple[float, float]:
        plist = groups[gi]["pois"]
        if not plist:
            return (0.0, 0.0)
        return (
            sum(p["lng"] for p in plist) / len(plist),
            sum(p["lat"] for p in plist) / len(plist),
        )

    # ── Step 2: split_pair 后修正（必须拆分但落同簇 → 移到其它簇）────────────
    for hint_a, hint_b in split_hints:
        ia, ib = _name_to_idx(hint_a, hint_b)
        if ia is None or ib is None or ia == ib:
            continue
        ga, gb = _which_group(ia), _which_group(ib)
        if ga is None or gb is None or ga != gb:
            continue
        # 同簇违规：把 b 移到与 b 经纬度最近的"非 a 所在簇"
        b_pt = (pois[ib]["lng"], pois[ib]["lat"])
        candidates = [
            (gi, _euclidean(b_pt, _cluster_centroid(gi)))
            for gi in range(travel_days) if gi != ga
        ]
        if not candidates:
            continue
        target_gi = min(candidates, key=lambda x: x[1])[0]
        if _move_poi(ib, ga, target_gi):
            logger.info(
                f"[kmeans split_fix] {pois[ib]['name']} 从 Day{ga+1} 移到 Day{target_gi+1} "
                f"（与 {pois[ia]['name']} 强制拆分）"
            )

    # ── Step 3: joint_pair 后修正（必须同天但跨簇 → 合并到同簇）──────────────
    for hint_a, hint_b in rag_joint_hints:
        ia, ib = _name_to_idx(hint_a, hint_b)
        if ia is None or ib is None or ia == ib:
            continue
        ga, gb = _which_group(ia), _which_group(ib)
        if ga is None or gb is None or ga == gb:
            continue
        # 跨簇违规：把 b 移到 a 所在簇（接受小幅 size 失衡，trim 阶段会重新缩到 quota）
        if _move_poi(ib, gb, ga):
            logger.info(
                f"[kmeans joint_fix] {pois[ib]['name']} 从 Day{gb+1} 合并到 Day{ga+1} "
                f"（与 {pois[ia]['name']} 同天约束）"
            )

    # ── Step 4: 候选池砍尾，每簇按"距质心远 + rating 低"裁到 pois_per_day ────
    # 候选池 > final_count 时（特种兵 1.5x、普通 1.2x），每簇约 pois_per_day + 1~2 个，
    # 这里裁掉地理离群且评分低的，让每天保持地理紧凑 + 高质量。
    def _trim_score(poi: Dict, centroid: Tuple[float, float]) -> float:
        """越大越优先保留：rating 高 + 距质心近。"""
        rating = poi.get("rating", 0.0) or 0.0
        if rating == 0.0:
            rank = poi.get("search_rank", 20)
            rating = max(0.0, (21 - rank) / 21 * 2.0)
        dist = _euclidean((poi["lng"], poi["lat"]), centroid)
        # rating 满分约 2.0，距离量级 0.01-0.1（经纬度差），距离权重放大
        return rating - dist * 50.0

    # trim 阶段被裁的 POI 收集到 dropped_pool，作为 _fill_undersized_days 的补充源
    dropped_pool: List[Dict] = []
    for gi, group in enumerate(groups):
        if len(group["pois"]) <= pois_per_day:
            continue
        cen = _cluster_centroid(gi)
        kept_sorted = sorted(group["pois"], key=lambda p: _trim_score(p, cen), reverse=True)
        dropped = kept_sorted[pois_per_day:]
        group["pois"] = kept_sorted[:pois_per_day]
        dropped_pool.extend(dropped)
        logger.info(
            f"[kmeans trim] Day{gi+1} 裁剪 {len(dropped)} 个 POI: "
            f"{[p['name'] for p in dropped]}"
        )

    return _kmeans_finalize(
        groups, pois, travel_style, transit_matrix,
        rag_joint_hints, split_hints, max_daily_hours,
        dropped_pool=dropped_pool,
    )


def _kmeans_finalize(
    groups: List[Dict[str, Any]],
    pois: List[Dict],
    travel_style: str,
    transit_matrix: Optional[List[List[float]]],
    rag_joint_hints: List[Tuple[str, str]],
    split_hints: List[Tuple[str, str]],
    max_daily_hours: float,
    dropped_pool: Optional[List[Dict]] = None,
) -> List[Dict[str, Any]]:
    """
    K-means 聚类的收尾流水线（与 _kmeans_cluster_by_geography 共用）：
      a. 时间预算修正：超时簇剔除最离群 POI
      b. 缺额天补丁（_fill_undersized_days）：从 dropped_pool / 富余天补足
      c. cross_day_swap 跨天 2-opt
      d. best_period 内部排序

    Args:
        dropped_pool: trim 阶段被裁的 POI，作为 _fill_undersized_days Step A 候选源
    """
    poi_id_to_idx: Dict[int, int] = {id(p): i for i, p in enumerate(pois)}
    pois_per_day = _POIS_PER_DAY.get(travel_style, 3)
    dropped_pool = dropped_pool if dropped_pool is not None else []

    # ── Step 5: 时间预算修正（超时则剔除最离群 POI 直至满足）──────────────
    def _day_total_hours(group: Dict[str, Any]) -> float:
        """当天游览时长 + 通勤时长（小时）。"""
        plist = group["pois"]
        visit = sum(p.get("estimated_hours", 1.5) for p in plist)
        if transit_matrix is None or len(plist) < 2:
            transit = max(0, len(plist) - 1) * 0.5
        else:
            idx_list = [poi_id_to_idx[id(p)] for p in plist]
            transit = sum(
                transit_matrix[idx_list[k]][idx_list[k + 1]] / 60.0
                for k in range(len(idx_list) - 1)
            )
        return visit + transit

    for gi, group in enumerate(groups):
        max_iter = 5
        while max_iter > 0 and _day_total_hours(group) > max_daily_hours and len(group["pois"]) > 1:
            cen_lng = sum(p["lng"] for p in group["pois"]) / len(group["pois"])
            cen_lat = sum(p["lat"] for p in group["pois"]) / len(group["pois"])
            outlier = max(
                group["pois"],
                key=lambda p: _euclidean((p["lng"], p["lat"]), (cen_lng, cen_lat)),
            )
            group["pois"] = [p for p in group["pois"] if p is not outlier]
            dropped_pool.append(outlier)  # budget 阶段剔除的也回流到 dropped_pool
            logger.info(
                f"[kmeans budget_fix] Day{group['day']} 超时，丢弃最离群 POI "
                f"「{outlier.get('name')}」"
            )
            max_iter -= 1

    # ── Step 5.5: 缺额天补丁（< pois_per_day - 1 的天从 dropped_pool / 富余天补足）──
    _fill_undersized_days(
        groups=groups,
        pois_per_day=pois_per_day,
        max_daily_hours=max_daily_hours,
        transit_matrix=transit_matrix,
        poi_id_to_idx=poi_id_to_idx,
        dropped_pool=dropped_pool,
    )

    # ── Step 6: cross_day_swap 跨天 2-opt（保留原有逻辑）────────────────────
    if transit_matrix is not None:
        # partner_of 映射：joint_pair 名称 → 索引对
        partner_of: Dict[int, int] = {}
        for ha, hb in rag_joint_hints:
            ia = next((i for i, p in enumerate(pois) if ha in p.get("name", "")), None)
            ib = next((i for i, p in enumerate(pois) if hb in p.get("name", "")), None)
            if ia is not None and ib is not None and ia != ib:
                partner_of[ia] = ib
                partner_of[ib] = ia

        split_partners: Dict[int, set] = {}
        for ha, hb in split_hints:
            ia = next((i for i, p in enumerate(pois) if ha in p.get("name", "")), None)
            ib = next((i for i, p in enumerate(pois) if hb in p.get("name", "")), None)
            if ia is not None and ib is not None and ia != ib:
                split_partners.setdefault(ia, set()).add(ib)
                split_partners.setdefault(ib, set()).add(ia)

        _cross_day_swap(
            groups=groups,
            transit_matrix=transit_matrix,
            poi_id_to_idx=poi_id_to_idx,
            pois=pois,
            partner_of=partner_of,
            split_partners=split_partners,
            max_daily_hours=max_daily_hours,
        )

    # ── Step 7: best_period 内部排序 ───────────────────────────────────────
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
    2. 按 best_period 分桶（morning → flexible → evening），
       在每个桶内独立做 TSP：
         - 桶内 n <= 4：暴力枚举
         - 桶内 n  > 4：最近邻贪心 + 2-opt 改善
       桶间按 _PERIOD_ORDER 顺序拼接，天然满足时段约束，
       避免 P4.5 time_slot_mismatch 违规触发回环重排。
    3. 按最终顺序调用 get_transit_route 获取相邻段路线

    Fallback：MCP 调用失败时降级为欧氏距离 TSP（同样按时段分桶）。
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

    # --- 2. 按 best_period 分桶，每桶内独立 TSP ---
    period_buckets: Dict[int, List[int]] = {}
    for idx, poi in enumerate(day_pois):
        order_key = _PERIOD_ORDER.get(poi.get("best_period", "flexible"), 1)
        period_buckets.setdefault(order_key, []).append(idx)

    final_order: List[int] = []
    for order_key in sorted(period_buckets.keys()):
        indices = period_buckets[order_key]
        bucket_size = len(indices)
        if bucket_size <= 1:
            final_order.extend(indices)
            continue

        if matrix is not None:
            sub_matrix = [
                [matrix[i][j] for j in indices]
                for i in indices
            ]
            if bucket_size <= 4:
                local_order = _tsp_brute_force_matrix(sub_matrix, bucket_size)
            else:
                nn_local = _tsp_nearest_neighbor_matrix(sub_matrix, bucket_size)
                local_order = _tsp_2opt_improve(nn_local, sub_matrix)
        else:
            sub_points = [(day_pois[i]["lng"], day_pois[i]["lat"]) for i in indices]
            local_order = (
                _tsp_brute_force_euclidean(sub_points)
                if bucket_size <= 4
                else _tsp_nearest_neighbor_euclidean(sub_points)
            )

        final_order.extend(indices[k] for k in local_order)

    ordered_pois = [day_pois[i] for i in final_order]
    if len(period_buckets) > 1:
        logger.info(
            "_optimize_daily_route: per-period TSP "
            f"{[(p['name'], p.get('best_period', 'flexible')) for p in ordered_pois]}"
        )

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


def _tsp_2opt_improve(
    order: List[int],
    matrix: List[List[float]],
    max_iter: int = 50,
) -> List[int]:
    """
    在最近邻贪心结果上做 2-opt 局部搜索，减少路径总耗时。

    算法：遍历所有 (i, j) 段反转组合，若反转后路径更短则接受，
    循环直至无改善或达到 max_iter 次。

    仅供 n > 4 时调用（n <= 4 已用暴力枚举，无需改善）。

    Args:
        order:    最近邻贪心生成的访问顺序（索引列表）。
        matrix:   N×N 通勤时间矩阵（分钟）。
        max_iter: 最大迭代轮数，防止大 n 时运行过长。

    Returns:
        改善后的访问顺序（索引列表）。
    """
    best = list(order)
    n = len(best)

    for _ in range(max_iter):
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                # 当前路径：best[i] -> best[i+1] ... best[j] -> best[j+1]
                # 反转后：  best[i] -> best[j]   ... best[i+1] -> best[j+1]
                a, b = best[i], best[i + 1]
                c, d = best[j], best[(j + 1) % n]
                # 开放路径（非环）：只考虑 i+1 到 j 段的两端连接
                gain = (matrix[a][b] + matrix[c][d]) - (matrix[a][c] + matrix[b][d])
                if gain > 1e-6:  # 有改善（用小 epsilon 避免浮点误差）
                    best[i + 1 : j + 1] = best[i + 1 : j + 1][::-1]
                    improved = True
        if not improved:
            break

    return best


# =============================================================================
# 孤岛 POI 检测辅助函数
# =============================================================================

def _find_isolated_pois(
    pois: List[Dict],
    transit_matrix: Optional[List[List[float]]],
    threshold_min: float = 60.0,
) -> List[int]:
    """
    识别与所有其他 POI 的最小通勤时间均超过阈值的"孤岛 POI"，返回索引列表。
    无 transit_matrix 时返回空列表（欧氏距离不可靠，不做判断）。

    返回索引而非名称，便于调用方同时切 selected_pois 与 transit_matrix；
    名称由调用方按索引派生。
    """
    if transit_matrix is None or len(pois) < 2:
        return []
    isolated: List[int] = []
    for i in range(len(pois)):
        min_transit = min(
            transit_matrix[i][j]
            for j in range(len(pois)) if j != i
        )
        if min_transit > threshold_min:
            isolated.append(i)
    return isolated


# =============================================================================
# P4.5 违规解析辅助函数
# =============================================================================

def _parse_violation_hints(
    violations: List,
) -> Tuple[set, List[Tuple[str, str]], set]:
    """
    将 P4.5 itinerary_review_node 输出的 RuleViolation 列表解析为 P3 可消费的三类 hints。

    对应 itinerary_review_node 中各 violation_type 的 suggestion 文案格式：
      - daily_time_overload   : "建议将「{outlier}」移至其他天..."
          → remove_hints 加入日内孤岛 outlier（本轮跳过该 POI）
      - long_transit_leg      : "建议将「A」和「B」拆分到不同天..."
          → split_hints 加入 (A, B)
      - time_slot_mismatch    : "建议将「{name}」调整为当天第1/第2个景点"
                                 或 "建议将「{name}」调整为当天最后一个景点"
          → reorder_hints 加入 name（TSP 后按 best_period 稳定重排）
      - isolated_poi_in_day   : "建议从行程中移除「{name}」"
          → remove_hints 加入 name（孤岛景点强制踢出，不再回到候选池）
      - remove_poi (通用)     : "建议...「{name}」..." → remove_hints 加入 name

    兼容 RuleViolation (pydantic) 和 dict 两种形态（checkpointer 可能序列化）。

    Returns:
        (remove_hints, split_hints, reorder_hints)
    """
    import re

    remove_hints: set = set()
    split_hints: List[Tuple[str, str]] = []
    reorder_hints: set = set()

    for v in violations:
        if hasattr(v, "model_dump"):
            v_dict = v.model_dump()
        elif isinstance(v, dict):
            v_dict = v
        else:
            continue

        vtype: str = v_dict.get("violation_type", "") or ""
        sugg: str = v_dict.get("suggestion", "") or ""
        names: List[str] = re.findall(r'「([^」]+)」', sugg)

        if vtype == "long_transit_leg":
            if len(names) >= 2:
                split_hints.append((names[0], names[1]))

        elif vtype == "daily_time_overload":
            if names:
                remove_hints.add(names[0])

        elif vtype == "time_slot_mismatch":
            if names:
                reorder_hints.add(names[0])

        elif vtype in ("isolated_poi_in_day", "remove_poi"):
            if names:
                remove_hints.add(names[0])

    return remove_hints, split_hints, reorder_hints


# =============================================================================
# LLM 结构化抽取 RAG 推荐 POI（替代/增强 jieba 提取）
# =============================================================================

async def _llm_extract_rag_recommendations(
    rag_snippets: List[Dict],
    city: str,
    travel_style: str,
    travel_days: int,
    llm,
    kb_must_visit: Optional[List[str]] = None,
) -> List[str]:
    """
    基于 RAG 攻略原始片段，让 LLM 抽取该城市核心景点推荐序列（按重要性排序）。

    用于 _select_pois 的 Phase-1 锚定种子（与知识库 must_visit 合并去重）。

    Args:
        rag_snippets:  rag_context.rag_snippets，每项 {"content": str, ...}
        city:          目的地城市
        travel_style:  旅行风格，用于差异化推荐
        travel_days:   旅游天数，用于计算目标景点数
        llm:           LangChain ChatOpenAI 实例
        kb_must_visit: 知识库已确认的城市核心景点列表，作为"已锁定的核心"
                       注入 prompt，避免 LLM 重复推荐同一批，引导其补充其他
                       同等级 5A/地标景点（如北京已含故宫/天安门，LLM 应补
                       国博/天坛/北海等而非动物园系列）。

    Returns:
        按重要性排序的景点名称列表，可能为空
    """
    if not rag_snippets or llm is None:
        return []

    # 拼接 snippet content（截断到约 4000 字防止 context 爆炸）
    parts: List[str] = []
    total_len = 0
    for s in rag_snippets:
        if not isinstance(s, dict):
            continue
        content = s.get("content", "") or ""
        if not content:
            continue
        parts.append(content)
        total_len += len(content)
        if total_len >= 4000:
            break
    rag_text = "\n---\n".join(parts)[:4000]

    if not rag_text.strip():
        return []

    # 目标候选数：约 2 倍最终需求，给 K-means 留筛选空间
    pois_per_day = _POIS_PER_DAY.get(travel_style, 3)
    target_count = max(pois_per_day * max(travel_days, 1) * 2, 6)

    style_hint_map = {
        "老人": (
            "节奏舒缓、单点停留时长充足；偏好文化古迹、园林、博物馆、"
            "宗教场所、温泉康养类；避免登山涉水/长时间徒步/极限项目"
        ),
        "亲子": (
            "兼顾互动性与教育性；偏好动物园、海洋馆、科技馆、儿童乐园、"
            "亲子农场、研学基地等；避免严肃纯文物类博物馆和高强度徒步"
        ),
        "情侣": (
            "风景优美、氛围浪漫；偏好夜景地标、海边湖畔、文艺街区、"
            "登高望远点、热门拍照打卡地标"
        ),
        "普通": (
            "综合知名度与体验；以城市核心 5A/4A 名片景点为主体，"
            "搭配少量有特色的小众或文化体验"
        ),
        "特种兵": (
            "高密度、高曝光，主打必看名片；偏好 5A 核心景点、"
            "城市地标、网红打卡点；接受紧凑时间安排"
        ),
    }
    style_hint = style_hint_map.get(travel_style, "综合知名度与体验，热门和小众均衡")

    # 已锁定核心：注入 prompt 让 LLM 明确知道哪些已无需重复推荐
    kb_locked = kb_must_visit or []
    kb_locked_str = "、".join(kb_locked) if kb_locked else "（无）"

    # 风格景点 vs 通用地标的混合配比（防止 LLM 被风格 hint 锁死）
    style_quota = max(int(target_count * 0.6), 1)
    landmark_quota = target_count - style_quota

    prompt = (
        f"你是一位资深的{city}本地旅游顾问。请为「{travel_style}」"
        f"旅行者推荐{city}的核心景点名单。\n\n"
        f"旅行天数：{travel_days}\n"
        f"风格说明：{style_hint}\n\n"
        f"== 已锁定的城市核心景点（已纳入行程，请勿重复推荐） ==\n"
        f"{kb_locked_str}\n\n"
        f"== 攻略参考原文 ==\n{rag_text or '（攻略未命中，请完全基于专家知识推荐）'}\n\n"
        f"== 推荐策略（两条并行通道，必须同时完成） ==\n"
        f"通道 A：风格契合景点（约 {style_quota} 个）\n"
        f"  - 优先从攻略原文中识别与「{travel_style}」风格强匹配的景点\n"
        f"  - 攻略不足时，基于专家知识补充该风格下{city}最知名的同类景点\n"
        f"通道 B：城市通用名片地标（约 {landmark_quota} 个）\n"
        f"  - 无论风格如何，都必须补充{city}最知名的 5A/4A 级地标或代表性"
        f"历史/自然名片，让非该风格爱好者也能识别这是「{city}必游」\n"
        f"  - 不允许仅因风格 hint 而省略此类通用地标\n\n"
        f"== 硬性约束 ==\n"
        f"  1. 不得输出已锁定列表中的景点，输出名单必须与之**完全不重叠**\n"
        f"  2. 仅输出真正的景点（不含餐厅、酒店、地铁站、景区内部子点）\n"
        f"  3. 景点名称使用通用标准写法（如\"灵隐寺\"而非\"灵隐\"）\n"
        f"  4. 必须是国家 5A/4A 级景区或该城市公认标志性地标，禁止编造\n"
        f"  5. 总数约 {target_count} 个，按游览价值与风格契合度综合排序输出\n\n"
        f'输出格式（严格 JSON，无任何额外文字）：'
        f'{{"recommended_pois": ["景点A", "景点B", ...]}}'
    )

    try:
        response = await retry_with_backoff(
            lambda: llm.ainvoke(prompt),
            max_retries=2,
        )
        raw_content = response.content if hasattr(response, "content") else str(response)
        # 容错：去除可能的 markdown 代码块包裹
        cleaned = raw_content.strip()
        if cleaned.startswith("```"):
            # 去掉 ```json ... ``` 包裹
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
        data = json.loads(cleaned)
        result = data.get("recommended_pois", []) or []
        # 防御性后过滤：即使在 prompt 中要求过，LLM 仍可能重复 kb_must_visit；
        # 在此剔除，避免 _select_pois 合并阶段被无效条目占据顺位
        kb_locked_set: set = set(kb_must_visit or [])
        # 去重保序、过滤空白、剔除 KB 重复
        seen: set = set()
        deduped: List[str] = []
        skipped_kb: List[str] = []
        for name in result:
            if not isinstance(name, str):
                continue
            name = name.strip()
            if not name or name in seen:
                continue
            if name in kb_locked_set:
                skipped_kb.append(name)
                continue
            deduped.append(name)
            seen.add(name)
        logger.info(
            f"_llm_extract_rag_recommendations: city={city}, style={travel_style}, "
            f"抽取到 {len(deduped)} 个景点: {deduped[:8]}{'...' if len(deduped) > 8 else ''}"
            + (f"; KB重复已剔除 {skipped_kb}" if skipped_kb else "")
        )
        return deduped
    except Exception as exc:
        logger.warning(
            f"_llm_extract_rag_recommendations: LLM 调用/解析失败: {exc}，返回空列表"
        )
        return []


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
        '美食', '民宿', '酒店', '商场', '售票处'
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
