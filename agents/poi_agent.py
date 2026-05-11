"""
POI 搜索智能体 POIFetchAgent
职责：根据目的地城市，搜索景区候选 POI，输出标准化列表，供后续 TSP 路线规划使用。

搜索策略（分 KB 城市 / 非 KB 城市两条路径）：

【KB 城市路径（城市在 CityKnowledgeDB 中）】
  高德只做"地理数据提供者"——由知识库决定推荐哪些景点，高德只负责返回坐标/评分。
  ① must_visit 精准查询（top_n=2）：保证所有必去景点进入候选池，供 Phase-1 锚定
  ② route_combo 额外子景点精准查询（top_n=2）：断桥/法喜寺/龙井村等细粒度坐标，
     支持 TSP 精确路由和 Phase-2 combo_boost 评分
  ②b llm_seed_pois 精准查询（top_n=2, trust_kb=True）：由 llm_seed_extract_node
     基于 RAG + KB 抽取的城市共识地标（如"天坛公园"/"北海公园"），
     剔除已被 ①② 覆盖的条目后逐一精搜
  ③ LLM attraction_hints 补充（top_n=3）：捕捉用户个性化兴趣（如"特别想去大熊猫基地"）
  ✗ 不做 "{city}景点" 泛搜：避免引入大量低质量噪声 POI

【非 KB 城市路径（城市不在 CityKnowledgeDB 中）】
  ②b llm_seed_pois 精准查询（top_n=2, trust_kb=True），作为种子主源
  ③ LLM attraction_hints 精准查询（top_n=5）
  ④ 兜底 "{city}景点" 泛搜（top_n=10），仅当 seed + hints 均为空时触发

全局去重：按 POI 名称去重，优先保留先搜到的条目（KB ①② > seed ②b > hints > 泛搜）。
非景点 POI（酒店/餐厅/附属设施等）由 _normalize_pois 通过高德 typecode 黑名单硬过滤。
"""
from __future__ import annotations

import logging
import math
import re
from typing import Any, Dict, List, Optional

from mcp.client.session import ClientSession
from mcp_clients.amap_client import amap_mcp_session, search_pois
from utils.knowledge_parser import CityKnowledgeDB
from utils.poi_category import is_attraction_typecode

logger = logging.getLogger(__name__)

# 非KB城市泛搜的默认 top_n（结果质量有限，不宜过多）
_FALLBACK_TOP_N = 10
# 特种兵模式下各路径 top_n 乘数（搜更多候选以满足高密度行程需求）
_SPECIAL_FORCES_MULTIPLIER = 2

# 景点泛搜兜底模板（非KB城市且无 LLM hints 时使用）
_FALLBACK_KEYWORD_TMPL = "{city}景点"

# 前缀+地理距离去重阈值
_GEO_DEDUP_THRESHOLD_M = 200.0
# 短名最少字符数：防止 "中" / "大" 这类单字前缀引发误合并
_MIN_PREFIX_LEN = 2
# 地球半径（米），haversine 公式用
_EARTH_RADIUS_M = 6_371_000.0


# =============================================================================
# 模块级辅助函数
# =============================================================================

def _parse_location(location: str) -> Optional[tuple[float, float]]:
    """
    将高德 "lng,lat" 字符串解析为 (lng, lat) 浮点元组。
    格式不合法或坐标为零时返回 None（过滤掉无效 POI）。
    """
    if not location:
        return None
    parts = location.split(",")
    if len(parts) != 2:
        return None
    try:
        lng, lat = float(parts[0].strip()), float(parts[1].strip())
    except ValueError:
        return None
    if lng == 0.0 and lat == 0.0:
        return None
    return lng, lat


def _normalize_rating(raw_rating: Any) -> float:
    """将各种评分格式统一转为 float，无法解析时返回 0.0。"""
    if isinstance(raw_rating, (int, float)):
        return float(raw_rating)
    try:
        return float(str(raw_rating).strip())
    except (ValueError, TypeError):
        return 0.0


def _normalize_pois(raw_pois: List[Dict], category: str, top_n: int, trust_kb: bool = False) -> List[Dict]:
    """
    将 search_pois() 的原始结果转换为标准 POI 格式，同时：
    - 过滤掉没有有效坐标的条目
    - 通过高德 typecode 黑名单硬过滤非景点 POI（酒店/餐厅/交通设施等）
    - 截取前 top_n 条

    字段说明：
      amap_type : 高德 typecode（6 位字符串，如 "110104"）。
                  amap_client.search_pois 已将原始 typecode 写入 item["type"]，
                  此处直接透传，供 itinerary_review_node Check 4 大类判断使用。
                  place/text 接口返回的均为真实 POI，typecode 字段必然存在。
    """
    result: List[Dict] = []
    dropped_by_typecode: List[tuple[str, str]] = []  # (name, typecode) for logging
    for item in raw_pois:
        coords = _parse_location(item.get("location", ""))
        if coords is None:
            continue
        typecode = item.get("type", "") or ""
        if not is_attraction_typecode(typecode, trust_kb=trust_kb):
            dropped_by_typecode.append((item.get("name", ""), typecode))
            continue
        lng, lat = coords
        result.append({
            "name": item.get("name", ""),
            "lng": lng,
            "lat": lat,
            "category": category,
            "rating": _normalize_rating(item.get("rating", 0.0)),
            "address": item.get("address", ""),
            # 记录在本次搜索结果中的排名（1-based），高德按相关性/热度排序，
            # 越靠前的 POI 通常越知名，供 rating=0 时作为评分代理指标
            "search_rank": len(result) + 1,
            # 高德 typecode：amap_client.search_pois 已将 typecode 写入 item["type"]
            "amap_type": typecode,
        })
        if len(result) >= top_n:
            break
    if dropped_by_typecode:
        # 仅展示前 5 条样本，避免单次日志过长
        sample = dropped_by_typecode[:5]
        suffix = f" ...等 {len(dropped_by_typecode)} 条" if len(dropped_by_typecode) > 5 else ""
        logger.info(
            f"_normalize_pois: typecode 黑名单过滤 {len(dropped_by_typecode)} 条非景点: "
            f"{sample}{suffix}"
        )
    return result


# 子 POI 名称归一化正则：
#   1. "天安门广场-国旗" / "故宫（西门）" → 去掉 "-..." 或 "（...）" 后缀
#   2. "故宫博物院检票处" / "颐和园游客中心" → 去掉附属设施后缀词
_SUBPOI_SUFFIX_PATTERN = re.compile(
    r'[-（(].*$|'
    r'(检票[处口]|售票[处口]|入口|出口|停车场|游客中心|游客服务中心|'
    r'东门|西门|南门|北门|正门)$'
)


def _canon_name(name: str) -> str:
    """对 POI 名称做归一化，去除附属设施/子点后缀。空值或纯噪声返回 ""。"""
    if not name:
        return ""
    return _SUBPOI_SUFFIX_PATTERN.sub('', name).strip()


def _pick_best_poi(group: List[Dict]) -> Dict:
    """
    从同一聚类中按优先级选最优 POI（同时供子POI合并和前缀+地理去重使用）：
      1. typecode 落在风景名胜大类（11xxxx）的优先（最像主景点）
      2. 名称越短越优先（更接近规范名）
      3. search_rank 越小越优先（高德相关性排序靠前）
    """
    return min(
        group,
        key=lambda p: (
            0 if p.get("amap_type", "").startswith("11") else 1,
            len(p.get("name", "")),
            p.get("search_rank", 9999),
        ),
    )


def _haversine_meters(lng1: float, lat1: float, lng2: float, lat2: float) -> float:
    """两点经纬度距离（米），haversine 公式。"""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * _EARTH_RADIUS_M * math.asin(math.sqrt(a))


def _is_prefix_pair(name_a: str, name_b: str) -> bool:
    """
    判断两名称是否构成有效前缀关系：
      - 短名严格短于长名
      - 短名长度 >= _MIN_PREFIX_LEN（防止 "中"/"大" 这类单字噪声）
      - 长名以短名开头
    """
    if not name_a or not name_b or name_a == name_b:
        return False
    short, long_ = (name_a, name_b) if len(name_a) < len(name_b) else (name_b, name_a)
    if len(short) < _MIN_PREFIX_LEN:
        return False
    return long_.startswith(short)


def _geo_prefix_dedup(pois: List[Dict]) -> List[Dict]:
    """
    前缀+地理距离去重：
      若 A.name 是 B.name 的前缀（A 严格短于 B），且二者距离 < 200m，
      认为是同一景点的不同细分（如 "颐和园" / "颐和园博物馆"），合并。

    实现：两两扫描 + 并查集聚簇，每簇按 _pick_best_poi 选代表。
    复杂度 O(n^2)，n 通常 < 30，无需空间索引。
    """
    n = len(pois)
    if n < 2:
        return pois

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    merged_pairs: List[tuple[str, str, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            name_i = pois[i].get("name", "")
            name_j = pois[j].get("name", "")
            if not _is_prefix_pair(name_i, name_j):
                continue
            dist = _haversine_meters(
                pois[i]["lng"], pois[i]["lat"], pois[j]["lng"], pois[j]["lat"]
            )
            if dist < _GEO_DEDUP_THRESHOLD_M:
                union(i, j)
                merged_pairs.append((name_i, name_j, dist))

    clusters: Dict[int, List[Dict]] = {}
    for idx, poi in enumerate(pois):
        clusters.setdefault(find(idx), []).append(poi)

    deduped = [_pick_best_poi(g) for g in clusters.values()]

    if len(deduped) < n:
        sample = [(a, b, f"{d:.0f}m") for a, b, d in merged_pairs[:5]]
        suffix = f" ...等 {len(merged_pairs)} 对" if len(merged_pairs) > 5 else ""
        logger.info(
            f"_geo_prefix_dedup: 前缀+{int(_GEO_DEDUP_THRESHOLD_M)}m 合并 "
            f"{n - len(deduped)} 条, {n} -> {len(deduped)} 条独立景点; "
            f"合并对样本: {sample}{suffix}"
        )

    return deduped


def _canonicalize_and_dedup(pois: List[Dict]) -> List[Dict]:
    """
    按规范化名称合并子 POI，同名组保留最优一条。

    合并示例：
      - "故宫博物院" + "故宫博物院检票处" → 保留 "故宫博物院"
      - "天安门广场" + "天安门广场-国旗"  → 保留 "天安门广场"

    选择规则（优先级从高到低）：
      1. typecode 落在风景名胜大类（11xxxx）的优先（最像主景点）
      2. 名称越短越优先（更接近规范名）
      3. search_rank 越小越优先（高德相关性排序靠前）

    Args:
        pois: 已通过 _extend_deduped 精确去重的 POI 列表。

    Returns:
        归一化去重后的 POI 列表，保持原始字段不变。
    """
    if not pois:
        return pois

    groups: Dict[str, List[Dict]] = {}
    for poi in pois:
        canon = _canon_name(poi.get("name", ""))
        if not canon:
            # 整个名字都是噪声后缀（如纯 "停车场"），通常已被 typecode 黑名单挡掉，
            # 这里兜底丢弃，避免污染候选池
            continue
        groups.setdefault(canon, []).append(poi)

    deduped = [_pick_best_poi(g) for g in groups.values()]

    if len(deduped) < len(pois):
        merged = len(pois) - len(deduped)
        logger.info(
            f"_canonicalize_and_dedup: 子POI/附属合并 {merged} 条, "
            f"{len(pois)} → {len(deduped)} 条独立景点"
        )

    return deduped


async def _search_single(
    city: str,
    keyword: str,
    top_n: int,
    session: ClientSession,
    trust_kb: bool = False,
) -> List[Dict]:
    """
    对单个关键词发起一次高德 MCP 搜索，返回标准化 POI 列表。
    session 由调用方（POIFetchAgent.run）统一建立并传入，避免每次重复握手开销。

    Args:
        trust_kb: KB 路径设为 True，放行 06/19 软黑名单（古街/知名街区）。
    """
    try:
        raw = await search_pois(session, city=city, keywords=keyword)
        normalized = _normalize_pois(raw, category="景点", top_n=top_n, trust_kb=trust_kb)
        logger.info(
            f"POIFetchAgent: 搜索 '{keyword}' (top_n={top_n}) → "
            f"原始 {len(raw)} 条，有效 {len(normalized)} 条"
        )
        return normalized
    except Exception as e:
        logger.warning(f"POIFetchAgent: 搜索 '{keyword}' 失败: {e}")
        return []


def _extend_deduped(
    all_pois: List[Dict],
    new_pois: List[Dict],
    seen_names: set,
) -> None:
    """
    将 new_pois 中名称未出现过的条目追加到 all_pois，同时更新 seen_names。
    保证全局名称去重：先搜到的条目优先保留（KB路径 > LLM hints > 泛搜）。
    """
    for poi in new_pois:
        name = poi.get("name", "").strip()
        if name and name not in seen_names:
            seen_names.add(name)
            all_pois.append(poi)


# =============================================================================
# POIFetchAgent
# =============================================================================

class POIFetchAgent:
    def __init__(self, name: str = "POIFetchAgent", model=None, **kwargs):
        self.name = name
        self.model = model  # 保留接口一致性，当前实现不需要 LLM

    async def run(self, input_data: dict) -> dict:
        context = input_data.get("context", {})
        key_entities = context.get("key_entities", {})

        city: str = (
            key_entities.get("destination", "")
            or context.get("destination", "")
        )
        if not city:
            return {"agent": "poi_fetch", "error": "缺少目的地城市，无法搜索 POI"}

        travel_style: str = context.get("travel_style", "普通")
        # 特种兵模式下各路径 top_n 翻倍，以满足高密度行程的候选需求
        style_multiplier = _SPECIAL_FORCES_MULTIPLIER if travel_style == "特种兵" else 1

        # LLM 生成的景点搜索提示词；intent_node 已用负面约束保证不含住宿/餐饮/交通词，
        # _normalize_pois 内的 typecode 黑名单作为兜底防线。
        attraction_hints: List[str] = [
            h for h in context.get("attraction_hints", []) if isinstance(h, str) and h.strip()
        ]

        # llm_seed_extract_node 抽取的具名 POI 种子（KB 必去 ⊕ LLM 抽取，前者保序在前）
        # 与 attraction_hints 的区别：
        #   - attraction_hints 来自用户 query，承载个性化兴趣（"大熊猫"）
        #   - llm_seed_pois 来自 RAG + KB，承载目的地共识地标（"天坛公园"）
        # 两者通过不同搜索路径（trust_kb 等级不同）喂给本 agent。
        llm_seed_pois: List[str] = [
            s for s in context.get("llm_seed_pois", []) if isinstance(s, str) and s.strip()
        ]

        all_pois: List[Dict] = []
        seen_names: set = set()  # 全局去重集合

        knowledge_db = CityKnowledgeDB.get_instance()

        async with amap_mcp_session() as session:
            if city and knowledge_db.has_city(city):
                # ═══════════════════════════════════════════════════════════════
                # KB 城市路径：知识库决定"去哪些景点"，高德只提供坐标/评分
                # ═══════════════════════════════════════════════════════════════

                # ── 路径①：must_visit 精准查询 ───────────────────────────────
                # top_n=2：名称明确，高德 top1 基本就是目标，取 2 作为安全冗余
                kb_must_visit = knowledge_db.get_must_visit_names(city)
                top_n_kb = 2 * style_multiplier
                for name in kb_must_visit:
                    pois = await _search_single(
                        city, f"{city} {name}", top_n=top_n_kb, session=session, trust_kb=True
                    )
                    _extend_deduped(all_pois, pois, seen_names)

                logger.info(
                    f"POIFetchAgent [KB路径①-must_visit]: city={city}, "
                    f"搜索 {len(kb_must_visit)} 个必去景点, 累计 {len(all_pois)} 个去重POI"
                )

                # ── 路径②：route_combo 额外子景点精准查询 ────────────────────
                # 只搜索不被 must_visit 覆盖的新增子景点（如断桥、法喜寺、龙井村等）
                # 用途：为 TSP 路由提供细粒度坐标；Phase-2 combo_boost 评分来源
                kb_extra = knowledge_db.get_extra_combo_spots(city)
                for name in kb_extra:
                    pois = await _search_single(
                        city, f"{city} {name}", top_n=top_n_kb, session=session, trust_kb=True
                    )
                    _extend_deduped(all_pois, pois, seen_names)

                logger.info(
                    f"POIFetchAgent [KB路径②-combo额外子景点]: "
                    f"搜索 {len(kb_extra)} 个子景点 {kb_extra}, 累计 {len(all_pois)} 个去重POI"
                )

                # ── 路径②b：llm_seed_pois 精准查询（剔除已被①②覆盖的） ─────
                # llm_seed_extract_node 已把 KB must_visit 前缀合并进 seed，
                # 此处用集合差剔除已经搜过的，避免重复 API 调用
                already_searched: set = set(kb_must_visit) | set(kb_extra)
                seed_remaining = [s for s in llm_seed_pois if s not in already_searched]
                top_n_seed = 2 * style_multiplier
                for name in seed_remaining:
                    pois = await _search_single(
                        city, f"{city} {name}", top_n=top_n_seed, session=session, trust_kb=True
                    )
                    _extend_deduped(all_pois, pois, seen_names)

                if seed_remaining:
                    logger.info(
                        f"POIFetchAgent [KB路径②b-llm_seed]: "
                        f"搜索 {len(seed_remaining)} 个种子 {seed_remaining}, "
                        f"累计 {len(all_pois)} 个去重POI"
                    )

                # ── 路径③：LLM hints 补充用户特定兴趣 ───────────────────────
                # top_n=3：hint 有一定模糊性，多取几条；名称与KB重叠的会被去重过滤
                top_n_hint = 3 * style_multiplier
                for hint in attraction_hints:
                    pois = await _search_single(city, hint, top_n=top_n_hint, session=session)
                    _extend_deduped(all_pois, pois, seen_names)

                if attraction_hints:
                    logger.info(
                        f"POIFetchAgent [KB路径③-LLM hints]: "
                        f"hints={attraction_hints}, 累计 {len(all_pois)} 个去重POI"
                    )

            else:
                # ═══════════════════════════════════════════════════════════════
                # 非 KB 城市路径：seed 主源 + LLM hints 补充，泛搜兜底
                # 结果质量较 KB 城市路径略低，但 seed 由 RAG+LLM 共识抽取，远好于裸泛搜
                # ═══════════════════════════════════════════════════════════════
                top_n_seed = 2 * style_multiplier
                top_n_hint = 5 * style_multiplier

                # ── 路径②b：llm_seed_pois 精准查询（trust_kb=True，作为种子主源）─
                for name in llm_seed_pois:
                    pois = await _search_single(
                        city, f"{city} {name}", top_n=top_n_seed, session=session, trust_kb=True
                    )
                    _extend_deduped(all_pois, pois, seen_names)
                if llm_seed_pois:
                    logger.info(
                        f"POIFetchAgent [非KB路径-llm_seed]: city={city}, "
                        f"seeds={llm_seed_pois}, 累计 {len(all_pois)} 个去重POI"
                    )

                # ── 路径③：LLM hints 补充用户个性化兴趣 ─────────────────────
                if attraction_hints:
                    for hint in attraction_hints:
                        pois = await _search_single(city, hint, top_n=top_n_hint, session=session)
                        _extend_deduped(all_pois, pois, seen_names)
                    logger.info(
                        f"POIFetchAgent [非KB路径-LLM hints]: city={city}, "
                        f"hints={attraction_hints}, 累计 {len(all_pois)} 个去重POI"
                    )

                # ── 路径④：泛搜兜底（仅当 seed 和 hints 均为空时触发）────────
                if not llm_seed_pois and not attraction_hints:
                    keyword = _FALLBACK_KEYWORD_TMPL.format(city=city)
                    pois = await _search_single(
                        city, keyword, top_n=_FALLBACK_TOP_N * style_multiplier, session=session
                    )
                    _extend_deduped(all_pois, pois, seen_names)
                    logger.info(
                        f"POIFetchAgent [非KB路径-泛搜兜底]: city={city}, "
                        f"keyword='{keyword}', 累计 {len(all_pois)} 个去重POI"
                    )

        # 子 POI/附属设施归一化合并（"故宫博物院" 与 "故宫博物院检票处" 合并）
        all_pois = _canonicalize_and_dedup(all_pois)

        # 前缀+200m 地理去重（"颐和园" 与 "颐和园博物馆" 合并）
        all_pois = _geo_prefix_dedup(all_pois)

        logger.info(f"POIFetchAgent: 最终 poi_candidates 共 {len(all_pois)} 条（已全局去重）")
        return {
            "agent": "poi_fetch",
            "result": {"pois": all_pois},
        }
