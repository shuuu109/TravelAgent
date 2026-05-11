"""
POI 体验描述与照片补充节点 poi_enrich_node (P3.5)
================================================
位置：itinerary_planning_node 之后，accommodation_node 之前。

职责（Post-Retrieval Augmentation）：
  1. 对 daily_routes 中每个已确定的 POI，以景点名本身为 query 检索知识库，
     提炼 1-2 句核心体验描述写入 state["poi_descriptions"]
  2. 同时为每个 POI 抓取 1-3 张缩略图 URL 写入 state["poi_photos"]
     来源链：高德 v5 place/text（show_fields=photos）-> Wikipedia REST 缩略图
     未命中时返回空列表，前端用占位图兜底

设计要点：
  1. Query = 景点名：语义对齐极精准，无需复杂构造，召回噪声低
  2. 批量并发：asyncio.gather 并行处理所有 POI，描述与照片在每个 POI 内部串行
  3. 去重：同名 POI 只检索一次，避免重复 LLM/HTTP 调用
  4. 容错：单 POI 任一步失败不影响其他 POI；description/photos 各自独立写入
"""
from __future__ import annotations

import asyncio
import logging
from typing import Dict, List
from urllib.parse import quote

import httpx

from graph.state import TravelGraphState, ensure_hard_constraints
from agents.rag_base_agent import RAGBaseAgent
from mcp_clients.amap_client import AMAP_KEY

logger = logging.getLogger(__name__)

# 单个 POI 最多保留的照片张数；前端表格展示 1 张主图 + 2 张备用
_MAX_PHOTOS_PER_POI: int = 3

# HTTP 请求超时（秒）：高德/Wikipedia 单次最长等待
_HTTP_TIMEOUT: float = 5.0


def create_poi_enrich_node(llm):
    """
    工厂函数：将 LLM 实例通过闭包注入。

    Args:
        llm: LangChain ChatXxx 实例，实现 ainvoke(messages) -> AIMessage

    Returns:
        async 节点函数 poi_enrich_node(state) -> dict
    """
    # RAGBaseAgent 只做检索，共享 ask-question 技能的 ChromaDB
    rag = RAGBaseAgent(name="poi_enrich_rag", model=None, top_k=3)

    async def poi_enrich_node(state: TravelGraphState) -> dict:
        """
        节点主流程：
        1. 从 daily_routes 提取所有唯一 POI 名称
        2. 并发处理：每个 POI 同时获取知识库描述 + 高德/Wikipedia 照片
        3. 返回 {"poi_descriptions": {...}, "poi_photos": {...}}
        """
        daily_routes: List[Dict] = state.get("daily_routes") or []
        if not daily_routes:
            logger.warning("[poi_enrich] daily_routes 为空，跳过 POI 增强")
            return {"poi_descriptions": {}, "poi_photos": {}}

        # 去重：从所有天的 ordered_pois 中提取唯一 POI 名称
        poi_names: List[str] = []
        seen: set = set()
        for day_route in daily_routes:
            for poi in day_route.get("ordered_pois", []):
                name = poi.get("name", "").strip()
                if name and name not in seen:
                    poi_names.append(name)
                    seen.add(name)

        if not poi_names:
            return {"poi_descriptions": {}, "poi_photos": {}}

        # 用于高德 photos API 的 region 参数（缩小搜索范围、提升命中精度）
        hc = ensure_hard_constraints(state.get("hard_constraints"))
        city = hc.destination or ""

        logger.info(f"[poi_enrich] 开始补充 {len(poi_names)} 个 POI 的描述与照片")

        # Semaphore 限制最多 5 个 POI 并发处理，避免 LLM/HTTP 速率限制
        sem = asyncio.Semaphore(5)

        async def _enrich_one(name: str, http: httpx.AsyncClient):
            """单 POI 任务：先描述（占用 LLM 配额），再照片（HTTP 不占 sem）"""
            async with sem:
                desc = await _enrich_single_poi(name, rag, llm)
            photos = await _fetch_poi_photos(name, city, http)
            return desc, photos

        # 共享一个 httpx client，复用 TCP 连接池
        async with httpx.AsyncClient(follow_redirects=True) as http:
            tasks = [_enrich_one(name, http) for name in poi_names]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        poi_descriptions: Dict[str, str] = {}
        poi_photos: Dict[str, List[str]] = {}
        for name, result in zip(poi_names, results):
            if isinstance(result, Exception):
                logger.warning(f"[poi_enrich] {name!r} 整体处理失败: {result}")
                poi_descriptions[name] = ""
                poi_photos[name] = []
                continue
            desc, photos = result
            poi_descriptions[name] = desc or ""
            poi_photos[name] = photos or []

        desc_hits = sum(1 for v in poi_descriptions.values() if v)
        photo_hits = sum(1 for v in poi_photos.values() if v)
        logger.info(
            f"[poi_enrich] 完成：描述 {desc_hits}/{len(poi_names)}，"
            f"照片 {photo_hits}/{len(poi_names)}"
        )
        return {
            "poi_descriptions": poi_descriptions,
            "poi_photos": poi_photos,
        }

    return poi_enrich_node


# =============================================================================
# 内部辅助函数
# =============================================================================

async def _enrich_single_poi(poi_name: str, rag: RAGBaseAgent, llm) -> str:
    """
    对单个 POI 完成检索 + LLM 描述提炼。

    Args:
        poi_name: 景点名称，直接作为检索 query（语义对齐精准）
        rag:      RAGBaseAgent 实例，提供 search_knowledge()
        llm:      LangChain LLM 实例

    Returns:
        提炼后的 1-2 句体验描述。
        - RAG 命中：从攻略片段提炼
        - RAG 未命中但有 LLM：让 LLM 基于自身知识生成（system_prompt 规则 3）
        - LLM 不可用且 RAG 命中：截取首个文档片段降级
        - LLM 不可用且 RAG 未命中：返回空字符串（前端用占位文案）
    """
    docs = rag.search_knowledge(poi_name, top_k=3)

    if not docs and not llm:
        # 双路皆不可用：兜底返回空，由 respond_node / 前端容错处理
        logger.debug(f"[poi_enrich] {poi_name!r}: RAG 未命中且无 LLM，返回空")
        return ""

    if not llm:
        # 无 LLM 时直接截取首个文档片段作为降级描述
        return docs[0]["content"][:100]

    # 拼接检索到的上下文（docs 为空时 context 为空串，触发 user_prompt 规则 3）
    if docs:
        context = "\n\n".join(
            f"【片段{i + 1}】\n{d['content']}" for i, d in enumerate(docs)
        )
    else:
        logger.debug(f"[poi_enrich] {poi_name!r}: RAG 未命中，让 LLM 基于自身知识生成")
        context = "（攻略文本无该景点相关片段）"

    system_prompt = (
        "你是专业的旅游行程文案撰写助手。\n"
        "输出要求：\n"
        "- 用 1-2 句中文（约 60-100 字）撰写景点游览体验描述\n"
        "- 语气自然流畅，避免“根据攻略”“值得一去”“总而言之”等 AI 痕迹明显的措辞\n"
        "- 不使用 emoji\n"
        "- 直接输出正文，不要前缀、不要引号包裹，也不要任何额外解释"
    )

    user_prompt = (
        f"请为景点【{poi_name}】撰写游览体验文案。\n\n"
        f"撰写要求：\n"
        f"1. 优先从下方攻略文本中提炼最具体的细节：最佳游览时段、独特景观、"
        f"票务与动线提醒等。\n"
        f"2. 文案需能自然衔接到行程介绍中，不要以“根据攻略”“这是一座”"
        f"“作为……”等通用句式开头。\n"
        f"3. 若攻略文本未涉及该景点，基于景点自身的代表性特征"
        f"生成专业且不生硬的景点介绍。\n\n"
        f"【攻略文本】\n{context}"
    )

    try:
        response = await llm.ainvoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])
        result = response.content.strip()
        # 拒绝过长输出（超过 150 字视为 LLM 未遵循指令）
        if len(result) > 150:
            result = result[:150]
        logger.debug(f"[poi_enrich] {poi_name!r}: {result!r}")
        return result
    except Exception as e:
        logger.error(f"[poi_enrich] {poi_name!r} LLM 调用失败: {e}")
        return ""


# =============================================================================
# 照片抓取：高德 v5 -> Wikipedia REST -> 空列表
# =============================================================================

async def _fetch_poi_photos(
    poi_name: str,
    city: str,
    http: httpx.AsyncClient,
) -> List[str]:
    """
    单 POI 照片抓取的两路兜底链。

    Args:
        poi_name: 景点名（用于关键字匹配 / Wikipedia title）
        city:     目的地城市名（高德 region 参数，缩小搜索范围）
        http:     共享的 httpx.AsyncClient

    Returns:
        URL 列表（最多 _MAX_PHOTOS_PER_POI 张）；全部失败时返回空列表。
    """
    photos = await _amap_photos(poi_name, city, http)
    if photos:
        return photos[:_MAX_PHOTOS_PER_POI]

    photos = await _wikipedia_thumbnail(poi_name, http)
    return photos[:_MAX_PHOTOS_PER_POI]


async def _amap_photos(
    poi_name: str,
    city: str,
    http: httpx.AsyncClient,
) -> List[str]:
    """
    高德 v5 place/text 搜索 + show_fields=photos。

    国内景点命中率较高（约 60%）。无 city 时不限定 region，召回噪声会增加。
    """
    if not AMAP_KEY:
        return []
    params = {
        "keywords": poi_name,
        "key": AMAP_KEY,
        "show_fields": "photos",
        "page_size": "1",
    }
    if city:
        params["region"] = city
        params["city_limit"] = "true"

    try:
        resp = await http.get(
            "https://restapi.amap.com/v5/place/text",
            params=params,
            timeout=_HTTP_TIMEOUT,
        )
        data = resp.json()
        pois = data.get("pois") or []
        if not pois:
            return []
        photos = pois[0].get("photos") or []
        return [p.get("url") for p in photos if isinstance(p, dict) and p.get("url")]
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"[poi_enrich] amap photos failed for {poi_name!r}: {exc}")
        return []


# Wikipedia 词条命中率优化：常见景点尾缀剥离后再查一次。
# 排序从长到短，确保 "国家森林公园" 优先匹配，再轮到 "公园"。
_WIKI_TITLE_SUFFIXES: tuple = (
    "国家森林公园",
    "国家公园",
    "森林公园",
    "风景名胜区",
    "风景区",
    "景区",
    "景点",
    "公园",
)


def _normalize_title(name: str) -> str | None:
    """
    剥离景点名末尾的通用后缀，返回更可能命中 Wikipedia 词条的核心名。

    Args:
        name: 原始 POI 名，例如 "雷峰塔景区" / "西湖风景名胜区"

    Returns:
        归一化后的核心名（如 "雷峰塔"），无可剥离尾缀时返回 None。
        剥离后长度 < 2 视为无效，避免 "公园" -> "" 之类越界。
    """
    for suffix in _WIKI_TITLE_SUFFIXES:
        if name.endswith(suffix) and len(name) - len(suffix) >= 2:
            return name[: -len(suffix)]
    return None


async def _wiki_query_summary(
    title: str,
    http: httpx.AsyncClient,
) -> List[str]:
    """请求 Wikipedia REST page summary，命中返回 [original, thumbnail]。"""
    url = (
        "https://zh.wikipedia.org/api/rest_v1/page/summary/"
        f"{quote(title, safe='')}"
    )
    headers = {
        # Wikimedia 强制要求 User-Agent 包含联系方式或项目名，否则 403
        "User-Agent": "AligoTravelAgent/1.0 (https://github.com/example/aligo)",
        "Accept": "application/json",
    }
    try:
        resp = await http.get(url, headers=headers, timeout=_HTTP_TIMEOUT)
        if resp.status_code != 200:
            return []
        data = resp.json()
        original = (data.get("originalimage") or {}).get("source")
        thumb = (data.get("thumbnail") or {}).get("source")
        return [u for u in (original, thumb) if u]
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"[poi_enrich] wikipedia query failed for {title!r}: {exc}")
        return []


async def _wikipedia_thumbnail(
    poi_name: str,
    http: httpx.AsyncClient,
) -> List[str]:
    """
    Wikipedia REST API 中文站 page summary，先用归一化标题再 fallback 原名。

    免费无限额度，但小众景点常无对应词条；返回 originalimage + thumbnail 两档。
    """
    normalized = _normalize_title(poi_name)
    if normalized:
        photos = await _wiki_query_summary(normalized, http)
        if photos:
            return photos
    return await _wiki_query_summary(poi_name, http)
