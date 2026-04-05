import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import List, Dict, Optional, Tuple
from mcp.client.sse import sse_client
from mcp.client.session import ClientSession

logger = logging.getLogger(__name__)

# 从环境变量读取你申请的高德 Web 服务 Key
AMAP_KEY = os.getenv("AMAP_KEY", "1dd13742a147224131022165e14d6d55")

@asynccontextmanager
async def amap_mcp_session():
    """
    管理与高德官方 MCP Server 的 SSE 长连接生命周期
    
    Note: 当前针对高德MCP服务的连接和工具调用需要进一步测试。
    如果无法连接，请确保：
    1. AMAP_KEY 环境变量或config中有有效的API Key
    2. 网络连接正常
    3. 高德MCP服务端口可访问
    """
    try:
        # 高德官方的 SSE 接入点
        url = f"https://mcp.amap.com/sse?key={AMAP_KEY}"
        
        # 建立与官方 MCP Server 的连接
        async with sse_client(url) as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                # 初始化 MCP 协议
                await session.initialize()
                yield session
    except Exception as e:
        print(f"[WARNING] Unable to connect to Amap MCP service: {e}")
        # 抛出异常让调用方处理
        raise

async def _fill_locations_via_detail(
    session: ClientSession,
    items: List[Dict],
) -> None:
    """
    通过 maps_search_detail 为缺失坐标的 POI 补充 location 字段（原地修改）。

    maps_search_detail 按 POI id 精确查询，返回包含三大坐标系字段的详情，
    比地理编码 API 准确率更高、无歧义。取 GCJ-02 的 location 字段（高德默认）。

    Args:
        session: 与调用方共享的同一 MCP ClientSession，避免重复建立 SSE 连接。
        items:   待补充坐标的 POI dict 列表，每条需包含 _id 字段。
    """
    import json

    for item in items:
        poi_id = item.get("_id", "")
        if not poi_id:
            continue
        try:
            result = await session.call_tool("maps_search_detail", {"id": poi_id})
            for block in result.content:
                text = getattr(block, "text", None)
                if not text:
                    continue
                try:
                    data = json.loads(text)
                except json.JSONDecodeError:
                    continue

                # maps_search_detail 返回结构兼容处理：
                # 常见格式1: {"location": "lng,lat", ...}        顶层直接有 location
                # 常见格式2: {"pois": [{"location": "lng,lat"}]} 嵌套在 pois 列表里
                loc = data.get("location") or ""
                if not loc:
                    inner = data.get("pois") or []
                    if isinstance(inner, list) and inner:
                        loc = inner[0].get("location", "")
                if loc:
                    item["location"] = loc
                    break   # 已找到坐标，不再遍历其他 block
        except Exception:
            continue  # 单条失败不中断整体流程


async def _geocode_single(
    client: "httpx.AsyncClient",
    item: Dict,
    city: str,
    geo_url: str,
) -> None:
    """
    单条 POI 地理编码（内部辅助函数）。
    原地将 location 字段写入 item，失败时静默跳过。
    """
    query = item.get("name", "").strip() or item.get("address", "").strip()
    if not query:
        return
    try:
        resp = await client.get(geo_url, params={
            "address": query,
            "city":    city,
            "key":     AMAP_KEY,
            "output":  "json",
        })
        data = resp.json()
        geocodes = data.get("geocodes", [])
        if geocodes:
            loc = geocodes[0].get("location", "")
            if loc:
                item["location"] = loc
    except Exception:
        pass  # 单条失败不影响其他 POI


async def _batch_geocode_rest(
    items: List[Dict],
    city: str,
) -> None:
    """
    并发地理编码（REST API 版）：对 location 为空的 POI，并发调用高德地理编码
    REST 接口补充坐标，原地修改 items 列表。

    仅作为 _fill_locations_via_detail 的兜底方案。
    此函数直接使用 httpx 发 HTTPS REST 请求，完全绕开 SSE 代理限制。

    [WARNING] Why not use batch "|" requests:
    AMap geocoding API only returns successful results in the geocodes array, and failed addresses receive no null placeholders.
    This causes geocodes[idx] to be unable to align with items[idx] in batch mode, so only the first record can match.
    After switching to row-by-row concurrent mode, the alignment problem is completely eliminated, and asyncio.gather ensures concurrent efficiency.

    Args:
        items: 需要补充坐标的 POI dict 列表（含 name/address 字段）。
        city:  城市名，用于缩小地理编码范围，提升结果准确率。
    """
    import asyncio
    import httpx

    GEO_URL = "https://restapi.amap.com/v3/geocode/geo"

    # trust_env=False：忽略系统代理环境变量，直连高德 REST 接口
    async with httpx.AsyncClient(timeout=15.0, trust_env=False) as client:
        await asyncio.gather(
            *[_geocode_single(client, item, city, GEO_URL) for item in items]
        )


async def search_pois(
    session: ClientSession,
    city: str,
    keywords: str,
    types: str = None,
) -> List[Dict]:
    """
    POI搜索（P2 技能使用）。

    调用高德 MCP 的 maps_text_search 工具搜索 POI。
    因该工具不返回坐标，按以下优先级逐步补充 location 字段：
      1. maps_search_detail（通过 POI id 精确查询，坐标准确率最高）
      2. _batch_geocode_rest（REST 地理编码，作为兜底）
    最终每个元素包含: name, location (lng,lat 字符串), address, rating, type。

    Args:
        session:  由调用方通过 amap_mcp_session() 建立并传入，不在此函数内新建连接。
        city:     城市名，如 "杭州"。
        keywords: 关键字，如 "杭州景点"。
        types:    高德 POI 类型码（可选），如 "060100"。
    """
    import json

    args = {"keywords": keywords, "city": city}
    if types:
        args["types"] = types

    result = await session.call_tool("maps_text_search", args)

    pois: List[Dict] = []
    for block in result.content:
        text = getattr(block, "text", None)
        if not text:
            continue
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            continue

        # 兼容两种常见外层结构：直接列表 / {"pois": [...]}
        raw_list = data if isinstance(data, list) else data.get("pois", [])
        for item in raw_list:
            # maps_text_search 不返回 location，先置空，后续分两步补充
            location = item.get("location") or ""
            pois.append({
                "_id":      item.get("id", ""),   # 临时保留，用于 detail 查询
                "name":     item.get("name", ""),
                "location": location,
                "address":  item.get("address", ""),
                "rating":   (
                    item.get("biz_ext", {}).get("rating", "")
                    if isinstance(item.get("biz_ext"), dict)
                    else ""
                ),
                "type":     item.get("typecode", item.get("type", "")),
            })

    # ── Step 1: 通过 maps_search_detail 补充缺失坐标（最优先，精度最高）────────
    missing = [p for p in pois if not p["location"] and p.get("_id")]
    if missing:
        await _fill_locations_via_detail(session, missing)

    # ── Step 2: REST 地理编码兜底（处理 detail 仍未能补充的 POI）────────────
    still_missing = [p for p in pois if not p["location"]]
    if still_missing:
        await _batch_geocode_rest(still_missing, city)

    # 清理临时字段，不暴露给调用方
    for p in pois:
        p.pop("_id", None)

    return pois


async def get_distance_matrix(
    session: ClientSession,
    origins: List[str],
    destinations: List[str],
) -> List[List[float]]:
    """
    距离矩阵（P3 TSP 使用）。

    maps_distance 是 N→1 模型：origins 支持 "|" 分隔的多个起点，但
    destination 只接受单个终点。因此无法一次拿到 N×M 矩阵。

    修复方案（方案 B）：
      对每个 destination 单独发起一次 N→1 请求（共 M 次），
      通过 asyncio.gather 并发执行，总耗时约等于单次请求延迟。
      _AMAP_SEMAPHORE 控制最大并发数，防止触发高德 QPS 限流。

    Args:
        session:      由调用方传入的同一 ClientSession，避免重复建立 SSE 连接。
        origins:      N 个出发点坐标，格式 "lng,lat"。
        destinations: M 个目的地坐标，格式 "lng,lat"。

    Returns:
        N×M 的时间矩阵（秒）。缺失或请求失败的格填 float('inf')。
    """
    import json

    n, m = len(origins), len(destinations)
    matrix: List[List[float]] = [[float("inf")] * m for _ in range(n)]
    origins_str = "|".join(origins)

    # 局部创建 Semaphore：绑定到当前运行的 event loop，避免模块级变量的 loop 绑定问题。
    # 语义上也更准确：限制的是本次矩阵计算内的并发，而非全局并发。
    semaphore = asyncio.Semaphore(5)

    async def _fetch_column(dest_idx: int) -> Tuple[int, List[float]]:
        """
        向第 dest_idx 个目的地发起一次 N→1 请求。
        返回 (列索引, 长度为 N 的时间列表)，失败格填 inf。
        """
        col: List[float] = [float("inf")] * n
        async with semaphore:
            result = await session.call_tool("maps_distance", {
                "origins": origins_str,
                "destination": destinations[dest_idx],
                "type": 1,  # 1=驾车，0=直线，3=步行
            })

        for block in result.content:
            text = getattr(block, "text", None)
            if not text:
                continue
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                continue

            # 高德返回: {"results": [{"origin_id": "1", "dest_id": "1", "duration": "xxx"}]}
            # dest_id 固定为 "1"（单终点），origin_id 对应 origins 列表的 1-based 下标
            raw_results = data if isinstance(data, list) else data.get("results", [])
            for item in raw_results:
                try:
                    i = int(item.get("origin_id", 1)) - 1
                    duration = item.get("duration")
                    if duration is not None and 0 <= i < n:
                        col[i] = float(duration)
                except (ValueError, TypeError):
                    continue
            break  # 解析到第一个有效 block 即止

        return dest_idx, col

    # 并发发起 M 次 N→1 请求，return_exceptions=True 保证单列失败不影响整体
    column_results = await asyncio.gather(
        *[_fetch_column(j) for j in range(m)],
        return_exceptions=True,
    )

    for item in column_results:
        if isinstance(item, Exception):
            logger.warning(f"get_distance_matrix: 某列请求异常: {item}")
            continue
        dest_idx, col = item
        for origin_idx, duration in enumerate(col):
            matrix[origin_idx][dest_idx] = duration

    return matrix


async def get_transit_route(
    session: ClientSession,
    origin: str,
    destination: str,
    city: str,
) -> Dict:
    """
    单日路线详情（P3 最终路线使用）。

    调用高德 MCP 的 maps_direction_transit_integrated 工具，返回推荐公交/地铁路线。

    Args:
        session: 由调用方传入的同一 ClientSession，避免重复建立 SSE 连接。
        origin: 出发地坐标 "lng,lat"。
        destination: 目的地坐标 "lng,lat"。
        city: 城市名，用于公共交通城市过滤。

    Returns:
        {
            "duration": int,          # 预计耗时（秒）
            "distance": int,          # 距离（米）
            "steps": List[str],       # 分段文字描述列表
            "recommended_mode": str,  # 推荐出行方式
        }
    """
    import json

    fallback = {
        "duration": 0,
        "distance": 0,
        "steps": [],
        "recommended_mode": "transit",
    }

    result = await session.call_tool("maps_direction_transit_integrated", {
        "origin": origin,
        "destination": destination,
        "city": city,
        "cityd": city,  # 目的地城市（跨城时不同）
    })

    for block in result.content:
        text = getattr(block, "text", None)
        if not text:
            continue
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            continue

        # 高德返回结构: {"route": {"transits": [{"duration":..., "distance":..., "segments":[...]}]}}
        route = data.get("route", data)
        transits = route.get("transits", [])
        if not transits:
            break

        # 取第一条（推荐）路线
        best = transits[0]
        steps: List[str] = []
        for seg in best.get("segments", []):
            bus = seg.get("bus", {})
            walking = seg.get("walking", {})
            if bus:
                for line in bus.get("buslines", []):
                    steps.append(
                        f"乘 {line.get('name', '公交')} 至 {line.get('arrival_stop', {}).get('name', '')}"
                    )
            elif walking:
                dist = walking.get("distance", "")
                steps.append(f"步行 {dist} 米")

        return {
            # 高德 API duration 返回单位为秒，转换为分钟供上层渲染使用
            "duration": int(best.get("duration", 0)) // 60,
            "distance": int(best.get("distance", 0)),
            "steps": steps,
            "recommended_mode": "transit",
        }

    return fallback


async def search_hotels_nearby(
    session: ClientSession,
    location: str,
    radius: int = 2000,
    keywords: str = "酒店",
    city: str = "",
    count: int = 10,
) -> List[Dict]:
    """
    Phase 1 地理发现：基于坐标 + 半径搜索周边酒店 POI。

    调用高德 MCP 的 maps_around_search 工具。与 maps_text_search 的关键区别是：
    返回结果包含 distance 字段（单位：米），精确表示每家酒店距搜索中心的距离，
    是"按日程重心找最近酒店"场景的最优工具。

    Args:
        session:  由调用方通过 amap_mcp_session() 建立并传入，不在此函数内新建连接。
        location: 搜索中心坐标，格式 "lng,lat"（高德 GCJ-02 坐标系）。
        radius:   搜索半径（米），默认 2000m，最大 50000m。
        keywords: 关键字，默认 "酒店"；可改为 "民宿"、"青旅" 等。
        city:     城市名（可选），辅助缩小范围。
        count:    返回结果上限，取距离最近的前 count 家，默认 10。

    Returns:
        酒店 POI 列表，每条包含：
          name, location, address, distance_m（米）, amap_rating, type, _amap_id
        已按 distance_m 升序排列。
    """
    import json

    args: dict = {
        "location": location,
        "radius": radius,
        "keywords": keywords,
        "types": "100103",   # 高德 POI 类型码：住宿服务 / 宾馆酒店
    }
    if city:
        args["city"] = city

    result = await session.call_tool("maps_around_search", args)

    hotels: List[Dict] = []
    for block in result.content:
        text = getattr(block, "text", None)
        if not text:
            continue
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            continue

        # 兼容两种常见外层结构：直接列表 / {"pois": [...]}
        raw_list = data if isinstance(data, list) else data.get("pois", [])
        for item in raw_list:
            # distance 字段：maps_around_search 专有，单位米（字符串或整数）
            try:
                distance_m = int(item.get("distance", 0) or 0)
            except (ValueError, TypeError):
                distance_m = 0

            # amap_rating：优先从 biz_ext 取，fallback 取顶层 rating
            biz = item.get("biz_ext") or {}
            amap_rating = (
                biz.get("rating", "") if isinstance(biz, dict) else ""
            ) or item.get("rating", "")

            hotels.append({
                "_amap_id":    item.get("id", ""),
                "name":        item.get("name", ""),
                "location":    item.get("location", ""),
                "address":     item.get("address", ""),
                "distance_m":  distance_m,
                "amap_rating": amap_rating,
                "type":        item.get("typecode", item.get("type", "")),
            })
        break   # 第一个有效 block 即可，不重复解析

    # 按距离升序，取前 count 条
    hotels.sort(key=lambda h: h["distance_m"])
    return hotels[:count]


async def search_restaurants_nearby(
    session: ClientSession,
    location: str,
    radius: int = 3000,
    city: str = "",
    count: int = 5,
) -> List[Dict]:
    """
    基于每天景点重心，搜索周边餐厅推荐（P3 行程规划后调用）。

    与 search_hotels_nearby 逻辑相同，但针对餐饮类 POI：
      - types="050000" 对应高德 POI 类型"餐饮服务"
      - 默认半径 3000m，适合覆盖午餐/晚餐范围
      - 默认返回 5 家，够每天午餐/晚餐推荐使用

    Args:
        session:  由调用方通过 amap_mcp_session() 建立并传入，不在此函数内新建连接。
        location: 搜索中心坐标，格式 "lng,lat"（当天景点地理重心）。
        radius:   搜索半径（米），默认 3000m。
        city:     城市名（可选），辅助缩小范围。
        count:    返回结果上限，取距离最近的前 count 家，默认 5。

    Returns:
        餐厅 POI 列表，每条包含：
          name, location, address, distance_m（米）, amap_rating, type
        已按 distance_m 升序排列。
    """
    import json

    args: dict = {
        "location": location,
        "radius": radius,
        "keywords": "餐厅",
        "types": "050000",  # 高德 POI 类型码：餐饮服务
    }
    if city:
        args["city"] = city

    result = await session.call_tool("maps_around_search", args)

    restaurants: List[Dict] = []
    for block in result.content:
        text = getattr(block, "text", None)
        if not text:
            continue
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            continue

        raw_list = data if isinstance(data, list) else data.get("pois", [])
        for item in raw_list:
            try:
                distance_m = int(item.get("distance", 0) or 0)
            except (ValueError, TypeError):
                distance_m = 0

            biz = item.get("biz_ext") or {}
            amap_rating = (
                biz.get("rating", "") if isinstance(biz, dict) else ""
            ) or item.get("rating", "")

            restaurants.append({
                "name":        item.get("name", ""),
                "location":    item.get("location", ""),
                "address":     item.get("address", ""),
                "distance_m":  distance_m,
                "amap_rating": amap_rating,
                "type":        item.get("typecode", item.get("type", "")),
            })
        break  # 第一个有效 block 即可

    # 按距离升序，取前 count 条
    restaurants.sort(key=lambda r: r["distance_m"])
    return restaurants[:count]


# 备用方案：如果你更倾向于本地运行 (需要安装 Node.js)
# from mcp.client.stdio import stdio_client, StdioServerParameters
# @asynccontextmanager
# async def amap_mcp_session_local():
#     server_params = StdioServerParameters(
#         command="npx",
#         args=["-y", "@amap/amap-maps-mcp-server"],
#         env={"AMAP_MAPS_API_KEY": AMAP_KEY, "PATH": os.environ["PATH"]}
#     )
#     async with stdio_client(server_params) as (read, write):
#         async with ClientSession(read, write) as session:
#             await session.initialize()
#             yield session