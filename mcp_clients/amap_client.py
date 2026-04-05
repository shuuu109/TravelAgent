import os
from contextlib import asynccontextmanager
from typing import List, Dict, Optional
from mcp.client.sse import sse_client
from mcp.client.session import ClientSession

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
        print(f"⚠️  无法连接到高德MCP服务: {e}")
        # 抛出异常让调用方处理
        raise

async def _batch_geocode_rest(
    items: List[Dict],
    city: str,
) -> None:
    """
    批量地理编码（REST API 版）：对 location 为空的 POI，调用高德地理编码 REST
    接口补充坐标，原地修改 items 列表。

    与 MCP session 版不同，此函数直接使用 httpx 发 HTTPS 请求，完全绕开 SSE
    代理限制——Windows 系统代理对同一 SSE 连接的第二次 POST 常常超时，
    而独立的 REST 请求不受此影响。

    高德地理编码 REST 接口支持用 "|" 分隔最多 10 个地址，结果按序对应输入。

    Args:
        items: 需要补充坐标的 POI dict 列表（含 name/address 字段）。
        city:  城市名，用于缩小地理编码范围，提升结果准确率。
    """
    import json
    import httpx

    GEO_URL = "https://restapi.amap.com/v3/geocode/geo"
    BATCH   = 10  # 高德单次最多 10 个地址

    # trust_env=False：忽略系统代理环境变量，直连高德 REST 接口
    async with httpx.AsyncClient(timeout=15.0, trust_env=False) as client:
        for batch_start in range(0, len(items), BATCH):
            batch = items[batch_start: batch_start + BATCH]

            # "POI名 + 地址" 拼接，比纯地址地理编码更准确
            addresses = [
                f"{it.get('name', '')} {it.get('address', '')}".strip()
                or it.get("name", "")
                for it in batch
            ]
            joined = "|".join(addresses)

            try:
                resp = await client.get(GEO_URL, params={
                    "address": joined,
                    "city":    city,
                    "key":     AMAP_KEY,
                    "output":  "json",
                })
                data = resp.json()
            except Exception:
                continue  # 单批次失败，跳过，不中断其他批次

            geocodes = data.get("geocodes", [])
            for idx, geocode in enumerate(geocodes):
                if idx >= len(batch):
                    break
                loc = geocode.get("location", "")
                if loc:
                    batch[idx]["location"] = loc


async def search_pois(
    session: ClientSession,
    city: str,
    keywords: str,
    types: str = None,
) -> List[Dict]:
    """
    POI搜索（P2 技能使用）。

    调用高德 MCP 的 maps_text_search 工具搜索 POI，
    因该工具不返回坐标，再通过 maps_geo 批量补充 location 字段。
    最终每个元素包含: name, location (lng,lat 字符串), address, rating, type。

    Args:
        session: 由调用方通过 amap_mcp_session() 建立并传入，不在此函数内新建连接。
        city:    城市名，如 "杭州"。
        keywords: 关键字，如 "杭州景点"。
        types:   高德 POI 类型码（可选），如 "060100"。
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
            # maps_text_search 不返回 location，先置空，后续批量补充
            location = item.get("location") or ""
            pois.append({
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

    # ── 批量补充缺失坐标（REST API，绕开 SSE 代理限制）────────────────────────
    missing = [p for p in pois if not p["location"]]
    if missing:
        await _batch_geocode_rest(missing, city)

    return pois


async def get_distance_matrix(
    session: ClientSession,
    origins: List[str],
    destinations: List[str],
) -> List[List[float]]:
    """
    距离矩阵（P3 TSP 使用）。

    批量调用高德 MCP 的 maps_distance 工具，利用高德支持多 origin/destination
    的能力将调用次数降到最低（理想情况下一次调用得到 N×N 矩阵）。

    Args:
        session: 由调用方传入的同一 ClientSession，避免重复建立 SSE 连接。
        origins: N 个出发点坐标，格式 "lng,lat"。
        destinations: M 个目的地坐标，格式 "lng,lat"。

    Returns:
        N×M 的时间矩阵（秒）。缺失或错误的格将填 float('inf')。
    """
    import json

    n, m = len(origins), len(destinations)
    # 初始化结果矩阵
    matrix: List[List[float]] = [[float("inf")] * m for _ in range(n)]

    # 高德 maps_distance 支持用 "|" 分隔多个 origin/destination
    origins_str = "|".join(origins)
    destinations_str = "|".join(destinations)

    result = await session.call_tool("maps_distance", {
        "origins": origins_str,
        "destinations": destinations_str,
        "type": "1",  # 1=驾车，0=直线，3=步行
    })

    for block in result.content:
        text = getattr(block, "text", None)
        if not text:
            continue
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            continue

        # 高德返回结构: {"results": [{"origin_id": "1", "dest_id": "1", "duration": "xxx", ...}]}
        raw_results = data if isinstance(data, list) else data.get("results", [])
        for item in raw_results:
            try:
                # origin_id / dest_id 是 1-based 字符串
                i = int(item.get("origin_id", 1)) - 1
                j = int(item.get("dest_id", 1)) - 1
                duration = item.get("duration")
                if duration is not None and 0 <= i < n and 0 <= j < m:
                    matrix[i][j] = float(duration)
            except (ValueError, TypeError):
                continue

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
            "duration": int(best.get("duration", 0)),
            "distance": int(best.get("distance", 0)),
            "steps": steps,
            "recommended_mode": "transit",
        }

    return fallback


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