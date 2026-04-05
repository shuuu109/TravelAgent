"""
maps_geo MCP 工具坐标补充测试
==============================
验证用高德 MCP 的 maps_geo 工具，能否比 REST 地理编码 API 更可靠地
为 maps_text_search 返回的无坐标 POI 补充经纬度。

测试分三组：
  组1 - 景点名称（灵隐寺 / 雷峰塔 / 西溪湿地公园 …）
  组2 - 模拟 MCP 真实返回的 POI（城市阳台 / 钱江新城灯光秀 …）
  组3 - maps_geo 工具详情（打印参数 schema 供后续集成参考）

运行（需关闭网络代理）：
    python test2/test_maps_geo_mcp.py
"""
import sys, os, asyncio, json

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from mcp_clients.amap_client import amap_mcp_session

CITY = "杭州"

# 组1：经典景点
GROUP1 = [
    {"name": "西湖"},
    {"name": "灵隐寺"},
    {"name": "雷峰塔"},
    {"name": "河坊街"},
    {"name": "西溪湿地公园"},
]

# 组2：MCP maps_text_search 真实返回的 POI 名
GROUP2 = [
    {"name": "城市阳台"},
    {"name": "钱江新城灯光秀"},
    {"name": "CBD公园"},
    {"name": "杭州Do都城"},
    {"name": "城市阳台江堤步道"},
]


async def call_maps_geo(session, address: str, city: str) -> str:
    """调用 maps_geo 工具，返回 location 字符串，失败返回空串。"""
    try:
        result = await session.call_tool("maps_geo", {
            "address": address,
            "city": city,
        })
        for block in result.content:
            text = getattr(block, "text", None)
            if not text:
                continue
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                continue

            # 高德 geocode 返回结构: {"geocodes": [{"location": "lng,lat", ...}]}
            geocodes = data.get("geocodes", [])
            if geocodes:
                loc = geocodes[0].get("location", "")
                if loc:
                    return loc
    except Exception as e:
        return f"[ERROR] {e}"
    return ""


async def test_group(session, label: str, pois: list):
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    success = 0
    for poi in pois:
        loc = await call_maps_geo(session, poi["name"], CITY)
        ok = bool(loc and not loc.startswith("[ERROR]"))
        icon = "✅" if ok else "❌"
        if ok:
            success += 1
        print(f"  {icon} {poi['name']:<16} → {loc[:35] if loc else '(空)'}")
    print(f"\n  命中率: {success}/{len(pois)}")
    return success


async def inspect_tool_schema(session):
    """打印 maps_geo 工具的参数 schema，供集成时参考。"""
    print(f"\n{'='*55}")
    print("  maps_geo 工具 Schema（供集成参考）")
    print(f"{'='*55}")
    try:
        tools = await session.list_tools()
        for tool in tools.tools:
            if tool.name == "maps_geo":
                print(f"  name        : {tool.name}")
                print(f"  description : {tool.description[:120]}")
                schema = tool.inputSchema
                props  = schema.get("properties", {})
                req    = schema.get("required", [])
                print(f"  参数:")
                for k, v in props.items():
                    r = "必填" if k in req else "可选"
                    desc = v.get("description", "")[:60]
                    print(f"    {k:<12} [{r}]  {desc}")
                return
        print("  ⚠️  未找到 maps_geo 工具，打印所有可用工具名：")
        for tool in tools.tools:
            print(f"    - {tool.name}")
    except Exception as e:
        print(f"  ❌ 获取工具列表失败: {e}")


async def main():
    print("[maps_geo MCP 坐标补充测试]  city=", CITY)
    print("建立 SSE 连接中...", end=" ", flush=True)

    try:
        async with amap_mcp_session() as session:
            print("✅\n")

            # 先打印工具 schema
            await inspect_tool_schema(session)

            # 测试两组 POI
            s1 = await test_group(session, "组1 · 经典景点", GROUP1)
            s2 = await test_group(session, "组2 · MCP 真实返回 POI", GROUP2)

            total = s1 + s2
            total_n = len(GROUP1) + len(GROUP2)
            print(f"\n{'='*55}")
            print(f"  总命中率: {total}/{total_n}  ({total/total_n:.0%})")

            if total / total_n >= 0.8:
                print("\n  ✅ maps_geo 可用！建议在 search_pois 中")
                print("     优先用 maps_geo 替代 REST 地理编码补坐标。")
            else:
                print("\n  ⚠️  maps_geo 命中率仍不理想，需换其他方案。")
                print("     备选：用 /v3/place/text POI 搜索 REST API（直接返回坐标）。")

    except Exception as e:
        print(f"❌ 连接失败: {type(e).__name__}: {e}")


if __name__ == "__main__":
    asyncio.run(main())
