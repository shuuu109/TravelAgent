"""
_batch_geocode_rest 根因诊断脚本
===================================
逐步定位为什么批量地理编码只有 1/5 成功：
  实验A - 直接打印 REST API 原始 JSON 响应
  实验B - 对比 3 种地址格式的命中率（name+addr / 纯name / 纯addr）
  实验C - 逐条单独调用 vs 批量调用，确认是"API不返回占位符"还是"格式问题"

运行：
    python test2/test_geocode_debug.py
"""
import sys, os, asyncio, json, httpx

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from mcp_clients.amap_client import AMAP_KEY

GEO_URL = "https://restapi.amap.com/v3/geocode/geo"
CITY    = "杭州"

# 与 test_geocode_pipeline.py 相同的测试 POI
TEST_POIS = [
    {"name": "西湖",         "address": "西湖景区"},
    {"name": "灵隐寺",       "address": "法云弄1号"},
    {"name": "雷峰塔",       "address": "南山路15号"},
    {"name": "河坊街",       "address": "清河坊历史街区"},
    {"name": "西溪湿地公园", "address": "天目山路518号"},
]


async def experiment_a_raw_response():
    """实验A: 打印批量 REST 响应原文，看 geocodes 数组长度与内容"""
    print("=" * 60)
    print("实验A · 原始 REST API 响应（name+address 批量）")
    print("=" * 60)

    addresses = [f"{p['name']} {p['address']}" for p in TEST_POIS]
    joined = "|".join(addresses)
    print(f"  请求 address 参数:\n    {joined}\n")

    async with httpx.AsyncClient(timeout=15.0, trust_env=False) as client:
        resp = await client.get(GEO_URL, params={
            "address": joined,
            "city":    CITY,
            "key":     AMAP_KEY,
            "output":  "json",
        })
    data = resp.json()

    print(f"  响应 status: {data.get('status')}  info: {data.get('info')}")
    geocodes = data.get("geocodes", [])
    print(f"  geocodes 数组长度: {len(geocodes)}（输入 {len(TEST_POIS)} 条）")
    print()

    # 核心问题：API 是否为失败地址返回占位符？
    if len(geocodes) != len(TEST_POIS):
        print("  ⚠️  geocodes 数量 ≠ 输入数量！")
        print("      → 高德不为失败地址返回占位符，index 对齐失效")
        print("      → 这是 _batch_geocode_rest 的 BUG 根源\n")
    else:
        print("  ✅ geocodes 数量与输入相等，index 对齐正常\n")

    for i, gc in enumerate(geocodes):
        loc  = gc.get("location", "")
        name = TEST_POIS[i]["name"] if i < len(TEST_POIS) else "?"
        formatted_address = gc.get("formatted_address", "")
        print(f"    [{i}] 预期={name:<12} location={loc!r:<30} formatted={formatted_address[:40]}")

    print()
    return len(geocodes), geocodes


async def experiment_b_format_comparison():
    """实验B: 对比 3 种地址格式，找出哪种命中率最高"""
    print("=" * 60)
    print("实验B · 3 种地址格式命中率对比（逐条单独调用）")
    print("=" * 60)

    formats = {
        "name+address": [f"{p['name']} {p['address']}" for p in TEST_POIS],
        "纯 name":      [p["name"]    for p in TEST_POIS],
        "纯 address":   [p["address"] for p in TEST_POIS],
    }

    async with httpx.AsyncClient(timeout=15.0, trust_env=False) as client:
        for fmt_name, addr_list in formats.items():
            hits = 0
            print(f"\n  格式: {fmt_name}")
            for poi, addr in zip(TEST_POIS, addr_list):
                resp = await client.get(GEO_URL, params={
                    "address": addr,
                    "city":    CITY,
                    "key":     AMAP_KEY,
                    "output":  "json",
                })
                data = resp.json()
                gcs  = data.get("geocodes", [])
                loc  = gcs[0].get("location", "") if gcs else ""
                ok   = "✅" if loc else "❌"
                hits += bool(loc)
                print(f"    {ok} {poi['name']:<12} addr={addr!r:<30} → location={loc!r}")
            print(f"  命中率: {hits}/{len(TEST_POIS)}")

    print()


async def experiment_c_batch_vs_single():
    """实验C: 确认批量 vs 单独调用的差异（当输入 5 条时 geocodes 返回几条）"""
    print("=" * 60)
    print("实验C · 批量 vs 单独调用对比（纯 name 格式）")
    print("=" * 60)

    names = [p["name"] for p in TEST_POIS]

    # 批量
    async with httpx.AsyncClient(timeout=15.0, trust_env=False) as client:
        resp = await client.get(GEO_URL, params={
            "address": "|".join(names),
            "city":    CITY,
            "key":     AMAP_KEY,
            "output":  "json",
        })
    batch_data = resp.json()
    batch_gcs  = batch_data.get("geocodes", [])
    print(f"  批量调用 geocodes 长度: {len(batch_gcs)}")
    for i, gc in enumerate(batch_gcs):
        print(f"    [{i}] location={gc.get('location','')!r}")

    print()

    # 逐条
    print("  逐条调用:")
    async with httpx.AsyncClient(timeout=15.0, trust_env=False) as client:
        for poi in TEST_POIS:
            resp = await client.get(GEO_URL, params={
                "address": poi["name"],
                "city":    CITY,
                "key":     AMAP_KEY,
                "output":  "json",
            })
            gcs = resp.json().get("geocodes", [])
            loc = gcs[0].get("location", "") if gcs else ""
            print(f"    {poi['name']:<12} → {loc!r}")

    print()


async def main():
    n, geocodes = await experiment_a_raw_response()
    await experiment_b_format_comparison()
    await experiment_c_batch_vs_single()

    print("=" * 60)
    print("诊断总结")
    print("=" * 60)
    if n != len(TEST_POIS):
        print(f"  根因: 高德批量地理编码接口在 {len(TEST_POIS)} 个输入中")
        print(f"        只返回了 {n} 个 geocodes（不补空占位符）")
        print(f"        导致 _batch_geocode_rest 的 idx 对齐全部偏移")
        print()
        print("  修复方向（供参考）：")
        print("    1. 改用逐条调用（最稳，命中率高，但调用次数多）")
        print("    2. 改用 maps_geo MCP 工具（POI 名称搜索，更准确）")
        print("    3. 批量调用后用 formatted_address 字段反向对齐")
    else:
        print(f"  geocodes 数量与输入一致，问题在于地址格式。")
        print(f"  请参考实验B的命中率，使用最优地址格式。")


if __name__ == "__main__":
    asyncio.run(main())
