"""
高德 POI 完整流程诊断脚本
===========================
分两步验证：
  Step 1 - _batch_geocode_rest：直接测试 REST 地理编码补充步骤，
            确认在无代理环境下能否为"空 location"POI 补充坐标。
  Step 2 - search_pois 全链路：测试完整的 MCP → geocode 流水线，
            确认最终返回的 POI 有无有效坐标。

运行：
    python test2/test_geocode_pipeline.py
"""
import sys
import os
import asyncio
import json

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 直接从 amap_client 导入内部函数用于隔离测试
from mcp_clients.amap_client import (
    amap_mcp_session,
    search_pois,
    _batch_geocode_rest,
)

CITY    = "杭州"
KEYWORD = "杭州景点"

# ── 模拟 maps_text_search 返回的无坐标 POI ──────────────────────────────────
# 格式与 amap_client.search_pois() 内部 pois 列表一致
MOCK_POIS_NO_LOCATION = [
    {"name": "西湖",         "location": "", "address": "西湖景区",    "rating": "", "type": "110200"},
    {"name": "灵隐寺",       "location": "", "address": "法云弄1号",   "rating": "", "type": "110200"},
    {"name": "雷峰塔",       "location": "", "address": "南山路15号",  "rating": "", "type": "110200"},
    {"name": "河坊街",       "location": "", "address": "清河坊历史街区", "rating": "", "type": "060100"},
    {"name": "西溪湿地公园", "location": "", "address": "天目山路518号", "rating": "", "type": "110200"},
]


async def step1_test_geocode_rest() -> bool:
    """
    Step 1: 单独测试 _batch_geocode_rest，不依赖 MCP SSE 连接。
    直接用已知 POI 验证 REST API 是否能补充坐标。
    """
    print("=" * 60)
    print("Step 1 · REST 地理编码补充（_batch_geocode_rest）")
    print("=" * 60)

    # 深拷贝，避免污染后续步骤
    import copy
    test_pois = copy.deepcopy(MOCK_POIS_NO_LOCATION)

    print(f"  输入: {len(test_pois)} 条无坐标 POI，调用 REST 地理编码...\n")

    try:
        await _batch_geocode_rest(test_pois, city=CITY)
    except Exception as e:
        print(f"  ❌ _batch_geocode_rest 抛出异常: {type(e).__name__}: {e}")
        return False

    success, fail = 0, 0
    for poi in test_pois:
        loc = poi.get("location", "")
        if loc:
            parts = loc.split(",")
            if len(parts) == 2:
                try:
                    lng, lat = float(parts[0]), float(parts[1])
                    print(f"  ✅ {poi['name']:<12} → {lng:.4f}, {lat:.4f}")
                    success += 1
                    continue
                except ValueError:
                    pass
        print(f"  ❌ {poi['name']:<12} → location 仍为空或格式错误: {repr(loc)}")
        fail += 1

    print(f"\n  汇总: 成功={success}, 失败={fail} / 共{len(test_pois)} 条")
    all_ok = fail == 0
    print(f"  Step 1 结论: {'✅ PASS' if all_ok else '⚠️  PARTIAL / FAIL'}\n")
    return all_ok


async def step2_test_full_pipeline() -> bool:
    """
    Step 2: 测试完整的 search_pois 流水线（MCP + REST geocode）。
    如果 Step 1 已通过，此步预期也能返回有坐标的 POI。
    """
    print("=" * 60)
    print("Step 2 · 完整流水线（MCP maps_text_search + REST geocode）")
    print("=" * 60)

    try:
        async with amap_mcp_session() as session:
            print(f"  ✅ SSE 连接建立，调用 search_pois(city={CITY!r}, keywords={KEYWORD!r})...")
            pois = await search_pois(session, city=CITY, keywords=KEYWORD)
    except Exception as e:
        print(f"  ❌ 连接/调用失败: {type(e).__name__}: {e}")
        return False

    print(f"  search_pois 返回 {len(pois)} 条 POI\n")

    if not pois:
        print("  ❌ 返回列表为空，无法验证")
        return False

    with_loc, without_loc = 0, 0
    for poi in pois:
        loc = poi.get("location", "")
        if loc:
            with_loc += 1
        else:
            without_loc += 1

    print(f"  有坐标: {with_loc} 条  |  无坐标: {without_loc} 条")

    # 打印前 5 条示例
    print("\n  [前 5 条结果示例]")
    for poi in pois[:5]:
        loc = poi.get("location", "(空)")
        print(f"    • {poi.get('name','?'):<15} location={loc}")

    # 判断标准：90% 以上的 POI 有有效坐标
    rate = with_loc / len(pois)
    ok = rate >= 0.9
    print(f"\n  坐标覆盖率: {rate:.1%}")
    print(f"  Step 2 结论: {'✅ PASS（≥90% 有坐标）' if ok else '❌ FAIL（<90% 有坐标）'}\n")
    return ok


async def main():
    step1_ok = await step1_test_geocode_rest()
    step2_ok = await step2_test_full_pipeline()

    print("=" * 60)
    print("总结")
    print("=" * 60)
    print(f"  Step 1 REST 地理编码: {'✅ PASS' if step1_ok else '❌ FAIL'}")
    print(f"  Step 2 完整流水线:    {'✅ PASS' if step2_ok else '❌ FAIL'}")

    if step1_ok and step2_ok:
        print("\n  🎉 所有步骤通过！POI 流水线功能正常，location=None 是 MCP 的")
        print("     已知限制，REST geocode 补充方案已生效，无需修改 amap_client.py。")
    elif step1_ok and not step2_ok:
        print("\n  ⚠️  REST geocode 单独测试通过，但完整流水线失败。")
        print("     请检查 SSE 连接稳定性，或确认 MCP maps_text_search 返回了 POI 列表。")
    elif not step1_ok:
        print("\n  ❌ REST 地理编码失败（AMAP_KEY 是否有效？网络是否可访问高德 REST API？）")
        print("     建议：curl https://restapi.amap.com/v3/geocode/geo?address=西湖&city=杭州&key=<your_key>")


if __name__ == "__main__":
    asyncio.run(main())
