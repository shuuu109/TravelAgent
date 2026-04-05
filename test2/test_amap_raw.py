"""
高德 MCP 原始响应诊断脚本
==========================
目的：打印 maps_text_search 工具调用的真实返回结构，
      帮助定位 location 字段为何导致"有效 0 条"。

运行：
    python test2/test_amap_raw.py
"""
import sys
import os
import asyncio
import json

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from mcp_clients.amap_client import amap_mcp_session, AMAP_KEY

CITY    = "杭州"
KEYWORD = "杭州景点"


async def diagnose():
    print(f"[高德 MCP 诊断] key={AMAP_KEY[:8]}...  city={CITY}  keyword={KEYWORD}\n")

    try:
        async with amap_mcp_session() as session:
            print("✅ SSE 连接建立成功，开始调用 maps_text_search ...\n")

            result = await session.call_tool("maps_text_search", {
                "keywords": KEYWORD,
                "city": CITY,
            })

    except Exception as e:
        print(f"❌ 连接/调用失败: {type(e).__name__}: {e}")
        return

    # ── 遍历所有 content block ─────────────────────────────────────────────
    print(f"[result.content 共 {len(result.content)} 个 block]")
    for idx, block in enumerate(result.content):
        text = getattr(block, "text", None)
        print(f"\n── Block[{idx}]  type={type(block).__name__} ──")

        if text is None:
            print("  text=None（非文本 block）")
            continue

        print(f"  text 长度: {len(text)} 字符")
        print(f"  text 前 300 字符:\n    {text[:300]}")

        # 尝试 JSON 解析
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            print(f"  ⚠️  JSON 解析失败: {e}（这是纯文本/Markdown 返回，需调整解析逻辑）")
            continue

        # 找 POI 列表
        raw_list = data if isinstance(data, list) else data.get("pois", [])
        print(f"  JSON 解析成功，raw_list 长度={len(raw_list)}")

        if raw_list:
            first = raw_list[0]
            print(f"\n  [第一条 POI 的完整字段]")
            for k, v in first.items():
                print(f"    {k!r:25s}: {repr(v)[:120]}")

            # 重点检查 location 字段
            loc = first.get("location")
            print(f"\n  [location 诊断]")
            print(f"    location 原始值   : {repr(loc)}")
            print(f"    location 类型     : {type(loc).__name__}")
            if isinstance(loc, str):
                parts = loc.split(",")
                print(f"    split(',') 结果  : {parts}  (期望 2 段)")
                if len(parts) == 2:
                    try:
                        lng, lat = float(parts[0].strip()), float(parts[1].strip())
                        print(f"    ✅ 解析成功: lng={lng}, lat={lat}")
                    except ValueError as e:
                        print(f"    ❌ float 转换失败: {e}")
                else:
                    print("    ❌ 不是 'lng,lat' 格式")
            elif isinstance(loc, dict):
                print(f"    ⚠️  location 是 dict: {loc}  （需改用 loc['lng']/loc['lat']）")
            elif loc is None or loc == "":
                print("    ❌ location 为空/None（高德 MCP 未返回坐标）")

        print(f"\n  [顶层 data 的所有 key]: {list(data.keys()) if isinstance(data, dict) else '(list)'}")


if __name__ == "__main__":
    asyncio.run(diagnose())
