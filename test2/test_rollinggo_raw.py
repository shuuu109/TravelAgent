"""
RollingGo MCP 原始返回诊断脚本

目的：验证 RollingGo MCP 是否真实被调用，以及
      _query_rollinggo_single 的 landmark 搜索是否会把不同酒店
      都映射到同一条结果（导致价格高度重合）。

用法：python test2/test_rollinggo_raw.py
"""
import sys
import os
import asyncio
import json

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

SEP = "-" * 60


async def test_search_hotels_direct():
    """直接调用 search_hotels，确认 MCP 进程能正常启动并返回数据。"""
    from mcp_clients.hotel_client import search_hotels

    print(f"\n{SEP}")
    print("[TEST 1] searchHotels 直接调用（城市级，南京，3条）")
    print(SEP)

    raw = await search_hotels(
        origin_query="南京酒店",
        place="南京",
        place_type="city",
        check_in_date="2026-05-15",
        stay_nights=2,
        adults=2,
        size=3,
    )

    if hasattr(raw, "content") and raw.content:
        text = raw.content[0].text if hasattr(raw.content[0], "text") else str(raw.content[0])
        print(f"[原始返回 (前500字符)]:\n{text[:500]}")
        try:
            parsed = json.loads(text)
            _print_hotels(parsed)
        except Exception as e:
            print(f"[解析失败] {e}")
    else:
        print(f"[无内容返回] raw={raw}")


async def test_landmark_dedup():
    """
    模拟 _query_rollinggo_single 的 landmark 搜索方式，
    对 3 家不同名称的酒店各查一次，观察是否返回同一条结果（价格重合根因）。
    """
    from mcp_clients.hotel_client import hotel_mcp_session

    # 模拟高德搜到的 3 家南京酒店（名称不同，坐标接近）
    fake_amap_hotels = [
        {"name": "南京香格里拉大酒店", "location": "118.7839,32.0484"},
        {"name": "南京丽思卡尔顿酒店", "location": "118.8000,32.0600"},
        {"name": "维也纳国际酒店南京新街口店", "location": "118.7945,32.0557"},
    ]

    check_in_date = "2026-05-15"
    stay_nights   = 2

    print(f"\n{SEP}")
    print("[TEST 2] landmark 搜索去重诊断（各酒店独立查询，观察结果是否相同）")
    print(SEP)

    async with hotel_mcp_session() as session:
        for hotel in fake_amap_hotels:
            arguments = {
                "originQuery": hotel["name"],
                "place":       hotel["name"],
                "placeType":   "landmark",
                "size":        1,
                "checkInParam": {
                    "checkInDate": check_in_date,
                    "stayNights":  stay_nights,
                    "adults":      2,
                },
                # location 已移除：RollingGo MCP 不接受该参数
            }
            print(f"\n  查询: [{hotel['name']}]  坐标={hotel['location']}")
            try:
                raw = await session.call_tool("searchHotels", arguments=arguments)
                if hasattr(raw, "content") and raw.content:
                    text = (
                        raw.content[0].text
                        if hasattr(raw.content[0], "text")
                        else str(raw.content[0])
                    )
                    parsed = json.loads(text)
                    hotels_list = _extract_hotels_list(parsed)
                    if hotels_list:
                        h = hotels_list[0]
                        name  = h.get("hotelName") or h.get("name", "?")
                        price = h.get("price") or h.get("minPrice") or h.get("lowestPrice") or "N/A"
                        hid   = h.get("hotelId") or h.get("id", "?")
                        print(f"  -> RollingGo返回: [{name}]  price={price}  id={hid}")
                    else:
                        print(f"  -> 空列表 (RollingGo 无匹配结果)")
                        print(f"     原始: {text[:200]}")
                else:
                    print(f"  -> 无内容返回")
            except Exception as e:
                print(f"  -> 异常: {e}")


async def test_get_hotel_detail():
    """
    用高德周边搜索会实际返回的酒店名（包含品牌连锁和小众酒店）测试 getHotelDetail，
    验证名称匹配是否可靠。
    """
    from mcp_clients.hotel_client import hotel_mcp_session

    # 模拟高德在南京新街口附近可能搜到的酒店（覆盖高端/连锁/小众三类）
    test_hotels = [
        "南京香格里拉大酒店",       # 高端品牌
        "维也纳国际酒店南京新街口店", # 连锁经济品牌
        "汉庭酒店南京新街口店",       # 连锁经济品牌
        "南京中心大酒店",             # 城市独立酒店
    ]
    check_in  = "2026-05-15"
    check_out = "2026-05-17"

    print(f"\n{SEP}")
    print("[TEST 3] getHotelDetail 名称匹配可靠性测试（4家不同类型酒店）")
    print(SEP)

    async with hotel_mcp_session() as session:
        for hotel_name in test_hotels:
            arguments = {
                "name": hotel_name,
                "dateParam": {"checkIn": check_in, "checkOut": check_out},
                "occupancyParam": {"adults": 2, "rooms": 1},
                "localeParam": {"countryCode": "CN", "currency": "CNY"},
            }
            print(f"\n  查询: [{hotel_name}]")
            try:
                raw = await session.call_tool("getHotelDetail", arguments=arguments)
                if not (hasattr(raw, "content") and raw.content):
                    print("  -> 无内容返回")
                    continue
                text = (
                    raw.content[0].text
                    if hasattr(raw.content[0], "text")
                    else str(raw.content[0])
                )
                parsed = json.loads(text)
                if not parsed.get("success"):
                    print(f"  -> 失败: {text[:200]}")
                    continue
                returned_name = parsed.get("name", "?")
                plans = parsed.get("roomRatePlans", [])
                prices = [p.get("totalPrice") for p in plans if p.get("totalPrice")]
                min_price = min(prices) if prices else None
                matched = _name_matched(hotel_name, returned_name)
                status = "OK" if matched else "名称不匹配"
                print(f"  -> [{status}] 返回: {returned_name}")
                print(f"     最低价: {min_price} CNY  房型数: {len(plans)}")
            except Exception as e:
                print(f"  -> 异常: {e}")


def _name_matched(query: str, returned: str) -> bool:
    """简单判断查询名和返回名是否命中同一家酒店（取汉字词交集）。"""
    import re
    q_tokens = set(re.findall(r"[一-鿿]{2,}", query.replace(" ", "")))
    r_tokens = set(re.findall(r"[一-鿿]{2,}", returned.replace(" ", "")))
    return bool(q_tokens & r_tokens) if (q_tokens and r_tokens) else False


# ── 辅助函数 ────────────────────────────────────────────────────

def _extract_hotels_list(parsed) -> list:
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        for key in ("hotelInformationList", "hotels", "data"):
            if key in parsed and isinstance(parsed[key], list):
                return parsed[key]
    return []


def _print_hotels(parsed):
    hotels = _extract_hotels_list(parsed)
    if not hotels:
        print(f"[无酒店列表，完整结构]: {json.dumps(parsed, ensure_ascii=False, indent=2)[:800]}")
        return
    print(f"[共 {len(hotels)} 家酒店]")
    for h in hotels:
        name  = h.get("hotelName") or h.get("name", "?")
        price = h.get("price") or h.get("minPrice") or h.get("lowestPrice") or h.get("minRoomPrice") or "N/A"
        hid   = h.get("hotelId") or h.get("id", "?")
        star  = h.get("star") or h.get("starLevel", "?")
        print(f"  [{name}]  price={price}  star={star}  id={hid}")


# ── 主入口 ──────────────────────────────────────────────────────

async def main():
    print("RollingGo MCP 诊断开始")
    print("=" * 60)

    try:
        await test_search_hotels_direct()
    except Exception as e:
        print(f"[TEST 1 失败] {e}")

    try:
        await test_landmark_dedup()
    except Exception as e:
        print(f"[TEST 2 失败] {e}")

    try:
        await test_get_hotel_detail()
    except Exception as e:
        print(f"[TEST 3 失败] {e}")

    print(f"\n{'=' * 60}")
    print("诊断完成")


if __name__ == "__main__":
    asyncio.run(main())
