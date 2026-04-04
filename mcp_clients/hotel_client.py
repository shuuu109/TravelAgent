"""
RollingGo 酒店 MCP 客户端

通过 stdio 启动 rollinggo-mcp，提供三个工具：
  - searchHotels       搜索酒店列表
  - getHotelDetail     获取酒店房型/价格/政策
  - getHotelSearchTags 获取标签元数据（可缓存）
"""

import os
from contextlib import asynccontextmanager
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession
from langchain_mcp_adapters.tools import load_mcp_tools

from config import ROLLINGGO_MCP_CONFIG


# ────────────────────────────────────────────────
# 底层：原始 MCP Session（供底层直接调用使用）
# ────────────────────────────────────────────────

@asynccontextmanager
async def hotel_mcp_session():
    """
    启动 rollinggo-mcp 进程，建立 stdio MCP 会话。

    用法：
        async with hotel_mcp_session() as session:
            tools = await session.list_tools()
    """
    api_key = ROLLINGGO_MCP_CONFIG.get("ROLLINGGO_API_KEY", "")
    if not api_key or api_key == "your_rollinggo_api_key_here":
        raise ValueError(
            "请先在 config.py 的 ROLLINGGO_MCP_CONFIG 中填写真实的 ROLLINGGO_API_KEY"
        )

    command = ROLLINGGO_MCP_CONFIG["command"]   # e.g. "rollinggo-mcp" or "npx"
    args = ROLLINGGO_MCP_CONFIG.get("args", []) # e.g. [] or ["-y", "rollinggo-mcp"]

    server_params = StdioServerParameters(
        command=command,
        args=args,
        env={
            **os.environ,                          # 继承系统环境变量（PATH 等）
            "ROLLINGGO_API_KEY": api_key,
        },
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


# ────────────────────────────────────────────────
# 高层：加载为 LangChain Tool 列表（供 Agent 使用）
# ────────────────────────────────────────────────

async def get_hotel_tools():
    """
    启动 MCP server，将三个工具包装为 LangChain BaseTool 列表返回。

    返回：
        list[BaseTool]  包含 searchHotels / getHotelDetail / getHotelSearchTags
    """
    api_key = ROLLINGGO_MCP_CONFIG.get("ROLLINGGO_API_KEY", "")
    if not api_key or api_key == "your_rollinggo_api_key_here":
        raise ValueError(
            "请先在 config.py 的 ROLLINGGO_MCP_CONFIG 中填写真实的 ROLLINGGO_API_KEY"
        )

    server_params = StdioServerParameters(
        command=ROLLINGGO_MCP_CONFIG["command"],
        args=ROLLINGGO_MCP_CONFIG.get("args", []),
        env={
            **os.environ,
            "ROLLINGGO_API_KEY": api_key,
        },
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            return tools


# ────────────────────────────────────────────────
# 便捷函数：直接调用 searchHotels
# ────────────────────────────────────────────────

async def search_hotels(
    origin_query: str,
    place: str,
    place_type: str = "city",
    check_in_date: str | None = None,
    stay_nights: int = 1,
    adults: int = 1,
    star_min: int | None = None,
    star_max: int | None = None,
    price_min: float | None = None,
    price_max: float | None = None,
    hotel_brands: list[str] | None = None,
    size: int | None = None,
) -> dict:
    """
    封装 searchHotels 调用，返回原始 JSON 结果。

    参数：
        origin_query  用户原始搜索词，如 "北京朝阳区酒店"
        place         目的地标识，如城市名 "北京"
        place_type    地点类型，枚举：city / district / landmark / airport 等
        check_in_date 入住日期，格式 "YYYY-MM-DD"
        stay_nights   入住晚数
        adults        成人数量
        star_min      最低星级（2-5）
        star_max      最高星级（2-5）
        price_min     最低价格（CNY）
        price_max     最高价格（CNY）
        hotel_brands  指定品牌列表，如 ["汉庭", "如家"]
        size          返回数量（1-20）

    返回：
        dict  酒店列表原始数据
    """
    _size = size or ROLLINGGO_MCP_CONFIG.get("default_size", 5)

    # 构造请求参数
    arguments: dict = {
        "originQuery": origin_query,
        "place": place,
        "placeType": place_type,
        "size": _size,
    }

    if check_in_date:
        arguments["checkInParam"] = {
            "checkInDate": check_in_date,
            "stayNights": stay_nights,
            "adults": adults,
        }

    filter_options: dict = {}
    if star_min is not None:
        filter_options["starMin"] = star_min
    if star_max is not None:
        filter_options["starMax"] = star_max
    if filter_options:
        arguments["filterOptions"] = filter_options

    hotel_tags: dict = {}
    if price_min is not None:
        hotel_tags["priceMin"] = price_min
    if price_max is not None:
        hotel_tags["priceMax"] = price_max
    if hotel_brands:
        hotel_tags["brands"] = hotel_brands
    if hotel_tags:
        arguments["hotelTags"] = hotel_tags

    async with hotel_mcp_session() as session:
        result = await session.call_tool("searchHotels", arguments=arguments)
        return result


async def get_hotel_detail(
    hotel_id: str | None = None,
    hotel_name: str | None = None,
    check_in: str | None = None,
    check_out: str | None = None,
    adults: int = 1,
    rooms: int = 1,
    currency: str | None = None,
) -> dict:
    """
    封装 getHotelDetail 调用，返回实时房型/价格/政策。

    参数：
        hotel_id    酒店 ID（与 hotel_name 二选一）
        hotel_name  酒店名称（与 hotel_id 二选一）
        check_in    入住日期 "YYYY-MM-DD"
        check_out   离店日期 "YYYY-MM-DD"
        adults      成人数量
        rooms       房间数量
        currency    货币代码，默认 CNY

    返回：
        dict  房型详情原始数据
    """
    if not hotel_id and not hotel_name:
        raise ValueError("hotel_id 和 hotel_name 至少提供一个")

    _currency = currency or ROLLINGGO_MCP_CONFIG.get("default_currency", "CNY")
    _country = ROLLINGGO_MCP_CONFIG.get("default_country", "CN")

    arguments: dict = {
        "localeParam": {
            "countryCode": _country,
            "currency": _currency,
        }
    }

    if hotel_id:
        arguments["hotelId"] = hotel_id
    if hotel_name:
        arguments["name"] = hotel_name

    if check_in and check_out:
        arguments["dateParam"] = {
            "checkIn": check_in,
            "checkOut": check_out,
        }

    if adults or rooms:
        arguments["occupancyParam"] = {
            "adults": adults,
            "rooms": rooms,
        }

    async with hotel_mcp_session() as session:
        result = await session.call_tool("getHotelDetail", arguments=arguments)
        return result
