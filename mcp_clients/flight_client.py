"""
航班 MCP 客户端（Streamable HTTP 传输）

对接远程航班 MCP Server，支持以下工具：
- searchFlightItineraries  航班行程价格查询（主力）
- searchFlightsByDepArr    OD对航班查询
- searchFlightsByNumber    航班实时动态
- searchFlightsTransferinfo 航班中转方案
- flightHappinessIndex     乘机舒适度
- getRealtimeLocationByAnum 飞机实时定位
- getFutureWeatherByAirport 机场未来天气
"""

import os
from contextlib import asynccontextmanager
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.session import ClientSession

# ── 配置 ─────────────────────────────────────────────────────────────────────
# 优先读取环境变量，其次读 config.py，方便生产与开发两种场景
# 认证方式：API Key 已内嵌于 URL 的 ?api_key= 参数中，无需额外请求头
try:
    from config import FLIGHT_MCP_CONFIG as _cfg
    _default_url = _cfg.get("url", "https://<mcp-server-host>/mcp")
except ImportError:
    _default_url = "https://<mcp-server-host>/mcp"

FLIGHT_MCP_URL = os.getenv("FLIGHT_MCP_URL", _default_url)


@asynccontextmanager
async def flight_mcp_session():
    """
    建立与航班 MCP Server 的 Streamable HTTP 连接，并初始化协议。

    认证：API Key 已内嵌于 FLIGHT_MCP_URL 的 ?api_key= 参数中。

    用法（async with）：
        async with flight_mcp_session() as session:
            result = await session.call_tool("searchFlightItineraries", {...})
    """
    async with streamablehttp_client(FLIGHT_MCP_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


# ── 高层封装类（供 TransportAgent 直接调用）────────────────────────────────────
class FlightMCPClient:
    """对航班 MCP 工具的高层封装，屏蔽底层协议细节。"""

    async def _call(self, tool_name: str, arguments: dict) -> str:
        """底层：建立连接 → 调用工具 → 提取文本 → 关闭连接。"""
        try:
            async with flight_mcp_session() as session:
                result = await session.call_tool(tool_name, arguments)
                return self._extract_text(result)
        except Exception as e:
            return f"[航班MCP调用失败] tool={tool_name}, error={e}"

    @staticmethod
    def _extract_text(result) -> str:
        """从 MCP CallToolResult 中提取纯文本。"""
        if hasattr(result, "content") and result.content:
            return result.content[0].text
        return str(result)

    # ── 主力工具：行程价格查询 ────────────────────────────────────────────────

    async def search_flight_itineraries(
        self,
        dep_city_code: str,
        dep_date: str,
        arr_city_code: str,
    ) -> str:
        """
        查询航班行程及最低价（主力接口）。

        :param dep_city_code: 出发城市三字码，如 "SHA"、"PEK"、"SZX"
        :param dep_date:      出发日期，格式 YYYY-MM-DD
        :param arr_city_code: 到达城市三字码，如 "BJS"、"CTU"
        :return: 包含最低价/最短耗时/推荐方案的文本
        """
        return await self._call(
            "searchFlightItineraries",
            {
                "depCityCode": dep_city_code,
                "depDate": dep_date,
                "arrCityCode": arr_city_code,
            },
        )

    # ── 其他工具（按需使用）──────────────────────────────────────────────────

    async def search_flights_by_dep_arr(self, dep: str, arr: str) -> str:
        """OD 对航班查询，返回当天所有航班列表。"""
        return await self._call("searchFlightsByDepArr", {"dep": dep, "arr": arr})

    async def search_flights_by_number(self, fnum: str, date: str) -> str:
        """根据航班号+日期查询实时动态。"""
        return await self._call("searchFlightsByNumber", {"fnum": fnum, "date": date})

    async def search_transfer_info(
        self, dep_city: str, arr_city: str, dep_date: str
    ) -> str:
        """查询中转方案（未来 48 小时内）。"""
        return await self._call(
            "searchFlightsTransferinfo",
            {"depcity": dep_city, "arrcity": arr_city, "depdate": dep_date},
        )

    async def get_happiness_index(
        self, fnum: str, date: str, dep: str, arr: str
    ) -> str:
        """查询乘机舒适度（机上设施、餐食等）。"""
        return await self._call(
            "flightHappinessIndex",
            {"fnum": fnum, "date": date, "dep": dep, "arr": arr},
        )

    async def get_realtime_location(self, anum: str) -> str:
        """根据飞机注册号获取实时定位。"""
        return await self._call("getRealtimeLocationByAnum", {"anum": anum})

    async def get_airport_weather(self, airport: str) -> str:
        """查询机场未来天气。"""
        return await self._call("getFutureWeatherByAirport", {"airport": airport})

    # ── 兼容旧接口（避免 TransportAgent 报错）───────────────────────────────

    async def query_tickets(
        self, date: str, from_city: str, to_city: str
    ) -> str:
        """
        旧接口兼容层：城市名 → 自动映射三字码 → 调用 searchFlightItineraries。

        注意：城市名到三字码的映射为常见城市简表，
        如遇罕见城市请直接调用 search_flight_itineraries()。
        """
        dep_code = CITY_TO_IATA.get(from_city, from_city)
        arr_code = CITY_TO_IATA.get(to_city, to_city)
        return await self.search_flight_itineraries(dep_code, date, arr_code)


# ── 城市名 → IATA 三字码映射表 ────────────────────────────────────────────────
CITY_TO_IATA: dict[str, str] = {
    "北京": "BJS", "上海": "SHA", "广州": "CAN", "深圳": "SZX",
    "成都": "CTU", "重庆": "CKG", "杭州": "HGH", "武汉": "WUH",
    "西安": "SIA", "南京": "NKG", "天津": "TSN", "青岛": "TAO",
    "厦门": "XMN", "昆明": "KMG", "三亚": "SYX", "哈尔滨": "HRB",
    "长春": "CGQ", "沈阳": "SHE", "大连": "DLC", "郑州": "CGO",
    "长沙": "CSX", "合肥": "HFE", "福州": "FOC", "南宁": "NNG",
    "贵阳": "KWE", "兰州": "LHW", "乌鲁木齐": "URC", "呼和浩特": "HET",
    "南昌": "KHN", "石家庄": "SJW", "太原": "TYN", "济南": "TNA",
    "海口": "HAK", "拉萨": "LXA", "银川": "INC", "西宁": "XNN",
}


# ── 全局单例 ──────────────────────────────────────────────────────────────────
flight_client = FlightMCPClient()
