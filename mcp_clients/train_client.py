import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class TrainTicketClient:
    def __init__(self):
        # 配置启动参数，因为你在 grad_pro 安装了，所以可以直接用命令 "mcp-server-12306"
        self.server_params = StdioServerParameters(
            command="mcp-server-12306",
            args=[]
        )

    async def _call_mcp_tool(self, tool_name: str, arguments: dict):
        """核心底层方法：连接 MCP Server 并调用工具"""
        try:
            async with stdio_client(self.server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    # 初始化连接
                    await session.initialize()
                    # 调用指定的工具
                    result = await session.call_tool(tool_name, arguments)
                    return result
        except Exception as e:
            return f"调用 12306 MCP 服务失败: {str(e)}"

    async def query_tickets(self, date: str, from_station: str, to_station: str) -> str:
        """
        查询火车票
        :param date: 出发日期，格式 YYYY-MM-DD (例如 "2026-04-10")
        :param from_station: 出发站 (例如 "南京")
        :param to_station: 到达站 (例如 "上海")
        """
        arguments = {
            "train_date": date,
            "from_station": from_station,
            "to_station": to_station
        }
        result = await self._call_mcp_tool("query-tickets", arguments)
        if hasattr(result, 'content') and len(result.content) > 0:
            return result.content[0].text
        return str(result)

    async def query_ticket_price(
        self,
        date: str,
        from_station: str,
        to_station: str,
        train_code: str = "",
        purpose_codes: str = "ADULT"
    ) -> str:
        """
        查询火车票价（含所有席别：无座/硬座/硬卧/软卧/高级软卧/二等座/一等座/商务座/特等座/动卧）
        :param date: 出发日期，格式 YYYY-MM-DD (例如 "2026-04-10")
        :param from_station: 出发站 (例如 "南京")
        :param to_station: 到达站 (例如 "上海")
        :param train_code: 车次号（可选，留空返回所有车次）
        :param purpose_codes: 乘客类型，ADULT=成人，0X=学生（默认 ADULT）
        """
        arguments = {
            "train_date": date,
            "from_station": from_station,
            "to_station": to_station,
            "purpose_codes": purpose_codes
        }
        if train_code:
            arguments["train_code"] = train_code
        result = await self._call_mcp_tool("query-ticket-price", arguments)
        if hasattr(result, 'content') and len(result.content) > 0:
            return result.content[0].text
        return str(result)

# 实例化一个全局单例供后续 Tool 或 Agent 使用
train_client = TrainTicketClient()