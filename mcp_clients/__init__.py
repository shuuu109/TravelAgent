"""
MCP (Model Context Protocol) 客户端集合
- train_client: 12306 火车票查询（STDIO）
- flight_client: 航班行程/价格/实时动态查询（Streamable HTTP）
- amap_client: 高德地图服务（SSE）
"""
from .train_client import TrainTicketClient, train_client
from .flight_client import FlightMCPClient, flight_client, flight_mcp_session
from .amap_client import amap_mcp_session

__all__ = [
    "TrainTicketClient", "train_client",
    "FlightMCPClient", "flight_client", "flight_mcp_session",
    "amap_mcp_session",
]
