"""
MCP (Model Context Protocol) 客户端集合
- train_client: 12306 火车票查询（STDIO）
- amap_client: 高德地图服务（SSE）

注：途牛 CLI 客户端（hotel/flight/...）请直接 from mcp_clients.tuniu_client import ...
"""
from .train_client import TrainTicketClient, train_client

from .amap_client import amap_mcp_session

__all__ = [
    "TrainTicketClient", "train_client",
    "amap_mcp_session",
]
