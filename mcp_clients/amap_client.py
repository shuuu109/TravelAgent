import os
from contextlib import asynccontextmanager
from mcp.client.sse import sse_client
from mcp.client.session import ClientSession

# 从环境变量读取你申请的高德 Web 服务 Key
AMAP_KEY = os.getenv("AMAP_KEY", "1dd13742a147224131022165e14d6d55")

@asynccontextmanager
async def amap_mcp_session():
    """
    管理与高德官方 MCP Server 的 SSE 长连接生命周期
    
    Note: 当前针对高德MCP服务的连接和工具调用需要进一步测试。
    如果无法连接，请确保：
    1. AMAP_KEY 环境变量或config中有有效的API Key
    2. 网络连接正常
    3. 高德MCP服务端口可访问
    """
    try:
        # 高德官方的 SSE 接入点
        url = f"https://mcp.amap.com/sse?key={AMAP_KEY}"
        
        # 建立与官方 MCP Server 的连接
        async with sse_client(url) as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                # 初始化 MCP 协议
                await session.initialize()
                yield session
    except Exception as e:
        print(f"⚠️  无法连接到高德MCP服务: {e}")
        # 抛出异常让调用方处理
        raise

# 备用方案：如果你更倾向于本地运行 (需要安装 Node.js)
# from mcp.client.stdio import stdio_client, StdioServerParameters
# @asynccontextmanager
# async def amap_mcp_session_local():
#     server_params = StdioServerParameters(
#         command="npx",
#         args=["-y", "@amap/amap-maps-mcp-server"],
#         env={"AMAP_MAPS_API_KEY": AMAP_KEY, "PATH": os.environ["PATH"]}
#     )
#     async with stdio_client(server_params) as (read, write):
#         async with ClientSession(read, write) as session:
#             await session.initialize()
#             yield session