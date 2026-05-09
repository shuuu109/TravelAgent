# test2/test_probe_mcp_sse.py
import asyncio, traceback
from mcp.client.sse import sse_client
from mcp.client.session import ClientSession

URL = 'https://mcp.amap.com/sse?key=1dd13742a147224131022165e14d6d55'

async def main():
    try:
        # mcp 库版本不同，sse_client 支持的参数也不同；这里两个都试
        try:
            ctx = sse_client(URL, timeout=30, sse_read_timeout=60*5)
        except TypeError:
            ctx = sse_client(URL)

        async with ctx as streams:
            print('[1] sse_client connected')
            async with ClientSession(streams[0], streams[1]) as session:
                print('[2] ClientSession created, initializing...')
                await session.initialize()
                print('[3] initialize ok')
                tools = await session.list_tools()
                print(f'[4] tools: {[t.name for t in tools.tools][:8]}')
                # 实际调一次 maps_text_search
                result = await session.call_tool(
                    'maps_text_search',
                    {'keywords': '杭州 西湖', 'city': '杭州'}
                )
                print(f'[5] call_tool ok, blocks: {len(result.content)}')
                for b in result.content[:1]:
                    text = getattr(b, 'text', '')
                    print('     sample:', text[:300])
    except Exception as e:
        print(f'ERR: {type(e).__name__}: {str(e)[:300]}')
        traceback.print_exc()

asyncio.run(main())