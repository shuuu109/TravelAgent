
import sys
import os

# 添加项目根目录到 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from mcp_clients.hotel_client import search_hotels

async def test():
    result = await search_hotels(
        origin_query='北京朝阳区酒店',
        place='北京',
        place_type='city',
        check_in_date='2026-04-10',
        stay_nights=2,
        adults=1,
        size=3
    )
    print(result)

asyncio.run(test())

