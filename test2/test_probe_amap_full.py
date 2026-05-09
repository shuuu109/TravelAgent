# test2/test_probe_amap_full.py
import asyncio, sys, traceback
sys.path.insert(0, '.')
from mcp_clients.amap_client import amap_mcp_session, search_pois

async def main():
    for i in range(3):  # 跑 3 次，看是否偶发
        print(f'--- attempt {i+1} ---')
        try:
            async with amap_mcp_session() as session:
                pois = await search_pois(session, city='杭州', keywords='杭州 西湖')
                print(f'  OK: {len(pois)} pois')
                for p in pois[:3]:
                    print('   -', p.get('name'), '|', p.get('location'))
        except Exception as e:
            print(f'  ERR: {type(e).__name__}: {str(e)[:200]}')
            traceback.print_exc()

asyncio.run(main())