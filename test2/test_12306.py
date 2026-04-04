import asyncio
from mcp_clients.train_client import train_client

async def main():
    print("正在唤醒 12306 MCP 服务，查询车票中...")
    # 注意：日期最好传明后天的日期，因为 12306 只能查未来 15 天的
    date = "2026-04-05"
    from_city = "南京"
    to_city = "上海"

    print(f"查询 {date} 从 {from_city} 到 {to_city} 的车次：")

    result = await train_client.query_tickets(date, from_city, to_city)

    print("\n======= 查询结果 =======")
    print(result)
    print("========================")

if __name__ == "__main__":
    asyncio.run(main())