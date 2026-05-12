#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
航班 MCP 专项测试
测试目标：验证航班 MCP 能否正确返回航班列表及价格信息
"""

import sys
import os
import asyncio
from datetime import datetime, timedelta

# 添加项目根目录到 sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from mcp_clients.flight_client import flight_client, FlightMCPClient, flight_mcp_session


async def test_list_tools():
    """测试：列出航班 MCP 所有可用工具"""
    print("\n" + "=" * 70)
    print("测试 1：列出航班 MCP 可用工具")
    print("=" * 70)
    
    try:
        async with flight_mcp_session() as session:
            resp = await session.list_tools()
            tools = [t.name for t in resp.tools]
            print(f"✓ 成功获取工具列表，共 {len(tools)} 个工具:")
            for t in tools:
                print(f"  - {t}")
            return tools
    except Exception as e:
        print(f"✗ 获取工具列表失败: {e}")
        return []


async def test_search_flight_itineraries():
    """测试：主力接口 searchFlightItineraries - 查询航班行程及价格"""
    print("\n" + "=" * 70)
    print("测试 2：searchFlightItineraries - 航班行程价格查询")
    print("=" * 70)
    
    # 使用未来几天的日期作为测试日期
    test_date = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
    
    test_cases = [
        ("SHA", test_date, "PEK", "上海 → 北京"),
        ("BJS", test_date, "CAN", "北京 → 广州"),
        ("SZX", test_date, "CTU", "深圳 → 成都"),
    ]
    
    for dep_code, date, arr_code, route_name in test_cases:
        print(f"\n--- 测试路线: {route_name} ({dep_code} → {arr_code}) ---")
        try:
            result = await flight_client.search_flight_itineraries(
                dep_city_code=dep_code,
                dep_date=date,
                arr_city_code=arr_code
            )
            
            print(f"✓ 请求成功")
            print(f"响应长度: {len(result)} 字符")
            print(f"响应前 800 字符:\n{result[:800]}")
            
            # 简单验证响应是否包含关键词
            keywords = ["价格", "price", "航班", "flight", "最低", "lowest"]
            found_keywords = [kw for kw in keywords if kw in result.lower()]
            if found_keywords:
                print(f"✓ 响应包含关键词: {found_keywords}")
            else:
                print(f"⚠ 响应中未检测到常见关键词")
                
        except Exception as e:
            print(f"✗ 查询失败: {e}")


async def test_query_tickets_compatibility():
    """测试：兼容接口 query_tickets（城市名自动转三字码）"""
    print("\n" + "=" * 70)
    print("测试 3：query_tickets - 旧接口兼容性测试")
    print("=" * 70)
    
    test_date = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
    
    try:
        result = await flight_client.query_tickets(
            date=test_date,
            from_city="上海",
            to_city="北京"
        )
        print(f"✓ query_tickets 调用成功")
        print(f"响应前 500 字符:\n{result[:500]}")
    except Exception as e:
        print(f"✗ query_tickets 调用失败: {e}")


async def test_other_tools():
    """测试：其他航班 MCP 工具（可选，不强制要求全部通过）"""
    print("\n" + "=" * 70)
    print("测试 4：其他航班工具（可选）")
    print("=" * 70)
    
    test_date = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
    
    # 测试 1：机场天气
    print("\n--- 测试: get_airport_weather ---")
    try:
        result = await flight_client.get_airport_weather(airport="SHA")
        print(f"✓ 机场天气查询成功")
        print(f"响应: {result[:300]}")
    except Exception as e:
        print(f"✗ 机场天气查询失败: {e}")
    
    # 测试 2：OD对航班查询
    print("\n--- 测试: search_flights_by_dep_arr ---")
    try:
        result = await flight_client.search_flights_by_dep_arr(dep="SHA", arr="PEK")
        print(f"✓ OD对航班查询成功")
        print(f"响应前 300 字符: {result[:300]}")
    except Exception as e:
        print(f"✗ OD对航班查询失败: {e}")


async def main():
    print("=" * 70)
    print("航班 MCP 专项测试")
    print("=" * 70)
    
    # 先检查配置
    from mcp_clients.flight_client import FLIGHT_MCP_URL
    print(f"\n配置检查:")
    print(f"  FLIGHT_MCP_URL: {FLIGHT_MCP_URL[:80]}...")
    if "<mcp-server-host>" in FLIGHT_MCP_URL:
        print("  ⚠ FLIGHT_MCP_URL 仍为占位符，请在 config.py 中配置真实 URL")
        return
    
    # 运行测试
    await test_list_tools()
    await test_search_flight_itineraries()
    await test_query_tickets_compatibility()
    await test_other_tools()
    
    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())