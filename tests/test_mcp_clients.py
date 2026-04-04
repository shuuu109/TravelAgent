#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MCP 客户端功能测试

覆盖范围：
  第一部分 - 客户端配置校验（无网络，秒级）
    · train_client 命令、类名、方法签名是否正确
    · FlightMCPClient 类名、方法签名是否正确
    · __init__.py 导出是否正确

  第二部分 - MCP 连通性测试
    · 12306 STDIO MCP 握手（列出工具）
    · 航班 Streamable HTTP MCP 握手（列出工具）

  第三部分 - 真实查询测试（联网，分钟级）
    · train_client.query_tickets
    · flight_client.query_tickets（→ searchFlightItineraries）

  第四部分 - TransportAgent 集成测试（Mock MCP，无需联网）
    · 正确从 context 取参数并并发调用 train/flight client
    · date 缺失时自动补明天
    · origin 缺失时直接返回 error 不调用 MCP
    · MCP 抛异常时降级为 LLM 通用分析

用法：
  python tests/test_mcp_clients.py            # 全部（含联网测试）
  python tests/test_mcp_clients.py --no-live  # 跳过真实查询，只跑配置+Mock
"""
import sys
import os
import asyncio
import inspect
import json
import time
import unittest
from unittest.mock import patch, AsyncMock, MagicMock

# ── 路径 ─────────────────────────────────────────────────────────────────────
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

RUN_LIVE = "--no-live" not in sys.argv

# ── 简单颜色/符号辅助 ─────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"
OK   = f"{GREEN}✓{RESET}"
FAIL = f"{RED}✗{RESET}"
SKIP = f"{YELLOW}○{RESET}"

_results: list = []


def _record(section: str, name: str, passed: bool, note: str = "", skipped: bool = False):
    _results.append({"section": section, "name": name,
                     "passed": passed, "note": note, "skipped": skipped})
    icon   = SKIP if skipped else (OK if passed else FAIL)
    status = "跳过" if skipped else ("通过" if passed else "失败")
    suffix = f"  ({note})" if note else ""
    print(f"  {icon} {name} — {status}{suffix}")


# ═════════════════════════════════════════════════════════════════════════════
# 第一部分：客户端配置校验
# ═════════════════════════════════════════════════════════════════════════════
def test_client_config():
    section = "配置校验"
    print(f"\n{'='*70}\n第一部分：客户端配置校验\n{'='*70}")

    # ── train_client ──────────────────────────────────────────────────────────
    from mcp_clients.train_client import TrainTicketClient, train_client

    tc = TrainTicketClient()
    _record(section, "TrainTicketClient 可实例化", isinstance(tc, TrainTicketClient))
    _record(section, "train_client 单例存在", train_client is not None)
    _record(section, "query_tickets 方法存在", callable(getattr(tc, "query_tickets", None)))
    params = list(inspect.signature(tc.query_tickets).parameters)
    _record(section, "query_tickets 参数: (date, from_station, to_station)",
            params == ["date", "from_station", "to_station"], note=str(params))

    # ── flight_client（新 Streamable HTTP 版）────────────────────────────────
    from mcp_clients.flight_client import FlightMCPClient, flight_client, FLIGHT_MCP_URL

    fc = FlightMCPClient()
    _record(section, "FlightMCPClient 可实例化", isinstance(fc, FlightMCPClient))
    _record(section, "flight_client 单例存在", flight_client is not None)
    _record(section, "FLIGHT_MCP_URL 已配置（非占位符）",
            "<mcp-server-host>" not in FLIGHT_MCP_URL, note=FLIGHT_MCP_URL[:60])
    _record(section, "FLIGHT_MCP_URL 含 api_key 参数",
            "api_key=" in FLIGHT_MCP_URL)

    for method in ("search_flight_itineraries", "search_flights_by_dep_arr",
                   "search_flights_by_number", "search_transfer_info",
                   "get_happiness_index", "get_realtime_location",
                   "get_airport_weather", "query_tickets"):
        _record(section, f"FlightMCPClient.{method} 存在",
                callable(getattr(fc, method, None)))

    qt_params = list(inspect.signature(fc.query_tickets).parameters)
    _record(section, "query_tickets 参数: (date, from_city, to_city)",
            qt_params == ["date", "from_city", "to_city"], note=str(qt_params))

    fi_params = list(inspect.signature(fc.search_flight_itineraries).parameters)
    _record(section, "search_flight_itineraries 参数: (dep_city_code, dep_date, arr_city_code)",
            fi_params == ["dep_city_code", "dep_date", "arr_city_code"], note=str(fi_params))

    # ── __init__.py 导出 ──────────────────────────────────────────────────────
    import mcp_clients as mc
    _record(section, "__all__ 包含 FlightMCPClient",   "FlightMCPClient"   in mc.__all__)
    _record(section, "__all__ 包含 flight_client",     "flight_client"     in mc.__all__)
    _record(section, "__all__ 包含 flight_mcp_session","flight_mcp_session" in mc.__all__)
    _record(section, "__all__ 不含旧 FlightTicketClient",
            "FlightTicketClient" not in mc.__all__)


# ═════════════════════════════════════════════════════════════════════════════
# 第二部分：MCP 连通性测试
# ═════════════════════════════════════════════════════════════════════════════
async def _list_tools_stdio(server_params) -> list:
    from mcp.client.stdio import stdio_client
    from mcp import ClientSession
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            resp = await session.list_tools()
            return [t.name for t in resp.tools]


async def _list_tools_http(url: str) -> list:
    from mcp.client.streamable_http import streamablehttp_client
    from mcp.client.session import ClientSession
    async with streamablehttp_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            resp = await session.list_tools()
            return [t.name for t in resp.tools]


def test_mcp_connectivity():
    section = "MCP连通性"
    print(f"\n{'='*70}\n第二部分：MCP 连通性测试\n{'='*70}")

    from mcp_clients.train_client import TrainTicketClient
    from mcp_clients.flight_client import FLIGHT_MCP_URL

    # ── 12306 STDIO ───────────────────────────────────────────────────────────
    print("  [12306 MCP server — STDIO]")
    try:
        tools = asyncio.run(_list_tools_stdio(TrainTicketClient().server_params))
        _record(section, "12306 握手成功", True, note=f"工具数: {len(tools)}")
        _record(section, "12306 包含 query-tickets",
                "query-tickets" in tools, note=str(tools))
    except Exception as e:
        _record(section, "12306 握手成功", False, note=str(e)[:120])
        _record(section, "12306 包含 query-tickets", False, note="server未启动")

    # ── 航班 Streamable HTTP ──────────────────────────────────────────────────
    print("  [航班 MCP server — Streamable HTTP]")
    EXPECTED_TOOLS = [
        "searchFlightItineraries", "searchFlightsByDepArr",
        "searchFlightsByNumber", "searchFlightsTransferinfo",
        "flightHappinessIndex", "getRealtimeLocationByAnum",
        "getFutureWeatherByAirport",
    ]
    try:
        tools = asyncio.run(_list_tools_http(FLIGHT_MCP_URL))
        _record(section, "航班MCP 握手成功", True, note=f"工具数: {len(tools)}")
        for t in EXPECTED_TOOLS:
            _record(section, f"航班MCP 包含 {t}", t in tools)
    except Exception as e:
        _record(section, "航班MCP 握手成功", False, note=str(e)[:120])
        for t in EXPECTED_TOOLS:
            _record(section, f"航班MCP 包含 {t}", False, note="连接失败")


# ═════════════════════════════════════════════════════════════════════════════
# 第三部分：真实查询测试（联网）
# ═════════════════════════════════════════════════════════════════════════════
def test_live_queries():
    section = "真实查询"
    print(f"\n{'='*70}\n第三部分：真实查询测试\n{'='*70}")

    if not RUN_LIVE:
        print(f"  {SKIP} 已通过 --no-live 跳过全部真实查询")
        for name in ("train_client.query_tickets", "flight_client.query_tickets"):
            _record(section, name, True, skipped=True)
        return

    from mcp_clients.train_client import train_client
    from mcp_clients.flight_client import flight_client
    from datetime import datetime, timedelta

    future_date = (datetime.now() + timedelta(days=3)).strftime("%Y-%m-%d")

    def _run_check(label, fn, *args):
        print(f"\n  [{label}]")
        t0 = time.time()
        try:
            coro = fn(*args)
            result = asyncio.run(coro) if asyncio.iscoroutine(coro) else coro
            elapsed = time.time() - t0
            ok = isinstance(result, str) and len(result) > 10
            _record(section, f"{label} 返回非空字符串", ok,
                    note=f"{elapsed:.1f}s, {len(result)}字符")
            if ok:
                print(f"    响应（前300字）: {result[:300]}...")
        except Exception as e:
            _record(section, f"{label} 返回非空字符串", False, note=str(e)[:120])

    _run_check("train_client.query_tickets",
               train_client.query_tickets, future_date, "南京", "上海")

    # 调用 searchFlightItineraries（城市名→IATA 映射由 query_tickets 兼容层处理）
    _run_check("flight_client.query_tickets → searchFlightItineraries",
               flight_client.query_tickets, future_date, "南京", "上海")


# ═════════════════════════════════════════════════════════════════════════════
# 第四部分：TransportAgent 集成测试（Mock MCP，无需联网）
# ═════════════════════════════════════════════════════════════════════════════
_TRAIN_STUB = json.dumps({
    "trains": [
        {"train_no": "G7001", "departure_time": "07:00", "arrival_time": "08:10",
         "duration": "1小时10分", "second_class_price": "54.5元", "second_class_seat": "有"},
    ]
}, ensure_ascii=False)

_FLIGHT_STUB = json.dumps({
    "total": 10,
    "lowest_price": 350,
    "shortest_duration": "2小时",
    "lowest_price_flight": {
        "flight_no": "CZ8882", "dep_time": "2026-04-10 23:05",
        "arr_time": "2026-04-11 01:05", "duration": "2小时", "price": 350,
    },
}, ensure_ascii=False)

_WEATHER_STUB = json.dumps({
    "airport": "SHA",
    "date": "2026-04-10",
    "weather": "晴",
    "temperature": "18~26℃",
    "wind": "东南风3级",
    "humidity": "55%",
}, ensure_ascii=False)

_LLM_TRANSPORT_RESP = json.dumps({
    "query_info": {"origin": "南京", "destination": "上海",
                   "date": "2026-04-10", "data_source": "实时火车与航班查询"},
    "analysis": "Mock分析",
    "options": [],
    "recommendation": {"fastest": "高铁G7001", "best_value": "高铁G7001",
                       "best_choice": "高铁G7001", "arrival_hub": "上海虹桥站",
                       "reason": "快捷"},
    "weather_reminder": "上海天气晴好，18~26℃，出行舒适。",
}, ensure_ascii=False)

_LLM_DEGRADED_RESP = json.dumps({
    "query_info": {"origin": "南京", "destination": "上海",
                   "date": "2026-04-10", "data_source": "LLM知识推断（实时查询不可用）"},
    "analysis": "Mock降级分析",
    "options": [],
    "recommendation": {"best_choice": "高铁", "arrival_hub": "上海虹桥站", "reason": "快"},
}, ensure_ascii=False)


class TestTransportAgentIntegration(unittest.IsolatedAsyncioTestCase):
    """TransportAgent 集成测试：校验 context 解析与客户端调用逻辑"""

    async def asyncSetUp(self):
        self.model = AsyncMock()

        async def mock_llm(messages):
            resp = MagicMock()
            content = str(messages)
            resp.content = (_LLM_DEGRADED_RESP if "实时查询不可用" in content
                            else _LLM_TRANSPORT_RESP)
            return resp

        self.model.ainvoke = AsyncMock(side_effect=mock_llm)

    @patch("agents.transport_agent.train_client.query_tickets",
           new_callable=AsyncMock, return_value=_TRAIN_STUB)
    @patch("agents.transport_agent.flight_client.query_tickets",
           new_callable=AsyncMock, return_value=_FLIGHT_STUB)
    @patch("agents.transport_agent.flight_client.get_airport_weather",
           new_callable=AsyncMock, return_value=_WEATHER_STUB)
    async def test_calls_correct_params(self, mock_weather, mock_flight, mock_train):
        """正确提取 origin/destination/date 后三路并发调用（train/flight/weather）"""
        from agents.transport_agent import TransportAgent

        agent = TransportAgent(name="TA", model=self.model)
        payload = {"context": {"key_entities": {
            "origin": "南京", "destination": "上海", "date": "2026-04-10"
        }}}
        result = await agent.run(payload)

        mock_train.assert_called_once()
        mock_flight.assert_called_once()
        mock_weather.assert_called_once()

        train_args = mock_train.call_args[0]
        self.assertEqual(train_args[0], "2026-04-10", f"date 错误: {train_args}")
        self.assertEqual(train_args[1], "南京",        f"origin 错误: {train_args}")
        self.assertEqual(train_args[2], "上海",        f"destination 错误: {train_args}")

        self.assertIn("transport_plan", result,
                      f"返回缺少 transport_plan: {result}")

    @patch("agents.transport_agent.train_client.query_tickets",
           new_callable=AsyncMock, return_value=_TRAIN_STUB)
    @patch("agents.transport_agent.flight_client.query_tickets",
           new_callable=AsyncMock, return_value=_FLIGHT_STUB)
    @patch("agents.transport_agent.flight_client.get_airport_weather",
           new_callable=AsyncMock, return_value=_WEATHER_STUB)
    async def test_missing_date_uses_tomorrow(self, mock_weather, mock_flight, mock_train):
        """date 为空时，TransportAgent 自动使用明天日期"""
        from agents.transport_agent import TransportAgent

        agent = TransportAgent(name="TA", model=self.model)
        payload = {"context": {"key_entities": {
            "origin": "北京", "destination": "上海", "date": ""
        }}}
        await agent.run(payload)

        mock_train.assert_called_once()
        date_used = mock_train.call_args[0][0]
        self.assertRegex(date_used, r"^\d{4}-\d{2}-\d{2}",
                         f"date 格式错误: {date_used}")

    @patch("agents.transport_agent.train_client.query_tickets",
           new_callable=AsyncMock)
    async def test_missing_origin_no_mcp_call(self, mock_train):
        """origin 为空时，不调用 MCP，直接返回 error"""
        from agents.transport_agent import TransportAgent

        agent = TransportAgent(name="TA", model=self.model)
        payload = {"context": {"key_entities": {
            "origin": "", "destination": "上海", "date": "2026-04-10"
        }}}
        result = await agent.run(payload)

        mock_train.assert_not_called()
        self.assertIn("error", result, f"期望 error 字段: {result}")

    @patch("agents.transport_agent.train_client.query_tickets",
           new_callable=AsyncMock, side_effect=Exception("MCP server down"))
    @patch("agents.transport_agent.flight_client.query_tickets",
           new_callable=AsyncMock, side_effect=Exception("MCP server down"))
    @patch("agents.transport_agent.flight_client.get_airport_weather",
           new_callable=AsyncMock, side_effect=Exception("MCP server down"))
    async def test_mcp_failure_graceful_degradation(self, mock_weather, mock_flight, mock_train):
        """三路 MCP 均抛异常时，降级为 LLM 通用分析"""
        from agents.transport_agent import TransportAgent

        agent = TransportAgent(name="TA", model=self.model)
        payload = {"context": {"key_entities": {
            "origin": "南京", "destination": "上海", "date": "2026-04-10"
        }}}
        result = await agent.run(payload)

        self.assertIn("transport_plan", result,
                      f"降级后仍应有 transport_plan: {result}")
        data_source = (result.get("transport_plan", {})
                             .get("query_info", {})
                             .get("data_source", ""))
        self.assertIn("LLM", data_source,
                      f"降级后 data_source 应含 'LLM': {data_source!r}")


# ═════════════════════════════════════════════════════════════════════════════
# 汇总报告
# ═════════════════════════════════════════════════════════════════════════════
def _print_summary():
    print(f"\n{'='*70}\n测试汇总\n{'='*70}")
    sections: dict = {}
    for r in _results:
        sections.setdefault(r["section"], []).append(r)

    total = skipped = passed = failed = 0
    for sec, items in sections.items():
        sp = sum(1 for i in items if i["passed"] and not i["skipped"])
        ss = sum(1 for i in items if i["skipped"])
        sf = sum(1 for i in items if not i["passed"] and not i["skipped"])
        total += len(items); passed += sp; skipped += ss; failed += sf
        print(f"  {sec:<16}  通过: {sp}  跳过: {ss}  失败: {sf}")

    print(f"{'─'*70}")
    print(f"  总计            通过: {GREEN}{passed}{RESET}  "
          f"跳过: {YELLOW}{skipped}{RESET}  "
          f"失败: {RED}{failed}{RESET}  / 共 {total} 项")
    msg = (f"\n  {GREEN}全部检测项通过（或已知跳过）！{RESET}"
           if failed == 0 else
           f"\n  {RED}有 {failed} 项失败，请检查上方详细日志。{RESET}")
    print(msg + f"\n{'='*70}\n")


# ═════════════════════════════════════════════════════════════════════════════
# 入口
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    test_client_config()
    test_mcp_connectivity()
    test_live_queries()

    print(f"\n{'='*70}\n第四部分：TransportAgent 集成测试（Mock MCP）\n{'='*70}")
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestTransportAgentIntegration))

    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    ut_result = runner.run(suite)

    section = "智能体集成"
    all_ids = [t.id().split(".")[-1]
               for t in loader.loadTestsFromTestCase(TestTransportAgentIntegration)]
    failed_ids = {str(t).split(" ")[0]
                  for t, _ in ut_result.failures + ut_result.errors}
    for tid in all_ids:
        _record(section, tid, tid not in failed_ids)

    _print_summary()
    sys.exit(0 if not any(not r["passed"] and not r["skipped"] for r in _results) else 1)
