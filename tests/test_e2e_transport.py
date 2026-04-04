#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
端到端测试：南京 → 上海交通查询

验证三条核心断言：
  1. intent_node  ── 正确识别 transport_query 意图
  2. MCP clients  ── 火车 / 航班查询能返回真实数据
  3. respond_node ── 能将交通结果格式化为用户可读文本

用法：
  python tests/test_e2e_transport.py            # 全部（含真实 MCP 调用）
  python tests/test_e2e_transport.py --no-live  # 跳过 MCP 真实调用
"""
import asyncio
import io
import json
import os
import sys
import unittest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

# Windows 终端 UTF-8 输出修复
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from langchain_core.messages import AIMessage, HumanMessage

RUN_LIVE = "--no-live" not in sys.argv

# ── 颜色 / 符号 ───────────────────────────────────────────────────────────────
GREEN  = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"; RESET = "\033[0m"
OK   = f"{GREEN}[OK]{RESET}"; FAIL = f"{RED}[X]{RESET}"; SKIP = f"{YELLOW}[-]{RESET}"

_results: list = []

def _record(section, name, passed, note="", skipped=False):
    _results.append({"section": section, "name": name,
                     "passed": passed, "note": note, "skipped": skipped})
    icon   = SKIP if skipped else (OK if passed else FAIL)
    status = "跳过" if skipped else ("通过" if passed else "失败")
    suffix = f"  ({note})" if note else ""
    print(f"  {icon} {name} — {status}{suffix}")


# =============================================================================
# 第一部分：intent_node 意图识别
# =============================================================================

USER_INPUT = "我要从南京去上海，明天出发"

# intent_node 返回的 JSON 模板（mock LLM 输出）
MOCK_INTENT_JSON = {
    "reasoning": "用户明确表达从南京到上海的跨城移动需求，属于交通查询意图。",
    "intents": [
        {
            "type": "transport_query",
            "confidence": 0.97,
            "description": "查询南京到上海的火车/航班信息",
            "reason": "用户提到出发地、目的地和出发时间，需要实时交通数据"
        }
    ],
    "key_entities": {
        "origin": "南京",
        "destination": "上海",
        "date": (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
        "duration": None,
        "other": None
    },
    "rewritten_query": f"查询{(datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')}南京到上海的火车和航班",
    "agent_schedule": [
        {
            "agent_name": "transport_query",
            "priority": 1,
            "reason": "用户有明确跨城移动需求，需查询实时车次和航班",
            "expected_output": "南京到上海的火车和航班列表及推荐方案"
        }
    ]
}


async def _run_intent_node_with_mock_llm():
    """用 Mock LLM 运行 intent_node，验证解析逻辑。"""
    from graph.nodes.intent_node import create_intent_node

    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = AIMessage(
        content=json.dumps(MOCK_INTENT_JSON, ensure_ascii=False)
    )

    intent_node = create_intent_node(mock_llm)
    state = {"messages": [HumanMessage(content=USER_INPUT)]}
    return await intent_node(state)


def test_intent_node_extracts_transport_query():
    """
    第 1 组：intent_node 意图识别
    """
    print("\n【第 1 组】intent_node — 意图识别")
    print("-" * 50)
    section = "intent_node"

    # 1-A: intent_node 运行不报错，返回三个必要字段
    try:
        result = asyncio.run(_run_intent_node_with_mock_llm())
        _record(section, "1-A 节点无异常退出", True)
    except Exception as e:
        _record(section, "1-A 节点无异常退出", False, str(e))
        return

    # 1-B: intent_schedule 包含 transport_query
    schedule = result.get("intent_schedule", [])
    has_transport = any(t.get("agent_name") == "transport_query" for t in schedule)
    _record(section, "1-B intent_schedule 含 transport_query", has_transport,
            f"schedule={[t.get('agent_name') for t in schedule]}")

    # 1-C: key_entities 提取了 origin / destination
    entities = result.get("intent_data", {}).get("key_entities", {})
    has_origin = bool(entities.get("origin"))
    has_dest   = bool(entities.get("destination"))
    _record(section, "1-C 识别 origin（出发地）", has_origin,
            f"origin={entities.get('origin')}")
    _record(section, "1-D 识别 destination（目的地）", has_dest,
            f"destination={entities.get('destination')}")

    # 1-E: user_query 与输入一致
    _record(section, "1-E user_query 与输入一致",
            result.get("user_query") == USER_INPUT,
            f"user_query='{result.get('user_query')}'")

    # 1-F: 日期被解析为 "明天" 对应的日期
    date_val = entities.get("date", "")
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    _record(section, f"1-F 日期解析为明天({tomorrow})",
            date_val == tomorrow,
            f"date='{date_val}'")


# =============================================================================
# 第二部分：MCP 查询（火车 + 航班）
# =============================================================================

TOMORROW = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
ORIGIN   = "南京"
DEST     = "上海"


async def _run_train_query():
    from mcp_clients.train_client import train_client
    return await train_client.query_tickets(TOMORROW, ORIGIN, DEST)


async def _run_flight_query():
    from mcp_clients.flight_client import flight_client
    return await flight_client.query_tickets(TOMORROW, ORIGIN, DEST)


def test_mcp_clients_return_data():
    """
    第 2 组：MCP 客户端 — 实时查询
    """
    print("\n【第 2 组】MCP 客户端 — 实时查询")
    print("-" * 50)
    section = "mcp_clients"

    if not RUN_LIVE:
        _record(section, "2-A train_client 查询", False, "跳过（--no-live）", skipped=True)
        _record(section, "2-B flight_client 查询", False, "跳过（--no-live）", skipped=True)
        _record(section, "2-C 两路数据均非空",   False, "跳过（--no-live）", skipped=True)
        return

    # 2-A: train_client
    try:
        train_raw = asyncio.run(_run_train_query())
        train_ok  = bool(train_raw)
        note      = f"类型={type(train_raw).__name__}, 长度={len(str(train_raw))}"
        _record(section, "2-A train_client 查询返回数据", train_ok, note)
    except Exception as e:
        train_raw = None
        _record(section, "2-A train_client 查询返回数据", False, str(e))

    # 2-B: flight_client
    try:
        flight_raw = asyncio.run(_run_flight_query())
        flight_ok  = bool(flight_raw)
        note       = f"类型={type(flight_raw).__name__}, 长度={len(str(flight_raw))}"
        _record(section, "2-B flight_client 查询返回数据", flight_ok, note)
    except Exception as e:
        flight_raw = None
        _record(section, "2-B flight_client 查询返回数据", False, str(e))

    # 2-C: 两路都有数据
    both_ok = bool(train_raw) and bool(flight_raw)
    _record(section, "2-C 火车+航班数据均非空", both_ok)


# =============================================================================
# 第三部分：respond_node 格式化展示
# =============================================================================

# 模拟 orchestrate_node 展平后写入 state 的 skill_results
MOCK_SKILL_RESULTS = [
    {
        "agent_name": "transport_query",
        "status": "success",
        "data": {
            "transport_plan": {
                "query_info": {
                    "origin": "南京",
                    "destination": "上海",
                    "date": TOMORROW,
                    "data_source": "实时火车与航班查询"
                },
                "analysis": "南京到上海高铁约1.5-2小时，航班加上值机时间总耗时更长，推荐高铁。",
                "options": [
                    {
                        "transport_type": "高铁",
                        "transport_no": "G7000",
                        "departure_time": "08:00",
                        "arrival_time": "09:30",
                        "duration": "1小时30分",
                        "departure_hub": "南京南站",
                        "arrival_hub": "上海虹桥站",
                        "price_range": "二等座¥134.5，一等座¥214.5",
                        "is_recommended": True,
                        "data_source": "realtime",
                        "pros": "快速、准时",
                        "cons": "需提前购票"
                    },
                    {
                        "transport_type": "高铁",
                        "transport_no": "G7002",
                        "departure_time": "10:00",
                        "arrival_time": "11:35",
                        "duration": "1小时35分",
                        "departure_hub": "南京南站",
                        "arrival_hub": "上海虹桥站",
                        "price_range": "二等座¥134.5，一等座¥214.5",
                        "is_recommended": False,
                        "data_source": "realtime",
                        "pros": "出发时间适中",
                        "cons": "余票可能不多"
                    }
                ],
                "recommendation": {
                    "fastest": "G7000 08:00出发，1.5小时直达",
                    "best_value": "高铁二等座¥134.5，性价比最高",
                    "best_choice": "G7000 高铁",
                    "arrival_hub": "上海虹桥站",
                    "reason": "南京到上海距离约300km，高铁是最优选择"
                }
            }
        }
    }
]

# mock LLM 兜底回复（当规则层无法格式化时触发）
MOCK_LLM_SUMMARY = (
    "为您查询了明天南京到上海的交通方案：\n"
    "推荐乘坐G7000高铁，08:00从南京南站出发，09:30抵达上海虹桥站，历时1.5小时，"
    "二等座¥134.5。高铁快速准时，是南京到上海的最佳选择。"
)


async def _run_respond_node_with_mock(mock_llm):
    """用 Mock LLM 运行 respond_node，验证格式化逻辑。"""
    from graph.nodes.respond_node import create_respond_node

    respond_node = create_respond_node(mock_llm)
    state = {
        "skill_results": MOCK_SKILL_RESULTS,
        "intent_data": {
            "rewritten_query": f"查询{TOMORROW}南京到上海的火车和航班",
            "key_entities": {"origin": "南京", "destination": "上海", "date": TOMORROW}
        }
    }
    return await respond_node(state)


def test_respond_node_formats_transport_result():
    """
    第 3 组：respond_node — 交通结果格式化
    """
    print("\n【第 3 组】respond_node — 交通结果格式化")
    print("-" * 50)
    section = "respond_node"

    # Mock LLM（respond_node 的 LLM 兜底汇总）
    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = AIMessage(content=MOCK_LLM_SUMMARY)

    # 3-A: 节点运行无异常
    try:
        result = asyncio.run(_run_respond_node_with_mock(mock_llm))
        _record(section, "3-A 节点无异常退出", True)
    except Exception as e:
        _record(section, "3-A 节点无异常退出", False, str(e))
        return

    final_response = result.get("final_response", "")

    # 3-B: final_response 非空
    _record(section, "3-B final_response 非空",
            bool(final_response.strip()),
            f"长度={len(final_response)}")

    # 3-C: messages 中有 AIMessage
    msgs = result.get("messages", [])
    has_ai_msg = any(isinstance(m, AIMessage) for m in msgs)
    _record(section, "3-C messages 中追加了 AIMessage", has_ai_msg,
            f"messages 数量={len(msgs)}")

    # 3-D: 回复中包含与交通相关的关键词
    transport_keywords = ["南京", "上海", "高铁", "火车", "航班", "出发", "到达", "方案"]
    matched = [kw for kw in transport_keywords if kw in final_response]
    _record(section, f"3-D 回复含交通关键词（{matched}）",
            len(matched) >= 2,
            f"共匹配 {len(matched)}/{len(transport_keywords)} 个")

    # 3-E: skill_results 为空时不报错（边界情况）
    try:
        empty_result = asyncio.run(_run_respond_node_with_empty_state(mock_llm))
        _record(section, "3-E 空 skill_results 时安全返回",
                bool(empty_result.get("final_response")))
    except Exception as e:
        _record(section, "3-E 空 skill_results 时安全返回", False, str(e))


async def _run_respond_node_with_empty_state(mock_llm):
    from graph.nodes.respond_node import create_respond_node
    respond_node = create_respond_node(mock_llm)
    return await respond_node({"skill_results": [], "intent_data": {}})


# =============================================================================
# 汇总报告
# =============================================================================

def _print_summary():
    print("\n" + "=" * 60)
    print("测试汇总")
    print("=" * 60)
    total   = len(_results)
    skipped = sum(1 for r in _results if r["skipped"])
    passed  = sum(1 for r in _results if not r["skipped"] and r["passed"])
    failed  = sum(1 for r in _results if not r["skipped"] and not r["passed"])

    print(f"  总计: {total}  通过: {passed}  失败: {failed}  跳过: {skipped}")

    if failed:
        print(f"\n{RED}失败项:{RESET}")
        for r in _results:
            if not r["skipped"] and not r["passed"]:
                print(f"  {FAIL} [{r['section']}] {r['name']}"
                      + (f"  ({r['note']})" if r["note"] else ""))

    overall = failed == 0
    icon    = OK if overall else FAIL
    print(f"\n{icon} 最终结果: {'全部通过' if overall else '存在失败项'}")
    return overall


# =============================================================================
# main
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("端到端测试：南京 → 上海交通查询")
    print(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"MCP 真实调用: {'启用' if RUN_LIVE else '禁用 (--no-live)'}")
    print("=" * 60)

    test_intent_node_extracts_transport_query()
    test_mcp_clients_return_data()
    test_respond_node_formats_transport_result()

    ok = _print_summary()
    sys.exit(0 if ok else 1)
