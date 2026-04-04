"""
带 Mock 数据的端到端测试 (LangGraph 架构)

使用预定义 Mock 数据替代真实 MCP 调用，验证完整工作流：
  - 高铁/飞机：mock train_client / flight_client 的方法
  - 酒店：mock mcp_clients.hotel_client.search_hotels
无需网络连接，适合在 CI/本地快速验证流程正确性。
"""
import sys
import os
import asyncio
import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

# Windows GBK -> UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from graph.workflow import build_graph
from context.memory_manager import MemoryManager

SEP  = "=" * 60
SEP2 = "-" * 60


# ── Mock 数据 ────────────────────────────────────────────────────────────────

MOCK_TRAIN_TICKETS = """
车次查询结果（上海→北京，2026-04-06）：
G2   上海虹桥  06:43 → 北京南站  11:32  历时4小时49分  二等座：有票  一等座：有票
G4   上海虹桥  07:00 → 北京南站  11:37  历时4小时37分  二等座：有票  一等座：有票
G20  上海虹桥  14:00 → 北京南站  18:36  历时4小时36分  二等座：有票  一等座：有票
"""

MOCK_TRAIN_PRICE = """
票价查询结果（上海→北京，2026-04-06）：
G2:  二等座¥662  一等座¥1032  商务座¥1999
G4:  二等座¥669  一等座¥1032  商务座¥1999
G20: 二等座¥662  一等座¥1032  商务座¥1999
"""

MOCK_FLIGHT_TICKETS = """
航班查询结果（上海→北京，2026-04-06）：
CA8322  上海虹桥机场(SHA) 06:55 → 北京首都机场(PEK) 09:25  历时2小时30分  经济舱¥500  余票充足（最低价）
CZ1925  上海虹桥机场(SHA) 10:30 → 北京首都机场(PEK) 12:20  历时1小时50分  经济舱¥1840 余票充足（最短耗时）
"""

MOCK_WEATHER = """
北京天气预报（2026-04-06）：
晴，气温8-18℃，东风2级，适合出行，航班延误风险低。
"""

MOCK_HOTELS_DATA = {
    "message": "success",
    "hotelInformationList": [
        {
            "hotelId": "H001",
            "hotelName": "北京王府井天伦王朝酒店",
            "starLevel": 5,
            "price": 853,
            "address": "北京市东城区王府井大街2号",
            "features": "地铁1/5/6/8号线直达，3800㎡会议场地，室内恒温泳池，商务中心"
        },
        {
            "hotelId": "H002",
            "hotelName": "北京天伦松鹤大饭店",
            "starLevel": 4,
            "price": 592,
            "address": "北京市东城区崇文门西大街5号",
            "features": "地铁1/5号线覆盖，会议室可容纳400人，健身房"
        },
        {
            "hotelId": "H003",
            "hotelName": "北京西单雅高美爵酒店",
            "starLevel": 4,
            "price": 602,
            "address": "北京市西城区西单北大街183号",
            "features": "地铁1/4号线直达，毗邻天安门/故宫，法式餐厅+中式餐厅"
        },
        {
            "hotelId": "H004",
            "hotelName": "国际艺苑大酒店",
            "starLevel": 4,
            "price": 603,
            "address": "北京市东城区王府井大街1号",
            "features": "地铁1/5/6/8号线覆盖，3个多功能会议室，靠近故宫"
        }
    ]
}


def _make_hotel_mcp_result():
    """
    构造模拟的 MCP CallToolResult 对象。
    accommodation_agent 会访问 raw.content[0].text 并 json.loads 解析。
    """
    mock_content = MagicMock()
    mock_content.text = json.dumps(MOCK_HOTELS_DATA, ensure_ascii=False)
    mock_result = MagicMock()
    mock_result.content = [mock_content]
    return mock_result


# ── 输出辅助 ─────────────────────────────────────────────────────────────────

def _short(text: str, n: int = 120) -> str:
    if not text:
        return ""
    return text[:n] + "..." if len(text) > n else text


def _print_results(result: dict):
    # 意图识别
    intent_data = result.get("intent_data", {})
    print(f"\n{SEP2}")
    print("[意图识别]")
    print(f"  意图类型: {[i.get('type') for i in intent_data.get('intents', [])]}")
    entities = intent_data.get("key_entities", {})
    print(f"  关键实体: origin={entities.get('origin')}  destination={entities.get('destination')}  date={entities.get('date')}")
    schedule = intent_data.get("agent_schedule", [])
    print(f"  调度计划:")
    for s in schedule:
        print(f"    priority={s.get('priority')}  [{s.get('agent_name')}]  {_short(s.get('reason',''), 60)}")

    # Skill 执行结果
    skill_results = result.get("skill_results", [])
    print(f"\n{SEP2}")
    print(f"[技能执行结果]  共 {len(skill_results)} 个 agent")

    for r in skill_results:
        agent_name = r.get("agent_name", "?")
        status = r.get("status", "?")
        data = r.get("data", {})
        print(f"\n  [{agent_name}]  status={status}")

        if agent_name == "transport_query":
            transport_plan = data.get("transport_plan", {})
            if transport_plan:
                qi = transport_plan.get("query_info", {})
                print(f"    data_source : {qi.get('data_source', '-')}")
                print(f"    查询区间    : {qi.get('origin')} -> {qi.get('destination')}  ({qi.get('date')})")
                print(f"    分析摘要    : {_short(transport_plan.get('analysis',''), 100)}")
                options = transport_plan.get("options", [])
                print(f"    方案数      : {len(options)}")
                for opt in options[:5]:
                    t_type = opt.get("transport_type", "")
                    t_no   = opt.get("transport_no") or "-"
                    dep    = opt.get("departure_time") or "-"
                    arr    = opt.get("arrival_time") or "-"
                    dur    = opt.get("duration", "-")
                    price  = opt.get("price_range", "暂无")
                    print(f"      {t_type:4s}  {t_no:8s}  {dep}->{arr}  {dur}  {price}")
                rec = transport_plan.get("recommendation", {})
                if rec:
                    print(f"    推荐        : {_short(rec.get('best_choice',''), 80)}")
                    print(f"    到达枢纽    : {rec.get('arrival_hub') or rec.get('arrival_station','未知')}")
            else:
                print(f"    ERROR: {data.get('error','未知')}")

        elif agent_name == "accommodation_query":
            acc_plan = data.get("accommodation_plan", {})
            mcp_cnt  = data.get("mcp_hotels_count", 0)
            if acc_plan:
                print(f"    MCP真实酒店 : {mcp_cnt} 条")
                print(f"    mcp_data_used: {acc_plan.get('mcp_data_used')}")
                print(f"    目的地      : {acc_plan.get('destination')}")
                print(f"    到达枢纽    : {acc_plan.get('arrival_station')}")
                print(f"    分析摘要    : {_short(acc_plan.get('analysis',''), 100)}")
                options = acc_plan.get("options", [])
                print(f"    酒店方案数  : {len(options)}")
                for opt in options[:4]:
                    print(f"      [{opt.get('tier','')}] {opt.get('hotel_name','')}  {opt.get('price_range','')}")
                rec = acc_plan.get("recommendation", {})
                if rec:
                    print(f"    推荐        : {_short(rec.get('best_choice',''), 80)}")
            else:
                print(f"    ERROR: {data.get('error','未知')}")

        elif agent_name == "event_collection":
            print(f"    origin      : {data.get('origin')}")
            print(f"    destination : {data.get('destination')}")
            print(f"    start_date  : {data.get('start_date')}")
            print(f"    end_date    : {data.get('end_date')}")
            missing = data.get("missing_info", [])
            if missing:
                print(f"    缺失信息    : {missing}")

        elif agent_name == "itinerary_planning":
            itin = data.get("itinerary", {})
            if itin:
                print(f"    标题        : {itin.get('title', '')}")
                print(f"    时长        : {itin.get('duration', '')}")
                days = itin.get("daily_plans", [])
                print(f"    天数        : {len(days)} 天")
            else:
                print(f"    data keys   : {list(data.keys())}")

        else:
            data_str = json.dumps(data, ensure_ascii=False)
            print(f"    data        : {_short(data_str, 150)}")

    # 最终回复
    final_response = result.get("final_response", "")
    print(f"\n{SEP2}")
    print("[最终回复 (final_response)]")
    print(final_response)
    print(SEP)


# ── 主测试入口 ───────────────────────────────────────────────────────────────

async def run_test(query: str):
    print(f"\n{SEP}")
    print(f"[测试查询] {query}")
    print(SEP)

    memory_manager = MemoryManager(user_id="test_user", session_id="test_mock_001")
    graph = build_graph(memory_manager=memory_manager, checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "test_mock_001"}}

    print("正在调用 graph.ainvoke（Mock 数据，无需真实 MCP 连接）...")

    # transport_agent 在模块顶层 import 了 train_client / flight_client 的单例，
    # 需要在 agents.transport_agent 模块命名空间中 patch 对应方法。
    # accommodation_agent 在方法内部 import search_hotels，
    # 需要在 mcp_clients.hotel_client 模块命名空间中 patch。
    from mcp_clients.train_client import train_client
    from mcp_clients.flight_client import flight_client

    with patch.object(train_client, "query_tickets",
                      new_callable=AsyncMock, return_value=MOCK_TRAIN_TICKETS), \
         patch.object(train_client, "query_ticket_price",
                      new_callable=AsyncMock, return_value=MOCK_TRAIN_PRICE), \
         patch.object(flight_client, "query_tickets",
                      new_callable=AsyncMock, return_value=MOCK_FLIGHT_TICKETS), \
         patch.object(flight_client, "get_airport_weather",
                      new_callable=AsyncMock, return_value=MOCK_WEATHER), \
         patch("mcp_clients.hotel_client.search_hotels",
               new_callable=AsyncMock, return_value=_make_hotel_mcp_result()):

        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=query)]},
            config=config,
        )

    _print_results(result)


if __name__ == "__main__":
    asyncio.run(run_test("我后天从上海去北京出差，帮我查下交通和住宿"))
