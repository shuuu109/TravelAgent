"""
简单端到端规划测试 (LangGraph 架构)
跳过 RAG，验证 train / flight MCP + hotel MCP + 大模型联动。
"""
import sys
import os

# Windows GBK -> UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import asyncio
import json
import logging

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 只显示 WARNING 以上，屏蔽 MCP 启动日志噪音
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from graph.workflow import build_graph
from context.memory_manager import MemoryManager


SEP  = "=" * 60
SEP2 = "-" * 60


def _short(text: str, n: int = 120) -> str:
    """截断长文本"""
    if not text:
        return ""
    return text[:n] + "..." if len(text) > n else text


async def run_test(query: str):
    print(f"\n{SEP}")
    print(f"[测试查询] {query}")
    print(SEP)

    memory_manager = MemoryManager(user_id="test_user", session_id="test_001")
    graph = build_graph(memory_manager=memory_manager, checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "test_001"}}

    print("正在调用 graph.ainvoke ... (首次加载 MCP 较慢，请稍候)")
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=query)]},
        config=config,
    )

    # ── 意图识别 ──────────────────────────────────────────────
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

    # ── Skill 执行结果 ────────────────────────────────────────
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
                    print(f"      {t_type:4s} {t_no:8s}  {dep}->{arr}  {dur}  {price}")
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
            # 其他 agent 简要打印
            data_str = json.dumps(data, ensure_ascii=False)
            print(f"    data        : {_short(data_str, 150)}")

    # ── 最终回复 ──────────────────────────────────────────────
    final_response = result.get("final_response", "")
    print(f"\n{SEP2}")
    print("[最终回复 (final_response)]")
    print(final_response)
    print(SEP)


if __name__ == "__main__":
    asyncio.run(run_test("我后天从上海去北京出差，帮我查下交通和住宿"))
