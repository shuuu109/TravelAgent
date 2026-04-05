"""
全链路真实 API 端到端测试 (LangGraph 新框架)
============================================

与 test_full_pipeline.py 的核心区别：
  ✗  不使用任何 Mock / Patch
  ✓  全部调用真实外部服务：
      - 高铁查询   : mcp-server-12306 (stdio)
      - 航班查询   : Variflight MCP (Streamable HTTP)
      - 酒店搜索   : RollingGo MCP (stdio)
      - POI 检索   : 高德地图 MCP (SSE)
      - 距离矩阵   : 高德地图 MCP (SSE)
      - 公交路线   : 高德地图 MCP (SSE)
      - 所有 LLM   : Doubao API (真实调用)

运行前置条件：
  1. conda activate grad_pro   (或对应虚拟环境)
  2. mcp-server-12306 已安装并在 PATH 中可调用
  3. config.py 中 ROLLINGGO_API_KEY / AMAP_KEY / FLIGHT_MCP_URL 均已填写
  4. 网络连通以下域名：
       - mcp.amap.com
       - ai.variflight.com
       - ark.cn-beijing.volces.com  (Doubao LLM)

测试场景：情侣出行，北京→杭州，2天行程 + 往返交通 + 住宿推荐
（与 test_full_pipeline.py 故意选用不同城市和旅行风格，方便对比回归）
"""

import sys
import os
import asyncio
import time
import json
import logging

# Windows GBK → UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# 将项目根目录加入 sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# ── 日志级别配置 ──────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
# 关键节点打开 INFO 方便跟踪进度
logging.getLogger("agents.transport_agent").setLevel(logging.INFO)
logging.getLogger("agents.accommodation_agent").setLevel(logging.INFO)
logging.getLogger("agents.poi_agent").setLevel(logging.INFO)
logging.getLogger("graph.nodes.orchestrate_node").setLevel(logging.INFO)
logging.getLogger("graph.nodes.itinerary_planning_node").setLevel(logging.INFO)

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from graph.workflow import build_graph
from context.memory_manager import MemoryManager

SEP  = "=" * 60
SEP2 = "-" * 60


# ─────────────────────────────────────────────────────────────────────────────
# 前置检查：快速验证外部服务可达性，减少因配置缺失导致的长时间卡顿
# ─────────────────────────────────────────────────────────────────────────────

def _check_prerequisites() -> list[str]:
    """
    检查运行环境是否满足要求。
    返回所有警告信息列表（非致命，仍会继续执行）。
    """
    warnings: list[str] = []

    # 1. mcp-server-12306 命令是否存在
    import shutil
    if not shutil.which("mcp-server-12306"):
        warnings.append("⚠️  mcp-server-12306 未在 PATH 中找到 → 高铁查询将失败")

    # 2. RollingGo API Key
    try:
        from config import ROLLINGGO_MCP_CONFIG
        api_key = ROLLINGGO_MCP_CONFIG.get("ROLLINGGO_API_KEY", "")
        if not api_key or "your_" in api_key:
            warnings.append("⚠️  ROLLINGGO_API_KEY 未配置 → 酒店查询将失败")
        cmd = ROLLINGGO_MCP_CONFIG.get("command", "")
        if not os.path.exists(cmd) and not shutil.which(cmd):
            warnings.append(f"⚠️  RollingGo MCP 命令不可用: {cmd}")
    except ImportError:
        warnings.append("⚠️  无法导入 config.py → 多项服务配置缺失")

    # 3. AMAP Key
    try:
        from config import AMAP_MCP_CONFIG
        amap_key = AMAP_MCP_CONFIG.get("AMAP_KEY", "")
        if not amap_key:
            warnings.append("⚠️  AMAP_KEY 未配置 → POI/路线查询将失败")
    except ImportError:
        pass

    return warnings


# ─────────────────────────────────────────────────────────────────────────────
# 输出辅助
# ─────────────────────────────────────────────────────────────────────────────

def _short(text: str, n: int = 120) -> str:
    """截断长文本，末尾加省略号。"""
    if not text:
        return ""
    return text[:n] + "..." if len(text) > n else text


def _print_results(result: dict, elapsed: float):
    """打印各阶段结果（与 test_full_pipeline.py 保持相同格式，额外展示耗时）。"""

    # ── P1 意图识别 ───────────────────────────────────────────────────────────
    intent_data = result.get("intent_data", {})
    print(f"\n{SEP2}")
    print(f"[P1 意图识别]")
    print(f"  旅行风格 : {result.get('travel_style', '(未写入state)')}")
    print(f"  旅行天数 : {result.get('travel_days', '(未写入state)')}")
    print(f"  意图类型 : {[i.get('type') for i in intent_data.get('intents', [])]}")
    entities = intent_data.get("key_entities", {})
    print(
        f"  关键实体 : origin={entities.get('origin')}  "
        f"destination={entities.get('destination')}  "
        f"date={entities.get('date')}  "
        f"duration={entities.get('duration')}"
    )
    schedule = intent_data.get("agent_schedule", [])
    print(f"  调度计划 ({len(schedule)} 个 agent):")
    for s in schedule:
        print(
            f"    priority={s.get('priority')}  "
            f"[{s.get('agent_name')}]  "
            f"{_short(s.get('reason', ''), 60)}"
        )

    # ── P2 技能执行结果 ───────────────────────────────────────────────────────
    skill_results = result.get("skill_results", [])
    print(f"\n{SEP2}")
    print(f"[P2 技能执行结果]  共 {len(skill_results)} 个 agent")

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
                print(f"    查询区间    : {qi.get('origin')} → {qi.get('destination')}  ({qi.get('date')})")
                print(f"    分析摘要    : {_short(transport_plan.get('analysis', ''), 100)}")
                options = transport_plan.get("options", [])
                print(f"    方案数      : {len(options)}")
                for opt in options[:5]:
                    t_type = opt.get("transport_type", "")
                    t_no   = opt.get("transport_no") or "-"
                    dep    = opt.get("departure_time") or "-"
                    arr    = opt.get("arrival_time") or "-"
                    dur    = opt.get("duration", "-")
                    price  = opt.get("price_range", "暂无")
                    print(f"      {t_type:4s}  {t_no:8s}  {dep}→{arr}  {dur}  {price}")
                rec = transport_plan.get("recommendation", {})
                if rec:
                    print(f"    推荐        : {_short(rec.get('best_choice', ''), 80)}")
                    arrival = rec.get("arrival_hub") or rec.get("arrival_station", "未知")
                    print(f"    到达枢纽    : {arrival}")
            else:
                err_msg = data.get("error", "未知")
                print(f"    ERROR       : {err_msg}")
                # 诊断提示
                if "Connection error" in str(err_msg):
                    print("   诊断    : LLM API 连接失败（model.ainvoke），非 MCP 问题")
                elif "mcp-server-12306" in str(err_msg).lower() or "FileNotFoundError" in str(err_msg):
                    print("   诊断    : mcp-server-12306 命令不存在，请确认已安装并在 PATH 中")

        elif agent_name == "poi_fetch":
            result_data = data.get("result", {})
            pois = result_data.get("pois", [])
            err = data.get("error")
            if err:
                print(f"    ERROR: {err}")
                if "mcp.amap.com" in str(err).lower() or "amap" in str(err).lower():
                    print("   诊断    : 高德 MCP 连接失败，请检查 AMAP_KEY 和网络")
            else:
                print(f"    POI 总数    : {len(pois)}")
                cats: dict = {}
                for p in pois:
                    cat = p.get("category", "?")
                    cats[cat] = cats.get(cat, 0) + 1
                for cat, cnt in cats.items():
                    print(f"      [{cat}] {cnt} 个")
                if pois:
                    top = pois[0]
                    print(
                        f"    首个 POI    : {top.get('name')}  "
                        f"rating={top.get('rating')}  "
                        f"({top.get('lng', 0):.4f}, {top.get('lat', 0):.4f})"
                    )

        elif agent_name == "accommodation_query":
            acc_plan = data.get("accommodation_plan", {})
            if acc_plan:
                print(f"    目的地      : {acc_plan.get('destination')}")
                print(f"    到达枢纽    : {acc_plan.get('arrival_station')}")
                print(f"    mcp_used    : {acc_plan.get('mcp_data_used')}")
                print(f"    分析摘要    : {_short(acc_plan.get('analysis', ''), 100)}")
                options = acc_plan.get("options", [])
                print(f"    酒店方案数  : {len(options)}")
                for opt in options[:4]:
                    print(
                        f"      [{opt.get('tier', '')}] "
                        f"{opt.get('hotel_name', '')}  "
                        f"{opt.get('price_range', '')}"
                    )
                rec = acc_plan.get("recommendation", {})
                if rec:
                    print(f"    推荐        : {_short(rec.get('best_choice', ''), 80)}")
            else:
                err_msg = data.get("error", "未知")
                print(f"    ERROR: {err_msg}")
                if "rollinggo" in str(err_msg).lower() or "searchHotels" in str(err_msg):
                    print("   诊断    : RollingGo MCP 失败，检查 ROLLINGGO_API_KEY 和命令路径")

        elif agent_name == "event_collection":
            print(f"    origin      : {data.get('origin')}")
            print(f"    destination : {data.get('destination')}")
            print(f"    start_date  : {data.get('start_date')}")
            print(f"    end_date    : {data.get('end_date')}")
            missing = data.get("missing_info", [])
            if missing:
                print(f"    缺失信息    : {missing}")

        else:
            data_str = json.dumps(data, ensure_ascii=False)
            if status == "error":
                print(f"    ERROR       : {data.get('error', data_str[:150])}")
            else:
                print(f"    data        : {_short(data_str, 150)}")

    # ── P3 行程规划结果 ───────────────────────────────────────────────────────
    daily_itinerary = result.get("daily_itinerary", [])
    daily_routes    = result.get("daily_routes", [])
    print(f"\n{SEP2}")
    print(
        f"[P3 行程规划]  "
        f"daily_itinerary={len(daily_itinerary)} 天  "
        f"daily_routes={len(daily_routes)} 天"
    )

    if not daily_itinerary:
        print("  ⚠️  daily_itinerary 为空（POI 未成功获取或 itinerary_planning_node 未执行）")
    else:
        for day_group in daily_itinerary:
            day  = day_group.get("day")
            pois = day_group.get("pois", [])
            names = " | ".join(p.get("name", "?") for p in pois)
            print(f"  第{day}天 ({len(pois)}个POI): {names}")

    if not daily_routes:
        print("  ⚠️  daily_routes 为空（TSP 路线优化未执行或高德 MCP 连接失败）")
    else:
        for day_route in daily_routes:
            day     = day_route.get("day")
            ordered = day_route.get("ordered_pois", [])
            legs    = day_route.get("legs", [])
            total_d = day_route.get("total_duration", 0)
            route_str = ""
            for i, poi in enumerate(ordered):
                route_str += poi.get("name", "?")
                if i < len(legs):
                    leg   = legs[i]
                    mode  = leg.get("mode", "?")
                    dur   = leg.get("duration", 0)
                    route_str += f" →({mode} {dur}min)→ "
            print(f"  [路线] 第{day}天: {route_str}  总交通: {total_d}min")

    # ── P4 住宿 geo-center 展示 ───────────────────────────────────────────────
    print(f"\n{SEP2}")
    print("[P4 住宿-每日地理重心]")
    if daily_routes:
        for day_route in daily_routes:
            day     = day_route.get("day")
            ordered = day_route.get("ordered_pois", [])
            if ordered:
                avg_lng = sum(p.get("lng", 0) for p in ordered) / len(ordered)
                avg_lat = sum(p.get("lat", 0) for p in ordered) / len(ordered)
                print(f"  第{day}天重心: ({avg_lng:.4f}, {avg_lat:.4f})")
    else:
        print("  (daily_routes 为空，无法计算重心)")

    # ── P5 最终回复 ───────────────────────────────────────────────────────────
    final_response = result.get("final_response", "")
    print(f"\n{SEP2}")
    print("[P5 最终回复 (final_response)]")
    print(final_response if final_response else "（空）")

    # ── 全局耗时 ─────────────────────────────────────────────────────────────
    print(f"\n{SEP2}")
    print(f"[总耗时] {elapsed:.1f} 秒（真实 API 调用，含网络 I/O）")
    print(SEP)


# ─────────────────────────────────────────────────────────────────────────────
# 断言验证：关键字段检查（与 test_full_pipeline.py 逻辑一致）
# ─────────────────────────────────────────────────────────────────────────────

def _assert_full_pipeline(result: dict):
    """对关键字段做断言，任何失败项以 ❌ 打印，全部通过则显示 ✅。"""
    errors: list[str] = []

    # P1
    if not result.get("intent_data"):
        errors.append("P1: intent_data 为空")
    if not result.get("travel_style"):
        errors.append("P1/P2: travel_style 未写入 state")

    # P2
    skill_results = result.get("skill_results", [])
    agents_ran = [r.get("agent_name") for r in skill_results]

    if "transport_query" not in agents_ran:
        errors.append("P2: transport_query 未执行")
    else:
        tr = next(r for r in skill_results if r["agent_name"] == "transport_query")
        if tr.get("status") != "success":
            errors.append(
                f"P2: transport_query 状态异常: {tr.get('status')} | {tr.get('message', '')}"
            )
        else:
            tp = tr.get("data", {}).get("transport_plan", {})
            if not tp.get("options"):
                errors.append("P2: transport_query 返回的 options 列表为空（真实API未返回数据？）")

    if "poi_fetch" not in agents_ran:
        errors.append("P2: poi_fetch 未执行")
    else:
        pf = next(r for r in skill_results if r["agent_name"] == "poi_fetch")
        pois = pf.get("data", {}).get("result", {}).get("pois", [])
        if not pois:
            errors.append("P2: poi_fetch 返回空 POI 列表（高德 MCP 连接失败？）")
        elif len(pois) < 3:
            errors.append(f"P2: poi_fetch 仅返回 {len(pois)} 个 POI（预期 ≥ 3）")

    # P3
    daily_itinerary = result.get("daily_itinerary", [])
    daily_routes    = result.get("daily_routes", [])

    if not daily_itinerary:
        errors.append("P3: daily_itinerary 为空（POI 聚类未执行）")
    if not daily_routes:
        errors.append("P3: daily_routes 为空（TSP 路线优化未执行）")
    else:
        for dr in daily_routes:
            day = dr.get("day")
            if not dr.get("ordered_pois"):
                errors.append(f"P3: 第{day}天 ordered_pois 为空")

    # P4
    if "accommodation_query" not in agents_ran:
        errors.append("P4: accommodation_query 未执行")
    else:
        ac = next(r for r in skill_results if r["agent_name"] == "accommodation_query")
        if ac.get("status") != "success":
            errors.append(f"P4: accommodation_query 状态异常: {ac.get('status')}")

    # P5
    if not result.get("final_response"):
        errors.append("P5: final_response 为空")

    # 输出结果
    print(f"\n{'!' * 60}" if errors else f"\n{'=' * 60}")
    if errors:
        print(f"[断言结果] {len(errors)} 项未通过：")
        for e in errors:
            print(f"  ❌ {e}")
    else:
        print("✅  全链路断言通过：P1 → P2 → P3 → P4 → P5 均正常联动")
    print(f"{'!' * 60}" if errors else f"{'=' * 60}")


# ─────────────────────────────────────────────────────────────────────────────
# 主测试入口
# ─────────────────────────────────────────────────────────────────────────────

async def run_test(query: str):
    print(f"\n{SEP}")
    print(f"[真实API全链路测试]")
    print(f"[测试查询] {query}")
    print(SEP)

    # 前置检查
    warnings = _check_prerequisites()
    if warnings:
        print("\n[前置检查警告]")
        for w in warnings:
            print(f"  {w}")
        print("  以上警告不会中断测试，但相关服务节点可能返回 error 状态\n")
    else:
        print("[前置检查] ✅ 环境依赖检查通过\n")

    # 构建 graph
    memory_manager = MemoryManager(
        user_id="test_user_real",
        session_id="test_real_pipeline_001"
    )
    graph = build_graph(
        memory_manager=memory_manager,
        checkpointer=MemorySaver()
    )
    config = {"configurable": {"thread_id": "test_real_pipeline_001"}}

    print("正在调用 graph.ainvoke（无 Mock，所有服务均为真实调用）...")
    print("预计耗时较长（30~120 秒），取决于网络和外部服务响应速度\n")

    t0 = time.time()
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=query)]},
        config=config,
    )
    elapsed = time.time() - t0

    _print_results(result, elapsed)
    _assert_full_pipeline(result)


if __name__ == "__main__":
    asyncio.run(run_test(
        "我下周六从北京出发，和女朋友去杭州玩2天，"
        "请帮我规划情侣行程，并推荐往返交通和住宿"
    ))
