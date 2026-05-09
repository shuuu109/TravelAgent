"""
全链路真实 API 端到端测试 (LangGraph 新框架)
"""

import sys
import os
import asyncio
import time
import json
import logging
from collections import defaultdict

# Windows GBK -> UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# 将项目根目录加入 sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# -- 日志级别配置 ------------------------------------------------------------------
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
# 关键节点打开 INFO 方便跟踪进度
logging.getLogger("agents.transport_agent").setLevel(logging.INFO)
logging.getLogger("agents.accommodation_agent").setLevel(logging.INFO)
logging.getLogger("agents.poi_agent").setLevel(logging.INFO)
logging.getLogger("graph.nodes.orchestrate_node").setLevel(logging.INFO)
logging.getLogger("graph.nodes.itinerary_planning_node").setLevel(logging.INFO)
logging.getLogger("graph.nodes.itinerary_planning_node_newcluster").setLevel(logging.INFO)

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from graph.workflow import build_graph
from context.memory_manager import MemoryManager

SEP  = "=" * 60
SEP2 = "-" * 60

# 节点名称（与 workflow.add_node 第一个参数一致）到可读标签的映射
PIPELINE_NODES: dict[str, str] = {
    "intent":               "P1   intent_node",
    "extract_constraints":  "P1.4 extract_constraints_node",
    "validate_constraints": "P1.5 validate_constraints_node",
    "negotiate":            "P1.5b negotiate_node",
    "rag":                  "P2   rag_node",
    "transport":            "P2   transport_node",
    "poi_fetch":            "P2   poi_fetch_node",
    "preference":           "P2   preference_node",
    "memory_query":         "P2   memory_query_node",
    "info_query":           "P2   info_query_node",
    "itinerary_planning":   "P3   itinerary_planning_node",
    "poi_enrich":           "P3.5 poi_enrich_node",
    "accommodation":        "P4   accommodation_node",
    "itinerary_review":     "P4.5 itinerary_review_node",
    "budget_check":         "P4.6 budget_check_node",
    "respond":              "P5   respond_node",
}


# -------------------------------------------------------------------------------
# 前置检查：快速验证外部服务可达性，减少因配置缺失导致的长时间卡顿
# -------------------------------------------------------------------------------

def _check_prerequisites() -> list[str]:
    """
    检查运行环境是否满足要求。
    返回所有警告信息列表（非致命，仍会继续执行）。
    """
    warnings: list[str] = []

    # 1. mcp-server-12306 命令是否存在
    import shutil
    if not shutil.which("mcp-server-12306"):
        warnings.append("[WARN]  mcp-server-12306 未在 PATH 中找到 -> 高铁查询将失败")

    # 2. RollingGo API Key
    try:
        from config import ROLLINGGO_MCP_CONFIG
        api_key = ROLLINGGO_MCP_CONFIG.get("ROLLINGGO_API_KEY", "")
        if not api_key or "your_" in api_key:
            warnings.append("[WARN]  ROLLINGGO_API_KEY 未配置 -> 酒店查询将失败")
        cmd = ROLLINGGO_MCP_CONFIG.get("command", "")
        if not os.path.exists(cmd) and not shutil.which(cmd):
            warnings.append(f"[WARN]  RollingGo MCP 命令不可用: {cmd}")
    except ImportError:
        warnings.append("[WARN]  无法导入 config.py -> 多项服务配置缺失")

    # 3. AMAP Key
    try:
        from config import AMAP_MCP_CONFIG
        amap_key = AMAP_MCP_CONFIG.get("AMAP_KEY", "")
        if not amap_key:
            warnings.append("[WARN]  AMAP_KEY 未配置 -> POI/路线查询将失败")
    except ImportError:
        pass

    return warnings


# -------------------------------------------------------------------------------
# 输出辅助
# -------------------------------------------------------------------------------

def _short(text: str, n: int = 120) -> str:
    """截断长文本，末尾加省略号。"""
    if not text:
        return ""
    return text[:n] + "..." if len(text) > n else text


def _get_hc_field(hc, field: str):
    """兼容 HardConstraints 对象和 dict 两种形态读取字段。"""
    if hc is None:
        return None
    if isinstance(hc, dict):
        return hc.get(field)
    return getattr(hc, field, None)


def _get_rag_field(rag_ctx, field: str):
    """兼容 RAGContext 对象和 dict 两种形态读取字段。"""
    if rag_ctx is None:
        return None
    if isinstance(rag_ctx, dict):
        return rag_ctx.get(field)
    return getattr(rag_ctx, field, None)


def _print_timing_summary(durations: dict[str, list[float]], total: float):
    """打印各节点耗时统计表，支持节点多次执行（review 回环）。"""
    node_order = [
        "intent", "extract_constraints", "validate_constraints",
        "negotiate",
        "rag", "transport", "poi_fetch",
        "preference", "memory_query", "info_query",
        "itinerary_planning",
        "poi_enrich", "accommodation", "itinerary_review",
        "budget_check", "respond",
    ]

    print(f"\n{SEP}")
    print("[各节点耗时统计]")
    print(f"  {'节点':<40} {'耗时':>7}  {'占比':>6}  进度条")
    print("  " + "-" * 72)

    accounted = 0.0
    for node in node_order:
        runs = durations.get(node)
        if not runs:
            continue
        label = PIPELINE_NODES[node]
        d = sum(runs)
        accounted += d
        pct = d / total * 100 if total > 0 else 0
        bar = "#" * max(1, int(pct / 2))  # 每个 # 约代表 2%
        runs_str = f" x{len(runs)}" if len(runs) > 1 else ""
        print(f"  {label:<40} {d:>6.1f}s  {pct:>5.1f}%  |{bar}{runs_str}")

    print("  " + "-" * 72)
    print(f"  {'总耗时 (wall clock)':<40} {total:>6.1f}s  100.0%")
    print(f"  {'各节点耗时之和':<40} {accounted:>6.1f}s")
    if total > 0 and accounted > total * 1.05:
        diff = accounted - total
        print(
            f"  [注] 耗时之和比总时间多 {diff:.1f}s，"
            "说明存在并行执行（orchestrate 内部 asyncio.gather）"
        )
    print(SEP)


def _print_results(result: dict, elapsed: float):
    """打印各阶段结果，覆盖 P1 -> P5 全链路及新增 P3.5 / P4.5 节点。"""

    # -- P1 意图识别 + P1.4 约束提取 -------------------------------------------
    intent_data = result.get("intent_data", {})
    print(f"\n{SEP2}")
    print("[P1 意图识别 + P1.4 硬约束提取]")
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

    # P1.4 hard_constraints（extract_constraints_node 写入）
    hc = result.get("hard_constraints")
    hc_origin = _get_hc_field(hc, "origin")
    hc_dest   = _get_hc_field(hc, "destination")
    hc_date   = _get_hc_field(hc, "start_date")
    print(f"  硬约束   : origin={hc_origin}  destination={hc_dest}  start_date={hc_date}")

    # P1 景点搜索提示词 + 住宿偏好 + 结构化知识库字段
    attr_hints   = result.get("attraction_hints", [])
    acc_prefs    = result.get("accommodation_prefs", {}) or {}
    best_season  = result.get("destination_best_season", "")
    hubs         = result.get("destination_transport_hubs", [])
    print(f"  景点搜索提示({len(attr_hints)} 条): {attr_hints}")
    if acc_prefs.get("brand_keywords") or acc_prefs.get("type") or acc_prefs.get("price_range"):
        print(f"  住宿偏好  : {acc_prefs}")
    if best_season:
        print(f"  最佳旅游季: {best_season}")
    if hubs:
        print(f"  交通枢纽  : {hubs}")

    # agent_schedule
    schedule = intent_data.get("agent_schedule", [])
    print(f"  调度计划 ({len(schedule)} 个 agent):")
    for s in schedule:
        print(
            f"    priority={s.get('priority')}  "
            f"[{s.get('agent_name')}]  "
            f"{_short(s.get('reason', ''), 60)}"
        )

    # -- P2 技能执行结果 --------------------------------------------------------
    skill_results = result.get("skill_results", [])
    print(f"\n{SEP2}")
    print(f"[P2 技能执行结果]  共 {len(skill_results)} 个 agent")

    for r in skill_results:
        agent_name = r.get("agent_name", "?")
        status     = r.get("status", "?")
        data       = r.get("data", {})
        print(f"\n  [{agent_name}]  status={status}")

        if agent_name == "transport_query":
            transport_plan = data.get("transport_plan", {})
            if transport_plan:
                qi = transport_plan.get("query_info", {})
                print(f"    data_source : {qi.get('data_source', '-')}")
                print(f"    查询区间    : {qi.get('origin')} -> {qi.get('destination')}  ({qi.get('date')})")
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
                    print(f"      {t_type:4s}  {t_no:8s}  {dep}->{arr}  {dur}  {price}")
                rec = transport_plan.get("recommendation", {})
                if rec:
                    print(f"    推荐        : {_short(rec.get('best_choice', ''), 80)}")
                    arrival = rec.get("arrival_hub") or rec.get("arrival_station", "未知")
                    print(f"    到达枢纽    : {arrival}")
            else:
                err_msg = data.get("error", "未知")
                print(f"    ERROR       : {err_msg}")
                if "Connection error" in str(err_msg):
                    print("    诊断    : LLM API 连接失败（model.ainvoke），非 MCP 问题")
                elif "mcp-server-12306" in str(err_msg).lower() or "FileNotFoundError" in str(err_msg):
                    print("    诊断    : mcp-server-12306 命令不存在，请确认已安装并在 PATH 中")

        elif agent_name == "poi_fetch":
            result_data = data.get("result", {})
            pois        = result_data.get("pois", [])
            err         = data.get("error")
            if err:
                print(f"    ERROR: {err}")
                if "mcp.amap.com" in str(err).lower() or "amap" in str(err).lower():
                    print("    诊断    : 高德 MCP 连接失败，请检查 AMAP_KEY 和网络")
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
                    print("    诊断    : RollingGo MCP 失败，检查 ROLLINGGO_API_KEY 和命令路径")

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

    # RAGContext（orchestrate_node 写入，聚合 rag_experience + rag_risk）
    rag_ctx    = result.get("rag_context")
    rag_exp    = _get_rag_field(rag_ctx, "rag_experience")
    rag_risk   = _get_rag_field(rag_ctx, "rag_risks")
    rag_snips  = _get_rag_field(rag_ctx, "rag_snippets") or []
    print(f"\n  [P2 RAG Context]  snippets={len(rag_snips)}")
    if rag_exp is not None:
        tips    = rag_exp.get("tips", []) if isinstance(rag_exp, dict) else getattr(rag_exp, "tips", [])
        best_for = rag_exp.get("best_for", []) if isinstance(rag_exp, dict) else getattr(rag_exp, "best_for", [])
        print(f"    rag_experience: {len(tips)} tips, {len(best_for)} best_for")
        for t in tips[:2]:
            print(f"      - {_short(t, 80)}")
    else:
        print("    rag_experience: (未写入 - 知识库可能为空或 rag_experience agent 未执行)")
    if rag_risk is not None:
        risks = rag_risk.get("risks", []) if isinstance(rag_risk, dict) else getattr(rag_risk, "risks", [])
        print(f"    rag_risk      : {len(risks)} 条")
        for rv in risks[:2]:
            print(f"      - {_short(rv, 80)}")
    else:
        print("    rag_risk      : (未写入 - 知识库可能为空或 rag_risk agent 未执行)")

    # poi_candidates（poi_fetch 写入 state）
    poi_candidates = result.get("poi_candidates", [])
    print(f"  [P2 POI候选总数]: {len(poi_candidates)}")

    # -- P3 行程规划结果 --------------------------------------------------------
    daily_itinerary = result.get("daily_itinerary", [])
    daily_routes    = result.get("daily_routes", [])
    print(f"\n{SEP2}")
    print(
        f"[P3 行程规划]  "
        f"daily_itinerary={len(daily_itinerary)} 天  "
        f"daily_routes={len(daily_routes)} 天"
    )

    if not daily_itinerary:
        print("  [WARN] daily_itinerary 为空（POI 未成功获取或 itinerary_planning_node 未执行）")
    else:
        for day_group in daily_itinerary:
            day  = day_group.get("day")
            pois = day_group.get("pois", [])
            names = " | ".join(p.get("name", "?") for p in pois)
            print(f"  第{day}天 ({len(pois)}个POI): {names}")

    if not daily_routes:
        print("  [WARN] daily_routes 为空（TSP 路线优化未执行或高德 MCP 连接失败）")
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
                    leg  = legs[i]
                    mode = leg.get("mode", "?")
                    dur  = leg.get("duration", 0)
                    route_str += f" ->({mode} {dur}min)-> "
            print(f"  [路线] 第{day}天: {route_str}  总交通: {total_d}min")

    # 每日餐厅推荐（P3 itinerary_planning_node 写入，可为空）
    daily_restaurants = result.get("daily_restaurants", [])
    if daily_restaurants:
        print(f"\n  [P3 餐厅推荐]  {len(daily_restaurants)} 天")
        for day_rest in daily_restaurants[:2]:
            day         = day_rest.get("day")
            restaurants = day_rest.get("restaurants", [])
            print(f"  第{day}天推荐餐厅 ({len(restaurants)} 家):")
            for rr in restaurants[:3]:
                print(
                    f"    {rr.get('name', '?')}  "
                    f"距离={rr.get('distance_m', '?')}m  "
                    f"评分={rr.get('amap_rating', '?')}"
                )
    else:
        print("  [P3 餐厅推荐] (空 - 高德周边搜索未执行或无结果)")

    # -- P3.5 POI 体验补充 (poi_enrich_node) ------------------------------------
    poi_descriptions = result.get("poi_descriptions")
    print(f"\n{SEP2}")
    if poi_descriptions is None:
        print("[P3.5 POI体验补充]  poi_descriptions 字段缺失（poi_enrich_node 未写入 state）")
    else:
        print(f"[P3.5 POI体验补充]  poi_descriptions={len(poi_descriptions)} 个")
        for name, desc in list(poi_descriptions.items())[:3]:
            print(f"  {name}: {_short(desc, 80)}")
        if not poi_descriptions:
            print("  (空 - 知识库无匹配文档或 daily_routes 为空)")

    # -- P4 住宿 geo-center 展示 ------------------------------------------------
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

    # -- P4.5 行程自检 (itinerary_review_node) ----------------------------------
    violations       = result.get("rule_violations") or []
    review_retry_cnt = result.get("review_retry_count", 0)
    print(f"\n{SEP2}")
    print(
        f"[P4.5 行程自检]  violations={len(violations)}  "
        f"review_retry_count={review_retry_cnt}"
    )
    if violations:
        for v in violations[:5]:
            vtype = v.get("violation_type", "?") if isinstance(v, dict) else getattr(v, "violation_type", "?")
            vdesc = v.get("description", "") if isinstance(v, dict) else getattr(v, "description", "")
            vsug  = v.get("suggestion", "") if isinstance(v, dict) else getattr(v, "suggestion", "")
            print(f"  [{vtype}] {_short(vdesc, 80)}")
            if vsug:
                print(f"    建议: {_short(vsug, 70)}")
    else:
        print("  自检通过，无违规")

    # -- P5 最终回复 ------------------------------------------------------------
    final_response = result.get("final_response", "")
    print(f"\n{SEP2}")
    print("[P5 最终回复 (final_response)]")
    print(final_response if final_response else "(空)")

    # -- 全局耗时 ---------------------------------------------------------------
    print(f"\n{SEP2}")
    print(f"[总耗时] {elapsed:.1f} 秒（真实 API 调用，含网络 I/O）")
    print(SEP)


# -------------------------------------------------------------------------------
# 断言验证：单节点字段检查 + graph 级别端到端断言
# -------------------------------------------------------------------------------

def _assert_full_pipeline(result: dict):
    """
    分三层对 pipeline 输出进行断言：

    Layer 1 - 节点字段检查：每个节点的关键输出字段是否存在且有效。
    Layer 2 - E2E 执行链检查：通过 state 字段存在性推断各节点是否实际执行。
    Layer 3 - 状态一致性检查：跨节点的数据流转是否一致（无数据污染/截断）。

    任何失败项以 [FAIL] 打印，全部通过则显示 [OK]。
    """
    errors: list[str] = []

    # =========================================================================
    # Layer 1 - 节点字段检查
    # =========================================================================

    # P1 intent_node
    if not result.get("intent_data"):
        errors.append("P1: intent_data 为空")
    if not result.get("travel_style"):
        errors.append("P1: travel_style 未写入 state")

    travel_days = result.get("travel_days", 0)
    if not travel_days:
        errors.append("P1: travel_days 未提取（duration 解析失败）")
    # travel_days 依赖 LLM 解析 duration 字段，可能将"往返各1天+游玩2天"解读为 4 天。
    # 此处只检查非零，具体数值由 daily_itinerary 天数一致性检查（Layer 3）保障。

    attr_hints = result.get("attraction_hints", [])
    if not attr_hints:
        errors.append("P1: attraction_hints 为空（LLM 未生成景点搜索提示词）")

    # P1.4 extract_constraints_node
    hc = result.get("hard_constraints")
    if hc is None:
        errors.append("P1.4: hard_constraints 字段缺失（extract_constraints_node 未写入 state）")
    else:
        hc_origin = _get_hc_field(hc, "origin")
        hc_dest   = _get_hc_field(hc, "destination")
        hc_date   = _get_hc_field(hc, "start_date")
        if not hc_origin:
            errors.append("P1.4: hard_constraints.origin 未提取（应为北京）")
        if not hc_dest:
            errors.append("P1.4: hard_constraints.destination 未提取（应为杭州）")
        if not hc_date:
            errors.append("P1.4: hard_constraints.start_date 未提取")

    # P2 transport_query
    skill_results = result.get("skill_results", [])
    agents_ran    = [r.get("agent_name") for r in skill_results]

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
                errors.append("P2: transport_query 返回的 options 列表为空（真实 API 未返回数据？）")

    # P2 poi_fetch
    if "poi_fetch" not in agents_ran:
        errors.append("P2: poi_fetch 未执行")
    else:
        pf   = next(r for r in skill_results if r["agent_name"] == "poi_fetch")
        pois = pf.get("data", {}).get("result", {}).get("pois", [])
        if not pois:
            errors.append("P2: poi_fetch 返回空 POI 列表（高德 MCP 连接失败？）")
        elif len(pois) < 3:
            errors.append(f"P2: poi_fetch 仅返回 {len(pois)} 个 POI（预期 >= 3）")

    # P3 itinerary_planning_node
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

    # P3.5 poi_enrich_node（字段必须存在，但允许空 dict - 知识库可无内容）
    if result.get("poi_descriptions") is None:
        errors.append("P3.5: poi_descriptions 字段缺失（poi_enrich_node 未写入 state）")

    # P4 accommodation_node（仍写入 skill_results）
    if "accommodation_query" not in agents_ran:
        errors.append("P4: accommodation_query 未执行")
    else:
        ac = next(r for r in skill_results if r["agent_name"] == "accommodation_query")
        if ac.get("status") != "success":
            errors.append(f"P4: accommodation_query 状态异常: {ac.get('status')}")

    # P5 respond_node
    if not result.get("final_response"):
        errors.append("P5: final_response 为空")

    # =========================================================================
    # Layer 2 - E2E 执行链检查
    # 通过 state 字段存在性推断各节点是否实际执行，确保图拓扑被完整遍历
    # =========================================================================
    chain_checks: list[tuple[str, bool, str]] = [
        (
            "P1  intent_node",
            bool(result.get("intent_data")),
            "intent_data 为空，intent_node 可能未执行",
        ),
        (
            "P1.4 extract_constraints_node",
            result.get("hard_constraints") is not None,
            "hard_constraints 字段缺失，extract_constraints_node 可能未执行",
        ),
        (
            "P2  fan-out (rag/transport/poi_fetch)",
            bool(result.get("skill_results")),
            "skill_results 为空，P2 fan-out 三节点可能未执行",
        ),
        (
            "P3  itinerary_planning_node",
            result.get("daily_itinerary") is not None,
            "daily_itinerary 字段缺失（None），itinerary_planning_node 可能未执行",
        ),
        (
            "P3.5 poi_enrich_node",
            result.get("poi_descriptions") is not None,
            "poi_descriptions 字段缺失（None），poi_enrich_node 可能未执行",
        ),
        (
            "P4  accommodation_node",
            any(r.get("agent_name") == "accommodation_query" for r in result.get("skill_results", [])),
            "skill_results 中无 accommodation_query，accommodation_node 可能未执行",
        ),
        (
            "P4.5 itinerary_review_node",
            "rule_violations" in result,
            "rule_violations 字段缺失，itinerary_review_node 可能未执行",
        ),
        (
            "P5  respond_node",
            bool(result.get("final_response")),
            "final_response 为空，respond_node 可能未执行",
        ),
    ]

    chain_errors: list[str] = []
    for node_label, passed, msg in chain_checks:
        if not passed:
            chain_errors.append(f"  [E2E链] {node_label}: {msg}")

    # =========================================================================
    # Layer 3 - 状态一致性检查
    # =========================================================================
    consistency_errors: list[str] = []

    # hard_constraints.destination 应与 intent_data.key_entities.destination 一致
    hc    = result.get("hard_constraints")
    id_kv = (result.get("intent_data") or {}).get("key_entities") or {}
    hc_dest_val = _get_hc_field(hc, "destination")
    id_dest_val = id_kv.get("destination", "")
    if hc_dest_val and id_dest_val and hc_dest_val != id_dest_val:
        consistency_errors.append(
            f"  [一致性] hard_constraints.destination={hc_dest_val!r} "
            f"!= intent_data.key_entities.destination={id_dest_val!r}"
        )

    # travel_days 应与 daily_itinerary / daily_routes 天数对齐
    if travel_days > 0:
        if daily_itinerary and len(daily_itinerary) != travel_days:
            consistency_errors.append(
                f"  [一致性] travel_days={travel_days} 与 daily_itinerary "
                f"天数 {len(daily_itinerary)} 不符"
            )
        if daily_routes and len(daily_routes) != travel_days:
            consistency_errors.append(
                f"  [一致性] travel_days={travel_days} 与 daily_routes "
                f"天数 {len(daily_routes)} 不符"
            )

    # daily_itinerary 与 daily_routes 天数必须一致
    if daily_itinerary and daily_routes and len(daily_itinerary) != len(daily_routes):
        consistency_errors.append(
            f"  [一致性] daily_itinerary {len(daily_itinerary)} 天 "
            f"!= daily_routes {len(daily_routes)} 天"
        )

    # transport_options 非空意味着 transport_query 成功写入 state
    transport_options = result.get("transport_options", [])
    if transport_options and "transport_query" not in agents_ran:
        consistency_errors.append(
            "  [一致性] transport_options 有数据但 skill_results 中无 transport_query"
        )

    # poi_candidates 非空意味着 poi_fetch 成功写入 state
    poi_candidates = result.get("poi_candidates", [])
    if poi_candidates and "poi_fetch" not in agents_ran:
        consistency_errors.append(
            "  [一致性] poi_candidates 有数据但 skill_results 中无 poi_fetch"
        )

    # =========================================================================
    # 汇总输出
    # =========================================================================
    all_errors = errors + chain_errors + consistency_errors
    divider = "!" * 60 if all_errors else "=" * 60
    print(f"\n{divider}")

    if all_errors:
        print(f"[断言结果] {len(all_errors)} 项未通过：")
        layer1_cnt = len(errors)
        layer2_cnt = len(chain_errors)
        layer3_cnt = len(consistency_errors)
        if errors:
            print(f"  -- Layer 1 节点字段检查 ({layer1_cnt} 项) --")
            for e in errors:
                print(f"  [FAIL] {e}")
        if chain_errors:
            print(f"  -- Layer 2 E2E执行链检查 ({layer2_cnt} 项) --")
            for e in chain_errors:
                print(f"  [FAIL] {e}")
        if consistency_errors:
            print(f"  -- Layer 3 状态一致性检查 ({layer3_cnt} 项) --")
            for e in consistency_errors:
                print(f"  [FAIL] {e}")
    else:
        print(
            "[OK]  全链路断言通过（Layer1 字段检查 + Layer2 E2E执行链 + Layer3 状态一致性）\n"
            "      P1 -> P1.4 -> P1.5 -> P2 -> P3 -> P3.5 -> P4 -> P4.5 -> P5"
        )
    print(divider)


# -------------------------------------------------------------------------------
# 主测试入口
# -------------------------------------------------------------------------------

async def run_test(query: str):
    print(f"\n{SEP}")
    print("[真实API全链路测试]")
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
        print("[前置检查] [OK] 环境依赖检查通过\n")

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

    print("正在执行 graph（无 Mock，所有服务均为真实调用）...")
    print("预计耗时较长（30~120 秒），取决于网络和外部服务响应速度")
    print("各节点进度将实时打印...\n")

    # node_name -> 本次节点开始时的绝对时间戳（用于计算本次耗时）
    node_start_ts: dict[str, float] = {}
    # node_name -> 历次耗时列表（支持 review 回环导致的多次执行）
    node_durations: dict[str, list[float]] = defaultdict(list)

    t0 = time.time()

    async for event in graph.astream_events(
        {"messages": [HumanMessage(content=query)]},
        config=config,
        version="v2",
    ):
        etype = event["event"]
        name  = event.get("name", "")
        now   = time.time()

        if etype == "on_chain_start" and name in PIPELINE_NODES:
            node_start_ts[name] = now
            run_num = len(node_durations[name]) + 1
            ts      = now - t0
            label   = PIPELINE_NODES[name]
            suffix  = f" (第{run_num}次)" if run_num > 1 else ""
            print(f"  [{ts:6.1f}s] >> {label}{suffix}")

        elif etype == "on_chain_end" and name in PIPELINE_NODES:
            if name in node_start_ts:
                duration = now - node_start_ts.pop(name)
                node_durations[name].append(duration)
                ts    = now - t0
                label = PIPELINE_NODES[name]
                print(f"  [{ts:6.1f}s]    {label} 完成  {duration:.1f}s")

    elapsed = time.time() - t0

    # 通过 checkpointer 获取最终 state（MemorySaver 在同一 graph 实例内有效）
    snapshot = await graph.aget_state(config)
    result   = snapshot.values if snapshot else {}

    _print_timing_summary(node_durations, elapsed)
    _print_results(result, elapsed)
    _assert_full_pipeline(result)


if __name__ == "__main__":
    asyncio.run(run_test(
        "我后天从南京出发，带小孩去北京玩3天，"
        "请帮我规划行程,预算5000元，住宿推荐连锁酒店，"
    ))
