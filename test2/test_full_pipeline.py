"""
全链路 Mock 端到端测试 (LangGraph 新框架)
==========================================

验证完整 P1→P2→P3→P4→P5 管道在 Mock 数据下正常运行：
  P1  intent_node              — 意图识别（LLM 真实调用）
  P2  orchestrate_node         — 并行编排（transport_query + poi_fetch + ...）
  P3  itinerary_planning_node  — POI 筛选 → 地理聚类 → TSP 路线优化
  P4  accommodation_node       — 按每日活动重心推荐酒店
  P5  respond_node             — 汇总生成自然语言回复

Mock 覆盖范围（均无需真实网络连接）：
  ✓  高铁查询      agents.transport_agent → train_client.query_tickets / query_ticket_price
  ✓  航班查询      agents.transport_agent → flight_client.query_tickets / get_airport_weather
  ✓  酒店搜索      mcp_clients.hotel_client.search_hotels
  ✓  高德POI搜索   agents.poi_agent.amap_mcp_session + search_pois
  ✓  高德距离矩阵  graph.nodes.itinerary_planning_node.get_distance_matrix
  ✓  高德公交路线  graph.nodes.itinerary_planning_node.get_transit_route

未 Mock（真实 LLM 调用）：
  ·  intent_node（意图识别）
  ·  transport_agent 内部 LLM 分析
  ·  accommodation_agent 内部 LLM 分析
  ·  respond_node（回复生成）

测试场景：亲子游，广州→成都，3天行程 + 往返交通 + 住宿推荐
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
# 打开 httpx/httpcore 的 WARNING 日志，以捕获 LLM API 连接异常
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
# 打开各 agent 的 INFO 日志，方便跟踪执行流程
logging.getLogger("agents.transport_agent").setLevel(logging.INFO)
logging.getLogger("agents.accommodation_agent").setLevel(logging.INFO)
logging.getLogger("graph.nodes.orchestrate_node").setLevel(logging.INFO)

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from graph.workflow import build_graph
from context.memory_manager import MemoryManager

SEP  = "=" * 60
SEP2 = "-" * 60


# ─────────────────────────────────────────────────────────────────────────────
# Mock 数据：交通（广州→成都）
# ─────────────────────────────────────────────────────────────────────────────

MOCK_TRAIN_TICKETS = """
车次查询结果（广州南→成都东，下周五）：
D2809  广州南站  08:00 → 成都东站  16:55  历时8小时55分  二等座：有票  一等座：有票
G1978  广州南站  09:30 → 成都东站  18:12  历时8小时42分  二等座：有票  一等座：有票
G2829  广州南站  13:00 → 成都东站  22:05  历时9小时05分  二等座：有票  一等座：无票
"""

MOCK_TRAIN_PRICE = """
票价查询结果（广州南→成都东）：
D2809: 二等座¥398  一等座¥624  商务座¥1168
G1978: 二等座¥429  一等座¥680  商务座¥1268
G2829: 二等座¥398  一等座¥624  商务座¥1168
"""

MOCK_FLIGHT_TICKETS = """
航班查询结果（广州白云→成都，下周五）：
CZ3475  广州白云机场(CAN) 07:20 → 成都天府机场(TFU) 09:10  历时1小时50分  经济舱¥680  余票充足（最早到达）
3U8901  广州白云机场(CAN) 14:10 → 成都双流机场(CTU) 16:05  历时1小时55分  经济舱¥520  余票充足（最低价）
CA4399  广州白云机场(CAN) 19:30 → 成都天府机场(TFU) 21:25  历时1小时55分  经济舱¥790  余票有限
"""

MOCK_WEATHER = """
成都天气预报（近期）：
多云转晴，气温15-22℃，微风，适合户外亲子活动，建议备薄外套和防晒用品。
"""


# ─────────────────────────────────────────────────────────────────────────────
# Mock 数据：酒店（成都，亲子友好）
# ─────────────────────────────────────────────────────────────────────────────

MOCK_HOTELS_DATA = {
    "message": "success",
    "hotelInformationList": [
        {
            "hotelId": "C001",
            "hotelName": "成都香格里拉大酒店",
            "starLevel": 5,
            "price": 988,
            "address": "成都市锦江区滨江东路9号",
            "features": "邻近春熙路/锦里，室内恒温亲子泳池，儿童活动中心，亲子主题套房"
        },
        {
            "hotelId": "C002",
            "hotelName": "成都锦江宾馆",
            "starLevel": 5,
            "price": 768,
            "address": "成都市锦江区人民南路二段80号",
            "features": "地铁1号线直达，儿童托管服务，近武侯祠，家庭房型"
        },
        {
            "hotelId": "C003",
            "hotelName": "成都宽窄巷子精品酒店",
            "starLevel": 4,
            "price": 459,
            "address": "成都市青羊区宽巷子26号",
            "features": "步行至宽窄巷子，川式装修，家庭房型，早餐含川味小吃"
        },
        {
            "hotelId": "C004",
            "hotelName": "成都天府希尔顿逸林酒店",
            "starLevel": 4,
            "price": 598,
            "address": "成都市天府新区兴隆湖公园旁",
            "features": "天府机场30分钟，户外泳池，儿童游乐区，景区距离适中"
        }
    ]
}


def _make_hotel_mcp_result():
    """
    构造模拟 MCP CallToolResult 对象。
    accommodation_agent 访问 result.content[0].text 并 json.loads 解析。
    """
    mock_content = MagicMock()
    mock_content.text = json.dumps(MOCK_HOTELS_DATA, ensure_ascii=False)
    mock_result = MagicMock()
    mock_result.content = [mock_content]
    return mock_result


# ─────────────────────────────────────────────────────────────────────────────
# Mock 数据：高德 POI（成都景点 / 餐厅 / 体验）
# ─────────────────────────────────────────────────────────────────────────────
# search_pois() 原始返回格式：name / location("lng,lat") / address / rating
# _normalize_pois() 负责将 location 拆成 lng/lat 浮点数，并按 top_n 截断

MOCK_POI_ATTRACTIONS = [
    {"name": "大熊猫繁育研究基地", "location": "104.142456,30.737946",
     "address": "成都市成华区熊猫大道1375号", "rating": "4.9"},
    {"name": "锦里古街",           "location": "104.041821,30.654289",
     "address": "成都市武侯区武侯祠大街231号", "rating": "4.7"},
    {"name": "武侯祠博物馆",       "location": "104.041432,30.647521",
     "address": "成都市武侯区武侯祠大街231号", "rating": "4.8"},
    {"name": "宽窄巷子",           "location": "104.059231,30.670891",
     "address": "成都市青羊区长顺上街104号",   "rating": "4.6"},
    {"name": "都江堰景区",         "location": "103.617123,30.998876",
     "address": "成都市都江堰市灌县古城",      "rating": "4.8"},
    {"name": "青城山前山",         "location": "103.572034,30.899213",
     "address": "成都市都江堰市青城山镇",      "rating": "4.7"},
    {"name": "成都博物馆",         "location": "104.063210,30.669870",
     "address": "成都市青羊区小河街3号",       "rating": "4.7"},
    {"name": "天府广场",           "location": "104.065723,30.657891",
     "address": "成都市青羊区天府广场",        "rating": "4.4"},
]

MOCK_POI_RESTAURANTS = [
    {"name": "陈麻婆豆腐（总店）", "location": "104.074523,30.672341",
     "address": "成都市青羊区西玉龙街197号", "rating": "4.5"},
    {"name": "龙抄手（春熙路店）", "location": "104.081234,30.659876",
     "address": "成都市锦江区春熙路步行街",  "rating": "4.4"},
    {"name": "蜀九香老火锅",       "location": "104.068912,30.661234",
     "address": "成都市锦江区红星路三段1号", "rating": "4.6"},
    {"name": "钟水饺（总府路）",   "location": "104.073456,30.663120",
     "address": "成都市锦江区总府路",        "rating": "4.3"},
]

MOCK_POI_EXPERIENCES = [
    {"name": "锦江夜游",           "location": "104.072345,30.654123",
     "address": "成都市锦江区合江亭",   "rating": "4.5"},
    {"name": "川剧变脸体验馆",     "location": "104.066789,30.668234",
     "address": "成都市青羊区琴台路",   "rating": "4.4"},
    {"name": "熊猫邮局（大慈寺）", "location": "104.082345,30.659123",
     "address": "成都市锦江区大慈寺路", "rating": "4.3"},
]


# ─────────────────────────────────────────────────────────────────────────────
# Mock 工具函数：高德 MCP
# ─────────────────────────────────────────────────────────────────────────────

def _make_amap_session_mock():
    """
    创建模拟高德 MCP 异步上下文管理器。
    amap_mcp_session() 是 @asynccontextmanager，yield 一个 ClientSession。
    这里 yield 一个 MagicMock，供 get_distance_matrix / get_transit_route 接收。
    """
    mock_session = MagicMock()
    mock_cm = MagicMock()
    mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
    mock_cm.__aexit__ = AsyncMock(return_value=None)
    return mock_cm


async def _mock_search_pois(session, city: str, keywords: str, **kwargs):
    """
    按关键词类型返回对应类别的 Mock POI 列表。
    命名规则与 _CATEGORY_KEYWORDS 对应：景点 / 餐厅 / 体验
    """
    if "餐厅" in keywords:
        return MOCK_POI_RESTAURANTS
    elif "体验" in keywords:
        return MOCK_POI_EXPERIENCES
    else:  # 景点（默认）
        return MOCK_POI_ATTRACTIONS


async def _mock_get_distance_matrix(session, origins, destinations, **kwargs):
    """
    根据坐标列表长度自动生成 n×n 时间矩阵（秒）。
    对角线为 0；非对角线模拟 10~50 分钟的市内交通时间。
    """
    n = len(origins)
    # 预设 4×4 基础矩阵（秒），按实际 n 截取
    base = [
        [0,    1800, 2400, 3000],
        [1800, 0,    1500, 2700],
        [2400, 1500, 0,    1200],
        [3000, 2700, 1200, 0   ],
    ]
    return [row[:n] for row in base[:n]]


async def _mock_get_transit_route(session, origin: str, destination: str,
                                  city: str, **kwargs):
    """模拟高德公交路线：固定返回地铁30分钟方案。"""
    return {
        "duration": 30,
        "recommended_mode": "地铁",
        "steps": ["乘坐地铁至目的地附近站，步行约5分钟"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# 输出辅助
# ─────────────────────────────────────────────────────────────────────────────

def _short(text: str, n: int = 120) -> str:
    if not text:
        return ""
    return text[:n] + "..." if len(text) > n else text


def _print_results(result: dict):
    # ── P1 意图识别 ───────────────────────────────────────────────────────────
    intent_data = result.get("intent_data", {})
    print(f"\n{SEP2}")
    print("[P1 意图识别]")
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
                # 诊断提示：区分 MCP 失败 vs LLM 失败
                if "Connection error" in str(err_msg):
                    print(f"   诊断    : 'Connection error' 通常来自 LLM API（Doubao）连接失败，")
                    print(f"               与 MCP Mock 无关。检查 Mock 调用统计——若 call_count>0")
                    print(f"               则 Mock 正常，错误发生在后续的 model.ainvoke() 阶段。")

        elif agent_name == "poi_fetch":
            result_data = data.get("result", {})
            pois = result_data.get("pois", [])
            err = data.get("error")
            if err:
                print(f"    ERROR: {err}")
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
                print(f"    ERROR: {data.get('error', '未知')}")

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
                err_msg = data.get("error", "")
                if "Connection error" in str(err_msg):
                    print(f"诊断    : LLM API 连接失败（model.ainvoke），非 Mock 问题")
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
        print("daily_itinerary 为空（POI 未成功获取或 itinerary_planning_node 未执行）")
    else:
        for day_group in daily_itinerary:
            day  = day_group.get("day")
            pois = day_group.get("pois", [])
            names = " | ".join(p.get("name", "?") for p in pois)
            print(f"  第{day}天 ({len(pois)}个POI): {names}")

    if not daily_routes:
        print("  ⚠️  daily_routes 为空（TSP 路线优化未执行）")
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
    print(SEP)


# ─────────────────────────────────────────────────────────────────────────────
# 断言验证：快速暴露各阶段联动问题
# ─────────────────────────────────────────────────────────────────────────────

def _assert_full_pipeline(result: dict):
    """对关键字段做断言，任何失败项以 ❌ 打印，全部通过则显示 ✅。"""
    errors: list = []

    # P1 —— 意图识别
    if not result.get("intent_data"):
        errors.append("P1: intent_data 为空")
    if not result.get("travel_style"):
        errors.append("P1/P2: travel_style 未写入 state（仅在 poi_fetch 执行后由 orchestrate_node 写入）")

    # P2 —— 技能执行
    skill_results = result.get("skill_results", [])
    agents_ran = [r.get("agent_name") for r in skill_results]

    if "transport_query" not in agents_ran:
        errors.append("P2: transport_query 未执行")
    else:
        tr = next(r for r in skill_results if r["agent_name"] == "transport_query")
        if tr.get("status") != "success":
            errors.append(f"P2: transport_query 状态异常: {tr.get('status')} | {tr.get('message', '')}")
        else:
            # 检查 transport_plan 结构
            tp = tr.get("data", {}).get("transport_plan", {})
            if not tp.get("options"):
                errors.append("P2: transport_query 返回的 options 列表为空")

    if "poi_fetch" not in agents_ran:
        errors.append("P2: poi_fetch 未执行（itinerary 类意图未触发自动注入？）")
    else:
        pf = next(r for r in skill_results if r["agent_name"] == "poi_fetch")
        pois = pf.get("data", {}).get("result", {}).get("pois", [])
        if not pois:
            errors.append("P2: poi_fetch 返回空 POI 列表（Mock 数据未生效？）")
        elif len(pois) < 3:
            errors.append(f"P2: poi_fetch 仅返回 {len(pois)} 个 POI（预期 ≥ 3）")

    # P3 —— 行程规划
    daily_itinerary = result.get("daily_itinerary", [])
    daily_routes    = result.get("daily_routes", [])

    if not daily_itinerary:
        errors.append("P3: daily_itinerary 为空（POI 聚类未执行）")
    if not daily_routes:
        errors.append("P3: daily_routes 为空（TSP 路线优化未执行）")
    else:
        # 验证路线结构完整性
        for dr in daily_routes:
            day = dr.get("day")
            if not dr.get("ordered_pois"):
                errors.append(f"P3: 第{day}天 ordered_pois 为空")
            # legs 可为空（MCP 失败 fallback），不强制要求

    # P4 —— 住宿
    if "accommodation_query" not in agents_ran:
        errors.append("P4: accommodation_query 未执行")
    else:
        ac = next(r for r in skill_results if r["agent_name"] == "accommodation_query")
        if ac.get("status") != "success":
            errors.append(f"P4: accommodation_query 状态异常: {ac.get('status')}")

    # P5 —— 最终回复
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
    print(f"[测试查询] {query}")
    print(SEP)

    memory_manager = MemoryManager(
        user_id="test_user",
        session_id="test_full_pipeline_001"
    )
    graph = build_graph(
        memory_manager=memory_manager,
        checkpointer=MemorySaver()
    )
    config = {"configurable": {"thread_id": "test_full_pipeline_001"}}

    print("正在调用 graph.ainvoke（全链路 Mock，无需真实 MCP 连接）...")
    print("注：LLM 为真实调用（Doubao API），并发时可能出现 Connection error\n")

    from mcp_clients.train_client import train_client
    from mcp_clients.flight_client import flight_client

    with \
        patch.object(train_client, "query_tickets",
                     new_callable=AsyncMock, return_value=MOCK_TRAIN_TICKETS), \
        patch.object(train_client, "query_ticket_price",
                     new_callable=AsyncMock, return_value=MOCK_TRAIN_PRICE), \
        patch.object(flight_client, "query_tickets",
                     new_callable=AsyncMock, return_value=MOCK_FLIGHT_TICKETS), \
        patch.object(flight_client, "get_airport_weather",
                     new_callable=AsyncMock, return_value=MOCK_WEATHER), \
        patch("mcp_clients.hotel_client.search_hotels",
              new_callable=AsyncMock, return_value=_make_hotel_mcp_result()), \
        patch("agents.poi_agent.amap_mcp_session",
              return_value=_make_amap_session_mock()), \
        patch("agents.poi_agent.search_pois",
              new_callable=AsyncMock, side_effect=_mock_search_pois), \
        patch("graph.nodes.itinerary_planning_node.amap_mcp_session",
              return_value=_make_amap_session_mock()), \
        patch("graph.nodes.itinerary_planning_node.get_distance_matrix",
              new_callable=AsyncMock, side_effect=_mock_get_distance_matrix), \
        patch("graph.nodes.itinerary_planning_node.get_transit_route",
              new_callable=AsyncMock, side_effect=_mock_get_transit_route):

        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=query)]},
            config=config,
        )

        # ── Mock 调用统计：验证 Mock 是否真的生效 ─────────────────────────────
        print("\n[Mock 调用统计]  (call_count > 0 = Mock 生效)")
        print(f"  train_client.query_tickets        : {train_client.query_tickets.call_count} 次")
        print(f"  train_client.query_ticket_price   : {train_client.query_ticket_price.call_count} 次")
        print(f"  flight_client.query_tickets       : {flight_client.query_tickets.call_count} 次")
        print(f"  flight_client.get_airport_weather : {flight_client.get_airport_weather.call_count} 次")

    _print_results(result)
    _assert_full_pipeline(result)


if __name__ == "__main__":
    asyncio.run(run_test(
        "我下周五从广州出发，带孩子去成都玩3天，"
        "请帮我规划亲子行程，并推荐往返交通和住宿"
    ))
