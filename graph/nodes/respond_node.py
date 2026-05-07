"""
回复生成节点 respond_node
职责：根据 skill_results 和 intent_data 生成用户可读的自然语言回复

来源：提取自 cli.py 的 _display_results / _generate_human_response 逻辑，
      并新增 LLM 兜底汇总（当规则生成无输出时）。

输入（从 state 读取）：
  - skill_results: List[Dict]   各 skill 的执行结果
  - intent_data: Dict           IntentionAgent 的完整输出

输出：
  {"final_response": response_text, "messages": [AIMessage(content=response_text)]}
"""
import json
import logging
from typing import List, Dict, Any, Optional

from langchain_core.messages import AIMessage
from graph.state import TravelGraphState, ensure_hard_constraints
from utils.knowledge_parser import CityKnowledgeDB

logger = logging.getLogger(__name__)


def create_respond_node(llm):
    """
    工厂函数：将 LLM 实例通过闭包注入，用于兜底汇总。

    Args:
        llm: LangChain ChatOpenAI 实例

    Returns:
        async 节点函数 respond_node(state) -> dict
    """

    async def respond_node(state: TravelGraphState) -> dict:
        """
        回复生成节点：
        1. 若 daily_routes 有值，用结构化路线格式渲染每日行程
        2. 优先用规则逻辑（从 cli.py 提取）生成各 skill 文字片段
        3. 若规则未产生输出，调用 LLM 做自然语言汇总
        4. 返回 final_response 和追加到 messages 的 AIMessage
        """
        skill_results: List[Dict] = state.get("skill_results", [])
        intent_data: Dict[str, Any] = state.get("intent_data", {})
        daily_routes: List[Dict] = state.get("daily_routes", [])
        # poi_descriptions 由 poi_enrich_node 写入：{景点名: 1-2句体验描述}
        poi_descriptions: Dict[str, str] = state.get("poi_descriptions") or {}

        # =====================================================================
        # 第一步：daily_routes 优先路径 — 结构化行程渲染
        # =====================================================================
        text_parts: List[str] = []
        has_daily_routes = bool(daily_routes)

        if has_daily_routes:
            daily_restaurants: List[Dict] = state.get("daily_restaurants", [])
            text_parts.append(_format_daily_routes(daily_routes, poi_descriptions, daily_restaurants))

        # =====================================================================
        # 第二步：用规则逻辑生成各 agent 的文字片段
        # =====================================================================
        if not skill_results and not has_daily_routes:
            text_parts.append("好的，我已记录下来。您可以继续补充信息，或尝试规划行程、查询信息。")
        else:
            seen_agents: set = set()
            for result in skill_results:
                agent_name = result.get("agent_name", "")
                # 同一 agent 多次出现时只渲染第一条，避免重复输出
                if agent_name in seen_agents:
                    continue
                seen_agents.add(agent_name)
                status = result.get("status", "")
                data = result.get("data", {})

                # daily_routes 已渲染行程，跳过 skill_results 中的 itinerary_planning 避免重复
                if agent_name == "itinerary_planning" and has_daily_routes:
                    continue

                # 行程已有时，event_collection 的摘要性文本无需重复展示
                if agent_name == "event_collection" and has_daily_routes:
                    continue

                if status == "error":
                    error_msg = data.get("error", "未知错误")
                    display_name = _get_agent_display_name(agent_name)
                    text_parts.append(f"{display_name}执行失败: {error_msg}")
                    continue

                if status != "success":
                    continue

                # rag_experience / rag_risk 的内容已从结构化 state 字段渲染，跳过 skill_results
                if agent_name in ("rag_experience", "rag_risk"):
                    continue

                part = _format_agent_result(
                    agent_name, data, skill_results,
                    has_daily_routes=has_daily_routes,
                )
                if part:
                    text_parts.append(part)

        # =====================================================================
        # 第二步：若规则无输出，用 LLM 做兜底汇总
        # =====================================================================
        if not text_parts and skill_results:
            llm_summary = await _llm_summarize(skill_results, intent_data, llm)
            text_parts.append(llm_summary)

        # =====================================================================
        # 末尾追加：结构化 RAG 输出区块（仅在有完整行程时附上）
        # =====================================================================
        if has_daily_routes:
            _rag_ctx = state.get("rag_context")
            raw_tips: List[str] = []
            raw_risks: List[str] = []

            # 收集并去重 rag_experience tips（过滤景点/路线推荐类条目）
            rag_experience = _rag_ctx.rag_experience if _rag_ctx else None
            if rag_experience and getattr(rag_experience, "tips", None):
                seen_keys: set = set()
                for t in rag_experience.tips:
                    if _is_poi_recommendation(t):
                        continue
                    key = _clean_tip(t)[:40].lower().strip()
                    if key not in seen_keys:
                        seen_keys.add(key)
                        raw_tips.append(_clean_tip(t))

            # 收集并去重 rag_risks
            rag_risks_data = _rag_ctx.rag_risks if _rag_ctx else None
            if rag_risks_data and getattr(rag_risks_data, "risks", None):
                seen_keys = set()
                for r in rag_risks_data.risks:
                    key = _clean_tip(r)[:40].lower().strip()
                    if key not in seen_keys:
                        seen_keys.add(key)
                        raw_risks.append(_clean_tip(r))

            if raw_tips or raw_risks:
                # 用 LLM 结合用户特征筛选润色，固定输出 3 条贴士 + 2 条避坑
                user_ctx = _build_user_context(state)
                tips_block, risks_block = await _llm_refine_tips_risks(
                    raw_tips, raw_risks, user_ctx, llm
                )
                if tips_block:
                    text_parts.append(tips_block)
                if risks_block:
                    text_parts.append(risks_block)
            else:
                # RAG 无数据时降级到 CityKnowledgeDB 静态 tips（最多3条）
                hard_constraints = ensure_hard_constraints(state.get("hard_constraints"))
                city = hard_constraints.destination or ""
                if not city:
                    intent_data_local: dict = state.get("intent_data") or {}
                    city = (intent_data_local.get("key_entities") or {}).get("destination", "")
                if city:
                    knowledge_db = CityKnowledgeDB.get_instance()
                    static_tips = knowledge_db.get_tips(city)
                    if static_tips:
                        tips_lines = "\n".join(
                            f"{i + 1}. {_clean_tip(t)}" for i, t in enumerate(static_tips[:3])
                        )
                        text_parts.append(f"## 旅行小贴士\n{tips_lines}")

            # ── 预算警告（交通超支 / 住宿预算过低）──────────────────────────
            budget_warning = _check_budget_warnings(state)
            if budget_warning:
                text_parts.insert(0, f"## 预算提示\n{budget_warning}")

            # ── 预算费用摘要 ────────────────────────────────────────────────
            budget_summary = _format_budget_summary(state)
            if budget_summary:
                text_parts.append(budget_summary)

            # ── 已知限制（P4.5 在 REVIEW_MAX_RETRIES 次回环后仍检出的违规）──────
            # route_after_review 将此类 state 路由到 respond_node，意味着自动修复
            # 已尽最大努力但仍存在次优之处，透明告知用户并附修正建议。
            rule_violations = state.get("rule_violations") or []
            if rule_violations:
                warning_lines: List[str] = []
                for i, v in enumerate(rule_violations):
                    if hasattr(v, "description"):
                        desc = v.description or ""
                        sugg = getattr(v, "suggestion", "") or ""
                    elif isinstance(v, dict):
                        desc = v.get("description", "") or ""
                        sugg = v.get("suggestion", "") or ""
                    else:
                        continue
                    if not desc:
                        continue
                    line = f"{i + 1}. {desc}"
                    if sugg:
                        line += f" 建议：{sugg}"
                    warning_lines.append(line)
                if warning_lines:
                    text_parts.append(
                        "## 已知限制\n"
                        "以下问题在自动调整后仍未完全消除，请结合实际情况灵活安排：\n"
                        + "\n".join(warning_lines)
                    )

        response_text = "\n\n".join(text_parts) if text_parts else "已处理您的请求。"

        return {
            "final_response": response_text,
            "messages": [AIMessage(content=response_text)]
        }

    return respond_node


# =============================================================================
# 内部辅助：各 agent 结果格式化（提取自 cli.py _generate_human_response）
# =============================================================================

def _format_agent_result(
    agent_name: str,
    data: dict,
    all_results: List[Dict],
    has_daily_routes: bool = False,
) -> str:
    """将单个 agent 的 data 格式化为纯文本段落，返回空字符串表示无内容。"""
    lines: List[str] = []

    # --- 行程规划 ---
    if agent_name == "itinerary_planning":
        itinerary = data.get("itinerary") or data.get("data", {}).get("itinerary")
        if itinerary:
            title = itinerary.get("title", "行程规划")
            lines.append(f"【{title}】")
            lines.append(f"时长: {itinerary.get('duration', '未知')}")
            for day_plan in itinerary.get("daily_plans", []):
                day_num = day_plan.get("day", 1)
                lines.append(f"\n第 {day_num} 天")
                activities = day_plan.get("activities") or day_plan.get("time_slots") or []
                for slot in activities:
                    time = slot.get("time", "")
                    activity = slot.get("activity") or slot.get("location") or ""
                    description = slot.get("description", "")
                    transport = slot.get("transport", "")
                    lines.append(f"  {time} - {activity}")
                    if description:
                        lines.append(f"    {description}")
                    if transport:
                        lines.append(f"    交通: {transport}")
                meals = day_plan.get("meals", {})
                if meals:
                    if meals.get("lunch"):
                        lines.append(f"  午餐: {meals['lunch']}")
                    if meals.get("dinner"):
                        lines.append(f"  晚餐: {meals['dinner']}")
            notes = itinerary.get("notes", [])
            if notes:
                lines.append("\n注意事项:")
                for note in notes:
                    lines.append(f"  • {note}")

    # --- 偏好管理 ---
    elif agent_name == "preference":
        raw_prefs = data.get("preferences") or data.get("data", {}).get("preferences")
        if isinstance(raw_prefs, dict):
            prefs_list = raw_prefs.get("preferences", [])
        else:
            prefs_list = raw_prefs if isinstance(raw_prefs, list) else []

        if prefs_list:
            lines.append("已更新您的偏好设置:")
            type_names = {
                "home_location": "常驻地",
                "transportation_preference": "交通偏好",
                "hotel_brands": "酒店偏好",
                "airlines": "航空公司偏好",
                "seat_preference": "座位偏好",
                "meal_preference": "餐食偏好",
                "budget_level": "预算等级"
            }
            for pref in prefs_list:
                pref_type = pref.get("type", "")
                pref_value = pref.get("value", "")
                action = pref.get("action", "replace")
                display_type = type_names.get(pref_type, pref_type)
                action_text = "追加" if action == "append" else "设置为"
                lines.append(f"  • {display_type} {action_text} {pref_value}")
            has_itinerary = any(r.get("agent_name") == "itinerary_planning" for r in all_results)
            if not has_itinerary:
                lines.append("下次规划行程时会参考这些偏好。")
        elif data.get("error"):
            lines.append(f"偏好未保存: {data['error']}")

    # --- 事项收集 ---
    elif agent_name == "event_collection":
        origin = data.get("origin") or data.get("data", {}).get("origin")
        destination = data.get("destination") or data.get("data", {}).get("destination")
        start_date = data.get("start_date") or data.get("data", {}).get("start_date")
        end_date = data.get("end_date") or data.get("data", {}).get("end_date")
        missing_info = data.get("missing_info") or data.get("data", {}).get("missing_info") or []

        has_itinerary = any(r.get("agent_name") == "itinerary_planning" for r in all_results)
        if not has_itinerary and (origin or destination):
            lines.append("已收集行程信息:")
            if origin:
                lines.append(f"  • 出发地: {origin}")
            if destination:
                lines.append(f"  • 目的地: {destination}")
            if start_date:
                lines.append(f"  • 出发日期: {start_date}")
            if end_date:
                lines.append(f"  • 返程日期: {end_date}")
        if missing_info:
            _MISSING_FIELD_NAMES = {
                "end_date": "返回日期",
                "duration_days": "行程天数",
                "return_location": "返程地点",
                "origin": "出发地",
                "destination": "目的地",
                "start_date": "出发日期",
                "trip_purpose": "出行目的",
            }
            missing_cn = [_MISSING_FIELD_NAMES.get(f, f) for f in missing_info]
            lines.append(f"还需要补充: {', '.join(missing_cn)}")

    # --- 信息查询 ---
    elif agent_name == "information_query":
        query_results = data.get("results") or data.get("data", {}).get("results") or data
        if not isinstance(query_results, dict):
            query_results = {}
        summary = query_results.get("summary", "")
        sources = query_results.get("sources", []) or []
        message = query_results.get("message", "")
        error = query_results.get("error", "")

        if summary:
            lines.append(summary)
        elif message:
            lines.append(message)
        elif error:
            lines.append(error)

        if sources:
            lines.append("\n参考来源:")
            for i, source in enumerate(sources[:3], 1):
                url = source.get("url", "") if isinstance(source, dict) else str(source)
                lines.append(f"  {i}. {url}")

    # --- 记忆查询 ---
    elif agent_name == "memory_query":
        query_result = (
            data.get("answer") or data.get("result") or data.get("content")
            or data.get("data", {}).get("answer")
            or data.get("data", {}).get("result")
            or data.get("data", {}).get("content")
        )
        if query_result:
            lines.append(str(query_result))

    # --- 交通查询 ---
    elif agent_name == "transport_query":
        transport_plan = data.get("transport_plan") or data.get("data", {}).get("transport_plan")
        if transport_plan:
            qi = transport_plan.get("query_info", {})
            date_str = qi.get("date", "")
            lines.append(f"## 交通方案 ({date_str})")
            analysis = transport_plan.get("analysis", "")
            if analysis:
                lines.append(analysis)
            options = transport_plan.get("options", [])
            if options:
                lines.append("")
                lines.append("| 类型 | 班次 | 时间 | 耗时 | 参考价格 |")
                lines.append("|------|------|------|------|----------|")
                for opt in options:
                    t_type = opt.get("transport_type", "")
                    t_no = opt.get("transport_no") or "-"
                    dep = opt.get("departure_time") or "-"
                    arr = opt.get("arrival_time") or "-"
                    dur = opt.get("duration", "-")
                    price = opt.get("price_range", "暂无")
                    lines.append(f"| {t_type} | {t_no} | {dep}→{arr} | {dur} | {price} |")
            rec = transport_plan.get("recommendation", {})
            if rec:
                best = rec.get("best_choice", "")
                reason = rec.get("reason", "")
                arrival_hub = rec.get("arrival_hub") or rec.get("arrival_station", "")
                if best:
                    lines.append(f"\n**推荐**: {best}")
                if reason:
                    lines.append(f"**理由**: {reason}")
                if arrival_hub:
                    lines.append(f"**到达枢纽**: {arrival_hub}")
        elif data.get("error"):
            lines.append(f"交通查询失败: {data['error']}")

    # --- 住宿推荐 ---
    elif agent_name == "accommodation_query":
        acc_plan = data.get("accommodation_plan") or data.get("data", {}).get("accommodation_plan")
        if acc_plan:
            dest = acc_plan.get("destination", "")
            arrival_station = acc_plan.get("arrival_station", "")
            lines.append(f"## 住宿推荐 - {dest}")

            # analysis 由 LLM 生成，包含区域选择逻辑（兼容旧字段 recommended_areas）
            analysis = acc_plan.get("analysis", "")
            if analysis:
                lines.append(analysis)

            if arrival_station and arrival_station not in ("未知", "null", "", None):
                lines.append(f"\n**到达枢纽**: {arrival_station}")

            # recommended_areas 已迁移至 analysis，此处兼容旧数据：仅在 analysis 为空时渲染
            if not analysis:
                areas = acc_plan.get("recommended_areas", [])
                if areas:
                    lines.append("\n**推荐区域**:")
                    for area in areas[:3]:
                        lines.append(f"  - {area.get('area_name', '')}：{area.get('reason', '')}")

            # 无效字段值集合：渲染时跳过这些值，避免输出"无（数据未提及）"等噪声
            _INVALID_FIELD_VALUES = {
                None, "null", "None", "", "无", "暂无", "无（数据未提及）",
                "数据未提及", "未知", "未提及", "暂无数据",
            }

            options = acc_plan.get("options", [])
            if options:
                lines.append("\n**酒店方案**:")
                for opt in options[:4]:
                    tier      = opt.get("tier", "")
                    name      = opt.get("hotel_name", "")
                    price     = opt.get("price_range")     # MCP 真实价格，null 时不渲染
                    highlights = opt.get("highlights")      # null 时跳过
                    distance  = opt.get("distance_info")   # MCP 距离数据，null 时跳过
                    src       = opt.get("data_source", "")

                    # 核心行：档次 + 名称（price 有效才追加）
                    hotel_line = f"  - [{tier}] **{name}**"
                    if price not in _INVALID_FIELD_VALUES:
                        hotel_line += f" {price}"
                    lines.append(hotel_line)

                    # 次行：highlights（有效才显示；distance_info 为内部计算字段，不对用户展示）
                    if highlights not in _INVALID_FIELD_VALUES:
                        lines.append(f"    {highlights}")

            rec = acc_plan.get("recommendation", {})
            if rec:
                best  = rec.get("best_choice", "")
                reason = rec.get("reason", "")
                tips  = rec.get("booking_tips", "")
                if best:
                    lines.append(f"\n**综合推荐**: {best}")
                if reason:
                    lines.append(f"**推荐理由**: {reason}")
                if tips:
                    lines.append(f"**预订建议**: {tips}")
        elif data.get("error"):
            lines.append(f"住宿查询失败: {data['error']}")

    # --- 通用兜底 ---
    if not lines:
        common_keys = ["answer", "content", "result", "message", "summary", "text", "description"]
        for k in common_keys:
            if k in data and isinstance(data[k], str) and data[k].strip():
                lines.append(data[k])
                break
        if not lines and "data" in data and isinstance(data["data"], dict):
            for k in common_keys:
                if k in data["data"] and isinstance(data["data"][k], str) and data["data"][k].strip():
                    lines.append(data["data"][k])
                    break

    return "\n".join(lines)


def _filter_rag_answer(answer: str, has_transport: bool, has_itinerary: bool) -> str:
    """
    过滤 RAG 答案中与已有真实数据重叠的段落，避免重复输出。

    规则：
    - has_transport=True  → 删除 RAG 答案中的【往返交通】段落
      （因为 transport_query agent 已输出了真实 MCP 交通方案）
    - has_itinerary=True  → 删除 RAG 答案中的【行程安排】段落
      （因为 itinerary_planning_node 已输出了基于 Amap+TSP 的结构化每日路线）

    段落识别：从「【xxx】」标题行开始，到下一个「【」标题行（不含）或文本结尾。
    清理多余空行后返回剩余内容；若全部被过滤则返回空字符串。
    """
    import re

    if not answer or not answer.strip():
        return answer

    # 要过滤的段落标题关键词
    skip_keywords: List[str] = []
    if has_transport:
        skip_keywords.extend(["往返交通", "交通方案", "去程", "返程"])
    if has_itinerary:
        skip_keywords.extend(["行程安排", "每日行程", "行程规划", "Day", "第.*天"])

    if not skip_keywords:
        return answer

    # 按【...】段落标题切分 RAG 答案
    # 匹配形如「【标题】」或「**标题**」开头的行作为段落分隔符
    section_header_pattern = re.compile(
        r'^(?:【[^】]*】|\*{1,2}[^\*]+\*{1,2}|#{1,3}\s+\S)',
        re.MULTILINE
    )

    # 找出所有段落标题的位置
    headers = list(section_header_pattern.finditer(answer))

    if not headers:
        # 无明确段落划分时，按行扫描：删除含有目标关键词的行及其后续直到下一空行的内容
        skip_kw_re = re.compile('|'.join(skip_keywords))
        result_lines: List[str] = []
        skipping = False
        for line in answer.splitlines():
            stripped = line.strip()
            if re.search(r'^【', stripped) or re.search(r'^\*{1,2}', stripped):
                # 新标题行：重新判断是否需要跳过
                skipping = bool(skip_kw_re.search(stripped))
            if not skipping:
                result_lines.append(line)
        cleaned = "\n".join(result_lines)
        return re.sub(r'\n{3,}', '\n\n', cleaned).strip()

    # 按标题位置将答案切为段落列表 [(header_text, body_text), ...]
    sections: List[tuple] = []
    for i, m in enumerate(headers):
        header_text = m.group()
        body_start = m.end()
        body_end = headers[i + 1].start() if i + 1 < len(headers) else len(answer)
        body_text = answer[body_start:body_end]
        sections.append((header_text, body_text))

    # 文档最前面（第一个标题之前）的序言部分
    preamble = answer[:headers[0].start()] if headers else ""

    # 过滤：跳过标题含目标关键词的段落
    skip_kw_re = re.compile('|'.join(skip_keywords))
    kept_parts: List[str] = []
    if preamble.strip():
        kept_parts.append(preamble.rstrip())

    for header_text, body_text in sections:
        if skip_kw_re.search(header_text):
            logger.debug(f"_filter_rag_answer: 过滤段落 '{header_text[:30]}'")
            continue
        kept_parts.append(header_text + body_text)

    joined = "\n\n".join(p.strip() for p in kept_parts if p.strip())
    # 清理多余空行
    cleaned = re.sub(r'\n{3,}', '\n\n', joined).strip()
    return cleaned


async def _llm_summarize(skill_results: List[Dict], intent_data: Dict, llm) -> str:
    """当规则生成无任何文本时，用 LLM 将 skill_results 汇总为自然语言。"""
    results_text = json.dumps(skill_results, ensure_ascii=False, indent=2)
    user_query = intent_data.get("rewritten_query", "")
    prompt = f"""你是一个旅行助手。请根据以下各智能体的执行结果，生成一段简洁、自然的中文回复给用户。

用户问题：{user_query}

智能体执行结果：
{results_text}

要求：直接输出给用户看的文字，不要有额外的解释或JSON。"""

    try:
        response = await llm.ainvoke([{"role": "user", "content": prompt}])
        return response.content.strip()
    except Exception as e:
        logger.error(f"LLM summarize failed: {e}")
        return "已处理您的请求。"


def _build_user_context(state: TravelGraphState) -> str:
    """从 state 提取出行季节、人数、风格，生成简短描述供 LLM 个性化润色。"""
    hard_constraints = ensure_hard_constraints(state.get("hard_constraints"))
    travel_style: str = state.get("travel_style") or "普通"
    pax: int = hard_constraints.pax or 1
    start_date: str = hard_constraints.start_date or ""

    season = ""
    if start_date:
        try:
            month = int(start_date[5:7])
            if month in (3, 4, 5):
                season = "春季"
            elif month in (6, 7, 8):
                season = "夏季"
            elif month in (9, 10, 11):
                season = "秋季"
            else:
                season = "冬季"
        except (ValueError, IndexError):
            pass

    parts: List[str] = []
    if season:
        parts.append(f"{season}出行")
    if pax > 1:
        parts.append(f"共{pax}人")
    style_map = {
        "亲子": "亲子游（带孩子）",
        "老人": "老人出行（腿脚不便/轻松游）",
        "情侣": "情侣旅行",
        "特种兵": "特种兵高强度打卡",
    }
    if travel_style in style_map:
        parts.append(style_map[travel_style])

    return "，".join(parts) if parts else "普通出行"


async def _llm_refine_tips_risks(
    raw_tips: List[str],
    raw_risks: List[str],
    user_ctx: str,
    llm,
) -> tuple:
    """
    用 LLM 将 RAG 原始条目润色为个性化贴士/避坑，固定输出 3 条贴士 + 2 条避坑。
    返回 (tips_markdown_block, risks_markdown_block)。
    """
    sections: List[str] = []
    if raw_tips:
        sections.append("【贴士原始条目】\n" + "\n".join(f"{i+1}. {t}" for i, t in enumerate(raw_tips)))
    if raw_risks:
        sections.append("【避坑原始条目】\n" + "\n".join(f"{i+1}. {r}" for i, r in enumerate(raw_risks)))

    prompt = (
        f"你是一个旅行顾问。请根据用户特征（{user_ctx}），"
        f"从以下原始条目中筛选并用口语化中文改写，输出最有价值的个性化建议。\n\n"
        + "\n\n".join(sections)
        + "\n\n要求：严格按以下格式输出，不加任何多余内容：\n"
        "TIPS:\n"
        "1. （第1条贴士，一句话，结合用户特征）\n"
        "2. （第2条贴士）\n"
        "3. （第3条贴士）\n"
        "RISKS:\n"
        "1. （第1条避坑，含场景+后果+建议）\n"
        "2. （第2条避坑）"
    )

    try:
        response = await llm.ainvoke([{"role": "user", "content": prompt}])
        return _parse_tips_risks_output(response.content.strip())
    except Exception as e:
        logger.error(f"_llm_refine_tips_risks failed: {e}")
        return "", ""


def _parse_tips_risks_output(content: str) -> tuple:
    """解析 LLM 的 TIPS:/RISKS: 块，返回两个 markdown 字符串。"""
    import re
    tips_block = ""
    risks_block = ""

    tips_m = re.search(r'TIPS:\s*\n((?:\d+\..+(?:\n|$)){1,5})', content, re.IGNORECASE)
    risks_m = re.search(r'RISKS:\s*\n((?:\d+\..+(?:\n|$)){1,3})', content, re.IGNORECASE)

    if tips_m:
        tips_block = "## 旅行小贴士\n" + tips_m.group(1).strip()
    if risks_m:
        risks_block = "## 避坑指南\n" + risks_m.group(1).strip()

    return tips_block, risks_block


def _format_daily_routes(
    daily_routes: List[Dict],
    poi_descriptions: Optional[Dict[str, str]] = None,
    daily_restaurants: Optional[List[Dict]] = None,
) -> str:
    """
    将 daily_routes 渲染为结构化行程文本，并为每个景点注入 poi_enrich_node 提炼的
    体验描述，每天末尾附上周边餐厅推荐（来自 daily_restaurants）。

    格式（每天）：
      **第 N 天**：区域名
      景点A → (步行15分钟) → 景点B → (地铁20分钟) → 景点C
      > 景点A：1-2句体验描述（来自 poi_descriptions）
      > 景点B：1-2句体验描述
      总交通时长: X小时Y分钟
      周边餐厅推荐：
        - 餐厅名（距活动区域Xm，评分Y）

    Args:
        poi_descriptions: poi_enrich_node 写入 state 的 {景点名: 描述} 字典，
                          直接按名查找，无需关键词匹配，准确率极高
    """
    poi_descriptions = poi_descriptions or {}
    daily_restaurants = daily_restaurants or []

    # 预处理：将 daily_restaurants 转为 {day -> restaurants} 方便按天查找
    restaurants_by_day: Dict[int, List[Dict]] = {
        item["day"]: item.get("restaurants", [])
        for item in daily_restaurants
        if isinstance(item, dict)
    }

    lines = ["## 每日行程路线"]

    for day_data in daily_routes:
        day_num = day_data.get("day", 1)
        ordered_pois = day_data.get("ordered_pois", [])
        legs = day_data.get("legs", [])
        total_duration = day_data.get("total_duration", 0)

        if not ordered_pois:
            continue

        region = _infer_region(ordered_pois)
        header = f"**第 {day_num} 天**"
        if region:
            header += f"：{region}"
        lines.append("")
        lines.append(header)

        # 构建 POI → (交通) → POI 链
        route_parts = [ordered_pois[0].get("name", "")]
        for i, leg in enumerate(legs):
            transport_str = _format_leg(leg)
            route_parts.append(f"({transport_str})")
            next_idx = i + 1
            if next_idx < len(ordered_pois):
                route_parts.append(ordered_pois[next_idx].get("name", ""))

        lines.append(" → ".join(route_parts))

        # 为每个景点注入 poi_enrich_node 提炼的体验描述（精确名称匹配，无噪声）
        tips_lines = []
        for poi in ordered_pois:
            poi_name = poi.get("name", "")
            description = poi_descriptions.get(poi_name, "").strip()
            if description:
                tips_lines.append(f"> **{poi_name}**：{description}")
        if tips_lines:
            lines.append("")
            lines.extend(tips_lines)

        if total_duration > 0:
            lines.append(_format_duration(total_duration, prefix="总交通时长: "))

        # 追加当天周边餐厅推荐
        day_restaurants = restaurants_by_day.get(day_num, [])
        if day_restaurants:
            lines.append("")
            lines.append("周边餐厅推荐：")
            for r in day_restaurants:
                name = r.get("name", "")
                distance_m = r.get("distance_m", 0)
                rating = r.get("amap_rating", "")
                # 构建描述：距离 + 评分（均为可选字段，缺失时不拼接）
                meta_parts = []
                if distance_m:
                    meta_parts.append(f"距活动区域约 {distance_m}m")
                if rating:
                    meta_parts.append(f"评分 {rating}")
                meta_str = f"（{', '.join(meta_parts)}）" if meta_parts else ""
                lines.append(f"  - {name}{meta_str}")

    return "\n".join(lines)



def _infer_region(pois: List[Dict]) -> str:
    """
    从 POI 地址列表中推断区域名（取各 POI 地址的区级前缀最长公共部分）。
    地址通常为 "XX市XX区XX路..."，尝试提取区名。
    """
    addresses = [p.get("address", "") for p in pois if p.get("address")]
    if not addresses:
        return ""

    # 尝试提取 "XX区" / "XX县" / "XX镇"
    import re
    district_pattern = re.compile(r"[\u4e00-\u9fa5]{1,5}[区县镇]")
    candidates: List[str] = []
    for addr in addresses:
        m = district_pattern.search(addr)
        if m:
            candidates.append(m.group())

    if not candidates:
        return ""

    # 返回出现次数最多的区名
    from collections import Counter
    most_common = Counter(candidates).most_common(1)[0][0]
    return most_common


def _format_leg(leg: Dict) -> str:
    """
    将单段交通 leg 格式化为简短说明，如 "步行15分钟"、"地铁20分钟"。
    若 steps 包含线路信息（如"地铁2号线"），则附加线路名。
    """
    mode = leg.get("mode", "") or "交通"
    duration = leg.get("duration", 0)
    steps = leg.get("steps", []) or []

    # 从 steps 中提取首条地铁/公交线路名
    line_name = ""
    if steps and isinstance(steps, list):
        for step in steps:
            if isinstance(step, dict):
                name = step.get("line_name") or step.get("lineName") or step.get("name", "")
            else:
                name = str(step)
            if name and any(kw in name for kw in ["号线", "路", "线路", "巴士"]):
                line_name = name
                break

    parts = [mode]
    if line_name:
        parts.append(line_name)
    if duration > 0:
        parts.append(_format_duration(duration))

    return "".join(parts)


def _format_duration(minutes: int, prefix: str = "") -> str:
    """将分钟数格式化为 'X小时Y分钟' 或 'Y分钟'。"""
    if minutes <= 0:
        return ""
    hours, mins = divmod(int(minutes), 60)
    if hours > 0 and mins > 0:
        result = f"{hours}小时{mins}分钟"
    elif hours > 0:
        result = f"{hours}小时"
    else:
        result = f"{mins}分钟"
    return f"{prefix}{result}"


def _get_agent_display_name(agent_name: str) -> str:
    """获取 agent 的中文显示名称"""
    agent_display_names = {
        "event_collection": "事项收集",
        "preference": "偏好管理",
        "itinerary_planning": "行程规划",
        "information_query": "信息查询",
        "rag_experience": "经验建议查询",
        "rag_risk": "避坑风险查询",
        "memory_query": "记忆查询",
        "transport_query": "交通查询",
        "accommodation_query": "住宿查询",
    }
    return agent_display_names.get(agent_name, agent_name)


def _clean_tip(tip: str) -> str:
    """
    剥离 tip/risk 条目开头的序号前缀，防止双重编号。

    处理的前缀格式（举例）：
      "1. 灵隐寺..."   -> "灵隐寺..."
      "①灵隐寺..."    -> "灵隐寺..."
      "(1) 灵隐寺..."  -> "灵隐寺..."
      "- 灵隐寺..."    -> "灵隐寺..."

    非序号前缀的内容保持不变。
    """
    import re
    if not tip:
        return tip
    # 匹配常见序号前缀：数字+点/括号、圆圈数字、短横线，后跟可选空格
    cleaned = re.sub(
        r'^(?:\d+[.\s）)]\s*|[①②③④⑤⑥⑦⑧⑨⑩]\s*|\([一二三四五六七八九十\d]+\)\s*|-\s+)',
        '',
        tip.strip(),
    )
    return cleaned.strip() or tip.strip()


def _is_poi_recommendation(tip: str) -> bool:
    """
    判断 tip 条目实质上是景点/路线推荐，而非实用操作建议。

    RAG 抽取时偶尔会将行程路线描述（含时间段、箭头、游览时长）错分到 tips 字段。
    出现以下两种或以上特征时，视为"景点推荐"条目并过滤掉：

      - 时间段格式  9:00-17:30 / 09:00~18:00
      - 路线箭头    -> / --> / =>
      - 游览时长    (2-2.5h) / (约3小时)
      - 景点名+括号数字开头  "灵隐寺（3" / "西湖(2"

    返回：
        True  = 是景点/路线推荐，应从 tips 中过滤
        False = 是实用建议，保留
    """
    import re
    if not tip:
        return False

    indicators = [
        r'\d{1,2}:\d{2}\s*[-~]\s*\d{1,2}:\d{2}',   # 时间段 9:00-17:30
        r'[-=]{1,2}>',                                # 路线箭头 -> / --> / =>
        r'\([\d.]+-[\d.]+\s*h\)',                     # 游览时长 (2-2.5h)
        r'\(约\s*[\d.]+\s*小时?\)',                    # 游览时长 (约3小时)
        r'^[\u4e00-\u9fa5]{2,8}[（(]\d',              # 景点名+括号数字开头
    ]
    matches = sum(1 for pattern in indicators if re.search(pattern, tip))
    return matches >= 2


# =============================================================================
# Budget helper functions
# =============================================================================

def _parse_min_transport_cost(transport_options):
    """Parse minimum single-person transport cost from transport_options price_range strings."""
    import re
    min_cost = None
    for opt in transport_options:
        raw = opt.get("price_range") or ""
        nums = re.findall(r"(\d+(?:\.\d+)?)", str(raw))
        if nums:
            candidate = float(min(nums, key=float))
            if min_cost is None or candidate < min_cost:
                min_cost = candidate
    return min_cost


def _check_budget_warnings(state):
    hard_constraints = ensure_hard_constraints(state.get("hard_constraints"))
    total_budget = hard_constraints.total_budget
    if not total_budget:
        return None

    transport_options = state.get("transport_options") or []
    min_transport = _parse_min_transport_cost(transport_options)

    if min_transport and min_transport > total_budget:
        return (
            f"往返交通最低报价约 {min_transport:.0f} 元/人，"
            f"已超出人均预算 {total_budget:.0f} 元。"
            f"建议适当提高预算，或选择更经济的出行方式。"
        )

    daily_budget = state.get("daily_budget_per_person")
    if daily_budget is not None and daily_budget > 0:
        acc_budget = daily_budget * 0.4
        if acc_budget < 50:
            return (
                f"当前落地预算约 {daily_budget:.0f} 元/天/人，其中住宿参考上限约 "
                f"{acc_budget:.0f} 元/晚，低于经济型住宿普遍水平。"
                f"建议提高预算，或考虑青年旅舎等共享住宿形式。"
            )

    return None


def _format_budget_summary(state):
    hard_constraints = ensure_hard_constraints(state.get("hard_constraints"))
    total_budget = hard_constraints.total_budget
    if not total_budget:
        return None

    travel_days = state.get("travel_days") or 0
    daily_budget = state.get("daily_budget_per_person")
    transport_options = state.get("transport_options") or []
    min_transport = _parse_min_transport_cost(transport_options)

    lines = ["费用参考"]
    lines[0] = "## " + lines[0]
    lines.append(f"人均总预算：{total_budget:.0f} 元")

    if min_transport:
        land = max(total_budget - min_transport, 0.0)
        lines.append(f"往返交通（最经济估算）：约 {min_transport:.0f} 元/人")
        lines.append(f"落地预算（交通后余额）：约 {land:.0f} 元/人")

    if daily_budget and travel_days:
        acc = daily_budget * 0.4
        spend = daily_budget * 0.6
        lines.append(f"每日落地预算：约 {daily_budget:.0f} 元/人（共 {travel_days} 天）")
        lines.append(f"  住宿参考上限：约 {acc:.0f} 元/晚")
        lines.append(f"  餐饮+景点+市内交通：约 {spend:.0f} 元/天")

    return "\n".join(lines)
