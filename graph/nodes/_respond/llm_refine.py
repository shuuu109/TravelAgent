"""LLM 兜底汇总 + tips/risks 个性化润色（few-shot 注入版）。"""
import json
import logging
from typing import Dict, List, Optional, Tuple

from graph.state import TravelGraphState, ensure_hard_constraints
from graph.nodes._respond.tips_risks import _clean_tip

logger = logging.getLogger(__name__)


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


_FEW_SHOT_EXAMPLES = """【示例 1】
用户特征：春季出行，共3人，亲子游（带孩子）
本次行程景点：故宫、天坛、王府井
TIPS:
1. 故宫人流大，带孩子建议提前 7 天预约早场首批，从神武门出来直奔景山看全景，孩子也走得动。
2. 天坛回音壁、圜丘台阶多，给孩子备一瓶水和小零食，祈年殿东侧长廊有遮阴可休息。
3. 王府井小吃街晚上灯亮后再去，孩子拍照氛围好，避开正午暴晒。
RISKS:
1. 故宫珍宝馆/钟表馆需另购票，很多家长到门口才发现娃想看却没票，建议预约时一起勾选。
2. 天坛公园南北门距离 1.5 公里以上，带娃别原路折返，按南门进北门出的单向动线走。

【示例 2】
用户特征：秋季出行，共2人，老人出行（腿脚不便/轻松游）
本次行程景点：黄山风景区、宏村
TIPS:
1. 黄山推荐云谷索道上、太平索道下，老人全程缆车 + 平路观景，省去 90% 台阶。
2. 山顶气温比山下低 8-10 度，山上酒店有军大衣租赁，比自己背厚衣服轻松。
3. 宏村南湖画桥晨雾最美，住村内民宿次日清晨拍，避开白天旅游团。
RISKS:
1. 黄山天气说变就变，索道遇大风/雷雨会停运，老人被困山顶很被动，出发前一晚务必查景区公众号公告。
2. 宏村景区内石板路湿滑且夜间无路灯，老人晚上别单独出门，雨后尤其注意。

【示例 3】
用户特征：夏季出行，共1人，特种兵高强度打卡
本次行程景点：外滩、南京路、豫园、东方明珠
TIPS:
1. 外滩日出 5:30 左右，先打卡空人版外滩再回酒店补觉，比晚上挤人墙效率高 10 倍。
2. 东方明珠 259 米观光层提前在小程序抢 9:00 首场，10 点后排队 1 小时起。
3. 豫园九曲桥工作日上午人最少，下午转南京路步行街，一天能打满 4 个点。
RISKS:
1. 南京路步行街很多"老字号"是后挂的牌，特种兵打卡前在大众点评看「本地人推荐榜」再去，别浪费时间踩雷。
2. 东方明珠和上海中心二选一即可，两个都上既贵又累，特种兵优先上海中心 118 层视野更高。

【示例 4】
用户特征：春季出行，共2人，情侣旅行
本次行程景点：西湖、灵隐寺、河坊街
TIPS:
1. 西湖断桥-白堤-苏堤一线骑行最适合情侣，公共自行车在游客中心租，1 小时绕半圈拍照点全打卡。
2. 灵隐寺飞来峰侧门进香火清净，正门人多且烧香排队 40 分钟以上，情侣拍照不出片。
3. 河坊街晚 7 点后灯笼全亮，找一家临街茶馆二楼，俯拍夜景比街上人挤人浪漫。
RISKS:
1. 西湖游船分手摇船和电瓶船，手摇船码头集中在湖滨，价格 150/小时浮动大，上船前问清是包船还是拼船，避免被坑。
2. 灵隐寺门口"高僧"主动搭讪算命/送手串都是套路，最后会要香火钱，直接绕开走。
"""


async def _llm_refine_tips_risks(
    raw_tips: List[str],
    raw_risks: List[str],
    user_ctx: str,
    llm,
    itinerary_pois: Optional[List[str]] = None,
) -> str:
    """
    用 LLM 将 RAG 原始条目润色为个性化贴士/避坑，固定输出 3 条贴士 + 2 条避坑。

    返回 LLM 的原始 content 字符串（含 TIPS:/RISKS: 块），由调用方用
    _parse_refined_tips_risks_lists 解析为 list[str] 喂给 _build_chat_summary。
    异常时返回空字符串。
    """
    sections: List[str] = []
    if itinerary_pois:
        # 截断到前 12 个，避免长行程 prompt 过大
        poi_preview = "、".join(itinerary_pois[:12])
        if len(itinerary_pois) > 12:
            poi_preview += f" 等共 {len(itinerary_pois)} 个景点"
        sections.append(f"【本次行程景点】\n{poi_preview}")
    if raw_tips:
        sections.append("【贴士原始条目】\n" + "\n".join(f"{i+1}. {t}" for i, t in enumerate(raw_tips)))
    if raw_risks:
        sections.append("【避坑原始条目】\n" + "\n".join(f"{i+1}. {r}" for i, r in enumerate(raw_risks)))

    prompt = (
        f"你是一个旅行顾问。请根据用户特征（{user_ctx}），"
        f"从以下原始条目中筛选并用口语化中文改写，输出最有价值的个性化建议。\n\n"
        + "\n\n".join(sections)
        + "\n\n【示例参考】\n"
        + _FEW_SHOT_EXAMPLES
        + "\n要求：严格按以下格式输出，不加任何多余内容：\n"
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
        return response.content.strip()
    except Exception as e:
        logger.error(f"_llm_refine_tips_risks failed: {e}")
        return ""
    
def _parse_refined_tips_risks_lists(content: str) -> Tuple[List[str], List[str]]:
    """
    解析 LLM 的 TIPS:/RISKS: 块，返回两个去序号的纯文本列表（供 chat_summary 使用）。

    与 _parse_tips_risks_output 的区别：后者返回 markdown 字符串供 final_response 拼接，
    本函数返回 list[str]，每条已去掉 "N. " 前缀，可直接塞入 chat_summary.tips / risks。
    """
    import re
    tips: List[str] = []
    risks: List[str] = []

    tips_m = re.search(r'TIPS:\s*\n((?:\d+\..+(?:\n|$)){1,5})', content, re.IGNORECASE)
    risks_m = re.search(r'RISKS:\s*\n((?:\d+\..+(?:\n|$)){1,3})', content, re.IGNORECASE)

    if tips_m:
        for line in tips_m.group(1).splitlines():
            cleaned = _clean_tip(line)
            if cleaned:
                tips.append(cleaned)
    if risks_m:
        for line in risks_m.group(1).splitlines():
            cleaned = _clean_tip(line)
            if cleaned:
                risks.append(cleaned)

    return tips, risks