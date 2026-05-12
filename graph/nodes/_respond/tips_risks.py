"""
tips/risks 清洗与收集工具。

职责（叶子模块，不依赖本包其他模块）：
  - _clean_tip            剥离条目开头的序号前缀，避免前端二次编号
  - _is_poi_recommendation 过滤 RAG 误抽到 tips 字段的景点/路线描述
  - _collect_raw_tips_risks 从 state.rag_context 收集去重后的原始条目
"""
import re
from typing import List, Tuple

from graph.state import TravelGraphState


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
    if not tip:
        return tip
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
    if not tip:
        return False

    indicators = [
        r'\d{1,2}:\d{2}\s*[-~]\s*\d{1,2}:\d{2}',
        r'[-=]{1,2}>',
        r'\([\d.]+-[\d.]+\s*h\)',
        r'\(约\s*[\d.]+\s*小时?\)',
        r'^[一-龥]{2,8}[（(]\d',
    ]
    matches = sum(1 for pattern in indicators if re.search(pattern, tip))
    return matches >= 2


def _collect_raw_tips_risks(state: TravelGraphState) -> Tuple[List[str], List[str]]:
    """
    从 state.rag_context 收集去重后的 raw_tips / raw_risks。

    tips 会过滤掉景点/路线推荐类条目（_is_poi_recommendation），
    两者均按前 40 字符做大小写无关去重。
    """
    rag_ctx = state.get("rag_context")
    raw_tips: List[str] = []
    raw_risks: List[str] = []

    rag_experience = rag_ctx.rag_experience if rag_ctx else None
    if rag_experience and getattr(rag_experience, "tips", None):
        seen_keys: set = set()
        for t in rag_experience.tips:
            if _is_poi_recommendation(t):
                continue
            key = _clean_tip(t)[:40].lower().strip()
            if key not in seen_keys:
                seen_keys.add(key)
                raw_tips.append(_clean_tip(t))

    rag_risks_data = rag_ctx.rag_risks if rag_ctx else None
    if rag_risks_data and getattr(rag_risks_data, "risks", None):
        seen_keys = set()
        for r in rag_risks_data.risks:
            key = _clean_tip(r)[:40].lower().strip()
            if key not in seen_keys:
                seen_keys.add(key)
                raw_risks.append(_clean_tip(r))

    return raw_tips, raw_risks
