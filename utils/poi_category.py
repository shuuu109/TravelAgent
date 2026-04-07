"""
utils/poi_category.py

POI 大类判断工具 —— 供 itinerary_review_node Check 4 使用。

判断优先级：
  1. 高德 typecode 前缀匹配（精准、稳定）
  2. 景点名称关键词匹配（typecode 未命中时的兜底方案）

typecode 说明：
  高德 POI 分类码为 6 位数字字符串，按"大类-子类-细类"三级编码。
  本模块只保留把握较高的两个精确前缀规则：
    11 01xx → 风景名胜/自然类（国家公园、植物园、综合景区等）→ 自然公园
    11 03xx → 博物馆/纪念馆/展览馆类                          → 博物馆
  其余大类（古镇古街、宗教寺庙、遗址遗迹）的名称关键词辨识度极高，
  未收录子码以防 Amap 文档与实测出入，直接由关键词匹配兜底。

维护说明：
  若后续积累了更多 typecode 实测样本，可在 _TYPECODE_PREFIX_RULES 中追加
  更精确的 6 位码规则；规则列表按前缀长度从长到短排列，越精确的规则越靠前。
"""
from __future__ import annotations

from typing import Dict, List, Optional


# ── typecode 前缀 → 大类标签 ───────────────────────────────────────────────────
# 规则列表按前缀长度从长到短排列，保证精确匹配优先于宽泛前缀。
# 每条规则格式：(typecode前缀, 大类标签)
_TYPECODE_PREFIX_RULES: List[tuple[str, str]] = [
    # 风景名胜/自然类（110100 - 110199）
    # 覆盖：综合景区、自然保护区、森林公园、国家公园、植物园、动物园等
    ("1101", "自然公园"),

    # 博物馆类（110300 - 110399）
    # 覆盖：综合博物馆、科技馆、纪念馆、展览馆、艺术馆等
    ("1103", "博物馆"),
]


# ── 名称关键词 → 大类标签（typecode 未命中时的兜底方案）────────────────────────
# 与旧版 _CATEGORY_KEYWORDS 保持一致，确保行为不退化
_CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "自然公园": ["公园", "湿地", "自然保护区", "森林公园", "地质公园", "风景区", "风光带", "植物园"],
    "古镇古街": ["古镇", "古街", "古城", "老街", "历史街区", "历史文化街区"],
    "博物馆":   ["博物馆", "纪念馆", "展览馆", "陈列馆", "艺术馆"],
    "宗教寺庙": ["寺", "庙", "观", "教堂", "清真寺", "道观", "佛寺"],
    "遗址遗迹": ["遗址", "遗迹", "故居", "旧址", "古遗址"],
}


def get_category_by_typecode(typecode: str) -> Optional[str]:
    """
    根据高德 typecode 返回大类标签。

    匹配规则：按 _TYPECODE_PREFIX_RULES 中的前缀从上到下依次尝试，
    命中即返回；typecode 为空或无法匹配时返回 None。

    Args:
        typecode: 6 位高德 POI 分类码字符串，如 "110104"。

    Returns:
        大类标签字符串（如 "自然公园"），无匹配时返回 None。
    """
    if not typecode:
        return None
    for prefix, label in _TYPECODE_PREFIX_RULES:
        if typecode.startswith(prefix):
            return label
    return None


def get_category_by_name(name: str) -> Optional[str]:
    """
    根据景点名称关键词返回大类标签（兜底方案）。

    Args:
        name: 景点名称，如 "西湖龙井村"。

    Returns:
        大类标签字符串，无匹配时返回 None。
    """
    for label, keywords in _CATEGORY_KEYWORDS.items():
        if any(kw in name for kw in keywords):
            return label
    return None


def get_category_for_poi(poi: Dict) -> Optional[str]:
    """
    综合判断 POI 所属大类标签（供 Check 4 分组使用）。

    判断策略：
      1. 优先读取 poi["amap_type"]（高德 typecode），按前缀规则匹配；
      2. typecode 未命中时，降级为 poi["name"] 关键词匹配。

    Args:
        poi: 标准化 POI 字典，预期包含 "amap_type"（str）和 "name"（str）字段。

    Returns:
        大类标签字符串（如 "自然公园"、"博物馆"）；
        两种方式均无匹配时返回 None（表示非需管控的大类景点）。
    """
    category = get_category_by_typecode(poi.get("amap_type", ""))
    if category is not None:
        return category
    return get_category_by_name(poi.get("name", ""))
