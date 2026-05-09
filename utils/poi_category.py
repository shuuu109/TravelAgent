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


# ── 候选 POI 硬过滤：非景点 typecode 黑名单 ───────────────────────────────────
# 高德 6 位 typecode 大类前缀（前 2 位）含义：
#   05 餐饮  06 购物  07 生活  10 住宿  12 商务住宅  13 政府社团
#   15 交通设施  16 金融  17 公司企业  18 道路附属  19 地名地址  20 公共设施
# 黑名单设计：仅拒绝明显与"景点"无关的大类；保守保留 08(体育休闲)、11(风景名胜)、
# 14(科教文化)、09(医疗) 等可能含景点子类的码段，由下游评分/review 进一步把关。
_NON_ATTRACTION_PREFIXES: tuple[str, ...] = (
    "05",    # 餐饮服务   ── 餐厅/小吃
    "06",    # 购物服务   ── 商场/便利店
    "10",    # 住宿服务   ── 酒店/民宿/连锁
    "12",    # 商务住宅   ── 写字楼/小区
    "15",    # 交通设施   ── 机场/车站/停车场/收费站
    "17",    # 公司企业
    "19",    # 地名地址   ── 部分子点/附属设施在此
    "99",    # 高德扩展码 ── 实测下"X-子点""检票处"等部分落此
)


def is_attraction_typecode(typecode: str) -> bool:
    """
    判断高德 typecode 是否可能代表景点（用于 POI 候选硬过滤）。

    策略：黑名单优先拒绝；typecode 缺失时保守放行（避免误伤）。

    Args:
        typecode: 6 位高德 POI 分类码字符串，如 "110104"、"100103"。

    Returns:
        False - typecode 命中黑名单（确认为非景点，应丢弃）
        True  - typecode 在白名单或缺失（保留，由下游评分判断）
    """
    if not typecode:
        return True
    return not any(typecode.startswith(p) for p in _NON_ATTRACTION_PREFIXES)


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
