"""
knowledge_parser.py

职责：将 01_china_tourist_knowledge_base.md 解析为按城市索引的结构化字典，
      供各节点（itinerary_planning / accommodation / respond）直接查表，
      避免通过 RAG 链路间接推断已有确定性信息。

设计原则：
  - 单例 + 懒加载：首次调用 CityKnowledgeDB.get_instance() 时解析文件，后续直接走缓存。
  - 零外部依赖：仅使用标准库 re / pathlib / dataclasses，不引入 jieba 或 NLP 框架。
  - 城市名模糊匹配：处理 "北京市"/"北京" 等变体，兼容 intent_node 提取的各种城市写法。
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# 知识库文件的默认路径（相对于本文件向上两级到项目根目录）
_DEFAULT_KB_PATH = (
    Path(__file__).parent.parent
    / ".claude/skills/ask-question/data/documents/01_china_tourist_knowledge_base.md"
)


# =============================================================================
# 数据结构
# =============================================================================

@dataclass
class MustVisitPOI:
    """必去景点条目"""
    name: str           # 景点名，如"西湖风景区"、"灵隐寺·飞来峰"
    duration: str       # 建议游玩时长，如"1天"、"2-2.5h"；可能为空字符串
    description: str    # 一行简介


@dataclass
class CityKnowledge:
    """单个城市的结构化知识"""
    must_visit: List[MustVisitPOI] = field(default_factory=list)
    # 顺路组合：每条为一组 POI 名称列表，如 ["断桥", "白堤", "苏堤", "雷峰塔"]
    route_combos: List[List[str]] = field(default_factory=list)
    # 住宿建议：每条为原文（含区域名+点评），如 "上城区湖滨/龙翔桥：紧邻西湖..."
    accommodation: List[str] = field(default_factory=list)
    # 本地美食：每条为原文
    food: List[str] = field(default_factory=list)
    # 避坑指南：每条为原文
    tips: List[str] = field(default_factory=list)
    # 最佳旅游季节摘要，如 "3-4月（春）；9-11月（秋）"
    best_season: str = ""


# =============================================================================
# 主类
# =============================================================================

class CityKnowledgeDB:
    """
    城市知识库，单例模式。

    用法（建议在服务启动时调用一次，之后各节点直接传入实例）：

        db = CityKnowledgeDB.get_instance()

        # 直接取必去景点名列表 → 替换原来的 jieba 提取
        must_visit_names = db.get_must_visit_names("杭州")

        # 取顺路组合 → 作为 _cluster_by_geography 的分组种子
        combos = db.get_route_combos("杭州")

        # 取住宿建议 → accommodation_node 直接输出
        accom = db.get_accommodation("杭州")

        # 取避坑指南 → respond_node 末尾追加
        tips = db.get_tips("杭州")
    """

    _instance: Optional["CityKnowledgeDB"] = None

    def __init__(self, md_path: Path) -> None:
        self._db: Dict[str, CityKnowledge] = {}
        self._parse(md_path)

    @classmethod
    def get_instance(
        cls,
        md_path: Optional[Path] = None,
    ) -> "CityKnowledgeDB":
        """
        获取单例。首次调用时解析文件；之后调用忽略 md_path 参数，直接返回缓存实例。
        """
        if cls._instance is None:
            path = md_path or _DEFAULT_KB_PATH
            cls._instance = cls(path)
        return cls._instance

    # ── 查询接口 ─────────────────────────────────────────────────────────────

    def get_must_visit(self, city: str) -> List[MustVisitPOI]:
        """返回必去景点完整列表（含时长和描述）"""
        key = self._find_city(city)
        return self._db[key].must_visit if key else []

    def get_must_visit_names(self, city: str) -> List[str]:
        """
        只返回景点名列表，直接替代 _extract_rag_preferred_pois 的输出。
        示例：["西湖风景区", "灵隐寺·飞来峰", "宋城旅游景区", ...]
        """
        return [p.name for p in self.get_must_visit(city)]

    def get_route_combos(self, city: str) -> List[List[str]]:
        """
        返回顺路组合列表，每条是一组按游览顺序排列的景点名。
        示例：[["断桥", "白堤", "孤山", "苏堤", "雷峰塔"], ...]
        """
        key = self._find_city(city)
        return self._db[key].route_combos if key else []

    def get_accommodation(self, city: str) -> List[str]:
        """返回住宿建议原文列表"""
        key = self._find_city(city)
        return self._db[key].accommodation if key else []

    def get_food(self, city: str) -> List[str]:
        """返回本地美食原文列表"""
        key = self._find_city(city)
        return self._db[key].food if key else []

    def get_tips(self, city: str) -> List[str]:
        """返回避坑指南原文列表"""
        key = self._find_city(city)
        return self._db[key].tips if key else []

    def get_best_season(self, city: str) -> str:
        """返回最佳旅游季节摘要字符串"""
        key = self._find_city(city)
        return self._db[key].best_season if key else ""

    def has_city(self, city: str) -> bool:
        """判断知识库中是否有该城市的数据"""
        return self._find_city(city) is not None

    def city_count(self) -> int:
        """已解析的城市数量"""
        return len(self._db)

    # ── 内部方法 ─────────────────────────────────────────────────────────────

    def _find_city(self, city: str) -> Optional[str]:
        """
        模糊匹配城市名，处理以下变体：
          - 精确命中："杭州" → "杭州"
          - 去尾缀命中："杭州市" → "杭州"
          - 子串命中："西湖区" 包含 "湖" 不会乱匹配，
            但 "北京市朝阳区" 会命中 "北京"
        """
        if not city:
            return None
        # 1. 精确匹配
        if city in self._db:
            return city
        # 2. 去 市/省/区/自治区/特别行政区 后缀后精确匹配
        normalized = re.sub(r'(?:市|省|区|自治区|特别行政区)$', '', city)
        if normalized and normalized in self._db:
            return normalized
        # 3. 包含关系：city 是 key 的子串（如"杭" in "杭州"——过短时可能误匹配，加长度保护）
        if len(city) >= 2:
            for key in self._db:
                if city in key or key in city:
                    return key
        return None

    def _parse(self, md_path: Path) -> None:
        """
        一次性将 MD 文件解析为城市字典，填充 self._db。

        MD 文件结构（每个城市一个 ## 块）：
          ## 城市名
          ### 交通枢纽
          ### 核心景点
          #### 必去：
          - 景点名 (时长)：描述
          #### 顺路组合：
          1. A→B→C（时间，主题）
          ### 住宿指南
          ### 本地美食
          ### 避坑指南
          ### 天气与最佳旅游时间
        """
        if not md_path.exists():
            logger.error(f"CityKnowledgeDB: 知识库文件不存在: {md_path}")
            return

        text = md_path.read_text(encoding="utf-8")

        # 按 "## 城市名" 切块；跳过首个空头部
        blocks = re.split(r'\n## ', '\n' + text)
        for block in blocks:
            if not block.strip():
                continue
            lines = block.split('\n', 1)
            city_name = lines[0].strip()
            content = lines[1] if len(lines) > 1 else ""

            if not city_name:
                continue

            knowledge = CityKnowledge()

            # ── 必去景点 ─────────────────────────────────────────────────
            m = re.search(
                r'####\s*必去[：:]?\s*\n(.*?)(?=####|###|^---|\Z)',
                content, re.DOTALL | re.MULTILINE,
            )
            if m:
                knowledge.must_visit = _parse_must_visit(m.group(1))

            # ── 顺路组合 ─────────────────────────────────────────────────
            m = re.search(
                r'####\s*顺路组合[：:]?\s*\n(.*?)(?=####|###|^---|\Z)',
                content, re.DOTALL | re.MULTILINE,
            )
            if m:
                knowledge.route_combos = _parse_route_combos(m.group(1))

            # ── 住宿指南 ─────────────────────────────────────────────────
            m = re.search(
                r'###\s*住宿指南\s*\n(.*?)(?=###|^---|\Z)',
                content, re.DOTALL | re.MULTILINE,
            )
            if m:
                knowledge.accommodation = _parse_list_items(m.group(1))

            # ── 本地美食 ─────────────────────────────────────────────────
            m = re.search(
                r'###\s*本地美食\s*\n(.*?)(?=###|^---|\Z)',
                content, re.DOTALL | re.MULTILINE,
            )
            if m:
                knowledge.food = _parse_list_items(m.group(1))

            # ── 避坑指南 ─────────────────────────────────────────────────
            m = re.search(
                r'###\s*避坑指南\s*\n(.*?)(?=###|^---|\Z)',
                content, re.DOTALL | re.MULTILINE,
            )
            if m:
                knowledge.tips = _parse_list_items(m.group(1))

            # ── 最佳旅游时间 ──────────────────────────────────────────────
            m = re.search(
                r'###\s*天气与最佳旅游时间\s*\n(.*?)(?=###|^---|\Z)',
                content, re.DOTALL | re.MULTILINE,
            )
            if m:
                season_text = m.group(1)
                # 匹配 "**最佳季节**：..." 或 "- **最佳季节**：..."
                bm = re.search(r'\*\*最佳季节\*\*[：:]\s*(.+)', season_text)
                if bm:
                    knowledge.best_season = bm.group(1).strip()

            self._db[city_name] = knowledge

        logger.info(
            f"CityKnowledgeDB: 解析完成，共 {len(self._db)} 个城市"
        )


# =============================================================================
# 内部解析函数（模块私有）
# =============================================================================

def _parse_must_visit(text: str) -> List[MustVisitPOI]:
    """
    解析必去景点段落，每条格式：
      - 景点名 (时长)：描述
    时长括号可选（少数条目没有时长），冒号为中文或英文。

    示例输入行：
      - 西湖风景区 (1天)：杭州名片，推荐环湖骑行...
      - 河坊街·南宋御街 (1-1.5h)：明清风格的历史商业街...
    """
    pois: List[MustVisitPOI] = []
    # 两种情况：带时长 "名称 (时长)：描述" 和不带时长 "名称：描述"
    pattern = re.compile(
        r'^-\s+'
        r'(.+?)'                         # 景点名（非贪婪，最短匹配）
        r'(?:\s+\(([^)]+)\))?'           # 可选时长，如 "(1天)" "(2-2.5h)"
        r'\s*[：:]\s*'                   # 中文或英文冒号
        r'(.+)$',                        # 描述（到行尾）
        re.MULTILINE,
    )
    for m in pattern.finditer(text):
        name = m.group(1).strip()
        duration = (m.group(2) or "").strip()
        description = m.group(3).strip()
        # 过滤过短的噪声词（单字词不是景点名）
        if name and len(name) >= 2:
            pois.append(MustVisitPOI(
                name=name,
                duration=duration,
                description=description,
            ))
    return pois


def _parse_route_combos(text: str) -> List[List[str]]:
    """
    解析顺路组合段落，每条格式：
      {序号}. 地点A→地点B→地点C（时间范围，主题描述）

    处理逻辑：
      1. 匹配以数字+点开头的行
      2. 去掉序号前缀
      3. 去掉行尾的中文括号内容（含时间和主题）
      4. 按 "→" 切分，得到景点名列表

    示例输入行：
      1. 断桥→白堤→孤山→苏堤→雷峰塔（9:00-17:30，西湖全景经典线）
      2. 灵隐寺→飞来峰→法喜寺→龙井村（8:00-15:00，禅意寻茶线）
    """
    combos: List[List[str]] = []
    for line in text.split('\n'):
        line = line.strip()
        # 必须以数字+点开头
        if not re.match(r'^\d+\.', line):
            continue
        # 去序号
        content = re.sub(r'^\d+\.\s*', '', line)
        # 去中文括号及英文括号中的时间/主题（保留景点名部分）
        content = re.sub(r'[（(][^）)]*[）)]', '', content).strip()
        # 按 → 切分，过滤空段
        spots = [s.strip() for s in re.split(r'→', content) if s.strip()]
        if len(spots) >= 2:
            combos.append(spots)
    return combos


def _parse_list_items(text: str) -> List[str]:
    """
    通用列表解析：提取 "- " 或 "{n}. " 开头的条目，保留完整原文。
    用于住宿指南、本地美食、避坑指南等段落。
    """
    items: List[str] = []
    for line in text.split('\n'):
        line = line.strip()
        m = re.match(r'^(?:-|\d+\.)\s+(.+)$', line)
        if m:
            content = m.group(1).strip()
            if content:
                items.append(content)
    return items
