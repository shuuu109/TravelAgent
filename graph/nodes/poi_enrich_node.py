"""
POI 体验描述补充节点 poi_enrich_node (P3.5)
==========================================
位置：itinerary_planning_node 之后，accommodation_node 之前。

职责（Post-Retrieval Augmentation）：
  对 daily_routes 中每个已确定的 POI，以景点名本身为 query 检索知识库，
  从命中文档里提炼 1-2 句核心体验描述，写入 state["poi_descriptions"]。

设计要点：
  1. Query = 景点名：语义对齐极精准，无需复杂构造，召回噪声低
  2. 批量并发：asyncio.gather 并行处理所有 POI，不产生串行等待
  3. 去重：同名 POI 只检索一次，避免重复 LLM 调用
  4. 无 LLM fallback：若 LLM 不可用或检索无结果，该 POI 的描述为空字符串，
     不阻断主流程（respond_node 应容错处理空描述）
"""
from __future__ import annotations

import asyncio
import logging
from typing import Dict, List

from graph.state import TravelGraphState
from agents.rag_base_agent import RAGBaseAgent

logger = logging.getLogger(__name__)


def create_poi_enrich_node(llm):
    """
    工厂函数：将 LLM 实例通过闭包注入。

    Args:
        llm: LangChain ChatXxx 实例，实现 ainvoke(messages) -> AIMessage

    Returns:
        async 节点函数 poi_enrich_node(state) -> dict
    """
    # RAGBaseAgent 只做检索，共享 ask-question 技能的 ChromaDB
    rag = RAGBaseAgent(name="poi_enrich_rag", model=None, top_k=3)

    async def poi_enrich_node(state: TravelGraphState) -> dict:
        """
        节点主流程：
        1. 从 daily_routes 提取所有唯一 POI 名称
        2. 并发查询知识库，每个 POI 提炼 1-2 句体验描述
        3. 返回 {"poi_descriptions": {poi_name: description, ...}}
        """
        daily_routes: List[Dict] = state.get("daily_routes") or []
        if not daily_routes:
            logger.warning("[poi_enrich] daily_routes 为空，跳过 POI 体验补充")
            return {"poi_descriptions": {}}

        # 去重：从所有天的 ordered_pois 中提取唯一 POI 名称
        poi_names: List[str] = []
        seen: set = set()
        for day_route in daily_routes:
            for poi in day_route.get("ordered_pois", []):
                name = poi.get("name", "").strip()
                if name and name not in seen:
                    poi_names.append(name)
                    seen.add(name)

        if not poi_names:
            return {"poi_descriptions": {}}

        logger.info(f"[poi_enrich] 开始补充 {len(poi_names)} 个 POI 的体验描述")

        # 并发处理所有 POI
        tasks = [_enrich_single_poi(name, rag, llm) for name in poi_names]
        descriptions: List[str] = await asyncio.gather(*tasks, return_exceptions=True)

        poi_descriptions: Dict[str, str] = {}
        for name, desc in zip(poi_names, descriptions):
            if isinstance(desc, Exception):
                logger.warning(f"[poi_enrich] {name!r} 描述提炼失败: {desc}")
                poi_descriptions[name] = ""
            else:
                poi_descriptions[name] = desc or ""

        logger.info(
            f"[poi_enrich] 完成，{sum(1 for v in poi_descriptions.values() if v)} / "
            f"{len(poi_names)} 个 POI 获得体验描述"
        )
        return {"poi_descriptions": poi_descriptions}

    return poi_enrich_node


# =============================================================================
# 内部辅助函数
# =============================================================================

async def _enrich_single_poi(poi_name: str, rag: RAGBaseAgent, llm) -> str:
    """
    对单个 POI 完成检索 + LLM 描述提炼。

    Args:
        poi_name: 景点名称，直接作为检索 query（语义对齐精准）
        rag:      RAGBaseAgent 实例，提供 search_knowledge()
        llm:      LangChain LLM 实例

    Returns:
        提炼后的 1-2 句体验描述；若无结果或 LLM 不可用，返回空字符串
    """
    docs = rag.search_knowledge(poi_name, top_k=3)
    if not docs:
        logger.debug(f"[poi_enrich] {poi_name!r}: 知识库无命中")
        return ""

    if not llm:
        # 无 LLM 时直接截取首个文档片段作为降级描述
        return docs[0]["content"][:100]

    # 拼接检索到的上下文
    context = "\n\n".join(
        f"【片段{i + 1}】\n{d['content']}" for i, d in enumerate(docs)
    )

    prompt = (
        f"从以下攻略文本中，提炼关于【{poi_name}】的核心游览体验。\n\n"
        f"要求：\n"
        f"1. 输出 1-2 句话，保留最具体的细节（如最佳游览时段、独特景观、注意事项）。\n"
        f"2. 语气自然、适合出现在行程介绍中，不要以'根据攻略'开头。\n"
        f"3. 若攻略中无该景点的直接描述，输出空字符串。\n\n"
        f"【攻略文本】\n{context}"
    )

    try:
        response = await llm.ainvoke([
            {"role": "system", "content": "你是旅游行程文案撰写助手，只输出景点体验描述，不输出其他内容。"},
            {"role": "user", "content": prompt}
        ])
        result = response.content.strip()
        # 拒绝过长输出（超过 150 字视为 LLM 未遵循指令）
        if len(result) > 150:
            result = result[:150]
        logger.debug(f"[poi_enrich] {poi_name!r}: {result!r}")
        return result
    except Exception as e:
        logger.error(f"[poi_enrich] {poi_name!r} LLM 调用失败: {e}")
        return ""
