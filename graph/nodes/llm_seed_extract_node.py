"""
LLM 种子景点抽取节点 llm_seed_extract_node (P2 中段)

职责：在 rag_node 完成、poi_fetch_node 启动之前，
      基于「城市知识库 must_visit + RAG 攻略 LLM 抽取」产出具名景点种子序列，
      写入 state["llm_seed_pois"]。

下游消费者：
  - poi_fetch_node：路径②b 用 seed 名单做精准 poi_search（trust_kb=True）
  - itinerary_planning_node._select_pois：Phase-1 锚定种子

设计取舍（与 attraction_hints 的分工）：
  - attraction_hints  ← 来自用户 query，承载个性化兴趣词（"大熊猫"、"灵隐寺"）
  - llm_seed_pois     ← 来自 RAG + KB，承载目的地共识地标（"天坛"、"北海"）
  二者通过不同搜索路径喂给 poi_agent，互不替代。

触发条件（满足所有，否则透传空 dict）：
  1. intent_type == "planning"
  2. intent_data.key_entities.destination 非空
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from graph.state import TravelGraphState, RAGContext
from utils.knowledge_parser import CityKnowledgeDB
from graph.nodes.itinerary_planning_node import _llm_extract_rag_recommendations

logger = logging.getLogger(__name__)


def create_llm_seed_extract_node(llm):
    """
    工厂函数：将 LLM 通过闭包注入。

    Args:
        llm: LangChain ChatOpenAI 实例，用于 RAG 抽取

    Returns:
        async 节点函数 llm_seed_extract_node(state) -> dict
    """

    async def llm_seed_extract_node(state: TravelGraphState) -> dict:
        if state.get("intent_type") != "planning":
            return {}

        intent_data: Dict[str, Any] = state.get("intent_data") or {}
        destination: str = (intent_data.get("key_entities") or {}).get("destination", "") or ""
        if not destination:
            logger.info("[llm_seed_extract_node] destination 为空，跳过种子抽取")
            return {}

        travel_style: str = state.get("travel_style") or intent_data.get("travel_style", "普通")
        travel_days: int = state.get("travel_days") or 1

        # ── KB 必去（直接查表，零 NLP 损耗）────────────────────────────────────
        knowledge_db = CityKnowledgeDB.get_instance()
        if destination and knowledge_db.has_city(destination):
            kb_seeds: List[str] = list(knowledge_db.get_must_visit_names(destination))
            logger.info(
                f"[llm_seed_extract_node] 知识库命中 city={destination!r}, "
                f"must_visit={kb_seeds}"
            )
        else:
            kb_seeds = []
            logger.info(
                f"[llm_seed_extract_node] 城市 {destination!r} 不在知识库, "
                f"种子主源依赖 LLM 抽取"
            )

        # ── LLM 注入：基于 RAG 攻略文本抽取景点序列 ───────────────────────────
        # kb_must_visit 注入 prompt 让 LLM 专注补充其他 5A/地标，避免风格 hint
        # 把推荐锁死在与 KB 同主题的景点（如亲子全是动物园）。
        rag_ctx: RAGContext = state.get("rag_context") or RAGContext()
        rag_snippets: List[Dict] = rag_ctx.rag_snippets or []

        llm_recommended: List[str] = await _llm_extract_rag_recommendations(
            rag_snippets=rag_snippets,
            city=destination,
            travel_style=travel_style,
            travel_days=travel_days,
            llm=llm,
            kb_must_visit=kb_seeds,
        )

        # KB 在前（保持原有命中优先级），LLM 补充未被 KB 覆盖的景点
        merged: List[str] = list(kb_seeds)
        seen: set = set(kb_seeds)
        for name in llm_recommended:
            if name and name not in seen:
                merged.append(name)
                seen.add(name)

        logger.info(
            f"[llm_seed_extract_node] city={destination}, style={travel_style}, "
            f"days={travel_days}, kb={len(kb_seeds)}, llm={len(llm_recommended)}, "
            f"merged seeds({len(merged)})={merged[:10]}{'...' if len(merged) > 10 else ''}"
        )

        return {"llm_seed_pois": merged}

    return llm_seed_extract_node
