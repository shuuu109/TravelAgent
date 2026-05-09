"""
RAG 节点 rag_node (P2 fan-out)
职责：并行执行 rag_experience + rag_risk 两个 RAG 检索 agent，将结果聚合为
      RAGContext 一次性写入 state，并把 skill_results 条目交给 reducer 追加。

触发条件（满足所有）：
  1. intent_type == "planning"           — 仅在行程规划链路中检索 RAG
  2. key_entities.destination 非空        — 没有目的地无法检索
  3. registry 中已注册 rag_experience + rag_risk 技能

异常处理：
  - 单个 RAG agent 失败不阻断流程，只记录错误条目；另一个仍写入。
  - destination 缺失或 intent 非 planning 时直接返回 {}（fan-out 跳过）。
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from graph.state import (
    TravelGraphState,
    RAGContext,
    ExperienceOutput,
    RiskOutput,
)

logger = logging.getLogger(__name__)


def create_rag_node(registry):
    """
    工厂函数：将 agent registry 通过闭包注入。

    Args:
        registry: LazyAgentRegistry 或 dict[str, agent]，需提供 rag_experience / rag_risk

    Returns:
        async 节点函数 rag_node(state) -> dict
    """

    async def rag_node(state: TravelGraphState) -> dict:
        intent_type: str = state.get("intent_type", "")
        if intent_type != "planning":
            return {}

        intent_data: Dict[str, Any] = state.get("intent_data") or {}
        destination: str = (intent_data.get("key_entities") or {}).get("destination", "") or ""
        if not destination:
            logger.info("[rag_node] destination 为空，跳过 RAG 检索")
            return {}

        # 构建传给 RAG agent 的 input_data（与原 orchestrate_node._prepare_context 字段一致）
        context: Dict[str, Any] = {
            "key_entities": intent_data.get("key_entities", {}),
            "travel_style": intent_data.get("travel_style", "普通"),
            "rewritten_query": intent_data.get("rewritten_query", ""),
        }
        input_data = {"context": context}

        try:
            exp_agent = registry["rag_experience"]
            risk_agent = registry["rag_risk"]
        except KeyError as e:
            logger.warning(f"[rag_node] RAG agent 未注册: {e}")
            return {}

        logger.info(f"[rag_node] 并行检索 destination={destination!r}")
        exp_result, risk_result = await asyncio.gather(
            exp_agent.run(input_data),
            risk_agent.run(input_data),
            return_exceptions=True,
        )

        skill_results: List[Dict] = []
        rag_snippets: List[Dict] = []
        rag_experience: Optional[ExperienceOutput] = None
        rag_risks: Optional[RiskOutput] = None

        # ── rag_experience ─────────────────────────────────────────
        if isinstance(exp_result, Exception):
            logger.error(f"[rag_node] rag_experience failed: {exp_result}")
            skill_results.append(_to_skill_result(
                "rag_experience", "error",
                {"error": str(exp_result)}, message=str(exp_result),
            ))
        else:
            status = exp_result.get("status", "success")
            skill_results.append(_to_skill_result("rag_experience", status, exp_result))
            if status == "success":
                rag_snippets = exp_result.get("retrieved_documents", []) or []
                exp_dict = exp_result.get("experience", {}) or {}
                if exp_dict and (exp_dict.get("tips") or exp_dict.get("best_for")):
                    rag_experience = ExperienceOutput(**exp_dict)
                    logger.info(
                        f"[rag_node] experience: {len(exp_dict.get('tips', []))} tips, "
                        f"{len(exp_dict.get('best_for', []))} best_for"
                    )

        # ── rag_risk ───────────────────────────────────────────────
        if isinstance(risk_result, Exception):
            logger.error(f"[rag_node] rag_risk failed: {risk_result}")
            skill_results.append(_to_skill_result(
                "rag_risk", "error",
                {"error": str(risk_result)}, message=str(risk_result),
            ))
        else:
            status = risk_result.get("status", "success")
            skill_results.append(_to_skill_result("rag_risk", status, risk_result))
            if status == "success":
                risks_dict = risk_result.get("risks", {}) or {}
                if risks_dict and risks_dict.get("risks"):
                    rag_risks = RiskOutput(**risks_dict)
                    logger.info(f"[rag_node] risks: {len(risks_dict['risks'])} items")

        updates: Dict[str, Any] = {"skill_results": skill_results}
        if rag_snippets or rag_experience or rag_risks:
            updates["rag_context"] = RAGContext(
                rag_snippets=rag_snippets,
                rag_experience=rag_experience,
                rag_risks=rag_risks,
            )
        return updates

    return rag_node


def _to_skill_result(
    agent_name: str,
    status: str,
    data: Dict[str, Any],
    message: str = "",
) -> Dict[str, Any]:
    """构建与 respond_node 期望对齐的 flat skill_result 条目。"""
    out = {"agent_name": agent_name, "status": status, "data": data}
    if message:
        out["message"] = message
    return out
