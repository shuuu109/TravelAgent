"""
协商终止节点 negotiate_node (P1.5b)

职责：当 hard_constraints 信息不全（missing_info 非空）或 validate_constraints_node
      检出物理时空冲突（rule_violations 非空）时，由本节点生成一段自然、委婉、
      带修正建议的回复，引导用户补充信息或调整计划。

工作流位置：
  validate_constraints → [route_after_validation] → negotiate → END
  （本轮对话到此终止，hard_constraints 会被 checkpointer 持久化，
   下一轮 extract_constraints_node 的 "新值优先" 合并会把已知信息续上。）

与 respond_node 区别：
  - negotiate_node：P1.5 阶段触发，未进入 P2-P4，只负责解释和追问
  - respond_node：  P5 阶段，完整行程已生成，负责最终结果渲染

此节点从 graph/node.py 迁移而来，LLM 改为 build_graph 注入。
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from langchain_core.messages import SystemMessage

from graph.state import TravelGraphState

logger = logging.getLogger(__name__)


def create_negotiate_node(llm):
    """
    工厂函数：返回 negotiate_node 异步节点。

    Args:
        llm: 从 build_graph 注入的 LangChain ChatOpenAI 实例。

    Returns:
        async 节点 negotiate_node(state) -> dict
    """

    async def negotiate_node(state: TravelGraphState) -> Dict[str, Any]:
        missing_info = state.get("missing_info") or []
        violations = state.get("rule_violations") or []

        prompt_parts = [
            "你是一个专业的智能出行规划师。当前用户的请求无法直接生成最终行程，"
            "请向用户委婉地解释原因并提问：\n"
        ]

        # 1. 信息缺失：指名要补充的字段（中文字段名，extract_constraints_node 输出）
        if missing_info:
            prompt_parts.append(
                f"- 缺失核心信息：我们需要知道用户的 {', '.join(missing_info)}。"
                f"请询问用户这些信息。"
            )

        # 2. 物理时空冲突：列出每条 RuleViolation 的 description + suggestion
        if violations:
            prompt_parts.append("- 存在以下物理时空约束冲突：")
            for v in violations:
                # v 可能是 RuleViolation 模型或已 dict 化（checkpointer 场景）
                desc = v.description if hasattr(v, "description") else v.get("description", "")
                sugg = v.suggestion if hasattr(v, "suggestion") else v.get("suggestion", "")
                prompt_parts.append(f"  * {desc} 建议：{sugg}")

        prompt_parts.append(
            "\n请基于以上情况，生成一段自然、专业、贴心的回复，引导用户补充信息或"
            "调整行程计划。不要输出其他多余的思考过程。"
        )

        system_msg = SystemMessage(content="\n".join(prompt_parts))
        history = state.get("messages") or []

        logger.info(
            f"[negotiate] 触发：missing={missing_info}, violations={len(violations)}"
        )
        response = await llm.ainvoke([system_msg] + history)

        # 将助手回复追加到 messages，交由 cli.py 输出给用户，本轮结束
        return {"messages": [response]}

    return negotiate_node
