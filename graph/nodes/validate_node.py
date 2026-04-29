"""
硬约束检查节点 validate_constraints_node (P1.5)

职责：轻量级硬约束缺失检查与日志输出。

历史背景：
  此节点原先通过高德 MCP + ReAct 子智能体对 origin/destination 做路网
  距离与耗时推理，并让 LLM 按 system_prompt 里的经验规则判定 is_valid。
  实测 LLM 经常把"北京->杭州/南京"等高铁可达的目的地误判为
  time_conflict / spatial_temporal_conflict，并阻塞 P2~P5 全流程。

  现已移除所有基于 LLM 的物理/时间判定与 MCP 调用，该节点只负责：
    - 读取 hard_constraints，记录当前状态
    - 返回空的 rule_violations（与 P1.4 的跨轮清理保持一致）

  真正的 missing_info 由 P1.4 extract_constraints_node 作为单一真源写入；
  route_after_validation 依赖 missing_info 驱动 negotiate 分支，逻辑不变。
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from graph.state import TravelGraphState, ensure_hard_constraints

logger = logging.getLogger(__name__)


def create_validate_constraints_node(llm=None):
    """
    工厂函数：返回轻量级 validate_constraints_node 异步节点。

    Args:
        llm: 为兼容 build_graph 的调用签名保留形参，本节点不再使用 LLM。

    Returns:
        async 节点 validate_constraints_node(state) -> dict
    """

    async def validate_constraints_node(state: TravelGraphState) -> Dict[str, Any]:
        hard_constraints = ensure_hard_constraints(state.get("hard_constraints"))
        missing_info = state.get("missing_info") or []

        if missing_info:
            logger.info(
                f"[validate_constraints] 硬约束缺失 {missing_info}; "
                f"origin={hard_constraints.origin!r}, "
                f"destination={hard_constraints.destination!r}, "
                f"start_date={hard_constraints.start_date!r} "
                f"-> 将由 route_after_validation 路由到 negotiate"
            )
        else:
            logger.info(
                f"[validate_constraints] 硬约束齐全: "
                f"{hard_constraints.origin} -> {hard_constraints.destination}, "
                f"{hard_constraints.start_date} ~ {hard_constraints.end_date or '未指定'}"
            )

        # 不追加任何 rule_violations；阻塞路由由 missing_info 驱动
        return {"rule_violations": []}

    return validate_constraints_node
