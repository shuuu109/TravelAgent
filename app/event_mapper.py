"""
LangGraph astream_events 输出 -> SSE Envelope 的映射规则。

核心数据：
  - NODE_LABELS:    节点名 -> (phase, 中文 label)，决定哪些节点会被推给前端
  - SILENT_NODES:   非 planning 分支的节点，不向前端推送
  - extract_node_data: 节点名 -> 从节点 output dict 中切片用户关心的字段


SSE 协议表（前端对接契约）
============================

所有事件统一封装为 SSEEnvelope（详见 app/schemas.py）：
    { event, node?, phase?, label?, data?, ts }

事件类型 × 字段：

  event = node_start
    触发：NODE_LABELS 中节点开始执行（SILENT_NODES 节点除外；
          intent 之后若识别为非 planning 分支，后续节点也被静默）
    字段：node, phase, label
    data ：null

  event = node_complete
    触发：节点执行结束；P3 子步骤通过 progress_cb 也走该事件，phase=P3
    字段：node, phase, label, data
    data：见下方 “按 node 切片” 表格

  event = node_retry
    触发：itinerary_review / budget_check 触发回环（占位事件，目前未推送）
    字段：node, phase, label, data?

  event = needs_input
    触发：negotiate 节点完成（缺信息或物理冲突），流程终止等待用户补充
    字段：data
    data：{ question: str, missing_info: list[str] }
          question 即 negotiate 生成的追问文案（来自 final_response）

  event = final
    触发：流结束信号；前端收到后关闭连接
    字段：data
    data：result_type=planning →
            { result_type, final_response, current_plan, daily_routes,
              daily_restaurants, daily_options_by_tier,
              poi_descriptions, poi_photos, rag_context,
              transport_options, budget_fit_message }
          result_type=text_only →
            { result_type, final_response }

  event = error
    触发：graph 流执行异常
    字段：data
    data：{ message: str }


node_complete data 按 node 切片：
  intent              : { intent_type, destination_best_season, attraction_hints }
  extract_constraints : { hard_constraints, travel_days, travel_style }
  validate_constraints: { missing_info, violations }
  negotiate           : { final_response }
  rag                 : { rag_context }
  transport           : { transport_options }
  poi_fetch           : { poi_count, sample_names }
  itinerary_planning  : { daily_routes, daily_itinerary, daily_budget_per_person }
  poi_enrich          : { poi_descriptions, poi_photos }
  itinerary_review    : { violations, review_retry_count }
  restaurant          : { daily_restaurants }
  accommodation       : { daily_options_by_tier, accommodation_downgrade_level }
  budget_check        : { budget_fit_message, accommodation_downgrade_level }
  respond             : { final_response }

修改本文件 NODE_LABELS / extract_node_data 前请同步更新本协议表。
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


# 节点元数据：phase 用于前端分组渲染，label 是 "正在 XXX" 的状态文字。
# 不在此 dict 中的节点（如 graph 顶层、子 chain）一律忽略。
NODE_LABELS: Dict[str, Tuple[str, str]] = {
    "intent": ("P1", "正在理解您的需求"),
    "extract_constraints": ("P1", "正在解析行程约束"),
    "validate_constraints": ("P1", "正在校验约束合理性"),
    "negotiate": ("P1", "需要您补充信息"),
    "rag": ("P2", "正在检索旅行经验与避坑信息"),
    "transport": ("P2", "正在查询往返交通方案"),
    "poi_fetch": ("P2", "正在搜索目的地景点"),
    "itinerary_planning": ("P3", "正在规划每日行程"),
    "poi_enrich": ("P3.5", "正在为景点补充介绍和照片"),
    "itinerary_review": ("P4.5", "正在自检行程合理性"),
    "restaurant": ("P3.6", "正在搜索每日周边餐厅"),
    "accommodation": ("P4", "正在推荐每日住宿"),
    "budget_check": ("P4.6", "正在校验预算"),
    "respond": ("P5", "正在生成最终行程"),
}


# 非 planning 分支节点：用户决定不暴露给前端
SILENT_NODES = {"preference", "memory_query", "info_query"}


def _dump_if_model(obj: Any) -> Any:
    """Pydantic 模型 -> dict，其它原样返回。"""
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return obj


def extract_node_data(node: str, output: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    从节点 output dict 中切片用户关心的字段。

    设计要点：
      - 控制 payload 体积：poi_candidates 可能上百条，只推 count + sample_names
      - Pydantic 模型一律 model_dump()，避免前端拿不到字段
      - 节点未涉及的字段不切，保持 envelope 紧凑
    """
    if not output:
        return None

    if node == "intent":
        return {
            "intent_type": output.get("intent_type"),
            "destination_best_season": output.get("destination_best_season"),
            "attraction_hints": output.get("attraction_hints"),
        }

    if node == "extract_constraints":
        return {
            "hard_constraints": _dump_if_model(output.get("hard_constraints")),
            "travel_days": output.get("travel_days"),
            "travel_style": output.get("travel_style"),
        }

    if node == "validate_constraints":
        return {
            "missing_info": output.get("missing_info") or [],
            "violations": [
                _dump_if_model(v) for v in (output.get("rule_violations") or [])
            ],
        }

    if node == "negotiate":
        return {"final_response": output.get("final_response")}

    if node == "rag":
        return {"rag_context": _dump_if_model(output.get("rag_context"))}

    if node == "transport":
        return {
            "transport_options": [
                _dump_if_model(t) for t in (output.get("transport_options") or [])
            ],
        }

    if node == "poi_fetch":
        candidates = output.get("poi_candidates") or []
        return {
            "poi_count": len(candidates),
            "sample_names": [
                c.get("name") for c in candidates[:8] if isinstance(c, dict)
            ],
        }

    if node == "itinerary_planning":
        return {
            "daily_routes": output.get("daily_routes") or [],
            "daily_itinerary": output.get("daily_itinerary") or [],
            "daily_budget_per_person": output.get("daily_budget_per_person"),
        }

    if node == "poi_enrich":
        return {
            "poi_descriptions": output.get("poi_descriptions") or {},
            "poi_photos": output.get("poi_photos") or {},
        }

    if node == "itinerary_review":
        return {
            "violations": [
                _dump_if_model(v) for v in (output.get("rule_violations") or [])
            ],
            "review_retry_count": output.get("review_retry_count", 0),
        }

    if node == "restaurant":
        return {"daily_restaurants": output.get("daily_restaurants") or []}

    if node == "accommodation":
        return {
            "daily_options_by_tier": output.get("daily_options_by_tier") or [],
            "accommodation_downgrade_level": output.get("accommodation_downgrade_level", 0),
        }

    if node == "budget_check":
        return {
            "budget_fit_message": output.get("budget_fit_message"),
            "accommodation_downgrade_level": output.get("accommodation_downgrade_level", 0),
        }

    if node == "respond":
        return {"final_response": output.get("final_response")}

    return None
