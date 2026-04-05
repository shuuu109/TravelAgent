"""
RAG 风险查询智能体 RAGRiskAgent
================================
职责：专门检索"避坑 / 踩雷 / 常见误区"，经结构化 LLM 抽取后返回 RiskOutput 格式数据。

与 RAGExperienceAgent 的关键区别：
  1. 查询语义聚焦于"避坑/踩雷/误区"，召回负面风险类内容更准确
  2. 每条风险项要求保留"具体场景 + 后果 + 规避建议"三要素，防止 LLM 泛化压缩
  3. 不返回 retrieved_documents（风险信息不用于 POI 权重偏移，仅供 P5 渲染）
"""
import os
import sys
import logging
from typing import List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser

from agents.rag_base_agent import RAGBaseAgent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 结构化抽取目标模型
# 字段定义与 graph/state.py 中 RiskOutput 保持一致
# ---------------------------------------------------------------------------
class RiskExtraction(BaseModel):
    risks: List[str] = Field(
        default_factory=list,
        description=(
            "避坑条目列表。每条必须包含'具体场景 + 潜在后果 + 规避建议'三要素。"
            "不合格示例：'注意交通'。"
            "合格示例：'西湖周边打车高峰期易堵，若赶班次可能误点，建议提前1小时出发或改乘地铁'。"
        )
    )


class RAGRiskAgent(RAGBaseAgent):
    """
    RAG 风险避坑查询智能体。

    查询语义："{destination} 避坑 踩雷 注意事项 常见误区"
    LLM 抽取：从检索片段中提炼每条含"场景+后果+建议"三要素的风险条目
    输出：
      - risks: {"risks": ["...", ...]}（供 respond_node 渲染"避坑提示"区块）
    """

    async def run(self, input_data: dict) -> dict:
        if not self.initialized:
            return {"status": "error", "message": "RAGRiskAgent not initialized"}

        context: dict = input_data.get("context", {})
        destination: str = (context.get("key_entities") or {}).get("destination", "") or ""

        # 构造风险检索 query：聚焦负面/避坑语义
        rag_query = " ".join(filter(None, [destination, "避坑 踩雷 注意事项 常见误区"]))
        logger.info(f"[RAGRisk] query: {rag_query!r}")

        # 先按城市精准过滤，若无命中则降级全局检索
        docs = self.search_knowledge(rag_query, city_filter=destination or None)
        if not docs and destination:
            logger.warning(f"[RAGRisk] city-filtered search returned nothing, falling back to global")
            docs = self.search_knowledge(rag_query)

        if not docs:
            return {
                "status": "no_knowledge",
                "risks": {"risks": []}
            }

        knowledge_context = "\n\n".join(
            f"【片段{i + 1}】\n{d['content']}" for i, d in enumerate(docs)
        )

        risks = await self._extract_risks(destination, knowledge_context)

        return {
            "status": "success",
            "risks": risks
        }

    async def _extract_risks(self, destination: str, knowledge_context: str) -> dict:
        """
        调用 LLM，从检索文档中结构化抽取风险/避坑信息。
        对三要素（场景+后果+建议）有明确约束，防止信息损失。
        """
        if not self.model:
            return {"risks": []}

        parser = PydanticOutputParser(pydantic_object=RiskExtraction)

        prompt = (
            f"你是一位旅游避坑专家。请从以下攻略文本中，"
            f"针对【{destination}】，抽取所有'避坑/踩雷/常见误区'相关信息。\n\n"
            f"抽取要求：\n"
            f"1. 每条 risk 必须包含'具体场景 + 潜在后果 + 规避建议'三要素。\n"
            f"   不合格：'注意交通'（过于笼统）\n"
            f"   合格：'西湖周边打车高峰期易堵，若赶班次可能误点，建议提前1小时出发或改乘地铁'\n"
            f"2. 仅抽取攻略中明确提及的避坑信息，不编造知识库外的内容。\n"
            f"3. 若攻略中避坑信息稀少，宁可返回空列表，不凑数。\n\n"
            f"【攻略文本】\n{knowledge_context}\n\n"
            f"{parser.get_format_instructions()}"
        )

        try:
            response = await self.model.ainvoke([
                {"role": "system", "content": "你是旅游避坑信息结构化抽取助手，严格按指定 JSON 格式输出，不输出其他内容。"},
                {"role": "user", "content": prompt}
            ])
            result = parser.parse(response.content)
            return result.model_dump()
        except Exception as e:
            logger.error(f"[RAGRisk] LLM extraction failed: {e}")
            return {"risks": []}
