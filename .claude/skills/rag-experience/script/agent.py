"""
RAG 经验查询智能体 RAGExperienceAgent
=====================================
职责：检索旅行风格相关的经验和建议，经结构化 LLM 抽取后返回 ExperienceOutput 格式数据。

与旧版 RAGKnowledgeAgent 的区别：
  1. 查询语义聚焦于"经验/建议"，召回经验类内容更准确
  2. LLM 抽取使用 PydanticOutputParser，约束输出为结构化字段，避免信息损失
  3. 同时返回 retrieved_documents（供 itinerary_planning_node 做 POI 权重偏移）
     和 experience（供 respond_node 渲染"旅行小贴士"区块）
"""
import os
import sys
import logging
from typing import List

# 确保项目根目录在 sys.path 中，使 agents/config 等包可正常导入
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser

from agents.rag_base_agent import RAGBaseAgent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 结构化抽取目标模型
# 字段定义与 graph/state.py 中 ExperienceOutput 保持一致，
# 此处独立定义以保持技能插件的自包含性（不引入 graph 层依赖）
# ---------------------------------------------------------------------------
class ExperienceExtraction(BaseModel):
    tips: List[str] = Field(
        default_factory=list,
        description=(
            "可操作的旅行建议列表。"
            "每条必须保留原文的具体细节，例如景点名称、时间段、票价等，"
            "禁止压缩为泛泛表达（如'注意门票'是不合格的）。"
        )
    )
    best_for: List[str] = Field(
        default_factory=list,
        description=(
            "该目的地特别适合当前旅行风格的理由，1-3 条。"
            "需说明具体原因，例如适合情侣的哪类场景、亲子游的哪类设施等。"
        )
    )


class RAGExperienceAgent(RAGBaseAgent):
    """
    RAG 经验查询智能体。

    查询语义："{destination} {travel_style} 旅游攻略 经验 建议"
    LLM 抽取：从检索片段中提炼 tips（可操作建议）和 best_for（风格适配理由）
    输出：
      - retrieved_documents: 原始检索片段（供 itinerary_planning_node 做 POI 权重偏移）
      - experience: {"tips": [...], "best_for": [...]}（供 respond_node 渲染小贴士）
    """

    async def run(self, input_data: dict) -> dict:
        if not self.initialized:
            return {"status": "error", "message": "RAGExperienceAgent not initialized"}

        context: dict = input_data.get("context", {})
        destination: str = (context.get("key_entities") or {}).get("destination", "") or ""
        travel_style: str = context.get("travel_style", "") or ""

        # 构造经验检索 query：在通用攻略基础上加"经验 建议"语义权重
        style_part = travel_style if travel_style and travel_style != "普通" else ""
        rag_query = " ".join(filter(None, [destination, style_part, "旅游攻略 经验 建议"]))
        logger.info(f"[RAGExperience] query: {rag_query!r}")

        # 先按城市 + 章节精准过滤（排除 city_overview 的串台干扰）
        # 经验建议主要来自"核心景点"章节，也可能出现在"天气与最佳旅游时间"中
        # 此处以宽松的 city_filter 为主，不限制 section，让语义相似度决定召回
        docs = self.search_knowledge(
            rag_query,
            city_filter=destination or None,
            section_filter=None,   # 经验类涉及多个章节，不做 section 限制
        )
        if not docs and destination:
            logger.warning("[RAGExperience] city-filtered search returned nothing, falling back to global")
            docs = self.search_knowledge(rag_query)

        if not docs:
            return {
                "status": "no_knowledge",
                "retrieved_documents": [],
                "experience": {"tips": [], "best_for": []}
            }

        # 构建知识上下文传给 LLM
        knowledge_context = "\n\n".join(
            f"【片段{i + 1}】\n{d['content']}" for i, d in enumerate(docs)
        )

        experience = await self._extract_experience(destination, travel_style, knowledge_context)

        return {
            "status": "success",
            # 截断过长内容，保持 state 紧凑；itinerary_planning_node 只需关键词级别的信息
            "retrieved_documents": [
                {
                    "content": d["content"][:200] + "..." if len(d["content"]) > 200 else d["content"],
                    "metadata": d["metadata"]
                }
                for d in docs
            ],
            "experience": experience
        }

    async def _extract_experience(
        self,
        destination: str,
        travel_style: str,
        knowledge_context: str
    ) -> dict:
        """
        调用 LLM，从检索文档中结构化抽取经验建议。
        使用 PydanticOutputParser 约束输出，避免 LLM 自由发挥导致细节丢失。
        """
        if not self.model:
            # 无 LLM 时直接返回空结构，不中断流程
            return {"tips": [], "best_for": []}

        parser = PydanticOutputParser(pydantic_object=ExperienceExtraction)
        style_desc = f"旅行风格：{travel_style}。" if travel_style and travel_style != "普通" else ""

        prompt = (
            f"你是一位旅游攻略专家。请从以下攻略文本中，"
            f"针对【{destination}】的旅行规划，{style_desc}"
            f"抽取结构化的经验建议。\n\n"
            f"抽取要求：\n"
            f"1. tips 每条必须保留原文的具体细节（景点名称、时间段、金额、操作步骤等），"
            f"禁止压缩为泛泛建议，例如不合格：'注意门票'，合格：'灵隐寺需先买飞来峰票(45元)再买香花券，路边带路者均为黑导游'。\n"
            f"2. tips 每条禁止以数字序号（如'1.'、'2.'、'①'）开头，直接输出建议内容本身。\n"
            f"3. tips 中严禁抽取住宿推荐（酒店/民宿/区域住宿建议）和大交通信息（高铁/飞机/去程/返程）；"
            f"这两类由专门模块处理，此处抽取会导致重复输出。\n"
            f"4. best_for 说明该目的地特别适合该旅行风格的具体场景或理由，1-3 条即可。\n"
            f"5. 仅基于以下攻略内容抽取，不编造知识库外的信息。\n\n"
            f"【攻略文本】\n{knowledge_context}\n\n"
            f"{parser.get_format_instructions()}"
        )

        try:
            response = await self.model.ainvoke([
                {"role": "system", "content": "你是旅游攻略结构化抽取助手，严格按指定 JSON 格式输出，不输出其他内容。"},
                {"role": "user", "content": prompt}
            ])
            result = parser.parse(response.content)
            return result.model_dump()
        except Exception as e:
            logger.error(f"[RAGExperience] LLM extraction failed: {e}")
            return {"tips": [], "best_for": []}
