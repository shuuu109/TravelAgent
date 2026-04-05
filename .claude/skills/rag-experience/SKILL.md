---
name: rag-experience
description: 检索目的地旅行经验与建议，结构化抽取可操作 tips 和旅行风格适配理由，输出供 respond_node 渲染"旅行小贴士"区块。当用户有明确旅行规划意图时调度，通常与 rag_risk 同时触发。
---

# rag-experience（RAG 经验建议查询）

从知识库中检索目的地旅行经验，使用 **RAGExperienceAgent** 经结构化 LLM 抽取后，
输出 `ExperienceOutput` 格式的建议数据。

## 触发场景

- 用户有明确的旅行规划意图（包含目的地 + 出行计划）
- 通常与 `rag_risk` 同时调度，优先级相同（并行执行）

## 返回格式

```json
{
  "status": "success",
  "retrieved_documents": [{"content": "...", "metadata": {...}}],
  "experience": {
    "tips": ["灵隐寺需先买飞来峰票再买香花券，路边带路者均为黑导游", "..."],
    "best_for": ["情侣适合在断桥残雪拍照打卡，人少时段为工作日上午9点前", "..."]
  }
}
```
