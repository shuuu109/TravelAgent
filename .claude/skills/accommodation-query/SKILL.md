---
name: accommodation-query
description: Use this skill when the user needs hotel/accommodation recommendations for a trip. Triggers when user asks "住哪里", "推荐酒店", "住宿怎么安排", or when a trip itinerary is being planned and accommodation needs to be assigned per day. This skill uses AccommodationAgent and applies smart hub-proximity logic for arrival/departure days.
---

# Accommodation Query（住宿智能推荐）

为用户规划每日住宿，核心逻辑：**到达/离开日靠近交通枢纽，中间日靠近当日游玩区域**。调用 **AccommodationAgent** + RollingGo MCP 获取真实酒店数据，LLM 二次分析生成结构化推荐。

## When to Use

- 用户问「住哪里」「推荐酒店」「住宿怎么安排」
- plan-trip 行程规划后需要补充每晚住宿安排
- 用户提供了到达/离开时间，需要按枢纽邻近原则分配住宿

---

## 核心选址逻辑

### 第一天（到达日）
| 到达时间 | 住宿选址 |
|----------|----------|
| **下午或晚上到达** | 住在**到达交通枢纽附近**（火车站/机场周边），减少拖行李奔波 |
| **早晨到达** | 无需靠近枢纽，直接住在**第一天游玩区域附近** |

### 最后一天（离开日）前一晚
| 离开时间 | 住宿选址 |
|----------|----------|
| **早晨出发（早班车/早班机）** | 最后一晚住在**出发交通枢纽附近**，方便次日早出 |
| **晚上或下午出发** | 无需靠近枢纽，住在**最后一天游玩区域附近**，出发前继续游玩 |

### 中间天数
- 住宿选址优先考虑**当天游玩的大区域**
- 优选**靠近地铁站**的位置，方便全天出行
- 如当天游玩景点分散，选居中地带或次日景点附近

---

## Agent

- **AccommodationAgent** (`agents/accommodation_agent.py`)
- 依赖 **RollingGo MCP** 查询真实酒店数据（不可用时降级为 LLM 推荐）
- 异步：`run()` 为 `async`，需 `await`

---

## 调用示例

```python
import asyncio
from agents.accommodation_agent import AccommodationAgent
from config import LLM_CONFIG
from langchain_openai import ChatOpenAI

async def query_accommodation(trip_info: dict):
    model = ChatOpenAI(
        model=LLM_CONFIG["model_name"],
        api_key=LLM_CONFIG["api_key"],
        base_url=LLM_CONFIG["base_url"],
        temperature=LLM_CONFIG.get("temperature", 0.7),
    )
    agent = AccommodationAgent(name="AccommodationAgent", model=model)

    input_data = {
        "context": {
            "key_entities": {
                "destination": trip_info["destination"],
                "date": trip_info["check_in_date"],      # YYYY-MM-DD
                "duration": trip_info["duration"],        # 如 "3天"
                "adults": trip_info.get("adults", 1),
            },
            "transport_recommendation": {
                "arrival_hub": trip_info.get("arrival_station", ""),   # 如 "北京南站"
            },
            "user_preferences": {
                "hotel_brands": trip_info.get("hotel_brands", []),     # 如 ["汉庭", "如家"]
                "budget_level": trip_info.get("budget_level", "舒适"), # 经济/舒适/高端
            },
            # 住宿选址策略标志（由调用方传入）
            "accommodation_strategy": trip_info.get("accommodation_strategy", {}),
        },
        "previous_results": trip_info.get("previous_results", []),
    }

    result = await agent.run(input_data)
    return result

# 使用示例
result = asyncio.run(query_accommodation({
    "destination": "北京",
    "check_in_date": "2026-05-01",
    "duration": "3天",
    "adults": 2,
    "arrival_station": "北京南站",
    "hotel_brands": ["汉庭"],
    "budget_level": "舒适",
    "accommodation_strategy": {
        "day_1": "hub_proximity",    # 下午到达，靠近枢纽
        "day_2": "activity_area",    # 中间日，靠近游玩区
        "day_3": "hub_proximity",    # 早晨离开，最后一晚靠近枢纽
    }
}))
```

---

## accommodation_strategy 字段说明

`accommodation_strategy` 由上游（orchestrate_node 或用户输入解析）传入，指导 AccommodationAgent 每晚的选址重点：

```json
{
    "day_1": "hub_proximity",     // 靠近到达交通枢纽
    "day_2": "activity_area",     // 靠近当日游玩区域（中间日默认值）
    "day_3": "hub_proximity"      // 靠近出发交通枢纽（早班离开）
}
```

| 策略值 | 含义 |
|--------|------|
| `hub_proximity` | 靠近交通枢纽（火车站/机场），适合到达当晚或早班离开前一晚 |
| `activity_area` | 靠近当天游玩的大区域，优选地铁沿线 |
| `transit_friendly` | 无特定枢纽要求，但强调交通便利（备用值） |

---

## 返回格式

```json
{
    "accommodation_plan": {
        "destination": "北京",
        "arrival_station": "北京南站",
        "mcp_data_used": true,
        "analysis": "第1晚靠近北京南站，减少到达当晚奔波；第2晚住东城区，方便游览故宫和胡同；第3晚提前返回南站附近，保障次日早班高铁。",
        "recommended_areas": [
            {
                "area_name": "北京南站周边",
                "reason": "下午/晚上到达后无需长途转移",
                "distance_to_station": "步行10分钟或地铁1站"
            },
            {
                "area_name": "东城区/王府井",
                "reason": "中间日游玩故宫、南锣鼓巷的核心区域，地铁2/5号线覆盖",
                "distance_to_station": "靠近地铁站"
            }
        ],
        "options": [
            {
                "tier": "舒适型",
                "hotel_name": "汉庭酒店（北京南站店）",
                "hotel_id": "12345",
                "area": "丰台区·北京南站",
                "price_range": "280-350元/晚",
                "star": "3星",
                "highlights": "含早餐，地铁4号线马家堡站步行5分钟",
                "distance_info": "距北京南站800米",
                "pros": "性价比高，交通便利",
                "cons": "周边餐饮选择较少"
            }
        ],
        "recommendation": {
            "best_choice": "汉庭酒店（北京南站店）第1晚 + 全季酒店（王府井店）第2晚",
            "reason": "到达日靠近枢纽免于奔波，中间日住景区核心提升游览效率",
            "booking_tips": "建议提前3天预订，五一节假日价格上浮约30%"
        }
    },
    "mcp_hotels_count": 5
}
```

---

## 选址决策流程

```
输入：arrival_time, departure_time, daily_itinerary
        ↓
判断第1天到达时间
  ├─ 下午/晚上 → day_1 strategy = hub_proximity（到达枢纽附近）
  └─ 早晨      → day_1 strategy = activity_area（第1天景点附近）
        ↓
中间天（day_2 … day_N-1）
  → strategy = activity_area（当天游玩区域 + 靠近地铁）
        ↓
判断最后一天离开时间
  ├─ 早晨出发  → day_N strategy = hub_proximity（出发枢纽附近）
  └─ 下午/晚上 → day_N strategy = activity_area（最后一天景点附近）
        ↓
AccommodationAgent.run() × N晚
```

---

## 与 plan-trip 的协作

`accommodation-query` 通常作为 plan-trip 的后置步骤，或由 `orchestrate_node` 统一调度：

1. **plan-trip** 生成 `daily_plans`（每日景点、城市、主题）
2. **orchestrate_node** 解析到达/离开时间，生成 `accommodation_strategy`
3. **accommodation-query** 依据策略逐晚调用 AccommodationAgent，填充每晚住宿方案

前序 `transport_query` 结果（`arrival_hub`、`arrival_station`）会自动透传给 AccommodationAgent，用于枢纽邻近判断。

---

## Prompt 指南

【住宿选址原则】
1. **到达当晚靠近交通枢纽**：用户拖着行李刚到，减少二次转移成本
2. **早班离开前一晚靠近枢纽**：避免早晨赶车时的交通压力
3. **中间日靠近游玩区域**：住在当天主要景点的大区域，优先地铁沿线
4. **地铁优先**：中间日住宿必须注明最近地铁站及步行距离
5. **连续性**：若连续两天在同一大区域游玩，同一酒店住两晚更佳（减少搬行李）

【特殊情况处理】
- 若到达和离开都在同一天（一日游）：不需要住宿推荐，告知用户
- 若行程信息不完整（缺少到达时间）：默认按下午/晚上到达处理，在建议中注明假设
- 若目的地只有一个交通枢纽：到达日和离开日前一晚可推荐同一区域的酒店

【禁止事项】
- 不得在无真实数据时虚构具体酒店名称和价格（无 MCP 数据时需注明"价格为估算"）
- 不得忽视用户的品牌偏好和预算等级
