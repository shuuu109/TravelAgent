# Aligo Travel Agent Wiki

Aligo is an intelligent travel planning assistant built on the **Doubao LLM** and **LangGraph** framework. It uses a multi-agent architecture with Plan-and-Execute scheduling to deliver personalized travel itineraries through natural language conversation.

## Wiki Contents

| Page | Description |
|------|-------------|
| [Architecture Overview](Architecture-Overview.md) | System architecture, LangGraph workflow, and data flow |
| [Getting Started](Getting-Started.md) | Installation, configuration, and first run |
| [Graph Nodes](Graph-Nodes.md) | Detailed reference for every LangGraph node |
| [Skill Plugins](Skill-Plugins.md) | Plugin architecture and available skill agents |
| [Memory System](Memory-System.md) | Two-layer memory: short-term and long-term |
| [State Reference](State-Reference.md) | `TravelGraphState` fields and data models |
| [MCP Integrations](MCP-Integrations.md) | Amap, flight, hotel, and train MCP clients |
| [Resilience](Resilience.md) | Circuit breaker, retry with backoff, and health checks |
| [CLI Guide](CLI-Guide.md) | Interactive CLI commands and usage |
| [Testing](Testing.md) | Test suite overview and how to run tests |

## Key Features

- **LLM-based intent recognition** with 90%+ accuracy across 6 intent categories
- **Two-layer memory** (short-term sliding window + long-term JSON persistence)
- **RAG knowledge base** using Milvus + BGE-small-zh-v1.5 embeddings
- **Priority-based parallel scheduling** via `asyncio.gather`
- **Skill plugin architecture** with lazy loading and dynamic discovery
- **MCP tool integration** for real-time map, flight, hotel, and train data
- **Circuit breaker + exponential backoff** for LLM service resilience
