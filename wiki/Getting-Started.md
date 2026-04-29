# Getting Started

## Prerequisites

- Python 3.10+
- A Doubao LLM API key (from Volcengine / ByteDance)

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Core dependencies include:

| Package | Purpose |
|---------|---------|
| `langgraph` | Graph-based workflow orchestration |
| `langchain-openai` | LLM interface (OpenAI-compatible) |
| `pydantic` | Structured output and data validation |
| `pymilvus[milvus_lite]` | Vector database for RAG |
| `sentence-transformers` | BGE embedding model |
| `rich` | CLI terminal interface |
| `ddgs` | DuckDuckGo web search |
| `mcp` | Model Context Protocol SDK |
| `jieba` | Chinese text segmentation |

### 2. Configure the LLM

Edit `config.py` and set your Doubao API key:

```python
LLM_CONFIG = {
    "api_key": "YOUR_API_KEY_HERE",
    "model_name": "doubao-seed-1-6-lite-251015",
    "base_url": "https://ark.cn-beijing.volces.com/api/v3",
    "temperature": 0.7,
    "max_tokens": 8192,
}
```

### 3. Initialize the RAG Knowledge Base

Before first run, build the Milvus vector index from the bundled travel documents:

```bash
python .claude/skills/ask-question/script/init_knowledge_base.py
```

This creates the Milvus database file at `.claude/skills/ask-question/data/milvus_travel_kb.db`.

### 4. Download the Embedding Model

The BGE-small-zh-v1.5 model must be available at `data/models/bge-small-zh-v1.5/`. If not already present, download it from HuggingFace:

```bash
mkdir -p data/models
# Use huggingface-cli or git clone
huggingface-cli download BAAI/bge-small-zh-v1.5 --local-dir data/models/bge-small-zh-v1.5
```

### 5. Start the CLI

```bash
python cli.py
```

You will be prompted for a user ID (default: `default_user`). The system initializes in approximately 3 seconds thanks to lazy loading.

## Optional: MCP Service Configuration

### Amap (Gaode Maps)

Set your Amap Web API key in `config.py`:

```python
AMAP_MCP_CONFIG = {
    "AMAP_KEY": "YOUR_AMAP_KEY",
    "sse_endpoint": "https://mcp.amap.com/sse",
    "timeout": 30,
}
```

### Flight MCP

The flight API key is embedded in the URL in `config.py` under `FLIGHT_MCP_CONFIG`.

### RollingGo Hotel MCP

Configure the RollingGo hotel MCP server in `config.py` under `ROLLINGGO_MCP_CONFIG`.

## Quick Verification

After starting the CLI, try these commands:

```
help          # Show available commands
health        # Check LLM service connectivity
status        # View current session state
```

Then try a natural language query:

```
I want to travel from Beijing to Hangzhou for 3 days next week.
```
