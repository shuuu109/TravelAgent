# MCP Integrations

Aligo integrates with external services via the **Model Context Protocol (MCP)**. MCP provides a standardized interface for LLM agents to invoke real-world tools.

## Amap (Gaode Maps)

**Client**: `mcp_clients/amap_client.py`

The primary geographic services provider. Connects to the Amap official MCP server via SSE.

### Connection

```python
from mcp_clients.amap_client import amap_mcp_session

async with amap_mcp_session() as session:
    # session is an MCP ClientSession
    tools = await load_mcp_tools(session)
```

The `amap_mcp_session()` context manager establishes an SSE connection to `https://mcp.amap.com/sse` with the configured API key.

### Available Functions

| Function | Description | Used By |
|----------|-------------|---------|
| `search_pois(session, keywords, city)` | Search POIs with coordinate enrichment via detail lookup and geocoding | POIFetchAgent (P2) |
| `get_distance_matrix(session, origins, destinations)` | N×M distance matrix with semaphore-controlled concurrency | Itinerary Planning (P3) |
| `get_transit_route(session, origin, destination)` | Single-pair transit route details | Itinerary Planning (P3) |
| `search_hotels_nearby(session, location, radius)` | Hotel search by coordinates and radius | Accommodation (P4) |
| `search_restaurants_nearby(session, location, radius)` | Restaurant search by coordinates and radius | Itinerary Planning (P3) |

### POI Search Details

`search_pois()` performs a multi-step enrichment:
1. Calls Amap keyword search MCP tool
2. For each result, attempts a detail lookup to get exact coordinates
3. Falls back to geocoding if detail lookup fails
4. Returns enriched POI dicts with `location` (lat,lng), `name`, `rating`, etc.

### Distance Matrix

`get_distance_matrix()` handles large origin-destination pairs by:
- Using an `asyncio.Semaphore` to limit concurrent requests
- Splitting large requests into batches if needed
- Returning a structured matrix of distances and durations

### Configuration

In `config.py`:

```python
AMAP_MCP_CONFIG = {
    "AMAP_KEY": "your_amap_web_api_key",
    "sse_endpoint": "https://mcp.amap.com/sse",
    "timeout": 30,
}
```

---

## Flight MCP

**Configuration**: `FLIGHT_MCP_CONFIG` in `config.py`

Connects to the Variflight aviation MCP server via Streamable HTTP. The API key is embedded in the URL.

```python
FLIGHT_MCP_CONFIG = {
    "url": "https://ai.variflight.com/servers/aviation/mcp/?api_key=...",
}
```

Used by `TransportAgent` to query real-time flight schedules between cities.

---

## RollingGo Hotel MCP

**Configuration**: `ROLLINGGO_MCP_CONFIG` in `config.py`

A local MCP server for hotel search. Launched as a subprocess via `rollinggo-mcp` CLI.

```python
ROLLINGGO_MCP_CONFIG = {
    "ROLLINGGO_API_KEY": "your_api_key",
    "command": "rollinggo-mcp",   # or via npx
    "args": [],
    "timeout": 30,
    "default_size": 5,
    "default_currency": "CNY",
    "default_country": "CN",
}
```

Used by the `accommodation_node` (P4) to search hotels near the arrival transport hub.

---

## MCP in the Validation Node

The `validate_rule_constraints` node (P1.5) uses Amap MCP tools differently — it creates a **ReAct sub-agent** with all Amap tools available, then lets the LLM autonomously decide which tools to call for distance and time validation:

```python
async with amap_mcp_session() as session:
    amap_tools = await load_mcp_tools(session)
    validator_agent = create_react_agent(llm, amap_tools)
    response = await validator_agent.ainvoke({"messages": [...]})
```

This agentic approach allows flexible tool usage without hardcoding the validation logic.
